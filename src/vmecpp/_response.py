# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The first-order response of a converged equilibrium to a change of its input.

At the converged state ``x0`` of an input ``p`` the force ``F(x0, p)`` vanishes. For the
input ``p + dp`` the equilibrium moves to ``x0 + dx`` with ``J dx = -(dF/dp) dp`` to
first order, ``J`` the Jacobian of the force with respect to the coordinates of the
state the solver evolves. ``J`` is assembled column by column from the central
differences of the exact force that :meth:`VmecModel.hessian_vector_product` takes,
and factorized once; each perturbation then costs one force derivative and one solve.
A change of the boundary enters through the fixed boundary coordinates of the state,
which the hot restart of the perturbed input sets; a change of the coil currents, the
toroidal flux, the pressure or the profiles enters through the force at the unchanged
state. The response comes back as the derivative of the geometry in the layout of the
``wout`` file, with the displacement of every surface along its normal, which the
freedom to relabel the poloidal angle leaves alone.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import scipy.linalg

from vmecpp import geometry as _geometry
from vmecpp._continuation import _state_mode_table
from vmecpp._pydantic_numpy import own_model_fields
from vmecpp.cpp import _vmecpp  # type: ignore

if typing.TYPE_CHECKING:
    from vmecpp import VmecInput, VmecOutput

# The geometry of the input, which the state carries; everything else acts on the
# force at a fixed state.
_BOUNDARY_FIELDS = frozenset(
    {"rbc", "zbs", "rbs", "zbc", "raxis_c", "zaxis_s", "raxis_s", "zaxis_c"}
)
# Resolution and run control, which a perturbation may not change.
_FIXED_FIELDS = frozenset(
    {
        "lasym",
        "nfp",
        "mpol",
        "ntor",
        "mpol_geometry",
        "ntor_geometry",
        "vacuum_mpol",
        "vacuum_ntor",
        "ntheta",
        "nzeta",
        "ns_array",
        "ftol_array",
        "niter_array",
        "lfreeb",
        "mgrid_file",
        "nstep",
        "nvacskip",
        "delt",
        "tcon0",
        "lforbal",
        "lbsubs",
        "free_boundary_method",
        "iteration_style",
        "return_outputs_even_if_not_converged",
        "always_fix_m1_gauge",
        "lambda_preconditioner_scale",
        "lambda_precondition_checkerboard_terms",
        "aphi",
        "signgs",
    }
)
_PRODUCT_BASIS = ("r_cc", "r_ss", "r_sc", "r_cs", "z_sc", "z_cs", "z_cc", "z_ss")
_LAMBDA_BASIS = ("lambda_sc", "lambda_cs", "lambda_cc", "lambda_ss")


def _product_basis(model) -> dict[str, np.ndarray]:
    geometry = _geometry.from_cpp(model.get_geometry())
    return {
        name: np.array(getattr(geometry, name), dtype=float)
        for name in (*_PRODUCT_BASIS, *_LAMBDA_BASIS)
    }


def _combined(
    xm: np.ndarray,
    xn: np.ndarray,
    nfp: int,
    first: np.ndarray,
    second: np.ndarray,
    *,
    cosine: bool,
) -> np.ndarray:
    """Product basis to the combined basis in the ``(mn, ns)`` layout of the wout.

    ``cosine`` reads ``first, second`` as the ``cc, ss`` coefficients of a cos(mu - nv)
    series, otherwise as the ``sc, cs`` coefficients of a sin(mu - nv) series; the
    rules are those of the output stage.
    """
    n = xn // nfp
    a = first[:, xm, np.abs(n)].T
    b = second[:, xm, np.abs(n)].T
    sign_b = 1.0 if cosine else -1.0
    m0_value = a if cosine else -b
    combined = np.where(
        (xm == 0)[:, None],
        m0_value,
        np.where((n == 0)[:, None], a, 0.5 * (a + sign_b * np.sign(n)[:, None] * b)),
    )
    ns = first.shape[0]
    axis_mask = ((xm > 0) & (n != 0))[:, None] & (np.arange(ns) == 0)[None, :]
    return np.where(axis_mask, 0.0, combined)


def _real_space(
    g: dict[str, np.ndarray], nfp: int, theta: np.ndarray, zeta: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """R, Z and their theta and zeta derivatives on every surface, ``[ns, ntheta,
    nzeta]``."""
    ns, mpol, ntor1 = g["r_cc"].shape
    m = np.arange(mpol)[:, None, None, None]
    n = (np.arange(ntor1) * nfp)[None, :, None, None]
    th = theta[None, None, :, None]
    ze = zeta[None, None, None, :]
    cm, sm = np.cos(m * th), np.sin(m * th)
    cn, sn = np.cos(n * ze), np.sin(n * ze)
    bases = {
        "cc": (cm * cn, -m * sm * cn, -n * cm * sn),
        "ss": (sm * sn, m * cm * sn, n * sm * cn),
        "sc": (sm * cn, m * cm * cn, -n * sm * sn),
        "cs": (cm * sn, -m * sm * sn, n * cm * cn),
    }

    def series(prefix):
        out = [np.zeros((ns, theta.size, zeta.size)) for _ in range(3)]
        for suffix, basis in bases.items():
            coefficients = g[f"{prefix}_{suffix}"]
            if np.any(coefficients):
                for i in range(3):
                    out[i] += np.einsum("jmn,mntz->jtz", coefficients, basis[i])
        return out

    return series("r"), series("z")


@dataclasses.dataclass(frozen=True)
class GeometryResponse:
    """The derivative of the geometry of an equilibrium with respect to the amplitude of
    a change of its input, in the layout of :class:`vmecpp.VmecWOut`.

    ``rmnc``, ``zmns`` and ``lmns_full`` hold ``d(coefficient)/d(amplitude)`` per
    mode and surface, and ``rmns``, ``zmnc``, ``lmnc_full`` the non-stellarator-
    symmetric parts, ``None`` unless the run has ``lasym``. The amplitude is 1 for
    the change the response was computed for, so the perturbed equilibrium is the base
    one plus ``amplitude`` times these arrays, to first order.
    """

    xm: np.ndarray
    xn: np.ndarray
    rmnc: np.ndarray
    zmns: np.ndarray
    lmns_full: np.ndarray
    rmns: np.ndarray | None
    zmnc: np.ndarray | None
    lmnc_full: np.ndarray | None
    nfp: int
    state: np.ndarray | None = dataclasses.field(repr=False)
    """The change of the solver's state per unit amplitude, in the internal layout of
    :meth:`VmecModel.get_state`; ``None`` for a difference of two outputs."""
    _base: dict[str, np.ndarray] = dataclasses.field(repr=False)
    _change: dict[str, np.ndarray] = dataclasses.field(repr=False)

    def mode(self, m: int, n: int) -> int:
        """The index of the mode ``cos(m theta - n nfp zeta)`` in the arrays."""
        matches = np.flatnonzero((self.xm == m) & (self.xn == n * self.nfp))
        if matches.size != 1:
            msg = f"no mode (m, n) = ({m}, {n}) in the spectrum"
            raise ValueError(msg)
        return int(matches[0])

    def normal_displacement(self, theta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """The displacement of every surface along its unit normal per unit amplitude,
        at the points ``(theta, zeta)``, as ``[ns, ntheta, nzeta]``.

        A relabeling of the poloidal angle leaves it unchanged.
        """
        theta = np.asarray(theta, dtype=float)
        zeta = np.asarray(zeta, dtype=float)
        (r, r_t, r_z), (_, z_t, z_z) = _real_space(self._base, self.nfp, theta, zeta)
        (dr, _, _), (dz, _, _) = _real_space(self._change, self.nfp, theta, zeta)
        cz, sz = np.cos(zeta)[None, None, :], np.sin(zeta)[None, None, :]
        x_t = np.stack([r_t * cz, r_t * sz, z_t])
        x_z = np.stack([r_z * cz - r * sz, r_z * sz + r * cz, z_z])
        normal = np.cross(x_t, x_z, axis=0)
        normal /= np.maximum(np.linalg.norm(normal, axis=0), 1.0e-300)
        return np.sum(np.stack([dr * cz, dr * sz, dz]) * normal, axis=0)


class Linearization:
    """The force Jacobian of a converged equilibrium, for its first-order response to
    changes of the input.

    ``output`` is a converged :class:`vmecpp.VmecOutput` whose input has a single
    radial resolution. Assembling the Jacobian costs two force evaluations per evolved
    coordinate, of up to ``6 ns mpol (ntor + 1)`` coordinates and twice as many with
    ``lasym``, with a vacuum solve in each for a free-boundary run; the dense
    factorization follows. ``difference_step`` is the relative step of the central
    differences the Jacobian is built from. The m = 1 gauge of the state stays where
    the run left it, as it does in a run with ``always_fix_m1_gauge``.
    """

    def __init__(self, output: VmecOutput, *, difference_step: float = 1.0e-7):
        import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

        self.input = vmecpp.VmecInput.model_validate(output.input)
        if len(self.input.ns_array) != 1:
            msg = "Linearization needs an input with a single ns_array entry"
            raise ValueError(msg)
        if not isinstance(self.input.mpol, int) or not isinstance(self.input.ntor, int):
            msg = "Linearization needs an input without Fourier continuation"
            raise ValueError(msg)
        self.ns = int(output.wout.ns)
        self.output = output
        self.difference_step = difference_step
        self._model = self._model_at(self.input, output)
        self._x0 = np.asarray(self._model.get_state()).copy()
        self.state = self._x0.copy()
        """The state of the equilibrium, in the layout of
        :meth:`VmecModel.get_state`."""
        self._settle(self._model)
        # The force on a coordinate the solver does not evolve, the boundary of a
        # fixed-boundary run, the axis above m = 0, the pinned m = 1 gauge and the modes
        # that vanish identically, is zero whatever the state; a generic direction finds
        # every other row of the Jacobian.
        generic = np.random.default_rng(0).standard_normal(self._x0.size)
        self.free = np.flatnonzero(
            np.asarray(self._model.hessian_vector_product(generic, difference_step))
            != 0.0
        )
        """The indices of the coordinates of the state the solver evolves."""
        self._model.set_state(self._x0)
        self._model.evaluate(2, 2, False)
        jacobian = np.empty((self.free.size, self.free.size))
        for column, k in enumerate(self.free):
            unit = np.zeros_like(self._x0)
            unit[k] = 1.0
            jacobian[:, column] = np.asarray(
                self._model.hessian_vector_product(unit, difference_step)
            )[self.free]
        self.jacobian = jacobian
        """The force Jacobian on the evolved coordinates, ``[free, free]``."""
        self._lu = scipy.linalg.lu_factor(jacobian)
        self._base_geometry = _product_basis(self._model)
        self.xm, self.xn = _state_mode_table(
            int(self.input.mpol), int(self.input.ntor), int(self.input.nfp)
        )

    def _model_at(self, vmec_input: VmecInput, restart: VmecOutput):
        """A model of ``vmec_input`` hot-started from the converged ``restart``."""
        initial_state = _vmecpp.HotRestartState(
            wout=restart.wout._to_cpp_wout(),
            indata=restart.input._to_cpp_vmecindata(),
        )
        return _vmecpp.VmecModel.create(
            vmec_input._to_cpp_vmecindata(), self.ns, initial_state
        )

    @staticmethod
    def _settle(model, max_evaluations: int = 1000) -> None:
        """Evaluate the force at the model's state until it is stationary.

        A free-boundary model switches its vacuum pressure on over the first evaluations
        and then decays the offsets of the spectral constraint by a tenth per
        evaluation, which the converged run has done long ago.
        """
        model.evaluate(2, 2, False)
        previous = np.asarray(model.get_forces()).copy()
        for _ in range(max_evaluations):
            model.evaluate(2, 2, False)
            current = np.asarray(model.get_forces()).copy()
            if np.abs(current - previous).max() <= 1.0e-13 * np.abs(current).max():
                return
            previous = current
        msg = "the force did not become stationary at the converged state"
        raise RuntimeError(msg)

    def _differing_fields(self, perturbed: VmecInput) -> tuple[list[str], list[str]]:
        """The geometry fields and the other fields that differ from the input, after
        checking that no fixed field does."""
        import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

        geometry, explicit = [], []
        for name in own_model_fields(vmecpp.VmecInput):
            base, other = getattr(self.input, name), getattr(perturbed, name)
            if base is None or other is None:
                same = base is other
            else:
                same = np.array_equal(np.asarray(base), np.asarray(other))
            if same:
                continue
            if name in _FIXED_FIELDS:
                msg = (
                    f"the perturbed input changes '{name}', which the response cannot "
                    "follow"
                )
                raise ValueError(msg)
            (geometry if name in _BOUNDARY_FIELDS else explicit).append(name)
        return geometry, explicit

    def _interpolated(
        self, perturbed: VmecInput, fraction: float, fields: list[str]
    ) -> VmecInput:
        """The input at ``fraction`` of the way from the input to ``perturbed`` in the
        given fields."""
        update = {}
        for name in fields:
            base = np.asarray(getattr(self.input, name), dtype=float)
            other = np.asarray(getattr(perturbed, name), dtype=float)
            value = base + fraction * (other - base)
            update[name] = value if base.ndim else float(value)
        return self.input.model_copy(update=update)

    def _forces_at(self, vmec_input: VmecInput, state: np.ndarray) -> np.ndarray:
        model = self._model_at(vmec_input, self.output)
        model.set_state(state)
        self._settle(model)
        return np.asarray(model.get_forces()).copy()

    def _explicit_force_derivative(
        self, perturbed: VmecInput, fields: list[str], step: float
    ) -> np.ndarray:
        """``dF/d(amplitude)`` at the fixed state ``x0``, by central differences over
        the non-geometry fields that differ."""
        plus = self._forces_at(self._interpolated(perturbed, step, fields), self._x0)
        minus = self._forces_at(self._interpolated(perturbed, -step, fields), self._x0)
        return (plus - minus) / (2.0 * step)

    def response_to(
        self, perturbed: VmecInput, *, step: float = 1.0e-4
    ) -> GeometryResponse:
        """The first-order response to the change from the input of the equilibrium to
        ``perturbed``, per unit amplitude of that change.

        ``perturbed`` may differ in the boundary and axis coefficients, the coil
        currents ``extcur``, ``phiedge``, ``pres_scale``, ``curtor`` and the profile
        coefficients; ``step`` is the fraction of the change over which the fixed
        coordinates of the state and the force are differenced.
        """
        import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

        perturbed = vmecpp.VmecInput.model_validate(perturbed)
        geometry, explicit = self._differing_fields(perturbed)
        # The restart of an input a fraction ``step`` of the way to ``perturbed`` sets
        # the fixed coordinates of the state, the boundary and the gauge pinned to it;
        # the fraction keeps the conversion of the input, which flips the poloidal angle
        # when the m = 1 coefficients call for it, in its linear range. The interior of
        # that state is an initial guess, which the linear system does not need.
        v = np.zeros_like(self._x0)
        rhs = np.zeros_like(self._x0)
        if geometry:
            shifted = self._model_at(
                self._interpolated(perturbed, step, geometry), self.output
            )
            v = (np.asarray(shifted.get_state()) - self._x0) / step
            v[self.free] = 0.0
            self._model.set_state(self._x0)
            self._model.evaluate(2, 2, False)
            rhs = np.asarray(
                self._model.hessian_vector_product(v, self.difference_step)
            )
        if explicit:
            rhs = rhs + self._explicit_force_derivative(perturbed, explicit, step)
        delta = v.copy()
        delta[self.free] -= scipy.linalg.lu_solve(self._lu, rhs[self.free])
        self._model.set_state(self._x0 + 1.0e-3 * delta)
        moved = _product_basis(self._model)
        self._model.set_state(self._x0)
        change = {k: (moved[k] - self._base_geometry[k]) / 1.0e-3 for k in moved}
        return self._geometry_response(change, delta)

    def force_of(
        self, vmec_input: VmecInput, state: np.ndarray | None = None
    ) -> np.ndarray:
        """The force of ``vmec_input`` at ``state``, the state of the equilibrium by
        default, in the layout of :meth:`VmecModel.get_forces`.

        The input has to agree with the equilibrium's in the fixed fields; its boundary
        enters only through the state.
        """
        import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

        self._differing_fields(vmecpp.VmecInput.model_validate(vmec_input))
        return self._forces_at(
            vmec_input, self._x0 if state is None else np.asarray(state)
        )

    def predicted_force(self, perturbed: VmecInput, amplitude: float) -> np.ndarray:
        """The force of the input ``amplitude`` of the way from the input of the
        equilibrium to ``perturbed``, at the state the response predicts, in the layout
        of :meth:`VmecModel.get_forces`.

        It is second order in ``amplitude`` where the force at the unmoved state is
        first order, which checks the linearization without another solve.
        """
        response = self.response_to(perturbed)
        geometry, explicit = self._differing_fields(perturbed)
        moved = self._interpolated(perturbed, amplitude, geometry + explicit)
        assert response.state is not None
        return self._forces_at(moved, self._x0 + amplitude * response.state)

    def geometry_difference(self, other: VmecOutput) -> GeometryResponse:
        """The geometry of another converged output minus the equilibrium's, in the
        layout of a response of amplitude 1: the secant that a response approximates
        when ``other`` solves a perturbed input."""
        model = self._model_at(other.input, other)
        change = {
            k: v - self._base_geometry[k] for k, v in _product_basis(model).items()
        }
        return self._geometry_response(change, None)

    def _geometry_response(
        self, change: dict[str, np.ndarray], state: np.ndarray | None
    ) -> GeometryResponse:
        nfp = int(self.input.nfp)
        lasym = bool(self.input.lasym)

        def combine(a, b, cosine):
            return _combined(self.xm, self.xn, nfp, change[a], change[b], cosine=cosine)

        return GeometryResponse(
            xm=self.xm,
            xn=self.xn,
            rmnc=combine("r_cc", "r_ss", True),
            zmns=combine("z_sc", "z_cs", False),
            lmns_full=combine("lambda_sc", "lambda_cs", False),
            rmns=combine("r_sc", "r_cs", False) if lasym else None,
            zmnc=combine("z_cc", "z_ss", True) if lasym else None,
            lmnc_full=combine("lambda_cc", "lambda_ss", True) if lasym else None,
            nfp=nfp,
            state=state,
            _base=self._base_geometry,
            _change=change,
        )


__all__ = ["GeometryResponse", "Linearization"]
