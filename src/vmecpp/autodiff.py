"""JAX access to an in-memory VMEC++ solve and its implicit VJP.

The solver is deliberately kept outside the JAX trace. A forward call runs
VMEC++ through the C++ model, while the reverse callback reruns the same model
and solves the transposed interior force system. This is the usual implicit
layer for a differentiable code: JAX differentiates the consumer objective,
and VMEC++ supplies the producer's residual transpose.

The solve pins the m=1 poloidal-origin gauge (``VmecModel.always_fix_m1_gauge``):
the native iteration lets that gauge drift under its force until ``fsqz`` drops
below 1e-6 and freezes it wherever it is, so the converged state would depend on
the iteration history and the fixed-gauge force Jacobian would not describe it.
With the gauge pinned, the gauge entries of the state are a linear function of
the boundary, and the VJP pulls them back to the boundary like the boundary
entries themselves.

The first public parameterization is the fixed-boundary case, with either a
prescribed iota or a prescribed toroidal current profile (``ncurr``). The
differentiable parameter is one dense array with rows ``rbc`` and ``zbs`` and
shape ``(2, mpol, 2 * ntor + 1)``. This first solver wrapper deliberately
supports the stellarator-symmetric fixed-boundary case. The geometry API
itself already supports asymmetric snapshots. Power-series profile parameters
are differentiable as well; free-boundary parameters are not.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from vmecpp import autodiff_wout, geometry
from vmecpp.cpp import _vmecpp  # type: ignore

_GEOMETRY_COEFFICIENTS = (
    "r_cc",
    "r_ss",
    "r_sc",
    "r_cs",
    "z_sc",
    "z_cs",
    "z_cc",
    "z_ss",
    "lambda_sc",
    "lambda_cs",
    "lambda_cc",
    "lambda_ss",
)


def _cpp_geometry_flat(value, ns: int, mpol: int, ntor: int) -> np.ndarray:
    shape = (ns, mpol, ntor + 1)
    arrays = [
        np.asarray(value.toroidal_flux, dtype=np.float64).reshape(ns),
        np.asarray(value.poloidal_flux, dtype=np.float64).reshape(ns),
    ]
    coefficients = value.coefficients
    for name in _GEOMETRY_COEFFICIENTS:
        raw = np.asarray(getattr(coefficients, name), dtype=np.float64)
        raw = np.zeros(shape, dtype=np.float64) if raw.size == 0 else raw.reshape(shape)
        arrays.append(raw.ravel())
    return np.concatenate(arrays)


def _make_indata(template, boundary: np.ndarray, profiles=None):
    indata = template.copy()
    indata.rbc[...] = boundary[0]
    indata.zbs[...] = boundary[1]
    for name, value in (profiles or {}).items():
        setattr(indata, name, float(value) if np.ndim(value) == 0 else value)
    return indata


def _key(solver_id: int, boundary: np.ndarray, profiles: dict) -> tuple[int, bytes]:
    values = [
        (name, np.shape(value), np.asarray(value).tobytes())
        for name, value in sorted(profiles.items())
    ]
    return solver_id, repr((np.asarray(boundary).tobytes(), values)).encode()


# VmecModel.status's integer value for vmecpp::VmecStatus::SUCCESSFUL_TERMINATION (see
# common/util/util.h); every other status, including NORMAL_TERMINATION (no fatal
# error, but the iteration budget was exhausted before ftol was met), means the step
# did not converge.
_VMEC_STATUS_SUCCESSFUL_TERMINATION = 11

# State vectors of recent forward solves, keyed by (solver id, boundary
# bytes), so the VJP linearizes at the solved state instead of solving again.
# The cache is module-level and holds NumPy data only: JAX keeps callback
# closures alive in its caches, so anything a solver instance owned would live
# as long as those, and C++ objects must not outlive the callback that made them.
_FORWARD_SOLVES: collections.OrderedDict[tuple[int, bytes], dict[str, Any]] = (
    collections.OrderedDict()
)
_FORWARD_SOLVES_SIZE = 2
# Unique per solver, so solvers of different inputs never share a cache entry.
_SOLVER_IDS = itertools.count()

_MU_0 = 4.0e-7 * np.pi


def _solve_model(template, boundary: np.ndarray, profiles=None):
    """Run all requested VMEC++ resolutions and return the final model.

    Each entry of ``ns_array`` converges to its own ``ftol_array`` entry.
    Coarse, non-final steps are allowed to exhaust their iteration budget
    without reaching ``ftol``: their only job is to hand a good initial guess
    to the next, finer step, exactly as ``vmecpp.run`` treats them. The final
    step is the one that must actually converge; a schedule truncated to its
    first steps for a cheap solve is only valid if its new last step is
    standalone-convergent.
    """
    indata = _make_indata(template, boundary, profiles)
    resolutions = [int(value) for value in np.asarray(indata.ns_array)]
    model = None
    for ns in resolutions:
        if ns < 3:
            continue
        if model is None:
            model = _vmecpp.VmecModel.create(indata, ns)
            model.always_fix_m1_gauge = True
        else:
            model.refine_to(ns)
        model.solve()
    if model is None:
        error_message = "VMEC input has no resolution with ns >= 3"
        raise ValueError(error_message)
    if model.status != _VMEC_STATUS_SUCCESSFUL_TERMINATION:
        error_message = (
            f"VMEC++ did not converge at the final multi-grid step (ns = {model.ns}): "
            f"status {model.status}, ftol = {model.ftolv:.3e}, final force residuals "
            f"fsqr = {model.fsqr:.3e}, fsqz = {model.fsqz:.3e}, fsql = {model.fsql:.3e}. "
            "If ns_array was truncated to its first steps, its new last entry must be "
            "standalone-convergent at its own ftol_array/niter_array entry."
        )
        raise RuntimeError(error_message)
    return model


def _span_slices(model) -> dict[str, slice]:
    """Return slices in VmecModel's canonical active-state ordering."""
    names: list[str] = ["r_cc"]
    if model.lthreed:
        names.append("r_ss")
    if model.lasym:
        names.append("r_sc")
    if model.lasym and model.lthreed:
        names.append("r_cs")
    names.append("z_sc")
    if model.lthreed:
        names.append("z_cs")
    if model.lasym:
        names.append("z_cc")
    if model.lasym and model.lthreed:
        names.append("z_ss")
    names.append("lambda_sc")
    if model.lthreed:
        names.append("lambda_cs")
    if model.lasym:
        names.append("lambda_cc")
    if model.lasym and model.lthreed:
        names.append("lambda_ss")
    span_size = model.ns * model.mpol * (model.ntor + 1)
    return {
        name: slice(index * span_size, (index + 1) * span_size)
        for index, name in enumerate(names)
    }


def _interior_and_boundary(model) -> tuple[np.ndarray, np.ndarray]:
    slices = _span_slices(model)
    state_size = int(np.asarray(model.get_state()).size)
    boundary: list[int] = []
    modes_per_surface = model.mpol * (model.ntor + 1)
    edge_start = (model.ns - 1) * modes_per_surface
    for name in (
        "r_cc",
        "r_ss",
        "r_sc",
        "r_cs",
        "z_sc",
        "z_cs",
        "z_cc",
        "z_ss",
    ):
        if name in slices:
            span = slices[name]
            boundary.extend(range(span.start + edge_start, span.stop))
    boundary_array = np.asarray(sorted(boundary), dtype=np.int64)
    interior_array = np.setdiff1d(np.arange(state_size), boundary_array)
    return interior_array, boundary_array


def _gauge_entries(model) -> np.ndarray:
    """State entries holding the m=1 poloidal-origin gauge on interior surfaces.

    ``FourierCoeffs::m1Constraint`` stores ``(r_ss - z_cs) / 2`` for ``m = 1``
    in the ``z_cs`` slot. Its force is zeroed when the gauge is fixed, so with
    ``always_fix_m1_gauge`` the solve leaves these entries at the initial guess:
    the boundary value scaled by ``sqrt(s)`` on every surface. The ``n = 0``
    entry is identically zero and is left to the structural deflation.
    """
    if not model.lthreed:
        return np.zeros(0, dtype=np.int64)
    slices = _span_slices(model)
    modes_per_surface = model.mpol * (model.ntor + 1)
    span = slices["z_cs"]
    entries = [
        span.start + j * modes_per_surface + 1 * (model.ntor + 1) + n
        for j in range(1, model.ns - 1)
        for n in range(1, model.ntor + 1)
    ]
    return np.asarray(entries, dtype=np.int64)


def _gauge_radial_weight(model, gauge: np.ndarray) -> np.ndarray:
    """``sqrt(s_j)`` for every gauge entry: the factor mapping the boundary gauge
    to surface ``j`` in ``FourierGeometry::interpFromBoundaryAndAxis``."""
    modes_per_surface = model.mpol * (model.ntor + 1)
    span_start = _span_slices(model)["z_cs"].start
    surface = (gauge - span_start) // modes_per_surface
    return np.sqrt(surface / (model.ns - 1.0))


def _pull_gauge_back_to_boundary(
    model, state_bar: np.ndarray, gauge: np.ndarray
) -> np.ndarray:
    """Fold the gauge cotangents onto the boundary gauge entries they derive from."""
    result = state_bar.copy()
    modes_per_surface = model.mpol * (model.ntor + 1)
    span_start = _span_slices(model)["z_cs"].start
    mode = (gauge - span_start) % modes_per_surface
    edge = span_start + (model.ns - 1) * modes_per_surface + mode
    np.add.at(result, edge, _gauge_radial_weight(model, gauge) * state_bar[gauge])
    result[gauge] = 0.0
    return result


def _boundary_from_state_vjp(model, state_bar: np.ndarray) -> np.ndarray:
    """Transpose the fixed-boundary parser for a symmetric input."""
    if model.lasym:
        error_message = (
            "DifferentiableVmec currently requires lasym=false; the asymmetric "
            "poloidal-origin shift is not a differentiable parameterization"
        )
        raise RuntimeError(error_message)
    slices = _span_slices(model)
    mpol = model.mpol
    ntor = model.ntor
    modes_per_surface = mpol * (ntor + 1)
    edge = model.ns - 1
    result = np.zeros((2, mpol, 2 * ntor + 1), dtype=np.float64)

    def edge_value(name: str, m: int, n: int) -> float:
        if name not in slices:
            return 0.0
        values = state_bar[slices[name]]
        return float(values[edge * modes_per_surface + m * (ntor + 1) + n])

    # Undo the state scaling and the m=1 gauge transform used by
    # Boundaries::ensureM1Constrained. The boundary parser itself is the
    # transpose of the positive/negative-toroidal-mode accumulation below.
    rbcc_bar = np.zeros((mpol, ntor + 1))
    zbsc_bar = np.zeros((mpol, ntor + 1))
    rbss_bar = np.zeros((mpol, ntor + 1))
    zbcs_bar = np.zeros((mpol, ntor + 1))
    for m in range(mpol):
        for n in range(ntor + 1):
            scale = (1.0 if m == 0 else np.sqrt(2.0)) * (
                1.0 if n == 0 else np.sqrt(2.0)
            )
            rbcc_bar[m, n] = edge_value("r_cc", m, n) / scale
            zbsc_bar[m, n] = edge_value("z_sc", m, n) / scale
            if model.lthreed:
                rbss_bar[m, n] = edge_value("r_ss", m, n) / scale
                zbcs_bar[m, n] = edge_value("z_cs", m, n) / scale
    if model.lthreed and mpol > 1:
        for n in range(ntor + 1):
            r_bar = rbss_bar[1, n]
            z_bar = zbcs_bar[1, n]
            rbss_bar[1, n] = 0.5 * (r_bar + z_bar)
            zbcs_bar[1, n] = 0.5 * (r_bar - z_bar)

    if model.have_to_flip_theta:
        for m in range(1, mpol):
            parity = 1.0 if m % 2 == 0 else -1.0
            rbcc_bar[m] *= parity
            zbsc_bar[m] *= -parity
            if model.lthreed:
                rbss_bar[m] *= -parity
                zbcs_bar[m] *= parity

    for m in range(mpol):
        for signed_n in range(-ntor, ntor + 1):
            source = ntor + signed_n
            target = abs(signed_n)
            sign = 1.0 if signed_n > 0 else -1.0 if signed_n < 0 else 0.0
            result[0, m, source] += rbcc_bar[m, target]
            if model.lthreed and m > 0:
                result[0, m, source] += sign * rbss_bar[m, target]
            if m > 0:
                result[1, m, source] += zbsc_bar[m, target]
            if model.lthreed:
                result[1, m, source] -= sign * zbcs_bar[m, target]
    return result


def _structural_nullfree_interior(
    model, interior: np.ndarray, n_probe: int = 6, tol: float = 1.0e-9, seed: int = 0
) -> np.ndarray:
    """Interior DOFs that actually enter the force.

    The augmented Hessian has a structural null space: state-independent gauge
    and parity modes that no force depends on and that produce no force. They
    make the transposed interior system singular, and the objective cotangent
    generally has a component outside its range, so the adjoint solve is
    inconsistent and stagnates rather than converging. In two dimensions the
    surviving null directions happen to stay orthogonal to the cotangent; in
    three dimensions the extra ``r_ss``, ``z_cs`` and ``lambda_cs`` blocks bring
    in modes that do not, which is why this deflation is not optional there.

    A DOF is kept when both its Hessian column and its row are nonzero. Column
    ``i`` is zero iff ``(H^T v)[i] = 0`` for random ``v``, and row ``i`` is zero
    iff ``(H v)[i] = 0``, so a handful of probes finds every structural zero. The
    set depends only on the mode structure, not on the state, so it is detected
    once per model and reused across adjoint solves.
    """
    state_size = int(np.asarray(model.get_state()).size)
    generator = np.random.default_rng(seed)
    column = np.zeros(state_size)
    row = np.zeros(state_size)
    for _ in range(n_probe):
        probe = np.ascontiguousarray(generator.standard_normal(state_size))
        column = np.maximum(
            column,
            np.abs(
                np.asarray(
                    model.exact_hessian_vector_product_transpose(probe),
                    dtype=np.float64,
                )
            ),
        )
        row = np.maximum(
            row,
            np.abs(
                np.asarray(model.exact_hessian_vector_product(probe), dtype=np.float64)
            ),
        )
    threshold = tol * max(column.max(), row.max(), 1.0)
    keep = [i for i in interior if column[i] > threshold and row[i] > threshold]
    if not keep:
        error_message = (
            "VMEC++ adjoint: the interior force operator is entirely structurally "
            "null; the model is not in a differentiable state"
        )
        raise RuntimeError(error_message)
    return np.asarray(keep, dtype=np.int64)


def _implicit_vjp(
    model, geometry_bar: np.ndarray, with_profiles: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The boundary cotangent and that of :func:`autodiff_wout.half_grid_profiles`."""
    if not getattr(model, "has_exact_force_jacobian", False):
        error_message = (
            "This VMEC++ build has no exact residual transpose. Rebuild with "
            "VMECPP_ENABLE_ENZYME to differentiate a solved equilibrium. "
            "No finite-difference derivative is used."
        )
        raise RuntimeError(error_message)
    if not model.always_fix_m1_gauge:
        error_message = (
            "VMEC++ adjoint: the model was solved with a free m=1 gauge; the exact "
            "force Jacobian describes the solve only with always_fix_m1_gauge"
        )
        raise RuntimeError(error_message)
    coefficient_bar = np.asarray(geometry_bar[2 * model.ns :], dtype=np.float64)
    poloidal_flux_bar = np.asarray(
        geometry_bar[model.ns : 2 * model.ns], dtype=np.float64
    )
    state_bar = np.asarray(
        model.geometry_state_vjp(coefficient_bar, poloidal_flux_bar), dtype=np.float64
    )
    state = np.asarray(model.get_state(), dtype=np.float64)
    interior, boundary = _interior_and_boundary(model)
    # The pinned gauge entries are prescribed by the boundary, like the boundary
    # entries themselves: they leave the interior system and are pulled back.
    gauge = _gauge_entries(model)
    interior = np.setdiff1d(interior, gauge)
    prescribed = np.concatenate([boundary, gauge])
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    state_size = state.size
    # Deflate the structural null space; without this the transposed
    # interior system is singular and inconsistent in 3D.
    interior = _structural_nullfree_interior(model, interior)

    def transpose(value: np.ndarray) -> np.ndarray:
        return np.asarray(
            model.exact_hessian_vector_product_transpose(np.ascontiguousarray(value)),
            dtype=np.float64,
        )

    def matvec(value: np.ndarray) -> np.ndarray:
        embedded = np.zeros(state_size)
        embedded[interior] = value
        return transpose(embedded)[interior]

    def precondition(value: np.ndarray) -> np.ndarray:
        embedded = np.zeros(state_size)
        embedded[interior] = value
        return np.asarray(
            model.apply_preconditioner(np.ascontiguousarray(embedded)),
            dtype=np.float64,
        )[interior]

    operator_factory: Any = LinearOperator
    operator = operator_factory(
        (interior.size, interior.size), matvec=matvec, dtype=np.float64
    )
    preconditioner = operator_factory(
        (interior.size, interior.size), matvec=precondition, dtype=np.float64
    )
    adjoint, info = gmres(
        operator,
        state_bar[interior],
        M=preconditioner,
        rtol=1.0e-8,
        restart=200,
        maxiter=400,
    )
    if info != 0:
        error_message = f"VMEC++ implicit adjoint solve failed with info={info}"
        raise RuntimeError(error_message)
    embedded = np.zeros(state_size)
    embedded[interior] = adjoint
    full_state_bar = np.zeros(state_size)
    full_state_bar[prescribed] = state_bar[prescribed] - transpose(embedded)[prescribed]
    full_state_bar = _pull_gauge_back_to_boundary(model, full_state_bar, gauge)
    profile_bar = np.zeros((3, model.ns - 1))
    if with_profiles:
        # dJ/dp = J_p - adjoint^T F_p, J_p through the poloidal flux
        profile_bar = np.stack(model.profile_vjp(-embedded, poloidal_flux_bar))
        profile_bar[1] *= -1.0 if model.have_to_flip_theta else 1.0
    return _boundary_from_state_vjp(model, full_state_bar), profile_bar


@dataclass(frozen=True)
class DifferentiableVmec:
    """A callable JAX view of one fixed-boundary VMEC++ input.

    The exact VJP covers the boundary coefficients and the power-series profile
    parameters passed as ``profiles`` (see
    :data:`autodiff_wout.PROFILE_PARAMETERS`).
    """

    vmec_input: Any
    _solver_id: int = field(
        default_factory=lambda: next(_SOLVER_IDS),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.vmec_input.lfreeb:
            error_message = "DifferentiableVmec currently requires lfreeb=false"
            raise ValueError(error_message)
        if self.vmec_input.lasym:
            error_message = (
                "DifferentiableVmec currently requires lasym=false; use the "
                "product-basis geometry API for asymmetric snapshots"
            )
            raise ValueError(error_message)
        if not isinstance(self.vmec_input.mpol, int) or not isinstance(
            self.vmec_input.ntor, int
        ):
            error_message = "DifferentiableVmec currently requires scalar mpol and ntor"
            raise ValueError(error_message)
        resolutions = np.asarray(self.vmec_input.ns_array)
        if resolutions.size == 0 or resolutions[-1] < 3:
            error_message = "DifferentiableVmec requires an ns_array entry >= 3"
            raise ValueError(error_message)

    @property
    def parameter_shape(self) -> tuple[int, int, int]:
        return (2, self.vmec_input.mpol, 2 * self.vmec_input.ntor + 1)

    @property
    def ns(self) -> int:
        return int(np.asarray(self.vmec_input.ns_array)[-1])

    @property
    def geometry_size(self) -> int:
        modes = self.vmec_input.mpol * (self.vmec_input.ntor + 1)
        return 2 * self.ns + len(_GEOMETRY_COEFFICIENTS) * self.ns * modes

    @property
    def extra_size(self) -> int:
        """Size of the non-differentiated outputs that follow the geometry."""
        return 0

    @property
    def output_shape(self) -> tuple[int]:
        return (self.geometry_size + self.extra_size,)

    def _remember(self, boundary: np.ndarray, profiles: dict, solve: dict) -> None:
        _FORWARD_SOLVES[_key(self._solver_id, boundary, profiles)] = solve
        while len(_FORWARD_SOLVES) > _FORWARD_SOLVES_SIZE:
            _FORWARD_SOLVES.popitem(last=False)

    def _forward_callback(self, boundary: np.ndarray, profiles=None) -> np.ndarray:
        profiles = profiles or {}
        model = _solve_model(self.vmec_input._to_cpp_vmecindata(), boundary, profiles)
        self._remember(boundary, profiles, {"state": np.array(model.get_state())})
        return _cpp_geometry_flat(
            model.get_geometry(), model.ns, model.mpol, model.ntor
        )

    def _solved_state(self, boundary: np.ndarray, profiles: dict) -> np.ndarray:
        """The state vector of a fresh forward solve, for a cache miss."""
        model = _solve_model(self.vmec_input._to_cpp_vmecindata(), boundary, profiles)
        return np.array(model.get_state())

    def _solved_model(self, boundary: np.ndarray, profiles: dict | None = None):
        """A model at the forward solve's state, consumed by the VJP.

        A model created cold and set to a solved state reproduces the solved model's
        forces, Hessian products and preconditioner exactly.
        """
        profiles = profiles or {}
        solve = _FORWARD_SOLVES.pop(_key(self._solver_id, boundary, profiles), None)
        state = (
            self._solved_state(boundary, profiles) if solve is None else solve["state"]
        )
        indata = _make_indata(self.vmec_input._to_cpp_vmecindata(), boundary, profiles)
        model = _vmecpp.VmecModel.create(indata, self.ns)
        model.always_fix_m1_gauge = True
        model.set_state(np.ascontiguousarray(state))
        return model

    def _backward_callback(
        self, boundary: np.ndarray, geometry_bar: np.ndarray, profiles=None
    ) -> tuple[np.ndarray, np.ndarray]:
        profiles = profiles or {}
        model = self._solved_model(boundary, profiles)
        return _implicit_vjp(model, geometry_bar, with_profiles=bool(profiles))

    def __call__(self, boundary, profiles=None) -> geometry.Geometry:
        """The geometry for ``boundary`` and optional ``profiles`` values."""
        return self._solve(boundary, profiles)[0]

    def _solve(self, boundary, profiles=None) -> tuple[geometry.Geometry, jax.Array]:
        """The geometry and the non-differentiated outputs of :attr:`extra_size`."""
        boundary = jnp.asarray(boundary, dtype=jnp.float64)
        if boundary.shape != self.parameter_shape:
            error_message = (
                f"boundary has shape {boundary.shape}, expected {self.parameter_shape}"
            )
            raise ValueError(error_message)
        ns = self.ns
        profiles = profiles or {}
        # the solve sees the coefficients; its VJP is taken with respect to the
        # half-grid profiles, from which JAX differentiates the coefficients
        values = jax.lax.stop_gradient(profiles)
        half = autodiff_wout.half_grid_profiles(self.vmec_input, ns, profiles)
        output_spec = jax.ShapeDtypeStruct(self.output_shape, boundary.dtype)
        cotangent_spec = (
            jax.ShapeDtypeStruct(boundary.shape, boundary.dtype),
            jax.ShapeDtypeStruct(half.shape, boundary.dtype),
        )

        def concrete(values):
            return {name: np.asarray(value) for name, value in values.items()}

        def forward_callback(value, values):
            flat = self._forward_callback(np.asarray(value), concrete(values))
            return flat.astype(np.asarray(value).dtype, copy=False)

        def backward_callback(value, values, cotangent):
            geometry_bar = np.asarray(cotangent)[: self.geometry_size]
            bars = self._backward_callback(
                np.asarray(value), geometry_bar, concrete(values)
            )
            return tuple(bar.astype(np.asarray(value).dtype) for bar in bars)

        def forward(value, values):
            return jax.pure_callback(
                forward_callback, output_spec, value, values, vmap_method="sequential"
            )

        @jax.custom_vjp
        def solve_flat(value, half, values):
            del half
            return forward(value, values)

        def solve_fwd(value, half, values):
            del half
            return forward(value, values), (value, values)

        def solve_bwd(residuals, cotangent):
            value, values = residuals
            bars = jax.pure_callback(
                backward_callback,
                cotangent_spec,
                value,
                values,
                cotangent,
                vmap_method="sequential",
            )
            return (*bars, jax.tree.map(jnp.zeros_like, values))

        solve_flat.defvjp(solve_fwd, solve_bwd)
        flat = solve_flat(boundary, half, values)
        mpol = self.vmec_input.mpol
        ntor = self.vmec_input.ntor
        modes = ns * mpol * (ntor + 1)
        arrays = [flat[:ns], flat[ns : 2 * ns]]
        offset = 2 * ns
        for _ in _GEOMETRY_COEFFICIENTS:
            arrays.append(flat[offset : offset + modes].reshape(ns, mpol, ntor + 1))
            offset += modes
        return geometry.Geometry(*arrays, nfp=self.vmec_input.nfp), flat[offset:]


def make_solver(vmec_input) -> DifferentiableVmec:
    """Return a JAX-compatible callable that runs VMEC++ for vmec_input.

    Example:

        from vmecpp import autodiff, simsopt_compat

        solver = autodiff.make_solver(input)
        objective = lambda boundary: simsopt_compat.quasisymmetry_total(
            solver(boundary), [0.6]
        )
        value, gradient = jax.value_and_grad(objective)(boundary)

    Forward execution and the VJP both invoke VMEC++ in memory. The VJP is
    available only in an Enzyme-enabled build, because it requires the exact
    transpose of the force residual. No finite-difference derivative is used.
    """
    if not _vmecpp.VMECPP_ENABLE_ENZYME:
        error_message = (
            "vmecpp.autodiff.make_solver requires a build with the exact force "
            "Jacobian; rebuild with the CMake option VMECPP_ENABLE_ENZYME=ON. "
            "No finite-difference derivative is used."
        )
        raise RuntimeError(error_message)
    return DifferentiableVmec(vmec_input)


@dataclass(frozen=True)
class _RunSolver(DifferentiableVmec):
    """:class:`DifferentiableVmec` whose forward solve is a pinned-gauge C++ run.

    The C++ output of the forward solve supplies what the JAX output stage does
    not compute (jxbout, mercier, threed1, the solver diagnostics) and, as the
    outputs after the geometry, the half-grid mass, iota and buco profiles and the
    sign the solver gives iota. The VJP
    linearizes at the state of a ``VmecModel`` hot-restarted from the forward
    solve's wout, which reproduces it to roundoff.
    """

    max_threads: int | None = None
    verbose: int = 0

    @property
    def extra_size(self) -> int:
        return 3 * (self.ns - 1) + 1

    def _run(self, boundary: np.ndarray, profiles: dict):
        indata = _make_indata(self.vmec_input._to_cpp_vmecindata(), boundary, profiles)
        output = _vmecpp.run(
            indata,
            max_threads=self.max_threads,
            verbose=_vmecpp.OutputMode(self.verbose),
            always_fix_m1_gauge=True,
        )
        initial_state = _vmecpp.HotRestartState(wout=output.wout, indata=indata)
        model = _vmecpp.VmecModel.create(indata, self.ns, initial_state=initial_state)
        flip = -1.0 if model.have_to_flip_theta else 1.0
        return output, np.array(model.get_state()), flip

    def _forward_callback(self, boundary: np.ndarray, profiles=None) -> np.ndarray:
        import vmecpp  # noqa: PLC0415

        profiles = profiles or {}
        output, state, flip = self._run(boundary, profiles)
        wout = vmecpp.VmecWOut._from_cpp_wout(output.wout)
        self._remember(
            boundary,
            profiles,
            {
                "state": state,
                "wout_fields": dict(wout.__dict__),
                "tables": vmecpp._output_tables_from_cpp(output),
            },
        )
        geometry_flat = _cpp_geometry_flat(
            _vmecpp.make_geometry(output),
            self.ns,
            self.vmec_input.mpol,
            self.vmec_input.ntor,
        )
        mass_half = np.asarray(wout.mass, dtype=np.float64)[1:] * _MU_0
        iota_half = np.asarray(wout.iotas, dtype=np.float64)[1:]
        buco_half = np.asarray(wout.buco, dtype=np.float64)[1:]
        extra = [mass_half, iota_half, buco_half, [flip]]
        return np.concatenate([geometry_flat, *extra])

    def _solved_state(self, boundary: np.ndarray, profiles: dict) -> np.ndarray:
        return self._run(boundary, profiles)[1]

    def forward_outputs(self) -> dict[str, Any] | None:
        """The C++ outputs of this solver's latest forward solve, if it has run."""
        for (solver_id, _), solve in reversed(_FORWARD_SOLVES.items()):
            if solver_id == self._solver_id:
                return solve
        return None


__all__ = ["DifferentiableVmec", "make_solver"]
