# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Bootstrap-consistent equilibria via the Redl formula.

A reactor-relevant current profile is set by the equilibrium itself: the
bootstrap current depends on the geometry, and the geometry depends on the
current. :func:`bootstrap_consistent` closes that loop for a current-constrained
(``ncurr = 1``) run. Each pass evaluates the Redl bootstrap current
:math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` (Redl et al., Physics of Plasmas
28, 022502 (2021)) on the converged equilibrium, converts it to an enclosed
toroidal current profile, under-relaxes toward it, and re-solves by hot restart
from the previous solution, so the extra solves cost a few iterations each
rather than the full count.

The Redl evaluation itself comes from ``simsopt.mhd.bootstrap``, which vmecpp
already depends on; this module contributes the geometry extraction from a
:class:`VmecOutput` and the consistency loop. The conversion between
:math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` and the enclosed current uses the
flux-function identity :math:`\\mu_0 \\left<\\vec{J}\\cdot\\vec{B}\\right> =
(G\\, dI/d\\psi - I\\, dG/d\\psi)\\left<B^2\\right>/(G + \\iota I)` with
:math:`G` and :math:`I` read from ``bvco`` and ``buco``, and the same identity
evaluates the current the equilibrium actually carries, so convergence is
declared on the mismatch between the achieved and the Redl
:math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` profiles rather than on the loop's
own inputs.

Density and temperature profiles are functions of the normalized toroidal flux
``s``: ``ne`` in 1/m^3, ``Te`` and ``Ti`` in eV, with ``Zeff`` a constant. Any
callable works; an object with a ``dfds`` method (such as a
``simsopt.mhd.profiles.Profile``) supplies its own derivative, and a plain
callable is differentiated by central differences.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from vmecpp import VmecInput, VmecOutput
    from vmecpp._free_boundary import MagneticFieldResponseTable

_MU_0 = 4.0e-7 * np.pi


class _FiniteDifferenceProfile:
    """Adapt a plain callable to the profile interface Redl evaluation needs."""

    def __init__(self, f: typing.Callable[[np.ndarray], np.ndarray]) -> None:
        self._f = f

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return np.asarray(self._f(np.asarray(s)), dtype=float)

    def dfds(self, s: np.ndarray, step: float = 1.0e-5) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        lo = np.clip(s - step, 0.0, 1.0)
        hi = np.clip(s + step, 0.0, 1.0)
        return (self(hi) - self(lo)) / (hi - lo)


def _as_profile(f: typing.Any) -> typing.Any:
    if hasattr(f, "dfds"):
        return f
    return _FiniteDifferenceProfile(f)


@dataclasses.dataclass
class _RedlGeometry:
    """Flux-surface quantities the Redl formula needs, on the half grid."""

    s: np.ndarray
    G: np.ndarray
    I: np.ndarray  # noqa: E741
    iota: np.ndarray
    epsilon: np.ndarray
    f_t: np.ndarray
    fsa_B2: np.ndarray
    R: np.ndarray
    psi_edge: float
    nfp: int


def _redl_geometry(
    output: VmecOutput, ntheta: int = 64, nphi: int = 65
) -> _RedlGeometry:
    """Extract the Redl geometry from a converged equilibrium.

    ``|B|`` and the Jacobian are reconstructed from the Nyquist-grid Fourier
    coefficients on a ``(ntheta, nphi)`` grid per half-grid surface, and the
    trapped fraction, effective inverse aspect ratio and flux-surface averages
    come from ``simsopt.mhd.bootstrap.compute_trapped_fraction``. ``G`` and
    ``I`` are the covariant field averages ``bvco`` and ``buco``, which are
    flux functions.
    """
    from simsopt.mhd.bootstrap import compute_trapped_fraction  # noqa: PLC0415

    w = output.wout
    ns = int(w.ns)
    s_half = (np.arange(1, ns) - 0.5) / (ns - 1)

    G = np.asarray(w.bvco, dtype=float)[1:]
    current = np.asarray(w.buco, dtype=float)[1:]
    iota = np.asarray(w.iotas, dtype=float)[1:]

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / w.nfp, nphi, endpoint=False)
    phi_2d, theta_2d = np.meshgrid(phi, theta)

    xm = np.asarray(w.xm_nyq)
    xn = np.asarray(w.xn_nyq)
    angle = (
        xm[:, None, None] * theta_2d[None, :, :]
        - xn[:, None, None] * phi_2d[None, :, :]
    )
    cos_angle = np.cos(angle)
    bmnc = np.asarray(w.bmnc, dtype=float)[:, 1:]
    gmnc = np.asarray(w.gmnc, dtype=float)[:, 1:]
    mod_b = np.einsum("jtp,js->tps", cos_angle, bmnc)
    sqrt_g = np.einsum("jtp,js->tps", cos_angle, gmnc)
    if w.lasym:
        sin_angle = np.sin(angle)
        bmns = np.asarray(w.bmns, dtype=float)[:, 1:]
        gmns = np.asarray(w.gmns, dtype=float)[:, 1:]
        mod_b += np.einsum("jtp,js->tps", sin_angle, bmns)
        sqrt_g += np.einsum("jtp,js->tps", sin_angle, gmns)

    _, _, epsilon, fsa_b2, fsa_1_over_b, f_t = compute_trapped_fraction(mod_b, sqrt_g)

    return _RedlGeometry(
        s=s_half,
        G=G,
        I=current,
        iota=iota,
        epsilon=epsilon,
        f_t=f_t,
        fsa_B2=fsa_b2,
        R=(G + iota * current) * fsa_1_over_b,
        psi_edge=float(-np.asarray(w.phi)[-1] / (2.0 * np.pi)),
        nfp=int(w.nfp),
    )


def redl_bootstrap_current(
    output: VmecOutput,
    *,
    ne: typing.Any,
    Te: typing.Any,
    Ti: typing.Any,
    Zeff: float = 1.0,
    helicity_n: int = 0,
    ntheta: int = 64,
    nphi: int = 65,
) -> tuple[np.ndarray, np.ndarray]:
    """The Redl bootstrap :math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` profile.

    Args:
        output: a converged equilibrium.
        ne: electron density profile in 1/m^3, as a callable of ``s``.
        Te: electron temperature profile in eV, as a callable of ``s``.
        Ti: ion temperature profile in eV, as a callable of ``s``.
        Zeff: the effective impurity charge, constant over the plasma.
        helicity_n: 0 for axisymmetry or quasi-axisymmetry, +/-1 for
            quasi-helical symmetry, as in ``simsopt.mhd.bootstrap``.
        ntheta: poloidal resolution of the geometry evaluation.
        nphi: toroidal resolution of the geometry evaluation.

    Returns:
        A tuple of two 1D arrays: the half-grid values of ``s`` and
        :math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` (SI units, T A/m^2) on
        those surfaces.
    """
    from simsopt.mhd.bootstrap import j_dot_B_Redl  # noqa: PLC0415

    geometry = _redl_geometry(output, ntheta=ntheta, nphi=nphi)
    j_dot_b, _ = j_dot_B_Redl(
        _as_profile(ne),
        _as_profile(Te),
        _as_profile(Ti),
        Zeff,
        helicity_n,
        s=geometry.s,
        G=geometry.G,
        R=geometry.R,
        iota=geometry.iota,
        epsilon=geometry.epsilon,
        f_t=geometry.f_t,
        psi_edge=geometry.psi_edge,
        nfp=geometry.nfp,
    )
    return geometry.s, j_dot_b


def _achieved_j_dot_b(geometry: _RedlGeometry) -> np.ndarray:
    """The :math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` the equilibrium carries.

    Evaluated from the flux-function identity with ``G``, ``I`` and their
    radial derivatives; reproduces ``wout.jdotb`` away from the axis.
    """
    d_i_ds = np.gradient(geometry.I, geometry.s)
    d_g_ds = np.gradient(geometry.G, geometry.s)
    return (
        (geometry.G * d_i_ds - geometry.I * d_g_ds)
        * geometry.fsa_B2
        / (_MU_0 * (geometry.G + geometry.iota * geometry.I) * geometry.psi_edge)
    )


def _enclosed_current(
    geometry: _RedlGeometry, j_dot_b: np.ndarray, resolution: int = 2000
) -> tuple[np.ndarray, float]:
    """Integrate a :math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` profile to the Boozer
    current flux function :math:`I(s)`.

    Solves :math:`dI/ds = (\\mu_0 \\psi_{edge} \\left<\\vec{J}\\cdot\\vec{B}
    \\right> (G + \\iota I)/\\left<B^2\\right> + I\\, dG/ds)/G` from
    :math:`I(0) = 0`, with the coefficients interpolated from the half grid.

    Returns:
        ``I`` on the half-grid surfaces of ``geometry`` and its value at
        ``s = 1``.
    """
    s_fine = np.linspace(0.0, 1.0, resolution + 1)
    ds = s_fine[1] - s_fine[0]
    g = np.interp(s_fine, geometry.s, geometry.G)
    d_g_ds = np.interp(s_fine, geometry.s, np.gradient(geometry.G, geometry.s))
    iota = np.interp(s_fine, geometry.s, geometry.iota)
    fsa_b2 = np.interp(s_fine, geometry.s, geometry.fsa_B2)
    j_dot_b_fine = np.interp(s_fine, geometry.s, j_dot_b)

    def d_i_ds(index: int, current: float) -> float:
        return (
            _MU_0
            * geometry.psi_edge
            * j_dot_b_fine[index]
            * (g[index] + iota[index] * current)
            / fsa_b2[index]
            + current * d_g_ds[index]
        ) / g[index]

    current = np.zeros(resolution + 1)
    for k in range(resolution):
        # Heun's method; dI/ds is linear in I, so this is plenty.
        slope = d_i_ds(k, current[k])
        predictor = current[k] + ds * slope
        current[k + 1] = current[k] + 0.5 * ds * (slope + d_i_ds(k + 1, predictor))
    return np.interp(geometry.s, s_fine, current), float(current[-1])


def _final_multigrid_step(input: VmecInput) -> VmecInput:
    """The input reduced to its final multigrid step, as hot restart requires."""
    return input.model_copy(
        update={
            "ns_array": np.asarray(input.ns_array[-1:]),
            "ftol_array": np.asarray(input.ftol_array[-1:]),
            "niter_array": np.asarray(input.niter_array[-1:]),
        }
    )


@dataclasses.dataclass
class BootstrapConsistentResult:
    """The converged equilibrium and the record of the consistency loop."""

    output: VmecOutput
    """The bootstrap-consistent equilibrium."""

    s: np.ndarray
    """Half-grid values of normalized toroidal flux."""

    j_dot_B: np.ndarray
    """The Redl :math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` on ``s``, evaluated on the
    final equilibrium."""

    residuals: list[float]
    """Per-pass mismatch between the achieved and the Redl
    :math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>`, relative to the profile maximum."""

    iterations: int
    """Number of equilibrium solves, the initial cold solve included."""


def bootstrap_consistent(
    input: VmecInput,
    *,
    ne: typing.Any,
    Te: typing.Any,
    Ti: typing.Any,
    Zeff: float = 1.0,
    helicity_n: int = 0,
    relaxation: float = 0.5,
    tolerance: float = 0.02,
    max_iterations: int = 12,
    ntheta: int = 64,
    nphi: int = 65,
    magnetic_field: MagneticFieldResponseTable | None = None,
    max_threads: int | None = None,
    verbose: bool | int | None = None,
) -> BootstrapConsistentResult:
    """Solve an equilibrium whose current profile is its own bootstrap current.

    The input is solved as given, with ``ncurr`` forced to 1. Each pass
    evaluates the Redl :math:`\\left<\\vec{J}\\cdot\\vec{B}\\right>` on the
    converged equilibrium, integrates it to an enclosed-current profile,
    under-relaxes the constraint toward it (``pcurr_type = "cubic_spline_i"``
    with matching ``curtor``), and re-solves by hot restart from the previous
    solution. The loop ends when the current the equilibrium carries matches
    the Redl profile to ``tolerance``, relative to the profile maximum.

    Args:
        input: as for :func:`vmecpp.run`; its current profile seeds the first
            pass and is then replaced.
        ne: electron density profile in 1/m^3, as a callable of ``s``.
        Te: electron temperature profile in eV, as a callable of ``s``.
        Ti: ion temperature profile in eV, as a callable of ``s``.
        Zeff: the effective impurity charge, constant over the plasma.
        helicity_n: as in :func:`redl_bootstrap_current`.
        relaxation: weight of the new Redl profile in each constraint update;
            1 is a full step.
        tolerance: convergence bound on the achieved-versus-Redl mismatch.
        max_iterations: bound on the number of equilibrium solves.
        ntheta: poloidal resolution of the geometry evaluation.
        nphi: toroidal resolution of the geometry evaluation.
        magnetic_field: as for :func:`vmecpp.run`.
        max_threads: as for :func:`vmecpp.run`.
        verbose: as for :func:`vmecpp.run`; ``None`` keeps its default.

    Raises:
        RuntimeError: the loop did not reach ``tolerance`` within
            ``max_iterations`` solves.
    """
    import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

    run_kwargs: dict[str, typing.Any] = {"max_threads": max_threads}
    if verbose is not None:
        run_kwargs["verbose"] = verbose

    work = vmecpp.VmecInput.model_validate(input).model_copy(deep=True)
    work.ncurr = 1
    output = vmecpp.run(work, magnetic_field, **run_kwargs)

    residuals: list[float] = []
    iterations = 1
    while True:
        geometry = _redl_geometry(output, ntheta=ntheta, nphi=nphi)
        from simsopt.mhd.bootstrap import j_dot_B_Redl  # noqa: PLC0415

        j_dot_b_target, _ = j_dot_B_Redl(
            _as_profile(ne),
            _as_profile(Te),
            _as_profile(Ti),
            Zeff,
            helicity_n,
            s=geometry.s,
            G=geometry.G,
            R=geometry.R,
            iota=geometry.iota,
            epsilon=geometry.epsilon,
            f_t=geometry.f_t,
            psi_edge=geometry.psi_edge,
            nfp=geometry.nfp,
        )
        if not np.all(np.isfinite(j_dot_b_target)):
            msg = (
                "the Redl evaluation produced non-finite values; the "
                "equilibrium's rotational transform may vanish, which the "
                "bootstrap formula divides by"
            )
            raise RuntimeError(msg)
        achieved = _achieved_j_dot_b(geometry)
        scale = max(float(np.max(np.abs(j_dot_b_target))), 1.0e-30)
        # The axis-side and edge-side half-grid points carry the one-sided
        # differences of the identity; judge convergence on the interior.
        residual = float(np.max(np.abs(achieved[1:-1] - j_dot_b_target[1:-1])) / scale)
        residuals.append(residual)
        if residual <= tolerance:
            return BootstrapConsistentResult(
                output=output,
                s=geometry.s,
                j_dot_B=j_dot_b_target,
                residuals=residuals,
                iterations=iterations,
            )
        if iterations >= max_iterations:
            msg = (
                f"bootstrap consistency not reached in {iterations} solves; "
                f"residuals {residuals}"
            )
            raise RuntimeError(msg)

        current_target, current_target_edge = _enclosed_current(
            geometry, j_dot_b_target
        )
        blended = (1.0 - relaxation) * geometry.I + relaxation * current_target
        achieved_edge = geometry.I[-1] + 0.5 * (geometry.I[-1] - geometry.I[-2])
        blended_edge = (
            1.0 - relaxation
        ) * achieved_edge + relaxation * current_target_edge

        knots_s = np.concatenate(([0.0], geometry.s, [1.0]))
        knots_i = np.concatenate(([0.0], blended, [blended_edge]))
        work = work.model_copy(
            update={
                "pcurr_type": "cubic_spline_i",
                "ac": np.zeros(0),
                "ac_aux_s": knots_s,
                "ac_aux_f": knots_i,
                "curtor": float(
                    output.wout.signgs * 2.0 * np.pi * blended_edge / _MU_0
                ),
            }
        )
        output = vmecpp.run(
            _final_multigrid_step(work),
            magnetic_field,
            restart_from=output,
            **run_kwargs,
        )
        iterations += 1
