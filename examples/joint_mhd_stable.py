# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""ConStellaration's MHD-stable quasi-isodynamic problem by vmecpp.joint or by the
nested baseline.

The benchmark scores a set of boundaries by the area that their magnetic gradient scale
lengths and aspect ratios cover, under limits on the rotational transform at the edge,
quasi-isodynamicity, the mirror ratio, the flux compression in regions of bad curvature
and the magnetic well. Here its residuals of quasi-isodynamicity are the objective at
one aspect ratio and the limits are constraints, with the gradient scale length held
above a bound, all as the benchmark computes them, in JAX. The start is the boundary
that the problem of examples/joint_quasi_isodynamic.py reaches from the near-axis
boundary within an elongation of 5, or a saved boundary. A cold, fully converged run on
the modes and surfaces of the benchmark checks the result.
"""

import argparse
import dataclasses
import functools
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import make_interp_spline

import vmecpp
from vmecpp import autodiff_wout
from vmecpp.joint import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_quasi_isodynamic import (  # type: ignore
    FEASIBILITY_TOLERANCE,
    QuasiIsodynamicProblem,
    converged_wout,
    near_axis_start,
)
from joint_simple_to_build import SimpleToBuildProblem  # type: ignore
from joint_simple_to_build import score as simple_to_build_score  # type: ignore

jax.config.update("jax_enable_x64", True)

_WOUT_QUANTITIES = (
    "rmnc",
    "zmns",
    "lmns",
    "gmnc",
    "bmnc",
    "bsupvmnc",
    "bsubsmns",
    "bsubumnc",
    "bsubvmnc",
    "iotas",
    "pres",
    "phi",
    "phipf",
    "vp",
    "Aminor_p",
)


def magnetic_well(vp):
    """The relative fall of ``dV/ds`` from the magnetic axis to the boundary, the
    ``vacuum_well`` of SIMSOPT, from ``vp`` on the half grid."""
    axis, edge = 1.5 * vp[1] - 0.5 * vp[2], 1.5 * vp[-1] - 0.5 * vp[-2]
    return (axis - edge) / axis


@functools.lru_cache
def _radial_weights(ns: int, surface: float) -> tuple[np.ndarray, ...]:
    """The weights of the values on the full grid and on the half grid of ``ns``
    surfaces that give the cubic spline through them and its radial derivative at the
    normalized toroidal flux ``surface``: those of the full grid, then those of the half
    grid."""
    full = np.linspace(0.0, 1.0, ns)
    weights = []
    for grid in (full, 0.5 * (full[1:] + full[:-1])):
        spline = make_interp_spline(grid, np.eye(grid.size))
        weights += [spline(surface), spline.derivative()(surface)]
    return tuple(weights)


def _poloidal_angles(lmns, xm, xn, phi, theta_pest):
    """The poloidal angle of VMEC of each point of the toroidal angles ``phi``, where
    ``theta + lambda`` is the straight-field-line angle ``theta_pest`` of its column.

    Newton steps from ``theta_pest`` stay between the last angles on either side of the
    root and halve that interval where they would leave it. One more step carries the
    derivatives.
    """

    def miss(theta, lmns, phi):
        """The straight-field-line angle of a poloidal angle less its target, and its
        derivative."""
        angle = xm[:, None, None] * theta - xn[:, None, None] * phi
        return (
            theta + jnp.tensordot(lmns, jnp.sin(angle), 1) - theta_pest,
            1.0 + jnp.tensordot(lmns * xm, jnp.cos(angle), 1),
        )

    fixed = jax.lax.stop_gradient(lmns), jax.lax.stop_gradient(phi)

    def refine(carry):
        theta, low, high, value, slope, count = carry
        low, high = (
            jnp.where(value < 0.0, theta, low),
            jnp.where(value > 0.0, theta, high),
        )
        newton = theta - value / slope
        theta = jnp.where((newton > low) & (newton < high), newton, 0.5 * (low + high))
        return theta, low, high, *miss(theta, *fixed), count + 1

    # lambda moves no angle further than the sum of the sizes of its coefficients, and
    # the step after the loop takes a miss of 1e-10 to roundoff
    reach = jnp.abs(fixed[0]).sum()
    start = jnp.broadcast_to(theta_pest, phi.shape)
    theta = jax.lax.while_loop(
        lambda carry: (jnp.abs(carry[3]).max() > 1.0e-10) & (carry[5] < 100),
        refine,
        (start, start - reach, start + reach, *miss(start, *fixed), 0),
    )[0]
    value, slope = miss(theta, lmns, phi)
    return theta - value / slope


def field_line_geometry(wout, modes, *, surface: float, field_lines: int, points: int):
    """The quantities of SIMSOPT's ``vmec_fieldlines`` that the flux compression takes,
    on the surface of the normalized toroidal flux ``surface``.

    Each of ``field_lines`` field lines of one field period holds ``points``
    straight-field-line poloidal angles from ``-pi`` to ``pi``. The radial dependence
    is that of cubic splines through the surfaces of the wout.
    """
    w_full, d_full, w_half, d_half = _radial_weights(wout["rmnc"].shape[1], surface)
    nfp = int(modes.nfp)
    xm, xn = np.asarray(modes.xm, dtype=float), np.asarray(modes.xn, dtype=float)
    xm_nyq = np.asarray(modes.xm_nyq, dtype=float)
    xn_nyq = np.asarray(modes.xn_nyq, dtype=float)

    def on_full(name):
        return wout[name] @ w_full, wout[name] @ d_full

    def on_half(name):
        return wout[name][..., 1:] @ w_half, wout[name][..., 1:] @ d_half

    rmnc, zmns = on_full("rmnc")[0], on_full("zmns")[0]
    lmns, lmns_s = on_half("lmns")
    bmnc, bmnc_s = on_half("bmnc")
    iota, iota_s = on_half("iotas")
    flux = -wout["phi"][-1] / (2.0 * jnp.pi)
    alpha = np.linspace(-np.pi / nfp, np.pi / nfp, field_lines, endpoint=False)
    theta_pest = np.linspace(-np.pi, np.pi, points, endpoint=False)
    phi = (theta_pest[None, :] - alpha[:, None]) / iota
    theta = _poloidal_angles(lmns, xm, xn, phi, theta_pest)

    def series(coefficients, table):
        return jnp.tensordot(coefficients, table, 1)

    angle = xm[:, None, None] * theta - xn[:, None, None] * phi
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    r = series(rmnc, cosine)
    r_t, r_p = series(-rmnc * xm, sine), series(rmnc * xn, sine)
    z_t, z_p = series(zmns * xm, cosine), series(-zmns * xn, cosine)
    lam_s = series(lmns_s, sine)
    lam_t, lam_p = series(lmns * xm, cosine), series(-lmns * xn, cosine)
    angle = xm_nyq[:, None, None] * theta - xn_nyq[:, None, None] * phi
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    jacobian = series(on_half("gmnc")[0], cosine)
    strength, strength_s = series(bmnc, cosine), series(bmnc_s, cosine)
    strength_t = series(-bmnc * xm_nyq, sine)
    strength_p = series(bmnc * xn_nyq, sine)
    b_sub_s = series(on_full("bsubsmns")[0], sine)
    b_sub_t = series(on_half("bsubumnc")[0], cosine)
    b_sub_p = series(on_half("bsubvmnc")[0], cosine)
    # the radial derivative of the field line label theta + lambda - iota phi
    shear = lam_s - phi * iota_s
    b_cross_grad_b_dot_grad_alpha = (
        b_sub_s * strength_t * (lam_p - iota)
        + b_sub_t * strength_p * shear
        + b_sub_p * strength_s * (1.0 + lam_t)
        - b_sub_p * strength_t * shear
        - b_sub_t * strength_s * (lam_p - iota)
        - b_sub_s * strength_p * (1.0 + lam_t)
    ) / jacobian
    return {
        "theta_vmec": theta,
        "B_sup_theta_pest": iota * series(on_half("bsupvmnc")[0], cosine),
        # from the cross product of the derivatives of the position along both angles
        "grad_s_dot_grad_s": (r**2 * (r_t**2 + z_t**2) + (z_t * r_p - r_t * z_p) ** 2)
        / jacobian**2,
        "B_cross_kappa_dot_grad_alpha": b_cross_grad_b_dot_grad_alpha / strength
        + autodiff_wout.MU_0 * on_half("pres")[1] / flux,
        "edge_toroidal_flux_over_2pi": flux,
    }


def unfavourable_part(drift):
    """The part of the cell of each point of a field line on which the curvature drift
    is unfavourable, with the drift taken as linear between neighbouring points."""

    def positive(first, last):
        """The part of an interval on which the line from ``first`` to ``last`` is
        positive."""
        size = jnp.abs(first) + jnp.abs(last)
        return jnp.where(
            size > 0.0,
            (jax.nn.relu(first) + jax.nn.relu(last)) / jnp.where(size > 0.0, size, 1.0),
            0.0,
        )

    middle = 0.5 * (drift[:, :-1] + drift[:, 1:])
    # the halves of the cells towards the next point and towards the last one, with
    # the outer halves of the two ends by the sign of their points
    ahead = jnp.concatenate([positive(drift[:, :-1], middle), drift[:, -1:] > 0.0], 1)
    behind = jnp.concatenate([drift[:, :1] > 0.0, positive(drift[:, 1:], middle)], 1)
    return 0.5 * (ahead + behind)


def flux_compression(
    wout,
    modes,
    *,
    surface: float = 0.49,
    field_lines: int = 101,
    points: int = 64,
    whole_cells: bool = False,
):
    """The flux compression in regions of bad curvature of ConStellaration: the mean
    over a surface of ``|grad r|^2`` where the curvature drift is unfavourable, with
    ``r`` the minor radius times the square root of the normalized toroidal flux,
    weighted by the Jacobian of the straight-field-line coordinates.

    The benchmark counts the whole cell of each point by the sign of the drift at the
    point, as ``whole_cells`` does, and its value jumps where a point changes sign.
    Without it each cell counts by its :func:`unfavourable_part`, which follows the
    boundary of the region continuously.
    """
    found = field_line_geometry(
        wout, modes, surface=surface, field_lines=field_lines, points=points
    )
    weight = jnp.abs(found["edge_toroidal_flux_over_2pi"] / found["B_sup_theta_pest"])
    compression = wout["Aminor_p"] ** 2 / (4.0 * surface) * found["grad_s_dot_grad_s"]
    drift = found["B_cross_kappa_dot_grad_alpha"] * jnp.sign(wout["phipf"][-1])
    part = drift > 0.0 if whole_cells else unfavourable_part(drift)
    return jnp.mean(part * compression * weight) / weight.mean()


@dataclasses.dataclass(kw_only=True)
class MHDStableProblem(SimpleToBuildProblem):
    """The problem ``MHDStableQIStellarator`` of ConStellaration at one aspect ratio:
    its residuals of quasi-isodynamicity under its limits, with the magnetic gradient
    scale length that it scores held above ``length``.

    Beside the constraints of the base class, the magnetic well stays above ``well``
    and the flux compression in regions of bad curvature below ``compression``, on the
    surface of the normalized toroidal flux ``surface`` and a grid of
    ``compression_grid`` field lines and poloidal angles, by default those of the
    benchmark. The benchmark limits no elongation.
    """

    mirror: float = 0.25
    elongation: float = np.inf
    quasi_isodynamicity: float = 10.0**-3.5
    well: float = 0.0
    compression: float = 0.9
    surface: float = 0.49
    compression_grid: tuple[int, int] = (101, 64)

    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = super().limits()
        return (
            np.concatenate([lower, [-np.inf, -np.inf]]),
            np.concatenate([upper, [1.0 - self.well, self.compression]]),
        )

    def stability(self, wout, modes, *, whole_cells: bool = False):
        """``dV/ds`` at the boundary over that at the magnetic axis, which is one less
        the magnetic well, and the flux compression in regions of bad curvature of a
        wout."""
        return jnp.stack(
            [
                1.0 - magnetic_well(wout["vp"]),
                flux_compression(
                    wout,
                    modes,
                    surface=self.surface,
                    field_lines=self.compression_grid[0],
                    points=self.compression_grid[1],
                    whole_cells=whole_cells,
                ),
            ]
        )

    def constraints(self, solved, boundary):
        wout = autodiff_wout.wout_quantities(solved, self.vmec_input)
        return jnp.concatenate(
            [super().constraints(solved, boundary), self.stability(wout, self._modes)]
        )


def score(problem: MHDStableProblem, wout) -> dict[str, float]:
    """The quantities that ConStellaration scores and limits, of a wout."""
    found = simple_to_build_score(problem, wout)
    ratio, compression = problem.stability(
        {name: np.asarray(getattr(wout, name)) for name in _WOUT_QUANTITIES},
        wout,
        whole_cells=True,
    )
    found["well"] = 1.0 - float(ratio)
    found["flux_compression"] = float(compression)
    return found


def check(problem: MHDStableProblem, boundary: np.ndarray, mpol: int, ns: int) -> None:
    """Print what ConStellaration scores and any unmet limit of a cold run on ``mpol``
    poloidal modes and ``ns`` surfaces.

    The benchmark scores a set of boundaries by the area that the rectangles ``(1 + L)
    (20 - A)`` of their gradient scale lengths ``L`` and aspect ratios ``A`` cover
    together. It takes a limit as met within FEASIBILITY_TOLERANCE of it, for
    quasi-isodynamicity of its logarithm and for the magnetic well of 0.1.
    """
    try:
        wout = converged_wout(problem, boundary, mpol, ns)
    except RuntimeError:
        print(f"  converged run on mpol {mpol}: VMEC++ did not converge")
        return
    found = score(problem, wout)
    limit = np.log10(problem.quasi_isodynamicity)
    misses = {
        "edge iota": 1.0 - found["edge_iota"] / problem.edge_iota,
        "quasi-isodynamicity": 1.0 - np.log10(found["quasi_isodynamicity"]) / limit,
        "mirror ratio": found["mirror"] / problem.mirror - 1.0,
        "flux compression": found["flux_compression"] / problem.compression - 1.0,
        "magnetic well": (problem.well - found["well"]) / 0.1,
    }
    unmet = [name for name, miss in misses.items() if miss > FEASIBILITY_TOLERANCE]
    length, aspect = found["gradient_scale_length"], found["aspect"]
    print(
        f"  converged run on mpol {mpol} and {ns} surfaces: gradient scale length"
        f" {length:.4f} at aspect {aspect:.4f}, an area of"
        f" {(1.0 + length) * (20.0 - aspect):.1f}, quasi-isodynamicity"
        f" {found['quasi_isodynamicity']:.3e}, edge iota per period"
        f" {found['edge_iota']:.5f}, mirror ratio {found['mirror']:.4f}, flux"
        f" compression {found['flux_compression']:.4f}, magnetic well"
        f" {found['well']:+.5f}, unmet: {', '.join(unmet) or 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, default=25)
    parser.add_argument("--mpol", type=int, default=5)
    parser.add_argument("--modes", type=int, default=3, help="highest boundary mode")
    parser.add_argument(
        "--length", type=float, default=12.0, help="smallest gradient scale length"
    )
    parser.add_argument("--aspect", type=float, default=10.0, help="largest")
    parser.add_argument(
        "--mirror", type=float, default=0.25, help="largest at the edge"
    )
    parser.add_argument(
        "--elongation", type=float, default=np.inf, help="largest, none by default"
    )
    parser.add_argument("--well", type=float, default=0.0, help="smallest")
    parser.add_argument(
        "--compression", type=float, default=0.9, help="largest flux compression"
    )
    parser.add_argument(
        "--reserve",
        type=float,
        default=0.03,
        help="fraction of the mirror ratio kept free for a run on more modes",
    )
    parser.add_argument(
        "--start",
        type=Path,
        help="rbc and zbs as .npy of the boundary to start from, in place of the"
        " quasi-isodynamic one from the near-axis boundary",
    )
    parser.add_argument(
        "--save", type=Path, help="file for rbc and zbs of the final boundary as .npy"
    )
    parser.add_argument(
        "--nested", action="store_true", help="run the nested baseline instead"
    )
    parser.add_argument(
        "--check",
        type=int,
        nargs=2,
        default=[10, 99],
        help="mpol and ntor, and the surfaces, of the converged run that checks the"
        " result",
    )
    args = parser.parse_args()
    vmec_input = near_axis_start(args.mpol, args.mpol, args.ns)
    limits = {"aspect": args.aspect, "mirror": args.mirror, "reserve": args.reserve}
    if args.start is None:
        # without a bound on the elongation the start ends on a boundary from which
        # VMEC++ does not start cold
        start = QuasiIsodynamicProblem(
            vmec_input,
            max_mode=args.modes,
            elongation=min(args.elongation, 5.0),
            **limits,
        )
        stages = tuple(range(1, args.modes + 1))
        boundary = optimize(start, stages, nested=args.nested, log=print).boundary
    else:
        boundary = np.load(args.start)
    vmec_input.rbc, vmec_input.zbs = (
        vmecpp.VmecInput.resize_2d_coeff(part, args.mpol, args.mpol)
        for part in boundary
    )
    problem = MHDStableProblem(
        vmec_input,
        max_mode=args.modes,
        length=args.length,
        elongation=args.elongation,
        well=args.well,
        compression=args.compression,
        **limits,
    )
    result = optimize(problem, (args.modes,), nested=args.nested, log=print)
    costs = problem.costs
    values = result.evaluation.constraints
    lengths = values[2 + problem.planes : -2]
    print(
        f"{len(result.steps)} steps, {costs.force_evaluations} force evaluations,"
        f" {result.steps[-1].seconds:.1f} s, gradient scale length"
        f" {lengths.min():.4f}, aspect {values[0]:.5f}, edge iota per period"
        f" {values[1]:.5f}, mirror ratio {values[2]:.5f}, magnetic well"
        f" {1.0 - values[-2]:+.5f} and flux compression {values[-1]:.5f} on these modes"
    )
    check(problem, result.boundary, *args.check)
    if args.save is not None:
        np.save(args.save, result.boundary)


if __name__ == "__main__":
    main()
