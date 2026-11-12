# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""ConStellaration's simple-to-build quasi-isodynamic problem by vmecpp.joint or by the
nested baseline.

The benchmark scores the smallest magnetic gradient scale length on the boundary under
limits on the aspect ratio, the rotational transform at the edge, quasi-isodynamicity,
the mirror ratio and the elongation. Here its residuals of quasi-isodynamicity are the
objective and the other limits are constraints, with the gradient scale length held
above a bound at every point of the benchmark's grid, all as the benchmark computes
them, in JAX. The start is the boundary that the problem of
examples/joint_quasi_isodynamic.py reaches from the near-axis boundary within the
limits, or a saved boundary. A cold, fully converged run on more modes checks the
result.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import vmecpp
from vmecpp import autodiff_wout
from vmecpp.joint import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_quasi_isodynamic import (  # type: ignore
    FEASIBILITY_TOLERANCE,
    QuasiIsodynamicProblem,
    converged_wout,
    measures,
    mirror_ratio,
    near_axis_start,
)

jax.config.update("jax_enable_x64", True)

_WOUT_QUANTITIES = ("rmnc", "zmns", "gmnc", "bmnc", "bsupumnc", "bsupvmnc", "Aminor_p")


def gradient_scale_lengths(wout, modes, nfp: int, *, n_theta: int, n_phi: int):
    """The magnetic gradient scale length ``B sqrt(2 / (grad B : grad B))`` on the
    boundary over the minor radius, times the number of field periods, as
    ConStellaration computes it.

    The grid holds ``n_theta`` poloidal angles and ``n_phi`` toroidal angles over half a
    field period, with both ends. Quantities of the half grid and the radial
    derivatives are extrapolated to the boundary from the outermost four surfaces.
    """
    theta, phi = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, n_theta),
        np.linspace(0.0, np.pi / nfp, n_phi),
        indexing="ij",
    )
    ds = 1.0 / (wout["rmnc"].shape[1] - 1)

    def on_full(x):
        """A quantity of the full grid and its radial derivative at the boundary."""
        slope = 11 / 6 * x[:, -1] - 3 * x[:, -2] + 3 / 2 * x[:, -3] - 1 / 3 * x[:, -4]
        return x[:, -1], slope / ds

    def on_half(x):
        """A quantity of the half grid and its radial derivative at the boundary."""
        outer = (1.5 * x[:, -1] - 2.0 * x[:, -2] + 0.5 * x[:, -3]) / ds
        inner = (x[:, -1] - x[:, -3]) / (2 * ds), (x[:, -2] - x[:, -4]) / (2 * ds)
        return (
            7 / 4 * x[:, -1] - x[:, -2] + 1 / 4 * x[:, -3],
            7 / 4 * outer - inner[0] + 1 / 4 * inner[1],
        )

    def tables(xm, xn):
        m, n = xm[:, None, None], xn[:, None, None]
        angle = m * theta - n * phi
        return m, n, np.cos(angle), np.sin(angle)

    def series(coefficients, table):
        return jnp.tensordot(coefficients, table, 1)

    m, n, c, s = tables(np.asarray(modes.xm), np.asarray(modes.xn))
    rmnc, rmnc_s = on_full(wout["rmnc"])
    zmns, zmns_s = on_full(wout["zmns"])
    r, r_t, r_p = series(rmnc, c), series(rmnc, -m * s), series(rmnc, n * s)
    r_tt, r_tp = series(rmnc, -m * m * c), series(rmnc, m * n * c)
    r_pp = series(rmnc, -n * n * c)
    z_t, z_p = series(zmns, m * c), series(zmns, -n * c)
    z_tt, z_tp = series(zmns, -m * m * s), series(zmns, m * n * s)
    z_pp = series(zmns, -n * n * s)
    r_s, z_s = series(rmnc_s, c), series(zmns_s, s)
    r_st, r_sp = series(rmnc_s, -m * s), series(rmnc_s, n * s)
    z_st, z_sp = series(zmns_s, m * c), series(zmns_s, -n * c)

    m, n, c, s = tables(np.asarray(modes.xm_nyq), np.asarray(modes.xn_nyq))
    strength = series(on_half(wout["bmnc"])[0], c)
    jacobian = series(on_half(wout["gmnc"])[0], c)
    bsupu, bsupu_s = on_half(wout["bsupumnc"])
    bsupv, bsupv_s = on_half(wout["bsupvmnc"])
    bu, bu_t, bu_p = series(bsupu, c), series(bsupu, -m * s), series(bsupu, n * s)
    bv, bv_t, bv_p = series(bsupv, c), series(bsupv, -m * s), series(bsupv, n * s)
    bu_s, bv_s = series(bsupu_s, c), series(bsupv_s, c)

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    def cartesian(radial, toroidal, vertical):
        """The Cartesian components of a gradient from its cylindrical ones."""
        return (
            radial * cos_phi - toroidal * sin_phi,
            radial * sin_phi + toroidal * cos_phi,
            vertical,
        )

    grad_s = cartesian(
        -z_t * r / jacobian, (r_p * z_t - r_t * z_p) / jacobian, r_t * r / jacobian
    )
    grad_theta = cartesian(
        z_s * r / jacobian, (r_s * z_p - r_p * z_s) / jacobian, -r_s * r / jacobian
    )
    grad_phi = cartesian(0.0 * jacobian, 1.0 / r, 0.0 * jacobian)

    # the derivatives of the Cartesian components of B = bu dx/dtheta + bv dx/dphi
    # along the radius, the poloidal angle and the toroidal angle
    bx = (
        bu_s * r_t * cos_phi
        + bu * r_st * cos_phi
        + bv_s * r_p * cos_phi
        + bv * r_sp * cos_phi
        - bv_s * r * sin_phi
        - bv * r_s * sin_phi,
        bu_t * r_t * cos_phi
        + bu * r_tt * cos_phi
        + bv_t * r_p * cos_phi
        + bv * r_tp * cos_phi
        - bv_t * r * sin_phi
        - bv * r_t * sin_phi,
        bu_p * r_t * cos_phi
        + bu * r_tp * cos_phi
        - bu * r_t * sin_phi
        + bv_p * r_p * cos_phi
        + bv * r_pp * cos_phi
        - bv * r_p * sin_phi
        - bv_p * r * sin_phi
        - bv * r_p * sin_phi
        - bv * r * cos_phi,
    )
    by = (
        bu_s * r_t * sin_phi
        + bu * r_st * sin_phi
        + bv_s * r_p * sin_phi
        + bv * r_sp * sin_phi
        + bv_s * r * cos_phi
        + bv * r_s * cos_phi,
        bu_t * r_t * sin_phi
        + bu * r_tt * sin_phi
        + bv_t * r_p * sin_phi
        + bv * r_tp * sin_phi
        + bv_t * r * cos_phi
        + bv * r_t * cos_phi,
        bu_p * r_t * sin_phi
        + bu * r_tp * sin_phi
        + bu * r_t * cos_phi
        + bv_p * r_p * sin_phi
        + bv * r_pp * sin_phi
        + bv * r_p * cos_phi
        + bv_p * r * cos_phi
        + bv * r_p * cos_phi
        - bv * r * sin_phi,
    )
    bz = (
        bu_s * z_t + bu * z_st + bv_s * z_p + bv * z_sp,
        bu_t * z_t + bu * z_tt + bv_t * z_p + bv * z_tp,
        bu_p * z_t + bu * z_tp + bv_p * z_p + bv * z_pp,
    )
    squares = sum(
        (
            component[0] * grad_s[axis]
            + component[1] * grad_theta[axis]
            + component[2] * grad_phi[axis]
        )
        ** 2
        for component in (bx, by, bz)
        for axis in range(3)
    )
    return strength * jnp.sqrt(2.0 / squares) / wout["Aminor_p"] * nfp


def _mean_of_cubic(points: int) -> np.ndarray:
    """The weights of the values at ``points`` evenly spaced points that give the mean,
    over their interval, of the cubic polynomial that fits them best."""
    fit = np.linalg.pinv(np.vander(np.linspace(-1.0, 1.0, points), 4, increasing=True))
    return fit[0] + fit[2] / 3.0


def _nested(first, last):
    """The two points of each level, from the lowest level up, with those of a level
    moved together until they lie between those of the level above."""

    def inward(outer, points):
        left, right = points
        shift = jnp.where(left < outer[0], outer[0] - left + 1.0e-12, 0.0)
        left, right = left + shift, right + shift
        shift = jnp.where(outer[1] < right, outer[1] - right - 1.0e-12, 0.0)
        left, right = left + shift, right + shift
        return (left, right), (left, right)

    _, (lefts, rights) = jax.lax.scan(
        inward, (first[-1], last[-1]), (first[-2::-1], last[-2::-1])
    )
    return (
        jnp.concatenate([lefts[::-1], first[-1:]]),
        jnp.concatenate([rights[::-1], last[-1:]]),
    )


def shuffled_well_residuals(
    a,
    d,
    iota,
    nfp: int,
    half_period,
    *,
    field_lines: int = 75,
    toroidal_points: int = 601,
    levels: int = 401,
    output_points: int = 2000,
):
    """ConStellaration's residuals of quasi-isodynamicity, after Goodman et al. (2023),
    of the field strength ``a``, ``d`` of ``boozer_strength`` on a surface of rotational
    transform ``iota``. The sum of their squares is the ``qi`` of the benchmark.

    Each field line runs over one field period, from half a period on where
    ``half_period`` holds, with the field strength scaled to span 0 to 1 over all of
    them. Its well is squashed to fall from its maximum to its minimum and rise again
    without a second well, and stretched to reach 1 at both ends. The distance between
    the two points of each level is then set to its mean over the field lines, weighted
    by the inverse of the mean squared change of each line so far, with the points of a
    level kept between those of the level above. The residuals are the differences of
    that field from the field strength at ``output_points`` points of each line.
    """
    period = 2.0 * jnp.pi / nfp
    start = jnp.where(half_period, 0.5 * period, 0.0)
    phi = start + period * jnp.linspace(0.0, 1.0, toroidal_points)
    alpha = -iota * start + 2.0 * jnp.pi * jnp.linspace(0.0, 1.0, field_lines)
    theta = alpha[:, None] + iota * phi[None, :]
    m = jnp.arange(a.shape[0], dtype=jnp.float64)
    n = jnp.arange(a.shape[1], dtype=jnp.float64) * nfp
    cos_n, sin_n = jnp.cos(jnp.outer(n, phi)), jnp.sin(jnp.outer(n, phi))

    def with_mode(total, mode):
        order, cosine, sine = mode
        angle = order * theta
        return total + jnp.cos(angle) * cosine + jnp.sin(angle) * sine, None

    # one poloidal mode at a time keeps the lines by points by modes out of memory
    strength, _ = jax.lax.scan(
        with_mode, jnp.zeros_like(theta), (m, a @ cos_n, d @ sin_n)
    )
    strength = (strength - strength.min()) / (strength.max() - strength.min())
    index = jnp.arange(toroidal_points)
    level = jnp.linspace(0.0, 1.0, levels)
    values = jnp.concatenate([level[::-1], level[1:]])
    output = start + period * jnp.linspace(0.0, 1.0, output_points)
    mean_of_fit = jnp.asarray(_mean_of_cubic(toroidal_points))

    def bounces(line):
        """The two points of each level on the squashed and stretched well of a line,
        and the weight of the line."""
        lowest = jnp.argmin(line)
        low = line[lowest]
        # the left side falls from its maximum, the right side rises to its maximum
        peak = jnp.argmax(jnp.where(index <= lowest, line, -jnp.inf))
        left = jnp.where(index < peak, line[peak], line)
        left = jax.lax.cummin(jnp.where(index > lowest, low, left))
        peak = jnp.argmax(jnp.where(index >= lowest, line, -jnp.inf))
        right = jnp.where(index > peak, line[peak], line)
        right = jax.lax.cummin(jnp.where(index < lowest, low, right), reverse=True)
        left = (left - low) / (left[0] - low)
        right = (right - low) / (right[-1] - low)
        change = (line - jnp.where(index < lowest, left, right)) ** 2
        bottom = phi[lowest][None]
        return (
            jnp.concatenate(
                [bottom, jnp.interp(level[1:-1], left[::-1], phi[::-1]), phi[:1]]
            ),
            jnp.concatenate([bottom, jnp.interp(level[1:-1], right, phi), phi[-1:]]),
            1.0 / (mean_of_fit @ change),
        )

    def ideal(first, last):
        """The field strength of a line with these points of its levels, at the output
        points."""
        points = jnp.concatenate([first[::-1], last[1:]])
        inner_first, inner_last = _nested(first, last)
        nested = jnp.concatenate([inner_first[::-1], inner_last[1:]])
        points = jnp.where(jnp.any(jnp.diff(points) <= 0.0), nested, points)
        return jnp.interp(output, points, values)

    first, last, weight = jax.vmap(bounces)(strength)
    distance = last - first
    excess = 0.5 * (distance - (weight / weight.sum()) @ distance)
    target = jax.vmap(ideal)(first + excess, last - excess)
    actual = jax.vmap(lambda line: jnp.interp(output, phi, line))(strength)
    return (target - actual).ravel() / jnp.sqrt(output_points * field_lines)


@dataclasses.dataclass(kw_only=True)
class SimpleToBuildProblem(QuasiIsodynamicProblem):
    """The problem ``SimpleToBuildQIStellarator`` of ConStellaration: its residuals of
    quasi-isodynamicity under its other limits, with the magnetic gradient scale length
    that it scores held above ``length``.

    The constraints are the aspect ratio below ``aspect``, the rotational transform at
    the edge per field period above ``edge_iota``, the mirror ratio and the elongations
    of the base class, and the gradient scale length above ``length`` at every point of
    a grid of ``grid`` poloidal and toroidal angles, by default that of the benchmark's
    runs at high fidelity, on 10 poloidal and toroidal modes. The benchmark grants its
    score while the sum of the squares of the residuals stays below
    ``quasi_isodynamicity``.
    """

    edge_iota: float = 0.25
    length: float = 8.0
    grid: tuple[int, int] = (26, 24)
    quasi_isodynamicity: float = 1.0e-4
    field_lines: int = 75
    toroidal_points: int = 601
    levels: int = 401
    output_points: int = 2000

    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = super().limits()
        # the last poloidal angle of the grid is the first again
        points = (self.grid[0] - 1) * self.grid[1]
        return (
            np.concatenate(
                [[-np.inf, self.edge_iota], lower, np.full(points, self.length)]
            ),
            np.concatenate([[self.aspect, np.inf], upper, np.full(points, np.inf)]),
        )

    def lengths(self, wout, modes):
        """The gradient scale lengths of a wout on the grid."""
        return gradient_scale_lengths(
            wout, modes, int(modes.nfp), n_theta=self.grid[0], n_phi=self.grid[1]
        )

    def wells(self, wout, tables):
        """The benchmark's residuals of quasi-isodynamicity on the outermost half-grid
        surface, from the quantities of a wout."""
        a, d, iota = self.spectrum(wout, tables)
        # the benchmark takes the place of the maximum from the sign of the mode
        # (0, 1) of the field strength on the second half-grid surface
        return shuffled_well_residuals(
            a,
            d,
            iota,
            tables.nfp,
            wout["bmnc"][1, 2] < 0,
            field_lines=self.field_lines,
            toroidal_points=self.toroidal_points,
            levels=self.levels,
            output_points=self.output_points,
        )

    def residuals(self, solved, boundary):  # noqa: ARG002
        return self.wells(
            autodiff_wout.wout_quantities(solved, self.vmec_input), self._tables
        )

    def constraints(self, solved, boundary):  # noqa: ARG002
        wout = autodiff_wout.wout_quantities(solved, self.vmec_input)
        _, strength, elongation = self.edge(wout, self._tables)
        return jnp.concatenate(
            [
                jnp.stack([wout["aspect"], wout["iotaf"][-1] / self._modes.nfp]),
                jnp.atleast_1d(mirror_ratio(strength)),
                elongation[:-1],
                self.lengths(wout, self._modes)[:-1].ravel(),
            ]
        )

    def parts(self, residuals: np.ndarray) -> dict[str, float]:
        return {"quasi_isodynamicity": float(residuals @ residuals)}


def score(problem: SimpleToBuildProblem, wout) -> dict[str, float]:
    """The quantities that ConStellaration scores and limits, of a wout."""
    found = measures(problem, wout)
    lengths = problem.lengths(
        {name: np.asarray(getattr(wout, name)) for name in _WOUT_QUANTITIES}, wout
    )
    found["gradient_scale_length"] = float(lengths.min())
    found["edge_iota"] = float(wout.iotaf[-1]) / wout.nfp
    return found


def check(problem: SimpleToBuildProblem, boundary: np.ndarray, mpol: int) -> None:
    """Print what ConStellaration scores and any unmet limit of a cold run on ``mpol``
    poloidal modes.

    The score is the gradient scale length over 20. The benchmark takes a limit as met
    within FEASIBILITY_TOLERANCE of it, for quasi-isodynamicity of its logarithm.
    """
    try:
        wout = converged_wout(problem, boundary, mpol)
    except RuntimeError:
        print(f"  converged run on mpol {mpol}: VMEC++ did not converge")
        return
    found = score(problem, wout)
    limit = np.log10(problem.quasi_isodynamicity)
    misses = {
        "aspect": found["aspect"] / problem.aspect - 1.0,
        "edge iota": 1.0 - found["edge_iota"] / problem.edge_iota,
        "quasi-isodynamicity": 1.0 - np.log10(found["quasi_isodynamicity"]) / limit,
        "mirror ratio": found["mirror"] / problem.mirror - 1.0,
        "elongation": found["elongation"] / problem.elongation - 1.0,
    }
    unmet = [name for name, miss in misses.items() if miss > FEASIBILITY_TOLERANCE]
    print(
        f"  converged run on mpol {mpol}: gradient scale length"
        f" {found['gradient_scale_length']:.4f}, a score of"
        f" {found['gradient_scale_length'] / 20.0:.3f}, quasi-isodynamicity"
        f" {found['quasi_isodynamicity']:.3e}, aspect {found['aspect']:.5f}, edge iota"
        f" per period {found['edge_iota']:.5f}, mirror ratio {found['mirror']:.4f},"
        f" elongation {found['elongation']:.3f}, unmet: {', '.join(unmet) or 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, default=25)
    parser.add_argument("--mpol", type=int, default=6)
    parser.add_argument("--modes", type=int, default=4, help="highest boundary mode")
    parser.add_argument(
        "--length", type=float, default=12.0, help="smallest gradient scale length"
    )
    parser.add_argument("--aspect", type=float, default=10.0, help="largest")
    parser.add_argument("--mirror", type=float, default=0.2, help="largest at the edge")
    parser.add_argument("--elongation", type=float, default=5.0, help="largest")
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
        "--check-mpol",
        type=int,
        default=10,
        help="mpol and ntor of the converged run that checks the result",
    )
    args = parser.parse_args()
    vmec_input = near_axis_start(args.mpol, args.mpol, args.ns)
    limits = {
        "aspect": args.aspect,
        "mirror": args.mirror,
        "elongation": args.elongation,
        "reserve": args.reserve,
    }
    if args.start is None:
        start = QuasiIsodynamicProblem(vmec_input, max_mode=args.modes, **limits)
        stages = tuple(range(1, args.modes + 1))
        boundary = optimize(start, stages, nested=args.nested, log=print).boundary
    else:
        boundary = np.load(args.start)
    vmec_input.rbc, vmec_input.zbs = (
        vmecpp.VmecInput.resize_2d_coeff(part, args.mpol, args.mpol)
        for part in boundary
    )
    problem = SimpleToBuildProblem(
        vmec_input, max_mode=args.modes, length=args.length, **limits
    )
    result = optimize(problem, (args.modes,), nested=args.nested, log=print)
    costs = problem.costs
    values = result.evaluation.constraints
    print(
        f"{len(result.steps)} steps, {costs.force_evaluations} force evaluations,"
        f" {result.steps[-1].seconds:.1f} s, gradient scale length"
        f" {values[2 + problem.planes :].min():.4f}, aspect {values[0]:.5f}, edge iota"
        f" per period {values[1]:.5f}, mirror ratio {values[2]:.5f} and elongation"
        f" {values[3 : 2 + problem.planes].max():.4f} on these modes"
    )
    check(problem, result.boundary, args.check_mpol)
    if args.save is not None:
        np.save(args.save, result.boundary)


if __name__ == "__main__":
    main()
