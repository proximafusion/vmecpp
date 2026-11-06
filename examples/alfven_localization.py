# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Shear-Alfven states along a field line, which of them localize, and how the states at
a line's end cross each gap."""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal

import vmecpp

POINTS_PER_PERIOD = 48  # grid points along the line per field period
BIN = 0.05  # width of the table's bins of omega R / v_A
CROSSING_PERIODS = 300  # field periods of the line for counting end-state crossings


def line_coefficients(wout, js, zeta, slope=None, alpha=0.0):
    """P = |grad s|^2 / B^2 and B^zeta on half-grid surface js along theta* = alpha +
    slope zeta."""
    slope = float(wout.iotas[js]) if slope is None else slope
    xm, xn = np.asarray(wout.xm, dtype=float), np.asarray(wout.xn, dtype=float)
    xm_nyq = np.asarray(wout.xm_nyq, dtype=float)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=float)
    lmns = np.asarray(wout.lmns)[:, js]
    rmnc = 0.5 * (np.asarray(wout.rmnc)[:, js] + np.asarray(wout.rmnc)[:, js - 1])
    zmns = 0.5 * (np.asarray(wout.zmns)[:, js] + np.asarray(wout.zmns)[:, js - 1])
    bmnc = np.asarray(wout.bmnc)[:, js]
    bsupvmnc = np.asarray(wout.bsupvmnc)[:, js]
    gmnc = np.asarray(wout.gmnc)[:, js]
    p, b_zeta = np.empty_like(zeta), np.empty_like(zeta)
    for start in range(0, len(zeta), 16384):
        z = zeta[start : start + 16384]
        star = alpha + slope * z
        theta = star.copy()
        for _ in range(8):
            phase = np.outer(xm, theta) - np.outer(xn, z)
            residual = theta + lmns @ np.sin(phase) - star
            theta -= residual / (1.0 + (lmns * xm) @ np.cos(phase))
        phase = np.outer(xm, theta) - np.outer(xn, z)
        cos, sin = np.cos(phase), np.sin(phase)
        r = rmnc @ cos
        r_t, r_v = -(rmnc * xm) @ sin, (rmnc * xn) @ sin
        z_t, z_v = (zmns * xm) @ cos, -(zmns * xn) @ cos
        cos = np.cos(np.outer(xm_nyq, theta) - np.outer(xn_nyq, z))
        b, jacobian = bmnc @ cos, gmnc @ cos
        metric = (r_t**2 + z_t**2) * (r_v**2 + z_v**2 + r**2)
        metric -= (r_t * r_v + z_t * z_v) ** 2
        p[start : start + 16384] = metric / jacobian**2 / b**2
        b_zeta[start : start + 16384] = bsupvmnc @ cos
    return p, b_zeta


def operator(p, b_zeta, h):
    """The symmetric tridiagonal form with natural ends of (P u_tau)_tau + W^2 P u = 0.

    Here d tau = d zeta / B^zeta and W^2 = omega^2 mu0 rho.
    """
    dtau = h / np.abs(b_zeta)
    conductance = (p[:-1] + p[1:]) / (dtau[:-1] + dtau[1:])
    weight = p * dtau
    weight[1:-1] = p[1:-1] * 0.25 * (dtau[:-2] + 2.0 * dtau[1:-1] + dtau[2:])
    diagonal = np.r_[conductance, 0.0] + np.r_[0.0, conductance]
    return diagonal / weight, -conductance / np.sqrt(weight[:-1] * weight[1:])


def states(diagonal, off_diagonal, low, high):
    """W and the extent in grid points of the states with W in low..high not pinned to a
    line end."""
    values, vectors = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="v",
        select_range=(low**2, high**2),
        lapack_driver="stebz",
    )
    n = len(diagonal)
    density = vectors**2
    extent = 1.0 / np.sum(density**2, axis=0)
    centroid = np.sum(density * np.arange(n)[:, None], axis=0) / n
    margin = 3.0 * extent / n
    pinned = (extent < n / 6.0) & ((centroid < margin) | (centroid > 1.0 - margin))
    return np.sqrt(np.maximum(values[~pinned], 0.0)), extent[~pinned]


def table(
    wout, js, periods, omega_max, slope=None, alpha=0.0, points=POINTS_PER_PERIOD
):
    """The states per bin of omega R / v_A and their median extent in field periods on
    lines of the given periods and half of them."""
    h = 2.0 * np.pi / int(wout.nfp) / points
    zeta = h * (0.5 + np.arange(points * periods))
    p, b_zeta = line_coefficients(wout, js, zeta, slope, alpha)
    half = len(p) // 2
    long_line = operator(p, b_zeta, h)
    half_line = operator(p[:half], b_zeta[:half], h)
    unit = abs(float(wout.b0)) / float(wout.Rmajor_p)
    rows = []
    for low in np.arange(0.0, omega_max, BIN):
        bounds = (low * unit, (low + BIN) * unit)
        extent = states(*long_line, *bounds)[1] / points
        extent_half = states(*half_line, *bounds)[1] / points
        if len(extent) >= 4 and len(extent_half) >= 2:
            rows.append((low, len(extent), np.median(extent), np.median(extent_half)))
    return rows


def widest_gaps(wout, js, periods, omega_max, count, points=POINTS_PER_PERIOD):
    """The omega R / v_A at the edges and the rotation number of the widest gaps between
    the states not pinned to a line end."""
    h = 2.0 * np.pi / int(wout.nfp) / points
    zeta = h * (0.5 + np.arange(points * periods))
    diagonal, off_diagonal = operator(*line_coefficients(wout, js, zeta), h)
    unit = abs(float(wout.b0)) / float(wout.Rmajor_p)
    values = np.sort(states(diagonal, off_diagonal, 0.05 * unit, omega_max * unit)[0])
    gaps = []
    for i in np.sort(np.argsort(np.diff(values))[-count:]):
        below = eigvalsh_tridiagonal(
            diagonal,
            off_diagonal,
            select="v",
            select_range=(0.0, values[i] ** 2 * (1.0 + 1e-7)),
            lapack_driver="stebz",
        )
        rotation = np.pi * len(below) / zeta[-1]
        gaps.append((values[i] / unit, values[i + 1] / unit, rotation))
    return gaps


def end_state_crossings(wout, js, periods, gaps, labels, points=POINTS_PER_PERIOD):
    """The net upward crossings of each gap's middle per line end as alpha advances by 2
    pi, and the mean number of states in the gap per end."""
    h = 2.0 * np.pi / int(wout.nfp) / points
    zeta = h * (0.5 + np.arange(points * periods))
    unit = abs(float(wout.b0)) / float(wout.Rmajor_p)
    below = np.zeros((len(gaps), 2, labels), dtype=int)
    above = np.zeros((len(gaps), 2, labels), dtype=int)
    for a, alpha in enumerate(2.0 * np.pi * np.arange(labels) / labels):
        diagonal, off_diagonal = operator(
            *line_coefficients(wout, js, zeta, None, float(alpha)), h
        )
        for g, gap in enumerate(gaps):
            low, high = gap[0] * unit, gap[1] * unit
            margin = 0.02 * (high - low)
            values, vectors = eigh_tridiagonal(
                diagonal,
                off_diagonal,
                select="v",
                select_range=((low + margin) ** 2, (high - margin) ** 2),
                lapack_driver="stebz",
            )
            centroid = np.sum(vectors**2 * np.arange(len(diagonal))[:, None], axis=0)
            lower = np.sqrt(np.maximum(values, 0.0)) < 0.5 * (low + high)
            left = centroid < 0.5 * len(diagonal)
            for end, mask in enumerate((left, ~left)):
                below[g, end, a] = np.sum(mask & lower)
                above[g, end, a] = np.sum(mask & ~lower)
    rises = np.minimum(
        below - np.roll(below, -1, axis=2), np.roll(above, -1, axis=2) - above
    )
    falls = np.minimum(
        np.roll(below, -1, axis=2) - below, above - np.roll(above, -1, axis=2)
    )
    net = np.sum(np.maximum(rises, 0) - np.maximum(falls, 0), axis=2)
    return net, np.mean(below + above, axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "data" / "w7x.json",
    )
    parser.add_argument("--surfaces", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--periods", type=int, default=1500)
    parser.add_argument("--omega-max", type=float, default=3.2)
    parser.add_argument("--slope", type=float, help="in place of the transform")
    parser.add_argument("--alpha", type=float, default=0.0, help="field-line label")
    parser.add_argument("--points", type=int, default=POINTS_PER_PERIOD)
    parser.add_argument(
        "--crossings",
        type=int,
        default=0,
        help="labels for counting the end states that cross the widest gaps",
    )
    args = parser.parse_args()

    wout = vmecpp.run(vmecpp.VmecInput.from_file(args.input_file), verbose=False).wout
    for s in args.surfaces:
        js = round(s * (int(wout.ns) - 1))
        print(f"s = {s}, iota = {float(wout.iotas[js]):.4f}")
        print(
            "  omega R / v_A, states, extent in periods on the line and on half of it"
        )
        for low, count, extent, extent_half in table(
            wout, js, args.periods, args.omega_max, args.slope, args.alpha, args.points
        ):
            print(
                f"  {low:4.2f}-{low + BIN:4.2f}  {count:4d}  {extent:7.1f}  {extent_half:7.1f}"
            )
        if args.crossings:
            print("  gap in omega R / v_A, rotation number, net crossings at each end")
            gaps = widest_gaps(
                wout, js, CROSSING_PERIODS, args.omega_max, 8, args.points
            )
            net, _ = end_state_crossings(
                wout, js, CROSSING_PERIODS, gaps, args.crossings, args.points
            )
            for gap, (left, right) in zip(gaps, net, strict=True):
                print(
                    f"  {gap[0]:.3f}-{gap[1]:.3f}  {gap[2]:.4f}  {left:+d} {right:+d}"
                )


if __name__ == "__main__":
    main()
