# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Trapped ions that keep time in a Huygens well of a quasi-isodynamic field.

:func:`construct` writes the inputs of the Huygens well and of a cosine well from pyQIC.
"""

import itertools
import tempfile
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal

import vmecpp

DATA = Path(__file__).parent / "data"
HUYGENS = DATA / "isochronous_bounce_huygens.json"
COSINE = DATA / "isochronous_bounce_cosine.json"
# The share of the half period in arc length over which the well is a parabola.
KNEE = 0.4
# Sample points of the arc length from the minimum in lengths of a field period.
SHARE = np.linspace(-0.5, 0.5, 8192, endpoint=False) + 0.5 / 8192
# Fractions of the depth of a well at which the bounce periods are compared.
BAND = np.linspace(0.25, 0.75, 41)
# Fractions of the depth over which the bounce periods of the designed well agree.
DESIGN_BAND = np.linspace(0.02, 0.75, 60)
# The nodes of the Gauss-Chebyshev quadrature of the bounce integrals.
NODES = np.cos((2.0 * np.arange(1, 129) - 1.0) * np.pi / 256)
# Samples of the Boozer toroidal angle over a field period.
BOOZER = 2.0 * np.pi * np.arange(8192) / 8192


def huygens_shape(share):
    """(B - B_{min}) / (B_{max} - B_{min}) of the Huygens well at the arc length
    ``share``."""
    t = np.clip((np.abs(share) - KNEE) / (0.5 - KNEE), 0.0, 1.0)
    cap = (
        0.75 * (2 * t**3 - 3 * t**2 + 1)
        + 1.5 * (0.5 - KNEE) / KNEE * (t**3 - 2 * t**2 + t)
        + (3 * t**2 - 2 * t**3)
    )
    return np.where(np.abs(share) <= KNEE, 0.75 * (share / KNEE) ** 2, cap)


def boozer_series(strength, harmonics=24):
    """The coefficients of \\cos(k \\varphi) of the axis field ``strength`` at SHARE in
    the Boozer angle \\varphi from the maximum."""
    count = len(SHARE)
    total = np.sum(strength)
    angle = (
        np.pi + 2.0 * np.pi * (np.cumsum(strength) - strength / 2 - total / 2) / total
    )
    grid = 2.0 * np.pi * np.arange(count) / count
    resampled = np.interp(
        grid, np.mod(angle, 2.0 * np.pi), strength, period=2.0 * np.pi
    )
    return cosine_coefficients(resampled, harmonics)


def cosine_coefficients(samples, harmonics):
    """The cosine coefficients of ``samples`` at evenly spaced \\varphi from 0."""
    coefficients = np.fft.rfft(samples).real / len(samples)
    coefficients[1:] *= 2.0
    return coefficients[: harmonics + 1]


def cosine_series(coefficients, modes=None):
    """B, dB / d\\phi and d^2 B / d\\phi^2 of \\sum_k c_k \\cos(n_k \\phi) as a function
    of \\phi."""
    c = np.asarray(coefficients, dtype=float)
    n = np.arange(len(c), dtype=float) if modes is None else np.asarray(modes, float)

    def strength(phi):
        angle = np.multiply.outer(phi, n)
        return (
            np.cos(angle) @ c,
            -np.sin(angle) @ (n * c),
            -np.cos(angle) @ (n * n * c),
        )

    return strength


def extremum(strength, phi):
    """The \\phi of the extremum of B nearest ``phi``."""
    for _ in range(8):
        _, slope, curvature = strength(phi)
        phi = phi - slope / curvature
    return phi


def crossings(strength, levels, lo, hi):
    """The \\phi in [``lo``, ``hi``] at which B equals each of ``levels``."""
    f_lo = strength(lo)[0] - levels
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        f_mid = strength(mid)[0] - levels
        same = np.sign(f_mid) == np.sign(f_lo)
        lo, f_lo = np.where(same, mid, lo), np.where(same, f_mid, f_lo)
        hi = np.where(same, hi, mid)
    phi = 0.5 * (lo + hi)
    for _ in range(3):
        value, slope, _ = strength(phi)
        phi = phi - (value - levels) / slope
    return phi


def bounce_periods(strength, speed, top, bottom, fractions, period=2.0 * np.pi):
    """The integrals of ``speed`` d\\phi / \\sqrt{level - B} across a well at
    ``fractions`` of its depth."""
    low, high = strength(bottom)[0], strength(top)[0]
    levels = low + (high - low) * np.asarray(fractions)
    ones = np.ones_like(levels)
    left = crossings(strength, levels, top * ones, bottom * ones)
    right = crossings(strength, levels, bottom * ones, (top + period) * ones)
    centre, half = 0.5 * (left + right), 0.5 * (right - left)
    phi = centre[:, None] + half[:, None] * NODES
    q = (levels[:, None] - strength(phi)[0]) / (1.0 - NODES**2)
    return np.pi / len(NODES) * np.sum(half[:, None] * speed(phi) / np.sqrt(q), 1)


def boozer_periods(coefficients, fractions):
    """The bounce periods over their mean and the depth of the axis field \\sum_k c_k
    \\cos(k \\varphi)."""
    strength = cosine_series(coefficients)
    grid = BOOZER[:: len(BOOZER) // 1024]
    bottom = extremum(strength, grid[np.argmin(strength(grid)[0])])
    # dl goes as d\varphi / B in the Boozer angle.
    periods = bounce_periods(
        strength, lambda phi: 1.0 / strength(phi)[0], 0.0, bottom, fractions
    )
    return periods / periods.mean(), strength(0.0)[0] - strength(bottom)[0]


def design_well(low, high, harmonics=40):
    """The coefficients of \\cos(k \\varphi) of an axis field from ``low`` to ``high``
    whose bounce periods agree over DESIGN_BAND."""

    def residual(coefficients):
        periods, depth = boozer_periods(coefficients, DESIGN_BAND)
        return np.concatenate([periods - 1.0, [depth / (high - low) - 1.0]])

    start = boozer_series(low + (high - low) * huygens_shape(SHARE), harmonics)
    return scipy.optimize.least_squares(
        residual, start, x_scale=1e-3, xtol=1e-15, ftol=1e-15, gtol=1e-15
    ).x


def bounce_integral(strength, length, level):
    """The integral of dl / \\sqrt{level - B} across the well about the least sampled
    B.

    B rises above ``level`` on both sides of the well.
    """
    u = level - strength
    left = right = int(np.argmin(strength))
    while u[left - 1] > 0.0:
        left -= 1
    while u[right + 1] > 0.0:
        right += 1
    h = np.diff(length)
    a, b = np.sqrt(u[left:right]), np.sqrt(u[left + 1 : right + 1])
    total = np.sum(2.0 * h[left:right] / (a + b))
    # The steps that hold the bounce points.
    for inside, outside, step in (
        (left, left - 1, h[left - 1]),
        (right, right + 1, h[right]),
    ):
        total += 2.0 * step * np.sqrt(u[inside]) / (u[inside] - u[outside])
    return total


def axis_series(wout):
    """B and dl / d\\phi on the magnetic axis of a wout as functions of \\phi."""
    ns = int(wout.ns)
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn, dtype=float)
    r_n, z_n, n = (
        np.asarray(wout.rmnc)[xm == 0, 0],
        np.asarray(wout.zmns)[xm == 0, 0],
        xn[xm == 0],
    )
    q = np.asarray(wout.xm_nyq) == 0
    # |B| at s = 0 from the three innermost half-grid surfaces by a quadratic in s.
    s = (np.arange(1, 4) - 0.5) / (ns - 1)
    weights = np.linalg.solve(np.vander(s, 3, increasing=True).T, [1.0, 0.0, 0.0])
    strength = cosine_series(
        np.asarray(wout.bmnc)[q, 1:4] @ weights, np.asarray(wout.xn_nyq)[q]
    )

    def speed(phi):
        angle = np.multiply.outer(phi, n)
        c, sn = np.cos(angle), np.sin(angle)
        return np.sqrt((c @ r_n) ** 2 + (sn @ (n * r_n)) ** 2 + (c @ (n * z_n)) ** 2)

    return strength, speed


def axis_periods(wout, fractions=BAND):
    """The bounce periods on the magnetic axis of a wout over their mean."""
    strength, speed = axis_series(wout)
    period = 2.0 * np.pi / int(wout.nfp)
    grid = np.linspace(0.0, period, 4096, endpoint=False)
    b = strength(grid)[0]
    top = extremum(strength, grid[np.argmax(b)])
    bottom = extremum(strength, grid[np.argmin(b)])
    bottom += period if bottom < top else 0.0
    periods = bounce_periods(strength, speed, top, bottom, fractions, period)
    return periods / periods.mean()


def axis_boozer(wout):
    """|B| on the magnetic axis of a wout at the Boozer angles BOOZER from its
    maximum."""
    strength, speed = axis_series(wout)
    phi = 2.0 * np.pi * np.arange(4 * len(BOOZER)) / (4 * len(BOOZER))
    b = strength(phi)[0]
    # d\varphi goes as B dl along the axis.
    weight = b * speed(phi)
    angle = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]))])
    angle *= 2.0 * np.pi / (angle[-1] + 0.5 * (weight[-1] + weight[0]))
    top = extremum(strength, phi[np.argmax(b)])
    angle = np.mod(angle - np.interp(top, phi, angle), 2.0 * np.pi)
    order = np.argsort(angle)
    spline = scipy.interpolate.CubicSpline(
        np.append(angle[order], angle[order][0] + 2.0 * np.pi),
        np.append(b[order], b[order][0]),
        bc_type="periodic",
    )
    return spline(BOOZER)


def rk4(rates, state, t, dt):
    """The state after a classical Runge-Kutta step from ``t`` to ``t + dt``, and |B| at
    its start.

    ``rates(state, t)`` returns d state / dt and |B|.
    """
    k1, strength = rates(state, t)
    k2 = rates(state + 0.5 * dt * k1, t + 0.5 * dt)[0]
    k3 = rates(state + 0.5 * dt * k2, t + 0.5 * dt)[0]
    k4 = rates(state + dt * k3, t + dt)[0]
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), strength


def field_lines(wout, s, alphas, periods, steps):
    """|B|, the arc length and \\theta along field lines from \\theta = ``alphas`` on
    the half-grid surface nearest ``s``, and its s."""
    ns = int(wout.ns)
    js = int(np.clip(np.round(s * (ns - 1) + 0.5), 1, ns - 1))
    m = np.asarray(wout.xm_nyq, dtype=int)
    n = np.asarray(wout.xn_nyq, dtype=int)
    poloidal = np.arange(m.max() + 1)
    coefficients = np.stack(
        [np.asarray(a)[:, js] for a in (wout.bmnc, wout.bsupumnc, wout.bsupvmnc)],
        axis=1,
    )

    def slope(state, phi):
        """The rates d / d\\phi of \\theta and the arc length of each line, and its
        |B|."""
        wave = np.exp(1j * np.outer(state[0], poloidal))[:, m] * np.exp(-1j * n * phi)
        strength, up, along = (wave.real @ coefficients).T
        return np.array([up / along, strength / np.abs(along)]), strength

    h = 2.0 * np.pi / int(wout.nfp) / steps
    state = np.array([alphas, np.zeros(len(alphas))], dtype=float)
    strengths, states = [], []
    for k in range(round(periods * steps) + 1):
        states.append(state)
        state, strength = rk4(slope, state, k * h, h)
        strengths.append(strength)
    theta, length = np.stack(states, axis=-1)
    s_found = (js - 0.5) / (ns - 1)
    return np.array(strengths).T, length, theta, s_found


def guiding_centers(wout, tolerance=1e-5):
    """The rates of (s, \\theta, \\zeta, v_\\parallel) of guiding centers and |B| in the
    vacuum field of a wout.

    ``moment`` is the magnetic moment over the mass and ``inertia`` the mass over the
    charge.
    """
    arrays = [
        np.asarray(a)
        for a in (
            wout.bmnc,
            wout.bsubumnc,
            wout.bsubvmnc,
            wout.bsupumnc,
            wout.bsupvmnc,
            wout.gmnc,
            wout.bsubsmns,
        )
    ]
    m = np.asarray(wout.xm_nyq, dtype=int)
    n = np.asarray(wout.xn_nyq, dtype=int)
    size = np.max([np.abs(a).max(1) / np.abs(a).max() for a in arrays], axis=0)
    size *= 1 + m + np.abs(n)
    keep = size > tolerance * size.max()
    m, n = m[keep], n[keep]
    *cosines, radial = (a[keep] for a in arrays)
    ns = int(wout.ns)
    half = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))
    full = np.sqrt(np.arange(ns) / (ns - 1))
    cosine_mn = scipy.interpolate.CubicSpline(
        half, np.stack([a[:, 1:].T for a in cosines], axis=1)
    )
    # B_s moves from the full grid of the wout to the half grid.
    radial_mn = scipy.interpolate.CubicSpline(
        half, scipy.interpolate.CubicSpline(full, radial.T)(half)
    )

    # The [piece, mode, column] coefficients of t^3, t^2, t and 1 of each quantity.
    cosine_pieces = np.transpose(cosine_mn.c, (1, 3, 0, 2))
    strength_pieces = cosine_pieces[..., 0]
    cosine_pieces = cosine_pieces.reshape(*cosine_pieces.shape[:2], -1)
    sine_pieces = np.concatenate(
        [
            -m[:, None] * strength_pieces,
            n[:, None] * strength_pieces,
            np.transpose(radial_mn.c, (1, 2, 0)),
        ],
        axis=-1,
    )
    powers = np.arange(3, -1, -1)

    def rates(state, moment, inertia):
        s, theta, zeta, v = state
        rho = np.sqrt(s)
        wave = (
            np.exp(1j * np.outer(theta, np.arange(m.max() + 1)))[:, m]
            * np.exp(-1j * np.outer(zeta, np.arange(n.min(), n.max() + 1)))[
                :, n - n.min()
            ]
        )
        piece = np.clip(np.searchsorted(half, rho) - 1, 0, len(half) - 2)
        order = np.argsort(piece, kind="stable")
        ordered = piece[order]
        cuts = [0, *(np.flatnonzero(np.diff(ordered)) + 1).tolist(), len(rho)]
        c, sn = wave.real[order], wave.imag[order]
        cosine_terms = np.empty((len(rho), cosine_pieces.shape[-1]))
        sine_terms = np.empty((len(rho), sine_pieces.shape[-1]))
        for first, last in itertools.pairwise(cuts):
            k = ordered[first]
            cosine_terms[first:last] = c[first:last] @ cosine_pieces[k]
            sine_terms[first:last] = sn[first:last] @ sine_pieces[k]
        back = np.argsort(order)
        t = (rho - half[piece])[:, None]
        scaled = t**powers
        terms = cosine_terms[back].reshape(len(rho), 4, 6)
        strength, b_theta, b_zeta, up, along, jacobian = (
            (scaled[:, :, None] * terms).sum(1).T
        )
        slopes = powers * t ** np.maximum(powers - 1, 0)
        d_s = np.sum(slopes * terms[:, :, 0], 1) / (2.0 * rho)
        d_theta, d_zeta, b_s = (
            (scaled[:, None, :] * sine_terms[back].reshape(len(rho), 3, 4)).sum(2).T
        )
        drift = inertia * (v**2 / strength + moment) / (strength**2 * jacobian)
        change = np.array(
            [
                drift * (b_theta * d_zeta - b_zeta * d_theta),
                v * up / strength + drift * (b_zeta * d_s - b_s * d_zeta),
                v * along / strength + drift * (b_s * d_theta - b_theta * d_s),
                -moment * (up * d_theta + along * d_zeta) / strength,
            ]
        )
        return change, strength

    return rates


def first_wells(wout, s, alphas, steps=500):
    """The first whole well of field lines from \\theta = ``alphas`` on the half-grid
    surface nearest ``s``, and its s."""
    strength, length, theta, s_found = field_lines(wout, s, alphas, 2.5, steps)
    wells = []
    for b, arc, angle in zip(strength, length, theta, strict=True):
        peaks, _ = scipy.signal.find_peaks(b, prominence=0.5 * np.ptp(b))
        i, j = peaks[:2]
        low = i + int(np.argmin(b[i : j + 1]))
        phi = 2.0 * np.pi / int(wout.nfp) * low / steps
        wells.append((b[i : j + 1], arc[i : j + 1], angle[low], phi))
    return wells, s_found


def clock(wout, surfaces, lines=4, ions=4, ticks=20, steps=100, gyroradius=1 / 250):
    """The [surface, line, ion, tick] times in bounce periods of ions released together
    at the bottoms of wells, and the s of the surfaces."""
    alphas = 2.0 * np.pi * np.arange(lines) / lines
    columns, scale, found = [], [], []
    for s in surfaces:
        wells, s_found = first_wells(wout, s, alphas)
        found.append(s_found)
        for b, arc, theta, phi in wells:
            bottom, top = b.min(), min(b[0], b[-1])
            middle = bounce_integral(b, arc, 0.5 * (bottom + top))
            fields = bottom + (top - bottom) * np.linspace(0.25, 0.75, ions)
            for field in fields:
                speed = np.sqrt(field / bottom - 1.0)
                columns.append([s_found, theta, phi, speed, bottom])
                scale.append(2.0 * np.sqrt(bottom) * middle)
    state, bottoms = np.array(columns).T[:4], np.array(columns).T[4]
    period = np.array(scale)
    rates = guiding_centers(wout)
    moment, inertia = 0.5 / bottoms, gyroradius * float(wout.Aminor_p) * bottoms

    def motion(state, _t):
        return rates(state, moment, inertia)

    dt = period.min() / steps
    times = [[] for _ in period]
    t = 0.0
    for _ in range(2 * ticks * steps):
        if min(len(x) for x in times) >= ticks:
            break
        after, _ = rk4(motion, state, t, dt)
        for ion in np.flatnonzero((state[3] < 0.0) & (after[3] >= 0.0)):
            times[ion].append(t + dt * state[3, ion] / (state[3, ion] - after[3, ion]))
        state, t = after, t + dt
    if min(len(x) for x in times) < ticks:
        message = f"an ion ticked {min(len(x) for x in times)} times in {t:.3g}"
        raise RuntimeError(message)
    times = np.array([x[:ticks] for x in times]) / period[:, None]
    return times.reshape(len(surfaces), lines, ions, ticks), found


def describe(wout, label, surfaces=(0.1, 0.2, 0.5)):
    """Text with the spreads of the bounce periods on the axis of a wout and on
    ``surfaces``."""
    spread = np.ptp(axis_periods(wout))
    lines = [f"{label}: bounce periods on the axis within {spread:.1e}"]
    times, found = clock(wout, surfaces)
    for clocks, s_found in zip(times, found, strict=True):
        periods = (clocks[..., -1] - clocks[..., 0]) / (clocks.shape[-1] - 1)
        spreads = np.ptp(periods, 1) / periods.mean(1)
        worst = int(np.argmax(spreads))
        lines.append(
            f"   s = {s_found:.2f}: periods of the ions within {spreads[worst]:.1e},"
            f" tick {clocks.shape[-1]} spread over"
            f" {np.ptp(clocks[worst, :, -1]) / periods[worst].mean():.2f} periods,"
            f" on the worst of {len(clocks)} field lines"
        )
    return "\n".join(lines)


def near_axis(b0_vals, ftol=1e-13, axis=None, r=0.02, ntor=40, buffer=8):
    """The VMEC++ input of the surface at ``r`` of pyQIC's QI NFP1 r2 at second order
    with the axis field ``b0_vals``."""
    from qic import Qic  # noqa: PLC0415

    stel = Qic.from_paper(
        "QI NFP1 r2", B0_vals=list(b0_vals), order="r2", k_buffer=buffer
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "input.near_axis"
        stel.to_vmec(str(path), r=r, ntorMax=ntor)
        vmec_input = vmecpp.VmecInput.from_file(path)
    update = {
        "ns_array": np.array([13, 25, 49]),
        "ftol_array": np.array([1e-10, 1e-10, ftol]),
        "niter_array": np.array([20000, 20000, 40000]),
    }
    if axis is not None:
        update["raxis_c"], update["zaxis_s"] = axis
    return vmec_input.model_copy(update=update)


def construct(corrections=6):
    """Write HUYGENS and COSINE from pyQIC's QI NFP1 r2 at second order."""
    from qic import Qic  # noqa: PLC0415

    cosine = Qic.from_paper("QI NFP1 r2").B0_vals
    COSINE.write_text(near_axis(cosine, ftol=1e-10).to_json() + "\n")
    low, high = cosine[0] - abs(cosine[1]), cosine[0] + abs(cosine[1])
    design = design_well(low, high)
    target = cosine_series(design)(BOOZER)[0]
    given = target.copy()
    # Each correction adds the miss of the VMEC++ axis in the Boozer angle.
    for _ in range(corrections):
        b0_vals = cosine_coefficients(given, len(design) - 1)
        given += target - axis_boozer(
            vmecpp.run(near_axis(b0_vals), verbose=False).wout
        )
    b0_vals = cosine_coefficients(given, len(design) - 1)
    # The stored input starts from the magnetic axis of an equilibrium at ftol 1e-15.
    wout = vmecpp.run(near_axis(b0_vals, ftol=1e-15), verbose=False).wout
    m0 = np.asarray(wout.xm) == 0
    axis = (np.asarray(wout.rmnc)[m0, 0], np.asarray(wout.zmns)[m0, 0])
    HUYGENS.write_text(near_axis(b0_vals, axis=axis).to_json() + "\n")


def main():
    for label, path in (("cosine well", COSINE), ("Huygens well", HUYGENS)):
        vmec_input = vmecpp.VmecInput.from_file(path)
        print(describe(vmecpp.run(vmec_input, verbose=False).wout, label), flush=True)


if __name__ == "__main__":
    main()
