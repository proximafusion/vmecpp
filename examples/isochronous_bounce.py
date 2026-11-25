# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Trapped ions that keep time in a quasi-isodynamic field.

In a Huygens well the field strength rises as the square of the arc length from the
bottom. A trapped particle bounces in it with the same period at every bounce height.
This example solves with VMEC++ a quasi-isodynamic field of one field period with a
Huygens well on its axis. A second field has a cosine well of the same depth. Ions of
one magnetic moment start together at the bottom of a well in each field, and their
guiding centers tick at every bounce. :func:`construct` writes the two inputs from
pyQIC's near-axis expansion.
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
# Bounce fields in fractions of the depth of a well above its minimum: where the
# periods are compared, and where the designed well holds them equal.
BAND = np.linspace(0.25, 0.75, 41)
DESIGN_BAND = np.linspace(0.02, 0.75, 60)
# The nodes of the Gauss-Chebyshev quadrature of the bounce integrals.
NODES = np.cos((2.0 * np.arange(1, 129) - 1.0) * np.pi / 256)
# Samples of the Boozer toroidal angle over a field period.
BOOZER = 2.0 * np.pi * np.arange(8192) / 8192


def huygens_shape(share):
    """(B - B_{min}) / (B_{max} - B_{min}) of the Huygens well at the arc length
    ``share``.

    It is the square of the arc length up to three quarters of the depth at ``KNEE`` of
    the half period. A cubic Hermite with zero slope at the top continues it.
    """
    t = np.clip((np.abs(share) - KNEE) / (0.5 - KNEE), 0.0, 1.0)
    cap = (
        0.75 * (2 * t**3 - 3 * t**2 + 1)
        + 1.5 * (0.5 - KNEE) / KNEE * (t**3 - 2 * t**2 + t)
        + (3 * t**2 - 2 * t**3)
    )
    return np.where(np.abs(share) <= KNEE, 0.75 * (share / KNEE) ** 2, cap)


def boozer_series(strength, harmonics=24):
    """The coefficients of \\cos(k \\varphi) of the axis field ``strength`` at SHARE.

    \\varphi is the Boozer toroidal angle from the maximum. Along the axis d\\varphi
    goes as B_0 dl.
    """
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
    """The coefficients of \\cos(k \\varphi) of ``samples`` at evenly spaced \\varphi from
    0."""
    coefficients = np.fft.rfft(samples).real / len(samples)
    coefficients[1:] *= 2.0
    return coefficients[: harmonics + 1]


def cosine_series(coefficients, modes=None):
    """B, dB / d\\phi and d^2 B / d\\phi^2 of \\sum_k c_k \\cos(n_k \\phi) as a function
    of \\phi.

    ``modes`` holds the n_k, 0, 1, 2 and so on by default.
    """
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
    """The \\phi of the extremum of B nearest ``phi``, by Newton's method."""
    for _ in range(8):
        _, slope, curvature = strength(phi)
        phi = phi - slope / curvature
    return phi


def crossings(strength, levels, lo, hi):
    """The \\phi in [``lo``, ``hi``] at which B equals each of ``levels``.

    B - level changes sign once in the interval. Bisection brackets each crossing and
    Newton's method ends it.
    """
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
    """The integrals of dl / \\sqrt{level - B} between the bounce points of a well.

    The well lies between the maxima of B at ``top`` and ``top`` + ``period``, with its
    minimum at ``bottom``. dl is ``speed`` d\\phi. The levels lie at ``fractions`` of the
    depth above the minimum. level - B vanishes linearly at both bounce points, and
    Gauss-Chebyshev quadrature takes its root into the weight.
    """
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
    """The bounce periods over their mean of the axis field \\sum_k c_k \\cos(k
    \\varphi), and its depth.

    \\varphi is the Boozer toroidal angle from the maximum, along which dl goes as
    d\\varphi / B.
    """
    strength = cosine_series(coefficients)
    grid = BOOZER[:: len(BOOZER) // 1024]
    bottom = extremum(strength, grid[np.argmin(strength(grid)[0])])
    periods = bounce_periods(
        strength, lambda phi: 1.0 / strength(phi)[0], 0.0, bottom, fractions
    )
    return periods / periods.mean(), strength(0.0)[0] - strength(bottom)[0]


def design_well(low, high, harmonics=40):
    """The coefficients of \\cos(k \\varphi) of an axis field from ``low`` to ``high``
    whose bounce periods agree from 0.02 to 0.75 of its depth.

    Least squares adjusts the series of :func:`huygens_shape` and leaves the field above
    three quarters of the depth free.
    """

    def residual(coefficients):
        periods, depth = boozer_periods(coefficients, DESIGN_BAND)
        return np.concatenate([periods - 1.0, [depth / (high - low) - 1.0]])

    start = boozer_series(low + (high - low) * huygens_shape(SHARE), harmonics)
    return scipy.optimize.least_squares(
        residual, start, x_scale=1e-3, xtol=1e-15, ftol=1e-15, gtol=1e-15
    ).x


def bounce_integral(strength, length, level):
    """The integral of dl / \\sqrt{level - B} across the well about the least B.

    The integral runs one way over the stretch where B < ``level``. It is exact for B
    linear between samples.
    """
    u = level - strength
    k0 = int(np.argmin(strength))
    left = k0
    while left > 0 and u[left - 1] > 0.0:
        left -= 1
    right = k0
    while right < len(u) - 1 and u[right + 1] > 0.0:
        right += 1
    if left == 0 or right == len(u) - 1:
        return np.nan
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
    """B and dl / d\\phi on the magnetic axis of a wout as functions of \\phi.

    The m = 0 coefficients of |B| are extrapolated to s = 0 from the three innermost
    half-grid surfaces by a quadratic in s.
    """
    ns = int(wout.ns)
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn, dtype=float)
    r_n, z_n, n = (
        np.asarray(wout.rmnc)[xm == 0, 0],
        np.asarray(wout.zmns)[xm == 0, 0],
        xn[xm == 0],
    )
    q = np.asarray(wout.xm_nyq) == 0
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
    """|B| on the magnetic axis of a wout at BOOZER, from its maximum.

    Along the axis d\\varphi goes as B dl.
    """
    strength, speed = axis_series(wout)
    phi = 2.0 * np.pi * np.arange(4 * len(BOOZER)) / (4 * len(BOOZER))
    b = strength(phi)[0]
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


def field_lines(wout, s, alphas, periods, steps):
    """|B|, the arc length and \\theta as [line, sample] arrays along field lines.

    The lines start at \\theta = ``alphas`` and \\phi = 0 on the half-grid surface
    nearest ``s``. RK4 follows them over ``periods`` field periods in ``steps`` steps a
    period. The s of the surface is returned last.
    """
    ns = int(wout.ns)
    js = int(np.clip(np.round(s * (ns - 1) + 0.5), 1, ns - 1))
    m = np.asarray(wout.xm_nyq, dtype=int)
    n = np.asarray(wout.xn_nyq, dtype=int)
    poloidal = np.arange(m.max() + 1)
    coefficients = np.stack(
        [np.asarray(a)[:, js] for a in (wout.bmnc, wout.bsupumnc, wout.bsupvmnc)],
        axis=1,
    )

    def slope(theta, phi):
        """The slopes d\\theta / d\\phi and dl / d\\phi and |B| of each line."""
        wave = np.exp(1j * np.outer(theta, poloidal))[:, m] * np.exp(-1j * n * phi)
        strength, up, along = (wave.real @ coefficients).T
        return up / along, strength / np.abs(along), strength

    h = 2.0 * np.pi / int(wout.nfp) / steps
    theta, length = np.array(alphas, dtype=float), np.zeros(len(alphas))
    strengths, lengths, thetas = [], [], []
    for k in range(round(periods * steps) + 1):
        phi = k * h
        k1, l1, strength = slope(theta, phi)
        strengths.append(strength)
        lengths.append(length.copy())
        thetas.append(theta.copy())
        k2, l2, _ = slope(theta + 0.5 * h * k1, phi + 0.5 * h)
        k3, l3, _ = slope(theta + 0.5 * h * k2, phi + 0.5 * h)
        k4, l4, _ = slope(theta + h * k3, phi + h)
        theta = theta + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        length = length + h / 6.0 * (l1 + 2.0 * l2 + 2.0 * l3 + l4)
    s_found = (js - 0.5) / (ns - 1)
    return np.array(strengths).T, np.array(lengths).T, np.array(thetas).T, s_found


def guiding_centers(wout, tolerance=1e-5):
    """The guiding-center equations in the vacuum field of a wout.

    The returned function gives the rate of change of the states (s, \\theta, \\zeta,
    v_\\parallel) of ions as a [quantity, ion] array, and |B| at each ion. ``moment``
    is the magnetic moment over the mass and ``inertia`` the mass over the charge. The
    Fourier coefficients are cubic splines in \\sqrt{s}, evaluated on the piece that
    holds each ion. A mode enters when the product of 1 + m + |n| and its largest
    coefficient over the largest of its quantity exceeds ``tolerance`` times the
    largest such product.
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
    # B_s, which the wout holds on the full grid, moves to the half grid.
    radial_mn = scipy.interpolate.CubicSpline(
        half, scipy.interpolate.CubicSpline(full, radial.T)(half)
    )

    # The coefficients of t^3, t^2, t and 1 of each piece, [piece, mode, column]. The
    # cosine series give the six cosine quantities, the sine series d|B| / d\theta,
    # d|B| / d\zeta and B_s.
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
    """|B|, the arc length, \\theta and \\phi along the first whole well of field lines.

    The field lines start at \\theta = ``alphas`` and \\phi = 0 on the half-grid
    surface nearest ``s``. RK4 follows them in ``steps`` steps a field period.
    \\theta and \\phi are those of the bottom of each well. The s of the surface is
    returned last.
    """
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
    """The tick times of ions released together from the bottoms of wells.

    On each of the half-grid surfaces nearest ``surfaces``, the wells are the first on
    ``lines`` field lines that start at \\phi = 0 and at evenly spaced \\theta. The
    ``ions`` of a well share one magnetic moment. Their energies set their bounce fields across the
    middle half of the depth of the well. RK4 follows their guiding centers at
    ``steps`` steps of the shortest bounce period. A tick is a turn of v_\\parallel
    from negative to positive. The gyroradius of the ions at the start is
    ``gyroradius`` times the minor radius. The times form a [surface, line, ion, tick]
    array in bounce periods of the middle of the depth of each well. The s of the
    surfaces are returned last.
    """
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
    dt = period.min() / steps
    times = [[] for _ in period]
    t = 0.0
    for _ in range(2 * ticks * steps):
        if min(len(x) for x in times) >= ticks:
            break
        k1, _ = rates(state, moment, inertia)
        k2, _ = rates(state + 0.5 * dt * k1, moment, inertia)
        k3, _ = rates(state + 0.5 * dt * k2, moment, inertia)
        k4, _ = rates(state + dt * k3, moment, inertia)
        after = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        for ion in np.flatnonzero((state[3] < 0.0) & (after[3] >= 0.0)):
            times[ion].append(t + dt * state[3, ion] / (state[3, ion] - after[3, ion]))
        state, t = after, t + dt
    if min(len(x) for x in times) < ticks:
        message = f"an ion ticked {min(len(x) for x in times)} times in {t:.3g}"
        raise RuntimeError(message)
    times = np.array([x[:ticks] for x in times]) / period[:, None]
    return times.reshape(len(surfaces), lines, ions, ticks), found


def describe(wout, label, surfaces=(0.1, 0.2, 0.5)):
    """Text with the spread of the bounce period on the axis of a wout and the worst of
    the clocks of the field lines on each of the surfaces nearest ``surfaces``."""
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
    """The VMEC++ input of the surface at ``r`` of pyQIC's QI NFP1 r2 at second order.

    ``b0_vals`` are the cosine coefficients of the axis field. The field-line label of
    pyQIC departs from first-order omnigenity as the power 2 ``buffer`` + 1 of the
    toroidal distance from the bottom of the well. The boundary has ``ntor`` toroidal
    harmonics, and the finest radial grid converges to ``ftol``. ``axis`` holds the
    cosine coefficients of R and the sine coefficients of Z of the initial magnetic
    axis.
    """
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
    """Write HUYGENS and COSINE from pyQIC's QI NFP1 r2 at second order.

    COSINE keeps pyQIC's own axis field. HUYGENS has the axis field of
    :func:`design_well` at the same depth. Each of the ``corrections`` adds to the axis
    field given to pyQIC, in the Boozer angle, the difference between the design and
    the axis of the VMEC++ equilibrium. HUYGENS starts its magnetic axis where the
    equilibrium solved to ftol 1e-15 has it.
    """
    from qic import Qic  # noqa: PLC0415

    cosine = Qic.from_paper("QI NFP1 r2").B0_vals
    COSINE.write_text(near_axis(cosine, ftol=1e-10).to_json() + "\n")
    low, high = cosine[0] - abs(cosine[1]), cosine[0] + abs(cosine[1])
    design = design_well(low, high)
    target = cosine_series(design)(BOOZER)[0]
    given = target.copy()
    for _ in range(corrections):
        b0_vals = cosine_coefficients(given, len(design) - 1)
        given += target - axis_boozer(
            vmecpp.run(near_axis(b0_vals), verbose=False).wout
        )
    b0_vals = cosine_coefficients(given, len(design) - 1)
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
