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

import tempfile
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.signal

import vmecpp

DATA = Path(__file__).parent / "data"
HUYGENS = DATA / "isochronous_bounce_huygens.json"
COSINE = DATA / "isochronous_bounce_cosine.json"
# The share of the half period in arc length over which the well is a parabola.
KNEE = 0.4
# Sample points of the arc length from the minimum in lengths of a field period.
SHARE = np.linspace(-0.5, 0.5, 8192, endpoint=False) + 0.5 / 8192


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
    coefficients = np.fft.rfft(resampled).real / count
    coefficients[1:] *= 2.0
    return coefficients[: harmonics + 1]


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


def profile_periods(strength, levels=21):
    """The bounce periods of the well ``strength`` at SHARE over their mean.

    The ``levels`` bounce fields span the middle half of the depth of the well.
    """
    bottom, top = strength.min(), min(strength[0], strength[-1])
    periods = np.array(
        [
            bounce_integral(strength, SHARE, bottom + (top - bottom) * f)
            for f in np.linspace(0.25, 0.75, levels)
        ]
    )
    return periods / periods.mean()


def profile_spread(strength, levels=21):
    """The relative spread of the bounce period of the well ``strength`` at SHARE over
    the middle half of its depth."""
    return np.ptp(profile_periods(strength, levels))


def axis_field(wout, count=8192):
    """|B| on the magnetic axis of a wout at SHARE.

    The m = 0 modes are extrapolated to s = 0 from the two innermost half-grid surfaces.
    """
    phi = 2.0 * np.pi / int(wout.nfp) * np.arange(count) / count
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn, dtype=float)
    r_n, z_n, n = (
        np.asarray(wout.rmnc)[xm == 0, 0],
        np.asarray(wout.zmns)[xm == 0, 0],
        xn[xm == 0],
    )
    speed = np.sqrt(
        (np.cos(np.outer(phi, n)) @ r_n) ** 2
        + (np.sin(np.outer(phi, n)) @ (n * r_n)) ** 2
        + (np.cos(np.outer(phi, n)) @ (n * z_n)) ** 2
    )
    q = np.asarray(wout.xm_nyq) == 0
    b_n = 1.5 * np.asarray(wout.bmnc)[q, 1] - 0.5 * np.asarray(wout.bmnc)[q, 2]
    strength = np.cos(np.outer(phi, np.asarray(wout.xn_nyq, dtype=float)[q])) @ b_n
    length = np.concatenate([[0.0], np.cumsum(speed[:-1])]) / np.sum(speed)
    share = np.mod(length - length[np.argmin(strength)] + 0.5, 1.0) - 0.5
    order = np.argsort(share)
    return np.interp(SHARE, share[order], strength[order], period=1.0)


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
    for k in range(periods * steps + 1):
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


def guiding_centers(wout, s, tolerance=1e-5):
    """The guiding-center equations in the vacuum field of a wout near a surface.

    The returned function gives the rate of change of the states (s, \\theta, \\zeta,
    v_\\parallel) of ions as a [quantity, ion] array, and |B| at each ion. ``moment``
    is the magnetic moment over the mass and ``inertia`` the mass over the charge. The
    Fourier coefficients are cubic splines in \\sqrt{s}, expanded to second order about
    the half-grid surface nearest ``s``. A mode enters when its largest coefficient
    over the largest of its quantity, times 1 + m + |n|, exceeds ``tolerance`` times
    the largest such value.
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
    js = int(np.clip(np.round(s * (ns - 1) + 0.5), 1, ns - 1))
    centre = np.sqrt((js - 0.5) / (ns - 1))
    cosine_mn = scipy.interpolate.CubicSpline(
        np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1)),
        np.stack([a[:, 1:].T for a in cosines], axis=1),
    )
    radial_mn = scipy.interpolate.CubicSpline(
        np.sqrt(np.arange(ns) / (ns - 1)), radial.T
    )
    cosine_taylor = [cosine_mn(centre, k) for k in range(3)]
    radial_taylor = [radial_mn(centre, k) for k in range(3)]

    def rates(state, moment, inertia):
        s, theta, zeta, v = state
        rho = np.sqrt(s)
        x = (rho - centre)[:, None]
        wave = (
            np.exp(1j * np.outer(theta, np.arange(m.max() + 1)))[:, m]
            * np.exp(-1j * np.outer(zeta, np.arange(n.min(), n.max() + 1)))[
                :, n - n.min()
            ]
        )
        c, sn = wave.real, wave.imag
        value, slope, curve = cosine_taylor
        cos_mn = value + x[:, :, None] * (slope + 0.5 * x[:, :, None] * curve)
        strength, b_theta, b_zeta, up, along, jacobian = np.einsum(
            "im,ijm->ji", c, cos_mn
        )
        b_mn = cos_mn[:, 0]
        d_s = np.sum(c * (slope[0] + x * curve[0]), 1) / (2.0 * rho)
        d_theta = -np.sum(sn * b_mn * m, 1)
        d_zeta = np.sum(sn * b_mn * n, 1)
        value, slope, curve = radial_taylor
        b_s = np.sum(sn * (value + x * (slope + 0.5 * x * curve)), 1)
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


def first_well(wout, s):
    """|B| and the arc length along the first whole well of a field line.

    The field line starts at \\theta = \\phi = 0 on the half-grid surface nearest
    ``s``. \\theta and \\phi at the bottom of the well and the s of the surface are
    returned last.
    """
    steps = 2000
    strength, length, theta, s_found = field_lines(wout, s, [0.0], 3, steps)
    b, arc = strength[0], length[0]
    peaks, _ = scipy.signal.find_peaks(b, prominence=0.5 * np.ptp(b))
    i, j = peaks[:2]
    low = i + int(np.argmin(b[i : j + 1]))
    phi = 2.0 * np.pi / int(wout.nfp) * low / steps
    return b[i : j + 1], arc[i : j + 1], theta[0, low], phi, s_found


def clock(wout, s, ions=8, ticks=30, steps=100, gyroradius=1 / 250):
    """The tick times of ions released together from the bottom of a well.

    The well is the first on the field line from \\theta = \\phi = 0 of the half-grid
    surface nearest ``s``. The ions share one magnetic moment. Their energies set their
    bounce fields across the middle half of the depth of the well. RK4 follows their
    guiding centers at ``steps`` steps a bounce period. A tick is a turn of
    v_\\parallel from negative to positive. The gyroradius of the ions at the start is
    ``gyroradius`` times the minor radius. The times form an [ion, tick] array in
    bounce periods of the middle of the depth. The s of the surface is returned last.
    """
    b, arc, theta, phi, s_found = first_well(wout, s)
    bottom, top = b.min(), min(b[0], b[-1])
    period = 2.0 * np.sqrt(bottom) * bounce_integral(b, arc, 0.5 * (bottom + top))
    fields = bottom + (top - bottom) * np.linspace(0.25, 0.75, ions)
    state = np.array(
        [
            np.full(ions, s_found),
            np.full(ions, theta),
            np.full(ions, phi),
            np.sqrt(fields / bottom - 1.0),
        ]
    )
    rates = guiding_centers(wout, s)
    moment, inertia = 0.5 / bottom, gyroradius * float(wout.Aminor_p) * bottom
    dt = period / steps
    times = [[] for _ in range(ions)]
    t = 0.0
    while min(len(x) for x in times) < ticks:
        k1, _ = rates(state, moment, inertia)
        k2, _ = rates(state + 0.5 * dt * k1, moment, inertia)
        k3, _ = rates(state + 0.5 * dt * k2, moment, inertia)
        k4, _ = rates(state + dt * k3, moment, inertia)
        after = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        for ion in np.flatnonzero((state[3] < 0.0) & (after[3] >= 0.0)):
            times[ion].append(t + dt * state[3, ion] / (state[3, ion] - after[3, ion]))
        state, t = after, t + dt
    return np.array([x[:ticks] for x in times]) / period, s_found


def describe(wout, label, surfaces=(0.05, 0.2)):
    """Text with the spread of the bounce period on the axis of a wout and the clocks on
    the surfaces nearest ``surfaces``."""
    spread = profile_spread(axis_field(wout))
    lines = [f"{label}: bounce periods on the axis within {spread:.1e}"]
    for s in surfaces:
        times, s_found = clock(wout, s)
        periods = (times[:, -1] - times[:, 0]) / (times.shape[1] - 1)
        lines.append(
            f"   s = {s_found:.2f}: periods of the ions within "
            f"{np.ptp(periods) / np.mean(periods):.1e}, tick {times.shape[1]} "
            f"spread over {np.ptp(times[:, -1]) / np.mean(periods):.2f} periods"
        )
    return "\n".join(lines)


def near_axis(b0_vals, r=0.08, ntor=28):
    """The VMEC++ input of the surface at ``r`` of pyQIC's QI NFP1 r2 at second order.

    ``b0_vals`` are the cosine coefficients of the axis field. The boundary has
    ``ntor`` toroidal harmonics.
    """
    from qic import Qic  # noqa: PLC0415

    stel = Qic.from_paper("QI NFP1 r2", B0_vals=list(b0_vals), order="r2")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "input.near_axis"
        stel.to_vmec(str(path), r=r, ntorMax=ntor)
        vmec_input = vmecpp.VmecInput.from_file(path)
    return vmec_input.model_copy(
        update={
            "ns_array": np.array([13, 25]),
            "ftol_array": np.array([1e-8, 1e-10]),
            "niter_array": np.array([5000, 10000]),
        }
    )


def construct(corrections=3):
    """Write HUYGENS and COSINE from pyQIC's QI NFP1 r2 at second order.

    COSINE keeps pyQIC's own axis field. HUYGENS has the Huygens well of the same
    depth. Each of the ``corrections`` adds to the axis field given to pyQIC the shape
    that the axis of the VMEC++ equilibrium lacks.
    """
    from qic import Qic  # noqa: PLC0415

    cosine = Qic.from_paper("QI NFP1 r2").B0_vals
    COSINE.write_text(near_axis(cosine).to_json() + "\n")
    low, high = cosine[0] - abs(cosine[1]), cosine[0] + abs(cosine[1])
    target = huygens_shape(SHARE)
    strength = low + (high - low) * target
    for _ in range(corrections):
        wout = vmecpp.run(near_axis(boozer_series(strength)), verbose=False).wout
        actual = axis_field(wout)
        strength += (high - low) * (
            target - (actual - actual.min()) / (actual.max() - actual.min())
        )
    HUYGENS.write_text(near_axis(boozer_series(strength)).to_json() + "\n")


def main():
    for label, path in (("cosine well", COSINE), ("Huygens well", HUYGENS)):
        vmec_input = vmecpp.VmecInput.from_file(path)
        print(describe(vmecpp.run(vmec_input, verbose=False).wout, label), flush=True)


if __name__ == "__main__":
    main()
