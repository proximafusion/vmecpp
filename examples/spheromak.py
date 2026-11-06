# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A spheromak, whose boundary reaches the major axis, against the Taylor state: the
axis, flux surfaces and enclosed current of psi = R^2 j_1(k r) / r in r < 1."""

import numpy as np

import vmecpp

MU0 = 4.0e-7 * np.pi
K = 4.493409457909064  # first zero of j_1, tan k = k
CENTER = 0.61  # centre of the polar boundary, near the magnetic axis
KAPPA = 0.1  # rate of the boundary angle at the corners over its mean rate
APHI = np.array([1.7, -0.7])  # s = 1.7 rho - 0.7 rho^2


def taylor_psi(r, z):
    """Poloidal flux of the Taylor state at cylindrical (r, z): zero on the sphere and
    on the axis, positive inside and negative just outside."""
    radius = np.maximum(np.hypot(r, z), 1e-300)
    x = K * radius
    return r**2 / radius * (np.sin(x) / x**2 - np.cos(x) / x)


def taylor_axis():
    """R of the magnetic axis, where x j_0(x) = j_1(x) with x = k R."""
    x = 2.74
    for _ in range(50):
        f = (x * x - 1.0) * np.sin(x) + x * np.cos(x)
        df = x * np.sin(x) + x * x * np.cos(x)
        x -= f / df
    return x / K


def taylor_profiles(cells=1600, levels=400, degree=8):
    """Psi on the axis, the toroidal flux, the levels of psi with their normalized
    toroidal flux s, and power-series coefficients of iota(s), from a grid."""
    r = (np.arange(cells) + 0.5) / cells
    z = -1.0 + (np.arange(2 * cells) + 0.5) / cells
    psi = taylor_psi(*np.meshgrid(r, z, indexing="ij"))
    flux = K * psi / r[:, None] / cells**2
    order = np.argsort(psi.ravel())[::-1]
    psi_axis = psi.max()
    psi_levels = np.linspace(0.0, psi_axis, levels + 1)[:-1]
    enclosed = np.interp(
        -psi_levels, -psi.ravel()[order], np.cumsum(flux.ravel()[order])
    )
    phi_edge = enclosed[0]
    s = enclosed / phi_edge
    iota = np.gradient(2.0 * np.pi * (psi_axis - psi_levels), enclosed)
    inner = slice(5, -5)
    coefficients = np.polynomial.polynomial.polyfit(s[inner], iota[inner], degree)
    return psi_axis, phi_edge, psi_levels, s, coefficients


def boundary(mpol, kappa=KAPPA, samples=1 << 15):
    """Cosine coefficients of R and sine coefficients of Z of the boundary, and the
    shift that keeps R >= 0."""
    corner = np.pi - np.arctan2(1.0, CENTER)  # polar angle of (0, 1)
    theta = 2.0 * np.pi * np.arange(samples) / samples
    t = np.where(theta > np.pi, theta - 2.0 * np.pi, theta)
    angle = np.empty_like(t)
    on_sphere = np.abs(t) <= corner
    u = (t[on_sphere] + corner) / (2.0 * corner)
    angle[on_sphere] = corner * (
        2.0 * u - 1.0 - (1.0 - kappa) * np.sin(2.0 * np.pi * u) / np.pi
    )
    t_axis = np.where(t[~on_sphere] > 0.0, t[~on_sphere], t[~on_sphere] + 2.0 * np.pi)
    u = (t_axis - corner) / (2.0 * np.pi - 2.0 * corner)
    angle[~on_sphere] = corner + (2.0 * np.pi - 2.0 * corner) * (
        u - (1.0 - kappa) * np.sin(2.0 * np.pi * u) / (2.0 * np.pi)
    )
    c, s = np.cos(angle), np.sin(angle)
    to_sphere = -CENTER * c + np.sqrt(CENTER**2 * c**2 - CENTER**2 + 1.0)
    to_axis = np.where(c < 0.0, CENTER / np.maximum(-c, 1e-300), np.inf)
    rho = np.minimum(to_sphere, to_axis)
    rbc = np.fft.rfft(CENTER + rho * c).real[:mpol] / samples
    rbc[1:] *= 2.0
    zbs = -2.0 * np.fft.rfft(rho * s).imag[:mpol] / samples
    fine = np.linspace(0.0, 2.0 * np.pi, 40001)
    shift = max(0.0, -(rbc @ np.cos(np.outer(np.arange(mpol), fine))).min())
    rbc[0] += shift
    return rbc, zbs, shift


def spheromak_input(profiles, mpol=24, ns_array=(17, 33)):
    """The fixed-boundary input of the spheromak."""
    _, phi_edge, _, _, iota = profiles
    rbc, zbs, _ = boundary(mpol)
    return vmecpp.VmecInput(
        mpol=mpol,
        ntor=0,
        ns_array=np.asarray(ns_array),
        ftol_array=np.asarray([1e-10] + [1e-14] * (len(ns_array) - 1)),
        niter_array=np.asarray([30000] + [100000] * (len(ns_array) - 1)),
        phiedge=phi_edge,
        aphi=APHI,
        ncurr=0,
        piota_type="power_series",
        ai=iota,
        pmass_type="power_series",
        am=np.zeros(1),
        pres_scale=0.0,
        rbc=rbc[:, None],
        zbs=zbs[:, None],
        raxis_c=np.array([CENTER]),
        zaxis_s=np.zeros(1),
    )


def deviations(wout, profiles):
    """The axis, psi on the full-grid surfaces over psi on the axis, and the enclosed
    current on the half-grid surfaces over the total, against the Taylor state."""
    psi_axis, phi_edge, psi_levels, s_levels, _ = profiles
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    angle = np.outer(np.asarray(wout.xm), theta)
    r = np.asarray(wout.rmnc).T @ np.cos(angle)
    z = np.asarray(wout.zmns).T @ np.sin(angle)
    ns = wout.ns
    s_full = np.polynomial.polynomial.polyval(np.linspace(0.0, 1.0, ns), [0.0, *APHI])
    expected = np.interp(s_full, s_levels[::-1], psi_levels[::-1])
    psi = np.abs(taylor_psi(np.abs(r), z) - expected[:, None]).max(axis=1) / psi_axis
    rho_half = (np.arange(ns - 1) + 0.5) / (ns - 1)
    s_half = np.polynomial.polynomial.polyval(rho_half, [0.0, *APHI])
    total = K * phi_edge / MU0
    current = 2.0 * np.pi * np.asarray(wout.buco)[1:] / MU0 - total * s_half
    return {
        "axis": float(np.asarray(wout.raxis_cc)[0] - taylor_axis()),
        "psi": psi,
        "s_half": s_half,
        "current": current / total,
        "total_current": abs(float(wout.ctor)) / total - 1.0,
        "min_r": float(r.min()),
    }


def main():
    profiles = taylor_profiles()
    psi_axis, phi_edge, _, _, iota = profiles
    print(
        f"Taylor state: psi {psi_axis:.6f} on the axis at R = {taylor_axis():.6f}, "
        f"toroidal flux {phi_edge:.6f}, iota "
        f"{np.polynomial.polynomial.polyval(0.0, iota):.4f} on the axis to "
        f"{np.polynomial.polynomial.polyval(1.0, iota):.4f} at the boundary"
    )
    for ns_array in ((17, 33), (17, 33, 65)):
        wout = vmecpp.run(
            spheromak_input(profiles, ns_array=ns_array), max_threads=1, verbose=False
        ).wout
        d = deviations(wout, profiles)
        quarter = (wout.ns - 1) // 4
        interior = (d["s_half"] > 0.1) & (d["s_half"] < 0.9)
        print(
            f"ns = {wout.ns}: {wout.itfsq} iterations, axis {d['axis']:+.1e}, psi "
            f"{d['psi'][quarter]:.1e}, {d['psi'][2 * quarter]:.1e}, "
            f"{d['psi'][3 * quarter]:.1e} at rho = 1/4, 1/2, 3/4 and at most "
            f"{d['psi'].max():.1e}, enclosed current within "
            f"{np.abs(d['current'][interior]).max():.1e} for 0.1 < s < 0.9, total "
            f"current {d['total_current']:+.1e}, smallest R {d['min_r']:.1e}"
        )


if __name__ == "__main__":
    main()
