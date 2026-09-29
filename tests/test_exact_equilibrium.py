# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ against an exact three-dimensional MHD equilibrium.

Landreman, "Analytic toroidal 3D MHD equilibria and steady Euler flows with
invariant surfaces" (arXiv:2609.26742), gives families of smooth equilibria with
exact nested flux surfaces and no continuous symmetry: the field, the flux label psi
and the pressure are elementary functions of the Cartesian position. The test takes
a member of the sheared-iota family of its section 3, with two field periods and
stellarator symmetry, writes the boundary psi = k_b^2 / 2, the pressure and the
rotational transform from the formulas, runs VMEC++ and compares, at VMEC++'s own
points, quantities that do not depend on its poloidal angle: the analytic psi on each
surface against the surface's flux label, the magnetic axis, the field vector at the
half-grid points against the analytic field, and the toroidal current enclosed by
each surface against the paper's. The error in the current falls as the square of
the radial grid spacing.

The paper's units have mu0 = 1: B is in tesla and lengths in metres, so the pressure
is p / mu0 in pascals. Its poloidal angle chi turns clockwise in an (R, Z) section,
the opposite of VMEC's theta, so VMEC's iota is minus the paper's.
"""

import math

import numpy as np
import pytest
from numpy.polynomial import Chebyshev, Polynomial

import vmecpp

MU0 = 4.0e-7 * math.pi
NFP = 2
# The paper's eps, S, lambda and boundary k_b. Lambda near 2 sqrt(S) makes the
# cross-section about as high as it is wide; iota then runs from 3.4289 on the axis
# to 3.4376 at the boundary, 0.06 below the nearest resonance of the retained modes,
# 7/2 of m = 4, n = 14, and the volume-averaged beta is 1.8 per cent.
EPS, S, LAM, K_B = 1.0, 1.75, 2.65, 0.2

NS_COARSE, NS, MPOL, NTOR = 13, 25, 12, 10
FTOL, NITER, DELT = 1.0e-16, 20000, 0.9
# Largest deviations from the exact solution at NS: psi relative to its boundary
# value, the axis in metres, the field relative to |B| at the point, and the enclosed
# current relative to its largest value. VMEC++ reaches 1.0e-4, 1.05e-5, 6.7e-5 and
# 4.2e-6. The deviation of the surfaces is a cos(4 theta + 14 phi) harmonic, resonant
# where iota = -7/2 just outside the profile, which the residual at FTOL constrains
# only to that level: it is 3.8e-4 at delt = 1.
TOL_PSI, TOL_AXIS, TOL_B, TOL_CURRENT = 5.0e-4, 2.0e-5, 1.2e-4, 8.0e-6


def _h(sigma):
    return np.sqrt(4.0 * sigma * sigma + EPS * EPS)


def _semiaxes(sigma):
    """The semiaxes a, b of the confocal ellipse sigma, with a b = sigma."""
    b = np.sqrt((_h(sigma) + EPS) / 2.0)
    return sigma / b, b


def _field(x, y, z):
    """The Cartesian field and the flux label psi (the paper's section 3.1)."""
    w = x + 1j * y
    wb = x - 1j * y
    k = wb * np.sqrt(1.0 + EPS / wb**2)
    xi = w * k + np.pi / 2.0 - S
    e = np.exp(-1j * LAM * z)
    bxy = e * 1j * np.sin(xi) / (2.0 * k)
    bz = np.real(e * np.cos(xi)) / LAM
    psi = (np.sin(LAM * z) ** 2 + (LAM * bz) ** 2) / 2.0
    return np.real(bxy), np.imag(bxy), bz, psi


def _section(k, phi, chi):
    """R and Z of the point X = -k cos(chi), Y = k sin(chi) in the plane phi, by
    bisection on the confocal coordinate sigma (the paper's section 3.2)."""
    chi, phi = np.broadcast_arrays(np.asarray(chi, float), np.asarray(phi, float))
    p, y = -k * np.cos(chi), k * np.sin(chi)
    root = np.sqrt(1.0 - y * y)
    lo = np.full(chi.shape, S - np.arcsin(k))
    hi = np.full(chi.shape, S + np.arcsin(k))

    def residual(sigma):
        a, b = _semiaxes(sigma)
        t = np.arctan2(a * np.sin(phi), b * np.cos(phi))
        nu = EPS / 2.0 * np.sin(2.0 * t)
        return (
            sigma
            - S
            - np.arctan(np.tanh(nu) * y / root)
            + np.arcsin(p / np.sqrt(np.cosh(nu) ** 2 - y * y))
        ), t

    for _ in range(60):
        mid = (lo + hi) / 2.0
        r, _ = residual(mid)
        lo = np.where(r < 0.0, mid, lo)
        hi = np.where(r >= 0.0, mid, hi)
    sigma = (lo + hi) / 2.0
    _, t = residual(sigma)
    a, b = _semiaxes(sigma)
    return np.hypot(a * np.cos(t), b * np.sin(t)), -np.arcsin(y) / LAM


def _boundary(mpol, ntor):
    """Rbc and zbs of the surface k = K_B, with VMEC's theta = -chi."""
    nth, nph = 2 * mpol + 6, 2 * ntor + 6
    theta = 2.0 * np.pi * np.arange(nth) / nth
    phi = 2.0 * np.pi * np.arange(nph) / (nph * NFP)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    r, z = _section(K_B, ph, -th)
    rbc = np.zeros((mpol, 2 * ntor + 1))
    zbs = np.zeros((mpol, 2 * ntor + 1))
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n < 0:
                continue
            arg = m * th - n * NFP * ph
            weight = 1.0 if m == 0 and n == 0 else 2.0
            rbc[m, n + ntor] = weight * np.mean(r * np.cos(arg))
            zbs[m, n + ntor] = weight * np.mean(z * np.sin(arg))
    return rbc, zbs


def _flux_rates(k, n=512):
    """Q'(k) / k and A'(k) / k, the derivatives of the toroidal and poloidal fluxes
    divided by k, whose ratio is the rotational transform."""
    k = np.atleast_1d(np.asarray(k, float))[:, None]
    u = 2.0 * np.pi * np.arange(n) / n
    c = np.sqrt(1.0 - k * k * np.sin(u) ** 2)
    sigma = S + np.arcsin(k * np.cos(u) / c)
    q = np.pi / LAM * np.mean(1.0 / (_h(sigma) * c), axis=1)
    nu = EPS / 2.0 * np.sin(2.0 * u)
    sigma = S + np.arcsin(k / np.cosh(nu))
    g = (_h(sigma) + EPS * np.cos(2.0 * u)) / 2.0
    root = np.sqrt(np.cosh(nu) ** 2 - k * k)
    a = 2.0 * np.pi / LAM * np.mean(g / (_h(sigma) * root), axis=1)
    return q, a


def _iota(k):
    q, a = _flux_rates(k)
    return a / q


def _current(k, n=1024):
    """The toroidal current enclosed by the surface k, in amperes."""
    k = np.atleast_1d(np.asarray(k, float))[:, None]
    u = 2.0 * np.pi * np.arange(n) / n
    p, y = -k * np.cos(u), k * np.sin(u)
    c = np.sqrt(1.0 - y * y)
    sigma = S - np.arcsin(p / c)
    density = y * y * (1.0 - k * k) / (2.0 * _h(sigma) * c**3) + p * p / (LAM**2 * c)
    return 2.0 * np.pi / MU0 * np.mean(density, axis=1)


def _profiles(degree=12, nodes=40):
    """The toroidal flux through the boundary, and k^2, the paper's iota and the
    enclosed current as power series in the normalized toroidal flux s."""
    kappa = K_B**2 * (1.0 - np.cos(np.linspace(0.0, np.pi, nodes))) / 2.0
    rate = Chebyshev.fit(kappa, _flux_rates(np.sqrt(kappa))[0], nodes - 1)
    flux = rate.integ(lbnd=0.0) / 2.0  # d flux / d kappa = Q'(k) / (2 k)
    phi_edge = float(flux(K_B**2))
    s = flux(kappa) / phi_edge
    k = np.sqrt(kappa)
    k2 = Polynomial.fit(s, kappa, degree).convert()
    iota = Polynomial.fit(s, _iota(k), degree).convert()
    current = Polynomial.fit(s, _current(k), degree).convert()
    return phi_edge, k2, iota, current


def _input(ns):
    """Fixed boundary at k = K_B with the pressure and the rotational transform of the
    exact solution; the current is left to the solver."""
    phi_edge, k2, iota, _ = _profiles()
    rbc, zbs = _boundary(MPOL, NTOR)
    a, b = _semiaxes(S)
    phi = 2.0 * np.pi * np.arange(64) / (64 * NFP)
    r_axis = 1.0 / np.sqrt(np.cos(phi) ** 2 / a**2 + np.sin(phi) ** 2 / b**2)
    raxis_c = np.array(
        [
            (1.0 if n == 0 else 2.0) * np.mean(r_axis * np.cos(n * NFP * phi))
            for n in range(NTOR + 1)
        ]
    )
    # p = (k_b^2 - k^2) / (2 lambda^2), which vanishes at the boundary
    am = -np.asarray(k2.coef) / (2.0 * LAM**2 * MU0)
    am[0] = (K_B**2 - k2.coef[0]) / (2.0 * LAM**2 * MU0)
    return vmecpp.VmecInput(
        nfp=NFP,
        lasym=False,
        mpol=MPOL,
        ntor=NTOR,
        ns_array=np.array([ns], dtype=np.int64),
        ftol_array=np.array([FTOL]),
        niter_array=np.array([NITER], dtype=np.int64),
        delt=DELT,
        phiedge=phi_edge,
        pmass_type="power_series",
        am=am,
        ncurr=0,
        piota_type="power_series",
        ai=-np.asarray(iota.coef),
        raxis_c=raxis_c,
        zaxis_s=np.zeros(NTOR + 1),
        rbc=rbc,
        zbs=zbs,
    )


def _modes_first(array, count):
    a = np.asarray(array, dtype=float)
    return a if a.shape[0] == count else a.T


def _series(coefficients, xm, xn, theta, phi, kind, d_theta=False, d_phi=False):
    """Sum of coefficients times cos (kind "c") or sin (kind "s") of m theta - n phi
    at the grid points, or its derivative in theta or phi."""
    arg = theta[..., None] * xm - phi[..., None] * xn
    if not (d_theta or d_phi):
        return (np.cos(arg) if kind == "c" else np.sin(arg)) @ coefficients
    factor = xm if d_theta else -xn
    return (-np.sin(arg) if kind == "c" else np.cos(arg)) @ (coefficients * factor)


def _current_error(wout):
    """Largest difference between the current the run carries inside each half-grid
    surface and the exact solution's, relative to the largest.

    buco is mu0 / (2 pi) times the current in VMEC's theta, which turns opposite to the
    paper's chi.
    """
    _, k2, _, _ = _profiles()
    s = np.linspace(0.0, 1.0, wout.ns)
    exact = _current(np.sqrt(k2(0.5 * (s[1:] + s[:-1]))))
    current = -2.0 * np.pi * np.asarray(wout.buco)[1:] / MU0
    return float(np.abs(current - exact).max() / np.abs(exact).max())


@pytest.fixture(scope="module")
def run():
    return vmecpp.run(_input(NS), max_threads=1, verbose=False).wout


@pytest.fixture(scope="module")
def grid():
    nth, nph = 24, 24
    theta = 2.0 * np.pi * (np.arange(nth) + 0.25) / nth
    phi = 2.0 * np.pi * (np.arange(nph) + 0.4) / (nph * NFP)
    return np.meshgrid(theta, phi, indexing="ij")


def test_flux_surfaces(run, grid):
    """On every surface of the run the analytic psi equals the surface's own flux label,
    psi = k(s)^2 / 2."""
    _, k2, _, _ = _profiles()
    th, ph = grid
    xm, xn = np.asarray(run.xm, float), np.asarray(run.xn, float)
    rmnc = _modes_first(run.rmnc, len(xm))
    zmns = _modes_first(run.zmns, len(xm))
    s = np.linspace(0.0, 1.0, run.ns)
    worst = 0.0
    for j in range(1, run.ns):
        r = _series(rmnc[:, j], xm, xn, th, ph, "c")
        z = _series(zmns[:, j], xm, xn, th, ph, "s")
        psi = _field(r * np.cos(ph), r * np.sin(ph), z)[3]
        worst = max(worst, float(np.abs(psi - k2(s[j]) / 2.0).max()))
    assert worst <= TOL_PSI * K_B**2 / 2.0, worst / (K_B**2 / 2.0)


def test_magnetic_axis(run, grid):
    """The run's axis is the ellipse of semiaxes a(S), b(S) in the plane Z = 0."""
    th, ph = grid
    xm, xn = np.asarray(run.xm, float), np.asarray(run.xn, float)
    r = _series(_modes_first(run.rmnc, len(xm))[:, 0], xm, xn, th, ph, "c")
    z = _series(_modes_first(run.zmns, len(xm))[:, 0], xm, xn, th, ph, "s")
    a, b = _semiaxes(S)
    r_exact = 1.0 / np.sqrt(np.cos(ph) ** 2 / a**2 + np.sin(ph) ** 2 / b**2)
    worst = float(np.hypot(r - r_exact, z).max())
    assert worst <= TOL_AXIS, worst


def test_magnetic_field(run, grid):
    """At the half-grid points the field of the run, B^u e_u + B^v e_v with the geometry
    interpolated as VMEC interpolates it, is the analytic field there."""
    th, ph = grid
    xm, xn = np.asarray(run.xm, float), np.asarray(run.xn, float)
    xm_nyq, xn_nyq = np.asarray(run.xm_nyq, float), np.asarray(run.xn_nyq, float)
    rmnc = _modes_first(run.rmnc, len(xm))
    zmns = _modes_first(run.zmns, len(xm))
    bsupu = _modes_first(run.bsupumnc, len(xm_nyq))
    bsupv = _modes_first(run.bsupvmnc, len(xm_nyq))
    s = np.linspace(0.0, 1.0, run.ns)
    odd = (xm % 2) == 1
    cp, sp = np.cos(ph), np.sin(ph)
    worst = 0.0
    for j in range(1, run.ns):
        # even m: the mean of the two nodes; odd m: sqrt(s) times the mean of the
        # nodes' coefficients over sqrt(s), the axis taking the first node's
        sh = 0.5 * (s[j - 1] + s[j])
        inner = 1 if j == 1 else j - 1

        def half(c, inner=inner, j=j, sh=sh):
            odd_part = c[:, inner] / np.sqrt(s[inner]) + c[:, j] / np.sqrt(s[j])
            return np.where(
                odd, 0.5 * np.sqrt(sh) * odd_part, 0.5 * (c[:, j - 1] + c[:, j])
            )

        rh, zh = half(rmnc), half(zmns)
        r = _series(rh, xm, xn, th, ph, "c")
        z = _series(zh, xm, xn, th, ph, "s")
        ru = _series(rh, xm, xn, th, ph, "c", d_theta=True)
        zu = _series(zh, xm, xn, th, ph, "s", d_theta=True)
        rv = _series(rh, xm, xn, th, ph, "c", d_phi=True)
        zv = _series(zh, xm, xn, th, ph, "s", d_phi=True)
        bu = _series(bsupu[:, j], xm_nyq, xn_nyq, th, ph, "c")
        bv = _series(bsupv[:, j], xm_nyq, xn_nyq, th, ph, "c")
        bx = bu * ru * cp + bv * (rv * cp - r * sp)
        by = bu * ru * sp + bv * (rv * sp + r * cp)
        bz = bu * zu + bv * zv
        ex, ey, ez, _ = _field(r * cp, r * sp, z)
        err = np.sqrt((bx - ex) ** 2 + (by - ey) ** 2 + (bz - ez) ** 2)
        worst = max(worst, float((err / np.sqrt(ex * ex + ey * ey + ez * ez)).max()))
    assert worst <= TOL_B, worst


def test_enclosed_current(run):
    """With iota prescribed, the toroidal current the run carries inside each half-grid
    surface is the exact solution's."""
    error = _current_error(run)
    assert error <= TOL_CURRENT, error


def test_current_converges_at_second_order(run):
    """Halving the radial grid spacing divides the error in the enclosed current by
    about four."""
    coarse = vmecpp.run(_input(NS_COARSE), max_threads=1, verbose=False).wout
    ratio = _current_error(coarse) / _current_error(run)
    assert ratio >= 3.5, ratio
