# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Verification of VMEC++ against exact three-dimensional MHD equilibria.

Landreman, "Analytic toroidal 3D MHD equilibria and steady Euler flows with invariant
surfaces" (arXiv:2609.26742, with the scripts and data of
https://github.com/landreman/analytic_3d_equilibria), gives two families of smooth
equilibria with exact nested flux surfaces and no continuous symmetry, the field, the
flux label psi and the pressure being elementary functions of the Cartesian position:
one with iota = 2 on every surface (the paper's section 2) and one with sheared iota
(section 3). Both have two field periods and stellarator symmetry.

This file writes VMEC++ inputs for members of both families from the formulas, runs
VMEC++ over sequences of radial and angular resolutions, and measures, at VMEC++'s own
points, quantities that do not depend on its poloidal angle:

- psi on each full-grid surface against the surface's flux label,
- the magnetic axis,
- the field vector at the half-grid points,
- the toroidal current enclosed by each half-grid surface (buco), and its radial
  derivative on the full grid (jcurv),

with their observed orders of convergence, and plots them. Each run starts from the
coarser grids of MULTIGRID below its ns and takes every grid to the residual ftol_for
gives it.

    python examples/exact_equilibria.py [--suite demo|quick|full] [--member NAME]
                                        [--out DIR] [--threads N] [--check]

The members are:

- "sheared" (eps = 0.6, S = 2.2, lambda = 2.97, k_b = 0.2): iota from 4.34 to 4.37,
  midway between the resonances at 4 of m = 1 and at 9/2 of m = 4, and beta 1.9 per
  cent; tests/test_exact_equilibrium.py uses it;
- "sheared-near-3.5" (eps = 1, S = 1.75, lambda = 2.65, k_b = 0.2): iota from 3.43
  to 3.44, 0.06 below the resonance at 7/2 of m = 4, n = 14, and beta 1.8 per cent;
- "sheared-A", the configuration A of the paper's supplement (eps = 1.08, S = 3,
  lambda = 3.5, k_b = 0.7): iota from 5.69 to 6.22, which crosses 6, and beta 20 per
  cent;
- "iota2" (eps = 0.25, delta = 1/64): every surface rational.

The demo suite runs "sheared" at two radial resolutions; the quick suite runs every
member up to ns = 100, with an angular scan of "sheared" at ns = 100; the full suite
runs them up to ns = 1000, with an angular scan at ns = 400. With --out the results go
to DIR as JSON, markdown tables and plots of the deviations against h and mpol and of
the enclosed current over s. The summary ends with checks on the scans of "sheared":
convergence in ns at O(h) and in mpol and ntor, the magnetic axis, B against its
derivative jcurv, and the enclosed current at O(h^2) with its limit at the axis; with
--check a failed check fails the run.

The paper's units have mu0 = 1: B is in tesla and lengths in metres, so the pressure
is p / mu0 in pascals. The paper's poloidal angles turn clockwise in an (R, Z)
section, the opposite of VMEC's theta, so VMEC's iota is minus the paper's.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

import vmecpp

MU0 = 4.0e-7 * math.pi
NFP = 2


def _dft_boundary(section, mpol, ntor):
    """Rbc and zbs of a stellarator-symmetric boundary given by section(theta, phi) ->
    (R, Z), from its values on a grid that resolves the retained modes."""
    nth, nph = 2 * mpol + 6, 2 * ntor + 6
    theta = 2.0 * np.pi * np.arange(nth) / nth
    phi = 2.0 * np.pi * np.arange(nph) / (nph * NFP)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    r, z = section(th, ph)
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


def _cosine_series(values_of_phi, count):
    """Cosine coefficients in n NFP phi of a function of phi, n < count."""
    phi = 2.0 * np.pi * np.arange(64) / (64 * NFP)
    values = values_of_phi(phi)
    return np.array(
        [
            (1.0 if n == 0 else 2.0) * np.mean(values * np.cos(n * NFP * phi))
            for n in range(count)
        ]
    )


def _sine_series(values_of_phi, count):
    """Coefficients zaxis_s of Z = -sum zaxis_s[n] sin(n NFP phi), n < count."""
    phi = 2.0 * np.pi * np.arange(64) / (64 * NFP)
    values = values_of_phi(phi)
    return np.array(
        [0.0]
        + [-2.0 * np.mean(values * np.sin(n * NFP * phi)) for n in range(1, count)]
    )


@dataclasses.dataclass(frozen=True)
class Sheared:
    """A member of the sheared-iota family, the paper's section 3.

    Its parameters are eps, S, lambda and the boundary k_b, and its surfaces are the
    circles of constant psi = k^2 / 2 in the paper's (X, Y) plane.
    """

    name: str
    eps: float
    S: float
    lam: float
    k_b: float

    def _h(self, sigma):
        return np.sqrt(4.0 * sigma * sigma + self.eps * self.eps)

    def semiaxes(self, sigma):
        """The semiaxes a, b of the confocal ellipse sigma, with a b = sigma."""
        b = np.sqrt((self._h(sigma) + self.eps) / 2.0)
        return sigma / b, b

    def field(self, x, y, z):
        """The Cartesian field and psi, the paper's equations 3.2 and 3.3."""
        w = x + 1j * y
        wb = x - 1j * y
        k = wb * np.sqrt(1.0 + self.eps / wb**2)
        xi = w * k + np.pi / 2.0 - self.S
        e = np.exp(-1j * self.lam * z)
        bxy = e * 1j * np.sin(xi) / (2.0 * k)
        bz = np.real(e * np.cos(xi)) / self.lam
        psi = (np.sin(self.lam * z) ** 2 + (self.lam * bz) ** 2) / 2.0
        return np.real(bxy), np.imag(bxy), bz, psi

    def section(self, k, phi, chi):
        """R and Z of X = -k cos(chi), Y = k sin(chi) in the plane phi, by bisection on
        the confocal coordinate sigma (the paper's section 3.2)."""
        chi, phi = np.broadcast_arrays(np.asarray(chi, float), np.asarray(phi, float))
        p, y = -k * np.cos(chi), k * np.sin(chi)
        root = np.sqrt(1.0 - y * y)
        lo = np.full(chi.shape, self.S - np.arcsin(k))
        hi = np.full(chi.shape, self.S + np.arcsin(k))

        def residual(sigma):
            a, b = self.semiaxes(sigma)
            t = np.arctan2(a * np.sin(phi), b * np.cos(phi))
            nu = self.eps / 2.0 * np.sin(2.0 * t)
            return (
                sigma
                - self.S
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
        a, b = self.semiaxes(sigma)
        return np.hypot(a * np.cos(t), b * np.sin(t)), -np.arcsin(y) / self.lam

    def boundary(self, mpol, ntor):
        """The surface k = k_b in VMEC's theta = -chi."""
        return _dft_boundary(lambda th, ph: self.section(self.k_b, ph, -th), mpol, ntor)

    def axis(self, phi):
        a, b = self.semiaxes(self.S)
        return 1.0 / np.sqrt(
            np.cos(phi) ** 2 / a**2 + np.sin(phi) ** 2 / b**2
        ), 0.0 * phi

    def _flux_rates(self, k, n=512):
        """Q'(k) / k and A'(k) / k, the derivatives of the toroidal and poloidal fluxes
        divided by k, whose ratio is the paper's iota."""
        k = np.atleast_1d(np.asarray(k, float))[:, None]
        u = 2.0 * np.pi * np.arange(n) / n
        c = np.sqrt(1.0 - k * k * np.sin(u) ** 2)
        sigma = self.S + np.arcsin(k * np.cos(u) / c)
        q = np.pi / self.lam * np.mean(1.0 / (self._h(sigma) * c), axis=1)
        nu = self.eps / 2.0 * np.sin(2.0 * u)
        sigma = self.S + np.arcsin(k / np.cosh(nu))
        g = (self._h(sigma) + self.eps * np.cos(2.0 * u)) / 2.0
        root = np.sqrt(np.cosh(nu) ** 2 - k * k)
        a = 2.0 * np.pi / self.lam * np.mean(g / (self._h(sigma) * root), axis=1)
        return q, a

    def _current(self, k, n=1024):
        """The toroidal current enclosed by the surface k along chi, in amperes."""
        k = np.atleast_1d(np.asarray(k, float))[:, None]
        u = 2.0 * np.pi * np.arange(n) / n
        p, y = -k * np.cos(u), k * np.sin(u)
        c = np.sqrt(1.0 - y * y)
        sigma = self.S - np.arcsin(p / c)
        density = y * y * (1.0 - k * k) / (2.0 * self._h(sigma) * c**3) + p * p / (
            self.lam**2 * c
        )
        return 2.0 * np.pi / MU0 * np.mean(density, axis=1)

    @functools.cached_property
    def profiles(self):
        """The toroidal flux through the boundary, and psi, VMEC's iota and VMEC's buco
        (mu0 / (2 pi) times the current enclosed along theta) as power series in the
        normalized toroidal flux s."""
        nodes, degree = 40, 12
        kappa = self.k_b**2 * (1.0 - np.cos(np.linspace(0.0, np.pi, nodes))) / 2.0
        k = np.sqrt(kappa)
        q, a = self._flux_rates(k)
        rate = Chebyshev.fit(kappa, q, nodes - 1)
        flux = rate.integ(lbnd=0.0) / 2.0  # d flux / d kappa = Q'(k) / (2 k)
        phi_edge = float(flux(self.k_b**2))
        s = flux(kappa) / phi_edge
        return {
            "phi_edge": phi_edge,
            "psi": Polynomial.fit(s, kappa / 2.0, degree).convert(),
            "iota": Polynomial.fit(s, -a / q, degree).convert(),
            "buco": Polynomial.fit(
                s, -MU0 * self._current(k) / (2.0 * np.pi), degree
            ).convert(),
        }

    def psi_of_s(self, s):
        return self.profiles["psi"](s)

    def buco_of_s(self, s):
        return self.profiles["buco"](s)

    def dbuco_ds(self, s):
        return self.profiles["buco"].deriv()(s)

    def vmec_input(self, ns, mpol, ntor):
        prof = self.profiles
        rbc, zbs = self.boundary(mpol, ntor)
        # p = (psi_b - psi) / lambda^2, which vanishes at the boundary
        am = -np.asarray(prof["psi"].coef) / (self.lam**2 * MU0)
        am[0] = (self.k_b**2 / 2.0 - prof["psi"].coef[0]) / (self.lam**2 * MU0)
        return vmecpp.VmecInput(
            nfp=NFP,
            lasym=False,
            mpol=mpol,
            ntor=ntor,
            ns_array=np.array([ns], dtype=np.int64),
            phiedge=prof["phi_edge"],
            pmass_type="power_series",
            am=am,
            ncurr=0,
            piota_type="power_series",
            ai=np.asarray(prof["iota"].coef),
            raxis_c=_cosine_series(lambda ph: self.axis(ph)[0], ntor + 1),
            zaxis_s=np.zeros(ntor + 1),
            rbc=rbc,
            zbs=zbs,
        )


@dataclasses.dataclass(frozen=True)
class Iota2:
    """A member of the family with iota = 2 on every surface, the paper's section 2.

    Its parameters are eps and the boundary psi = delta; its surfaces are circles of
    radius sqrt(psi) about (-eps / 2, 0) in the plane of the field-line labels (u, v).
    """

    name: str
    eps: float
    delta: float

    @property
    def _ab(self):
        return math.sqrt(1.0 + self.eps), math.sqrt(1.0 - self.eps)

    def field(self, x, y, z):
        """The Cartesian field and psi, the paper's equations 2.1 and 2.3."""
        a, b = self._ab
        s = (x / a) ** 2 + (y / b) ** 2
        f = np.sqrt(2.0 * s - s * s - 4.0 * z * z)
        bx = (2.0 * z * x - (a / b) * f * y) / s
        by = (2.0 * z * y + (b / a) * f * x) / s
        bz = 1.0 - s
        psi = (x * x + y * y + 4.0 * z * z + bx * bx + by * by + bz * bz - 2.0) / 4.0
        return bx, by, bz, psi + self.eps**2 / 4.0

    def embedding(self, u, v, t):
        """The field line (u, v) at the paper's zeta = t, its equation 2.9."""
        a, b = self._ab
        ell = np.sqrt((1.0 + np.sqrt(1.0 - 4.0 * (u * u + v * v))) / 2.0)
        ct, st = np.cos(t), np.sin(t)
        return (
            a * (ell * ct + (u * ct + v * st) / ell),
            b * (ell * st + (v * ct - u * st) / ell),
            v * np.cos(2.0 * t) - u * np.sin(2.0 * t),
        )

    def section(self, psi, theta, phi):
        """R and Z at VMEC's theta = beta - 2 phi, a straight-field-line angle with
        iota = -2, in the plane phi (after the paper's supplement)."""
        a, b = self._ab
        beta = theta + 2.0 * phi
        u = -self.eps / 2.0 + np.sqrt(psi) * np.cos(beta)
        v = np.sqrt(psi) * np.sin(beta)
        ell = np.sqrt((1.0 + np.sqrt(1.0 - 4.0 * (u * u + v * v))) / 2.0)
        aa, cc = a * (ell + u / ell), a * v / ell
        dd, ee = b * v / ell, b * (ell - u / ell)
        ct, st = (
            ee * np.cos(phi) - cc * np.sin(phi),
            aa * np.sin(phi) - dd * np.cos(phi),
        )
        norm = np.hypot(ct, st)
        ct, st = ct / norm, st / norm
        r = np.hypot(aa * ct + cc * st, dd * ct + ee * st)
        z = v * (ct * ct - st * st) - 2.0 * u * st * ct
        return r, z

    def boundary(self, mpol, ntor):
        return _dft_boundary(
            lambda th, ph: self.section(self.delta, th, ph), mpol, ntor
        )

    def axis(self, phi):
        return math.sqrt(1.0 - self.eps**2) + 0.0 * phi, self.eps / 2.0 * np.sin(
            2.0 * phi
        )

    def _buco(self, psi, n=128):
        """The buco of the surface psi, from the field along a loop on it.

        buco is mu0 / (2 pi) times the current enclosed along theta, the integral of B
        along a loop of the label angle alpha at fixed zeta divided by 2 pi.
        """
        alpha = 2.0 * np.pi * np.arange(n) / n
        h = 1e-30
        rho = np.sqrt(np.atleast_1d(np.asarray(psi, float)))[:, None]
        u = -self.eps / 2.0 + rho * np.cos(alpha + 1j * h)
        v = rho * np.sin(alpha + 1j * h)
        pos = self.embedding(u, v, 0.0)
        x, y, z = (np.real(c) for c in pos)
        dx, dy, dz = (np.imag(c) / h for c in pos)
        bx, by, bz, _ = self.field(x, y, z)
        return np.mean(bx * dx + by * dy + bz * dz, axis=1)

    @functools.cached_property
    def profiles(self):
        s = (1.0 - np.cos(np.linspace(0.0, np.pi, 41))) / 2.0
        return {"buco": Polynomial.fit(s, self._buco(s * self.delta), 14).convert()}

    def psi_of_s(self, s):
        return np.asarray(s) * self.delta

    def buco_of_s(self, s):
        return self._buco(np.asarray(s) * self.delta)

    def dbuco_ds(self, s):
        return self.profiles["buco"].deriv()(s)

    def vmec_input(self, ns, mpol, ntor):
        a, b = self._ab
        rbc, zbs = self.boundary(mpol, ntor)
        return vmecpp.VmecInput(
            nfp=NFP,
            lasym=False,
            mpol=mpol,
            ntor=ntor,
            ns_array=np.array([ns], dtype=np.int64),
            phiedge=math.pi * a * b * self.delta,
            pmass_type="power_series",
            # p = 2 (delta - psi), which vanishes at the boundary
            am=np.array([2.0 * self.delta, -2.0 * self.delta]) / MU0,
            ncurr=0,
            piota_type="power_series",
            ai=np.array([-2.0]),
            raxis_c=_cosine_series(lambda ph: self.axis(ph)[0], ntor + 1),
            zaxis_s=_sine_series(lambda ph: self.axis(ph)[1], ntor + 1),
            rbc=rbc,
            zbs=zbs,
        )


# The member the tests use; one near the resonance at iota = 7/2; configuration A of
# the paper's supplement; and the iota = 2 member of the supplement's DESC script.
SHEARED = Sheared("sheared", eps=0.6, S=2.2, lam=2.97, k_b=0.2)
SHEARED_72 = Sheared("sheared-near-3.5", eps=1.0, S=1.75, lam=2.65, k_b=0.2)
SHEARED_A = Sheared("sheared-A", eps=1.08, S=3.0, lam=3.5, k_b=0.7)
IOTA2 = Iota2("iota2", eps=0.25, delta=1.0 / 64.0)
MEMBERS = {m.name: m for m in (SHEARED, SHEARED_72, SHEARED_A, IOTA2)}


def ftol_for(ns):
    """The residual a run at ns is taken to: 1e-18 up to ns = 200, 1e-16 above."""
    return 1e-18 if ns <= 200 else 1e-16


def run(member, ns, mpol, ntor, niter=20000, delt=0.9, max_threads=1, grids=()):
    """VMEC++ on a member at one resolution, after the coarser radial grids given, if
    any, each grid run to ftol_for its ns; the output is returned whether or not the
    residual reached it."""
    ns_array = [*grids, ns]
    vmec_input = member.vmec_input(ns, mpol, ntor).model_copy(
        update={
            "ns_array": np.array(ns_array, dtype=np.int64),
            "ftol_array": np.array([ftol_for(n) for n in ns_array]),
            "niter_array": np.array([niter] * len(ns_array), dtype=np.int64),
            "delt": delt,
            "return_outputs_even_if_not_converged": True,
        }
    )
    return vmecpp.run(vmec_input, max_threads=max_threads, verbose=False).wout


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


def angle_grid(nth=24, nph=24):
    theta = 2.0 * np.pi * (np.arange(nth) + 0.25) / nth
    phi = 2.0 * np.pi * (np.arange(nph) + 0.4) / (nph * NFP)
    return np.meshgrid(theta, phi, indexing="ij")


def surface_error(member, wout, grid=None):
    """Largest |psi - psi(s)| at the points of each full-grid surface, relative to the
    boundary psi."""
    th, ph = grid if grid is not None else angle_grid()
    xm, xn = np.asarray(wout.xm, float), np.asarray(wout.xn, float)
    rmnc = _modes_first(wout.rmnc, len(xm))
    zmns = _modes_first(wout.zmns, len(xm))
    s = np.linspace(0.0, 1.0, wout.ns)
    worst = 0.0
    for j in range(1, wout.ns):
        r = _series(rmnc[:, j], xm, xn, th, ph, "c")
        z = _series(zmns[:, j], xm, xn, th, ph, "s")
        psi = member.field(r * np.cos(ph), r * np.sin(ph), z)[3]
        worst = max(worst, float(np.abs(psi - member.psi_of_s(s[j])).max()))
    return worst / float(member.psi_of_s(1.0))


def axis_error(member, wout, grid=None):
    """Largest distance of the run's axis from the exact axis, in metres."""
    th, ph = grid if grid is not None else angle_grid()
    xm, xn = np.asarray(wout.xm, float), np.asarray(wout.xn, float)
    r = _series(_modes_first(wout.rmnc, len(xm))[:, 0], xm, xn, th, ph, "c")
    z = _series(_modes_first(wout.zmns, len(xm))[:, 0], xm, xn, th, ph, "s")
    r_exact, z_exact = member.axis(ph)
    return float(np.hypot(r - r_exact, z - z_exact).max())


def field_error(member, wout, grid=None, s_min=0.0):
    """Largest |B - B_exact| / |B_exact| at the half-grid points with s >= s_min, the
    run's field being B^u e_u + B^v e_v with the geometry interpolated as VMEC
    interpolates it: even m the mean of the two nodes, odd m sqrt(s) times the mean of
    the nodes' coefficients over sqrt(s), the axis taking the first node's."""
    th, ph = grid if grid is not None else angle_grid()
    xm, xn = np.asarray(wout.xm, float), np.asarray(wout.xn, float)
    xm_nyq, xn_nyq = np.asarray(wout.xm_nyq, float), np.asarray(wout.xn_nyq, float)
    rmnc = _modes_first(wout.rmnc, len(xm))
    zmns = _modes_first(wout.zmns, len(xm))
    bsupu = _modes_first(wout.bsupumnc, len(xm_nyq))
    bsupv = _modes_first(wout.bsupvmnc, len(xm_nyq))
    s = np.linspace(0.0, 1.0, wout.ns)
    odd = (xm % 2) == 1
    cp, sp = np.cos(ph), np.sin(ph)
    worst = 0.0
    for j in range(1, wout.ns):
        sh = 0.5 * (s[j - 1] + s[j])
        if sh < s_min:
            continue
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
        ex, ey, ez, _ = member.field(r * cp, r * sp, z)
        err = np.sqrt((bx - ex) ** 2 + (by - ey) ** 2 + (bz - ez) ** 2)
        worst = max(worst, float((err / np.sqrt(ex * ex + ey * ey + ez * ez)).max()))
    return worst


def current_error(member, wout, s_min=0.0):
    """Largest |buco - buco_exact| on the half grid with s >= s_min, relative to the
    largest |buco_exact|: the error in the enclosed toroidal current."""
    s = np.linspace(0.0, 1.0, wout.ns)
    sh = 0.5 * (s[1:] + s[:-1])
    exact = member.buco_of_s(sh)
    error = np.abs(np.asarray(wout.buco)[1:] - exact)[sh >= s_min]
    return float(error.max() / np.abs(exact).max())


def current_derivative_error(member, wout, s_min=0.0):
    """Largest |jcurv - jcurv_exact| on the interior full-grid points with s >= s_min,
    relative to the largest |jcurv_exact|: the error in the radial derivative of the
    enclosed current, jcurv being signgs / mu0 times the centred difference of buco."""
    s = np.linspace(0.0, 1.0, wout.ns)[1:-1]
    exact = float(wout.signgs) * member.dbuco_ds(s) / MU0
    run_value = np.asarray(wout.jcurv)[1:-1]
    keep = s >= s_min
    return float(np.abs(run_value - exact)[keep].max() / np.abs(exact).max())


def current_near_axis(member, wout):
    """Buco / s on the innermost half-grid point against the exact limit d(buco)/ds at
    the axis, as a relative difference: how the enclosed current leaves the axis."""
    s1 = 0.5 / (wout.ns - 1)
    run_slope = float(np.asarray(wout.buco)[1]) / s1
    exact_slope = float(member.dbuco_ds(0.0))
    return abs(run_slope - exact_slope) / abs(exact_slope)


MEASURES = {
    "psi": surface_error,
    "axis": axis_error,
    "B": field_error,
    "B, s >= 1/4": functools.partial(field_error, s_min=0.25),
    "current": current_error,
    "current, s >= 1/4": functools.partial(current_error, s_min=0.25),
    "jcurv": current_derivative_error,
    "jcurv, s >= 1/4": functools.partial(current_derivative_error, s_min=0.25),
    "current near axis": current_near_axis,
}


def measure(member, wout):
    out = {name: fn(member, wout) for name, fn in MEASURES.items()}
    out["fsq"] = float(wout.fsqr + wout.fsqz + wout.fsql)
    return out


def orders(ns_list, values):
    """Observed orders log(e_1 / e_2) / log(h_1 / h_2) between consecutive
    resolutions, h = 1 / (ns - 1)."""
    h = 1.0 / (np.asarray(ns_list, float) - 1.0)
    e = np.asarray(values, float)
    return list(np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:]))


def fitted_order(ns_list, values):
    """The observed order over all resolutions, the slope of log e against log h."""
    h = 1.0 / (np.asarray(ns_list, float) - 1.0)
    return float(np.polyfit(np.log(h), np.log(np.asarray(values, float)), 1)[0])


def checks(results):
    """The checks on the scans of "sheared", as (check, criterion, measured, passed).

    They are empty when the results hold no radial and angular scan of "sheared".
    """
    entry = results.get("sheared")
    if entry is None or not entry["angular"]:
        return []
    radial, angular = entry["radial"], entry["angular"]
    ns = [r["ns"] for r in radial]
    fit = {k: fitted_order(ns, [r[k] for r in radial]) for k in MEASURES}
    out = []

    keys = ("psi", "axis", "B", "current")
    out.append(
        (
            "ns convergence at O(h)",
            "psi, the axis, B and the current converge at fitted order >= 0.9",
            ", ".join(f"{k} {fit[k]:.2f}" for k in keys),
            all(fit[k] >= 0.9 for k in keys),
        )
    )

    bulk = [r["B, s >= 1/4"] for r in angular]
    falls = {k: angular[0][k] / angular[2][k] for k in ("psi", "axis", "B, s >= 1/4")}
    out.append(
        (
            "mpol, ntor convergence",
            "B for s >= 1/4 falls at every step of the angular scan, and psi, the axis "
            "and B for s >= 1/4 fall at least 100-fold over its first two steps",
            "B, s >= 1/4 "
            + ", ".join(f"{v:.1e}" for v in bulk)
            + "; falls "
            + ", ".join(f"{k} {v:.0f}" for k, v in falls.items()),
            all(b < a for a, b in itertools.pairwise(bulk))
            and all(v >= 100.0 for v in falls.values()),
        )
    )

    h = 1.0 / (radial[-1]["ns"] - 1)
    out.append(
        (
            "magnetic axis",
            "the axis converges at fitted order >= 0.9 and lies within 2e-4 h metres "
            "of the exact axis at the finest ns",
            f"order {fit['axis']:.2f}, {radial[-1]['axis']:.1e} m at ns "
            f"{radial[-1]['ns']}, where 2e-4 h is {2e-4 * h:.1e} m",
            fit["axis"] >= 0.9 and radial[-1]["axis"] <= 2e-4 * h,
        )
    )

    out.append(
        (
            "B and its derivatives",
            "B for s >= 1/4 converges at fitted order >= 1.5, and jcurv, the radial "
            "derivative of the current, at a lower order than the current, over the "
            "whole profile and for s >= 1/4",
            f"B, s >= 1/4 {fit['B, s >= 1/4']:.2f}; jcurv {fit['jcurv']:.2f} against "
            f"the current's {fit['current']:.2f}, for s >= 1/4 "
            f"{fit['jcurv, s >= 1/4']:.2f} against {fit['current, s >= 1/4']:.2f}",
            fit["B, s >= 1/4"] >= 1.5
            and fit["jcurv"] < fit["current"]
            and fit["jcurv, s >= 1/4"] < fit["current, s >= 1/4"],
        )
    )

    pairs = entry["orders"]["current, s >= 1/4"]
    near = [r["current near axis"] for r in radial]
    out.append(
        (
            "enclosed current",
            "the current for s >= 1/4 converges at order >= 1.8 between every pair of "
            "resolutions, and buco / s at the innermost half-grid point approaches the "
            "exact limit at every step",
            "orders "
            + ", ".join(f"{o:.2f}" for o in pairs)
            + "; buco / s off by "
            + ", ".join(f"{v:.1e}" for v in near),
            all(o >= 1.8 for o in pairs)
            and all(b < a for a, b in itertools.pairwise(near)),
        )
    )
    return out


@dataclasses.dataclass(frozen=True)
class Scan:
    """A radial scan of a member at fixed mpol, ntor and an angular scan at fixed ns.

    The runs of a member that converges start from the coarser grids of MULTIGRID below
    their ns; the others run each grid to niter on its own.
    """

    member: str
    ns: tuple[int, ...]
    mpol: int
    ntor: int
    niter: int = 50000
    multigrid: bool = True
    angular_ns: int | None = None
    angular_modes: tuple[tuple[int, int], ...] = ()


MULTIGRID = (25, 100, 400)
MODES = ((4, 2), (6, 4), (8, 6), (10, 8), (12, 10), (14, 12), (16, 14))
QUICK, FULL = (13, 25, 50, 100), (25, 50, 100, 200, 400, 1000)

SUITES = {
    "demo": (Scan("sheared", (13, 25), 12, 10),),
    "quick": (
        Scan("sheared", QUICK, 12, 10, angular_ns=100, angular_modes=MODES),
        Scan("sheared-near-3.5", QUICK, 12, 10, niter=20000, multigrid=False),
        Scan("sheared-A", QUICK, 12, 10, niter=20000, multigrid=False),
        Scan("iota2", QUICK, 10, 8, niter=20000, multigrid=False),
    ),
    "full": (
        Scan("sheared", FULL, 12, 10, angular_ns=400, angular_modes=MODES),
        Scan("sheared-near-3.5", FULL, 12, 10, niter=20000, multigrid=False),
        Scan("sheared-A", FULL, 12, 10, niter=20000, multigrid=False),
        Scan("iota2", FULL, 10, 8, niter=20000, multigrid=False),
    ),
}


def _row(member, scan, ns, mpol, ntor, max_threads):
    grids = tuple(n for n in MULTIGRID if n < ns) if scan.multigrid else ()
    t0 = time.time()
    wout = run(
        member, ns, mpol, ntor, niter=scan.niter, max_threads=max_threads, grids=grids
    )
    row = {"ns": ns, "mpol": mpol, "ntor": ntor, **measure(member, wout)}
    row["seconds"] = time.time() - t0
    row["current profile"] = {
        "s": list(0.5 * (np.linspace(0, 1, ns)[1:] + np.linspace(0, 1, ns)[:-1])),
        "buco": list(np.asarray(wout.buco)[1:]),
    }
    print(
        f"{member.name} ns {ns} mpol {mpol} ntor {ntor}: "
        + ", ".join(f"{k} {row[k]:.2e}" for k in (*MEASURES, "fsq"))
        + f", {row['seconds']:.0f} s",
        flush=True,
    )
    return row


def run_suite(suite, members=None, out_dir=None, max_threads=None):
    """Every scan of the suite, or of the members named; the results, their observed
    orders and plots go to out_dir when it is given."""
    results = {}
    for scan in SUITES[suite]:
        if members and scan.member not in members:
            continue
        member = MEMBERS[scan.member]
        entry = {
            "radial": [
                _row(member, scan, ns, scan.mpol, scan.ntor, max_threads)
                for ns in scan.ns
            ],
            "angular": [
                _row(member, scan, scan.angular_ns, mpol, ntor, max_threads)
                for mpol, ntor in scan.angular_modes
            ],
        }
        entry["orders"] = {
            key: orders(
                [r["ns"] for r in entry["radial"]], [r[key] for r in entry["radial"]]
            )
            for key in MEASURES
        }
        results[scan.member] = entry
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{suite}.json").write_text(json.dumps(results, indent=1))
        (out_dir / f"{suite}.md").write_text(summary(results, suite))
        plot(results, out_dir, suite)
    print(summary(results, suite))
    return results


def summary(results, suite):
    """The errors and observed orders as markdown tables."""
    lines = [f"## Exact equilibria, {suite} suite", ""]
    keys = [*MEASURES, "fsq"]
    for name, entry in results.items():
        rows = entry["radial"]
        lines += [
            f"### {name}, mpol {rows[0]['mpol']}, ntor {rows[0]['ntor']}",
            "",
            "| ns | " + " | ".join(keys) + " |",
            "|---" * (len(keys) + 1) + "|",
        ]
        lines += [
            f"| {r['ns']} | " + " | ".join(f"{r[k]:.2e}" for k in keys) + " |"
            for r in rows
        ]
        lines += [
            "| order | "
            + " | ".join(
                ", ".join(f"{o:.2f}" for o in entry["orders"][k]) for k in MEASURES
            )
            + " | |",
            "",
        ]
        if entry["angular"]:
            rows = entry["angular"]
            lines += [
                f"{name} at ns {rows[0]['ns']}:",
                "",
                "| mpol, ntor | " + " | ".join(keys) + " |",
                "|---" * (len(keys) + 1) + "|",
            ]
            lines += [
                f"| {r['mpol']}, {r['ntor']} | "
                + " | ".join(f"{r[k]:.2e}" for k in keys)
                + " |"
                for r in rows
            ]
            lines.append("")
    found = checks(results)
    if found:
        lines += [
            "### Checks on sheared",
            "",
            "| check | criterion | measured | result |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {name} | {criterion} | {measured} | {'pass' if ok else 'FAIL'} |"
            for name, criterion, measured, ok in found
        ]
        lines.append("")
    return "\n".join(lines) + "\n"


def plot(results, out_dir, suite):
    """For each member the errors against h, against mpol, and the enclosed current over
    s against its exact limit at the axis."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for name, entry in results.items():
        member = MEMBERS[name]
        rows = entry["radial"]
        fig, ax = plt.subplots(figsize=(7, 5))
        h = np.array([1.0 / (r["ns"] - 1) for r in rows])
        for key in MEASURES:
            ax.loglog(h, [r[key] for r in rows], "o-", label=key)
        scale = rows[-1]["current"]
        for order, style in ((1, ":"), (2, "--")):
            ax.loglog(h, scale * (h / h[-1]) ** order, "k" + style, label=f"h^{order}")
        ax.set_xlabel("h = 1 / (ns - 1)")
        ax.set_ylabel("deviation from the exact solution")
        ax.set_title(f"{name}, mpol {rows[0]['mpol']}, ntor {rows[0]['ntor']}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{suite}_{name}_radial.png", dpi=120)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in rows:
            s = np.asarray(r["current profile"]["s"])
            ax.plot(
                s,
                np.asarray(r["current profile"]["buco"]) / s,
                ".-",
                ms=3,
                label=f"ns {r['ns']}",
            )
        s = np.linspace(1e-3, 1.0, 400)
        ax.plot(s, member.buco_of_s(s) / s, "k-", label="exact")
        ax.set_xlabel("s")
        ax.set_ylabel("buco / s")
        ax.set_title(f"{name}: enclosed current over s")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{suite}_{name}_current.png", dpi=120)
        plt.close(fig)
        if entry["angular"]:
            rows = entry["angular"]
            fig, ax = plt.subplots(figsize=(7, 5))
            for key in MEASURES:
                ax.semilogy(
                    [r["mpol"] for r in rows], [r[key] for r in rows], "o-", label=key
                )
            ax.set_xlabel("mpol, with ntor = mpol - 2")
            ax.set_ylabel("deviation from the exact solution")
            ax.set_title(f"{name}, ns {rows[0]['ns']}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"{suite}_{name}_angular.png", dpi=120)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=list(SUITES), default="demo")
    parser.add_argument(
        "--member",
        action="append",
        choices=list(MEMBERS),
        help="run only this member's scans; may be repeated",
    )
    parser.add_argument("--out", default=None, help="directory for the plots and data")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="VMEC++ threads per run, all available by default",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit with an error when a check on sheared fails",
    )
    args = parser.parse_args()
    results = run_suite(args.suite, args.member, args.out, args.threads)
    failed = [name for name, _, _, ok in checks(results) if not ok]
    if args.check and failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
