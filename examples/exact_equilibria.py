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
(section 3). Both have two field periods and stellarator symmetry. Issan, Citrin,
Landreman and Tracey, "Analytic non-axisymmetric 3D MHD equilibria without stellarator
symmetry" (arXiv:2610.07304), add to each family a parameter that breaks stellarator
symmetry, tau0 to the first (their section 2) and tau1 to the second (their section
3), Landreman's families being those at zero; members with a nonzero one run with
lasym. Sengupta et al., "Exact Nonsymmetric Vacuum Magnetic Fields with Nested Toroidal
Surfaces" (arXiv:2610.28466), give vacuum fields B = grad Phi with nested surfaces and
no continuous symmetry, Phi and the surface label psi being Jacobi elliptic functions
of confocal ellipsoidal coordinates (u, v, zeta) of the Cartesian position; a member of
their ellipsoidal family (appendix A) runs with zero pressure and zero toroidal current,
inside a given boundary and in free boundary, its field then being the external field.

This file writes VMEC++ inputs for these members from the formulas, runs VMEC++ over
sequences of radial and angular resolutions, and measures, at VMEC++'s own points,
quantities that do not depend on its poloidal angle:

- psi on each full-grid surface against the surface's flux label,
- the magnetic axis,
- the field vector at the half-grid points,
- the toroidal current enclosed by each half-grid surface (buco), and its radial
  derivative on the full grid (jcurv),
- for the vacuum members, whose enclosed current is zero, iota on the full grid and psi
  on the boundary,

with their observed orders of convergence, and plots them. Each run starts from the
coarser grids of MULTIGRID below its ns and takes every grid to the residual ftol_for
gives it.

    python examples/exact_equilibria.py [--suite demo|quick|full] [--member NAME]
                                        [--out DIR] [--threads N] [--check]
                                        [--reference DIR]

The members are:

- "sheared" (eps = 0.6, S = 2.2, lambda = 2.97, k_b = 0.2): iota from 4.34 to 4.37,
  midway between the resonances at 4 of m = 1 and at 9/2 of m = 4, and beta 1.9 per
  cent; tests/test_exact_equilibrium.py uses it;
- "sheared-asym", "sheared" with tau1 = 0.15: iota from 4.30 to 4.32, beta 1.9 per
  cent, and rbs up to 15 per cent of the largest m = 1 rbc; the tests use it too;
- "sheared-near-3.5" (eps = 1, S = 1.75, lambda = 2.65, k_b = 0.2): iota from 3.43
  to 3.44, 0.06 below the resonance at 7/2 of m = 4, n = 14, and beta 1.8 per cent;
- "sheared-A", the configuration A of Landreman's supplement (eps = 1.08, S = 3,
  lambda = 3.5, k_b = 0.7): iota from 5.69 to 6.22, which crosses 6, and beta 20 per
  cent;
- "iota2" (eps = 0.25, delta = 1/64): every surface rational;
- "iota2-asym", "iota2" with tau0 = 0.5, whose axis, centred at Z = -1/3, has no
  stellarator symmetry;
- "vacuum", a member of the ellipsoidal family of Sengupta et al. (k = m_g = 1/5, u0 =
  v0 = 2 F(pi/4 | 1/5), delta = 0) inside the surface psi = psi_0 + 0.1, psi_0 = 4
  being the label on the axis: two field periods, stellarator symmetry and iota from
  -0.1437 on the axis to -0.1368 at the boundary;
- "vacuum-free", "vacuum" in free boundary, starting from the surface psi_0 + 0.05: the
  run has to find the boundary.

The demo suite runs "sheared" at two radial resolutions; the quick suite runs every
member but "iota2-asym" up to ns = 100, "vacuum-free" at ns = 25 and 50, with an
angular scan of "sheared" at ns = 100; the full suite runs every member up to ns =
1000, "vacuum-free" up to ns = 400, with an angular scan at ns = 400. With --out the
results go to DIR as JSON, markdown tables and plots of the deviations against h and
mpol and of the enclosed current, or iota, over s. The summary ends with checks on the
convergence in ns of the vacuum members and on the scans of "sheared": convergence in
ns at O(h) and in mpol and ntor, the magnetic axis, B against its derivative jcurv, and
the enclosed current at O(h^2) with its limit at the axis; with --check a failed check
fails the run. With --reference DIR, a checkout of the repository at REFERENCE_COMMIT,
a further check compares the formulas here with the repository's own functions.

The papers' units have mu0 = 1: B is in tesla and lengths in metres, so the pressure
is p / mu0 in pascals. The poloidal angles of Landreman and of Issan et al. turn
clockwise in an (R, Z) section, the opposite of VMEC's theta, so VMEC's iota is minus
theirs.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.special import ellipj, ellipk, ellipkinc

import vmecpp

MU0 = 4.0e-7 * math.pi
NFP = 2


def _dft_boundary(section, mpol, ntor):
    """Rbc, zbs, rbs and zbc of a boundary given by section(theta, phi) -> (R, Z), from
    its values on a grid that resolves the retained modes."""
    nth, nph = 2 * mpol + 6, 2 * ntor + 6
    theta = 2.0 * np.pi * np.arange(nth) / nth
    phi = 2.0 * np.pi * np.arange(nph) / (nph * NFP)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    r, z = section(th, ph)
    rbc, zbs, rbs, zbc = (np.zeros((mpol, 2 * ntor + 1)) for _ in range(4))
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n < 0:
                continue
            arg = m * th - n * NFP * ph
            weight = 1.0 if m == 0 and n == 0 else 2.0
            rbc[m, n + ntor] = weight * np.mean(r * np.cos(arg))
            zbs[m, n + ntor] = weight * np.mean(z * np.sin(arg))
            rbs[m, n + ntor] = weight * np.mean(r * np.sin(arg))
            zbc[m, n + ntor] = weight * np.mean(z * np.cos(arg))
    return rbc, zbs, rbs, zbc


def _cosine_series(values_of_phi, count):
    """Cosine coefficients in n NFP phi of a function of phi, n < count: raxis_c of R
    and zaxis_c of Z."""
    phi = 2.0 * np.pi * np.arange(64) / (64 * NFP)
    values = values_of_phi(phi)
    return np.array(
        [
            (1.0 if n == 0 else 2.0) * np.mean(values * np.cos(n * NFP * phi))
            for n in range(count)
        ]
    )


def _sine_series(values_of_phi, count):
    """Coefficients c of f = -sum c[n] sin(n NFP phi), n < count, of a function f of
    phi: zaxis_s of Z and raxis_s of R."""
    phi = 2.0 * np.pi * np.arange(64) / (64 * NFP)
    values = values_of_phi(phi)
    return np.array(
        [0.0]
        + [-2.0 * np.mean(values * np.sin(n * NFP * phi)) for n in range(1, count)]
    )


def _geometry(section, axis, mpol, ntor, lasym):
    """The boundary and axis fields of a VmecInput, from section(theta, phi) -> (R, Z)
    on the boundary and axis(phi) -> (R, Z), with the coefficients that break
    stellarator symmetry when lasym is set."""
    rbc, zbs, rbs, zbc = _dft_boundary(section, mpol, ntor)
    fields = {
        "lasym": lasym,
        "rbc": rbc,
        "zbs": zbs,
        "raxis_c": _cosine_series(lambda ph: axis(ph)[0], ntor + 1),
        "zaxis_s": _sine_series(lambda ph: axis(ph)[1], ntor + 1),
    }
    if lasym:
        fields |= {
            "rbs": rbs,
            "zbc": zbc,
            "raxis_s": _sine_series(lambda ph: axis(ph)[0], ntor + 1),
            "zaxis_c": _cosine_series(lambda ph: axis(ph)[1], ntor + 1),
        }
    return fields


@dataclasses.dataclass(frozen=True)
class Sheared:
    """A member of the sheared-iota family, Landreman's section 3, with the parameter
    tau1 of Issan et al.'s section 3.

    Its parameters are eps, S, lambda, the boundary k_b and tau1, and its surfaces are
    the circles of constant psi = k^2 / 2 in the paper's (X, Y) plane.
    """

    name: str
    eps: float
    S: float
    lam: float
    k_b: float
    tau1: float = 0.0

    def _h(self, sigma):
        return np.sqrt(4.0 * sigma * sigma + self.eps * self.eps)

    def semiaxes(self, sigma):
        """The semiaxes a, b of the confocal ellipse sigma, with a b = sigma."""
        b = np.sqrt((self._h(sigma) + self.eps) / 2.0)
        return sigma / b, b

    def field(self, x, y, z):
        """The Cartesian field and psi, Landreman's equations 3.2 and 3.3 with Xi
        shifted by i tau1 (Issan et al.'s equations 15 to 17)."""
        w = x + 1j * y
        wb = x - 1j * y
        k = wb * np.sqrt(1.0 + self.eps / wb**2)
        xi = w * k + np.pi / 2.0 - self.S + 1j * self.tau1
        e = np.exp(-1j * self.lam * z)
        bxy = e * 1j * np.sin(xi) / (2.0 * k)
        bz = np.real(e * np.cos(xi)) / self.lam
        psi = (np.sin(self.lam * z) ** 2 + (self.lam * bz) ** 2) / 2.0
        return np.real(bxy), np.imag(bxy), bz, psi

    def _sigma(self, p, y, nu):
        """The confocal coordinate sigma at X = p, Y = y where the paper's nu, which is
        eps sin(2 zeta) / 2 + tau1, takes the value nu."""
        return (
            self.S
            + np.arctan(np.tanh(nu) * y / np.sqrt(1.0 - y * y))
            - np.arcsin(p / np.sqrt(np.cosh(nu) ** 2 - y * y))
        )

    def section(self, k, phi, chi):
        """R and Z of X = -k cos(chi), Y = k sin(chi) in the plane phi, by bisection on
        the confocal coordinate sigma (Landreman's section 3.2)."""
        chi, phi = np.broadcast_arrays(np.asarray(chi, float), np.asarray(phi, float))
        p, y = -k * np.cos(chi), k * np.sin(chi)
        lo = np.full(chi.shape, self.S - np.arcsin(k))
        hi = np.full(chi.shape, self.S + np.arcsin(k))

        def residual(sigma):
            a, b = self.semiaxes(sigma)
            t = np.arctan2(a * np.sin(phi), b * np.cos(phi))
            nu = self.eps / 2.0 * np.sin(2.0 * t) + self.tau1
            return sigma - self._sigma(p, y, nu), t

        for _ in range(60):
            mid = (lo + hi) / 2.0
            r, _ = residual(mid)
            lo = np.where(r < 0.0, mid, lo)
            hi = np.where(r >= 0.0, mid, hi)
        sigma = (lo + hi) / 2.0
        _, t = residual(sigma)
        a, b = self.semiaxes(sigma)
        return np.hypot(a * np.cos(t), b * np.sin(t)), -np.arcsin(y) / self.lam

    def boundary(self, theta, phi):
        """R and Z of the surface k = k_b at VMEC's theta = -chi."""
        return self.section(self.k_b, phi, -theta)

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
        sigma = self._sigma(-k * np.cos(u), k * np.sin(u), self.tau1)
        q = np.pi / self.lam * np.mean(1.0 / (self._h(sigma) * c), axis=1)
        nu = self.eps / 2.0 * np.sin(2.0 * u) + self.tau1
        sigma = self.S + np.arcsin(k / np.cosh(nu))
        g = (self._h(sigma) + self.eps * np.cos(2.0 * u)) / 2.0
        root = np.sqrt(np.cosh(nu) ** 2 - k * k)
        a = 2.0 * np.pi / self.lam * np.mean(g / (self._h(sigma) * root), axis=1)
        return q, a

    def _current(self, k, n=1024):
        """The toroidal current enclosed by the surface k along chi, in amperes, from
        the circulation of the field around the section phi = 0."""
        k = np.atleast_1d(np.asarray(k, float))[:, None]
        u = 2.0 * np.pi * np.arange(n) / n
        p, y = -k * np.cos(u), k * np.sin(u)
        c = np.sqrt(1.0 - y * y)
        sh, ch, th = np.sinh(self.tau1), np.cosh(self.tau1), np.tanh(self.tau1)
        sigma = self._sigma(p, y, self.tau1)
        d_sigma = -th * p / (c * (c * c + th * th * y * y)) - y * np.sqrt(
            ch * ch - k * k
        ) / (ch * ch - y * y)
        xi = sigma - self.S
        density = (c * sh * np.sin(xi) - y * ch * np.cos(xi)) * d_sigma / (
            2.0 * self._h(sigma)
        ) + p * p / (self.lam**2 * c)
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
        # p = (psi_b - psi) / lambda^2, which vanishes at the boundary
        am = -np.asarray(prof["psi"].coef) / (self.lam**2 * MU0)
        am[0] = (self.k_b**2 / 2.0 - prof["psi"].coef[0]) / (self.lam**2 * MU0)
        return vmecpp.VmecInput(
            nfp=NFP,
            mpol=mpol,
            ntor=ntor,
            ns_array=np.array([ns], dtype=np.int64),
            phiedge=prof["phi_edge"],
            pmass_type="power_series",
            am=am,
            ncurr=0,
            piota_type="power_series",
            ai=np.asarray(prof["iota"].coef),
            **_geometry(self.boundary, self.axis, mpol, ntor, self.tau1 != 0.0),
        )


@dataclasses.dataclass(frozen=True)
class Iota2:
    """A member of the family with iota = 2 on every surface, Landreman's section 2,
    with the parameter tau0 of Issan et al.'s section 2.

    Its parameters are eps, the boundary psi = delta and tau0; with Omega^2 = 1 -
    tau0^2, its surfaces are circles of radius sqrt(psi) about (-Omega^2 eps / 2, 0) in
    the plane of the field-line labels (u, v).
    """

    name: str
    eps: float
    delta: float
    tau0: float = 0.0

    @property
    def _ab(self):
        return math.sqrt(1.0 + self.eps), math.sqrt(1.0 - self.eps)

    @property
    def _omega2(self):
        return 1.0 - self.tau0**2

    @property
    def _u_axis(self):
        return -self._omega2 * self.eps / 2.0

    def field(self, x, y, z):
        """The Cartesian field and psi, Issan et al.'s equations 4 to 7, which are
        Landreman's 2.1 and 2.3 at tau0 = 0."""
        a, b = self._ab
        t, o2 = self.tau0, self._omega2
        s = (x / a) ** 2 + (y / b) ** 2
        f = np.sqrt(2.0 * s - s * s - 4.0 * z * z - 4.0 * t * s * z)
        bx = ((2.0 * z + t * s) * x - (a / b) * f * y) / s
        by = ((2.0 * z + t * s) * y + (b / a) * f * x) / s
        bz = 1.0 - s - 2.0 * t * z
        q = o2 / 2.0 * (x * x + y * y + 4.0 * z * z) + 2.0 * t * z
        b2 = bx * bx + by * by + bz * bz
        psi = (
            o2
            / 2.0
            * (q + b2 / 2.0 - 1.0 + t * t / (2.0 * o2) + o2 * self.eps**2 / 2.0)
        )
        return bx, by, bz, psi

    def embedding(self, u, v, t):
        """The field line (u, v) at the paper's zeta = t, Issan et al.'s equation 10."""
        a, b = self._ab
        o2 = self._omega2
        ell = np.sqrt((1.0 + np.sqrt(1.0 - 4.0 * (u * u + v * v))) / 2.0)
        ct, st = np.cos(t), np.sin(t)
        g = 2.0 * t + math.asin(self.tau0)
        return (
            a / math.sqrt(o2) * (ell * ct + (u * ct + v * st) / ell),
            b / math.sqrt(o2) * (ell * st + (v * ct - u * st) / ell),
            (v * np.cos(g) - u * np.sin(g) - self.tau0 / 2.0) / o2,
        )

    def section(self, psi, theta, phi):
        """R and Z at VMEC's theta = beta - 2 phi, a straight-field-line angle with
        iota = -2, in the plane phi (after Landreman's supplement)."""
        a, b = self._ab
        o2 = self._omega2
        beta = theta + 2.0 * phi
        u = self._u_axis + np.sqrt(psi) * np.cos(beta)
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
        r = np.hypot(aa * ct + cc * st, dd * ct + ee * st) / math.sqrt(o2)
        # cos and sin of 2 t + arcsin(tau0)
        c2, s2 = ct * ct - st * st, 2.0 * st * ct
        cg = c2 * math.sqrt(o2) - s2 * self.tau0
        sg = s2 * math.sqrt(o2) + c2 * self.tau0
        z = (v * cg - u * sg - self.tau0 / 2.0) / o2
        return r, z

    def boundary(self, theta, phi):
        """R and Z of the surface psi = delta at VMEC's theta."""
        return self.section(self.delta, theta, phi)

    def axis(self, phi):
        """R and Z of the magnetic axis, Issan et al.'s equation 12, in the plane
        phi."""
        o2 = self._omega2
        ax = math.sqrt((1.0 + self.eps) * (1.0 - o2 * self.eps) / o2)
        ay = math.sqrt((1.0 - self.eps) * (1.0 + o2 * self.eps) / o2)
        t = np.arctan2(ax * np.sin(phi), ay * np.cos(phi))
        z = self.eps * o2 * np.sin(2.0 * t + math.asin(self.tau0)) - self.tau0
        return ax * ay / np.hypot(ay * np.cos(phi), ax * np.sin(phi)), z / (2.0 * o2)

    def _buco(self, psi, n=128):
        """The buco of the surface psi, from the field along a loop on it.

        buco is mu0 / (2 pi) times the current enclosed along theta, the integral of B
        along a loop of the label angle alpha at fixed zeta divided by 2 pi.
        """
        alpha = 2.0 * np.pi * np.arange(n) / n
        h = 1e-30
        rho = np.sqrt(np.atleast_1d(np.asarray(psi, float)))[:, None]
        u = self._u_axis + rho * np.cos(alpha + 1j * h)
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
        o2 = self._omega2
        return vmecpp.VmecInput(
            nfp=NFP,
            mpol=mpol,
            ntor=ntor,
            ns_array=np.array([ns], dtype=np.int64),
            phiedge=math.pi * a * b * self.delta / o2,
            pmass_type="power_series",
            # p = 2 (delta - psi) / Omega^2, which vanishes at the boundary
            am=np.array([2.0 * self.delta, -2.0 * self.delta]) / (o2 * MU0),
            ncurr=0,
            piota_type="power_series",
            ai=np.array([-2.0]),
            **_geometry(self.boundary, self.axis, mpol, ntor, self.tau0 != 0.0),
        )


def _sn(re, im, m):
    """Jacobi's sn(re + i im | m) for real re, im (Abramowitz and Stegun 16.21.2)."""
    s, c, d, _ = ellipj(re, m)
    s1, c1, d1, _ = ellipj(im, 1.0 - m)
    return (s * d1 + 1j * c * d * s1 * c1) / (c1 * c1 + m * s * s * s1 * s1)


@dataclasses.dataclass(frozen=True)
class Vacuum:
    """A vacuum field of the ellipsoidal family of Sengupta et al. with one repetition
    of the profiles, the axis at u = v = u0 and the phase delta = 0, inside the surface
    psi = psi_0 + dpsi.

    Its parameters are the modulus k of the profiles, the shape parameter m_g of the
    confocal ellipsoids, u0 and dpsi, psi_0 = 1/k - 1 being the label on the axis; the
    scale kappa = K(1 - k^2) / (2 K(1 - m_g)) makes the field single valued. A
    free-boundary member takes its field as the external field and starts from the
    surface psi_0 + dpsi / 2.
    """

    name: str
    k: float
    m_g: float
    u0: float
    dpsi: float
    free_boundary: bool = False

    @property
    def _psi0(self):
        return 1.0 / self.k - 1.0

    @property
    def _kappa(self):
        return ellipk(1.0 - self.k**2) / (2.0 * ellipk(1.0 - self.m_g))

    def _position(self, u, v, zeta):
        """X(u, v, zeta), its derivatives in u, v and zeta, and P and E of the metric
        (Sengupta et al.'s equations A1 and A4)."""
        m_t = 1.0 - self.m_g
        sr, cr, dr, _ = ellipj((u + v) / 4.0, self.m_g)
        ss, cs, ds, _ = ellipj((u - v) / 4.0, self.m_g)
        st, ct, dt, _ = ellipj((zeta + ellipk(m_t)) / 2.0, m_t)
        x = np.stack([cr * st * ds, dr * ct * cs, dt * ss]) / sr
        # derivatives in the arguments (u + v) / 4, (u - v) / 4 and tau_g
        d_r = -np.stack([dr * st * ds, cr * ct * cs, cr * dr * dt * ss]) / sr**2
        d_s = (
            np.stack([-self.m_g * cr * st * ss * cs, -dr * ct * ss * ds, dt * cs * ds])
            / sr
        )
        d_t = (
            np.stack([cr * ds * ct * dt, -dr * cs * st * dt, -m_t * st * ct * ss]) / sr
        )
        lam, nu, mu = sr**-2, self.m_g * ss**2, 1.0 - m_t * st**2
        return (
            x,
            (d_r + d_s) / 4.0,
            (d_r - d_s) / 4.0,
            d_t / 2.0,
            (lam - nu) / 4.0,
            (lam + nu - 2.0 * mu) / 4.0,
        )

    def _covariant(self, u, v, zeta):
        """B_u, B_v and B_zeta, the derivatives of the potential of Sengupta et al.'s
        equation 1: the imaginary parts of sn(z_1) and sn(z_2) and the real part of
        their sum."""
        m, kappa = self.k**2, self._kappa
        tau = kappa * zeta
        p1 = _sn(ellipk(m) + kappa * (u - self.u0), tau, m)
        p2 = _sn(ellipk(m) + kappa * (v - self.u0), ellipk(1.0 - m) + tau, m)
        return p1.imag, p2.imag, (p1 + p2).real

    def _label(self, u, v, zeta):
        """The label psi of Sengupta et al.'s equation 2."""
        m, kappa = self.k**2, self._kappa
        _, cx, dx, _ = ellipj(kappa * (u - self.u0), m)
        _, cy, dy, _ = ellipj(kappa * (v - self.u0), m)
        U, V = cx / dx, dy / (self.k * cy)
        s, c, d, _ = ellipj(kappa * zeta, 1.0 - m)
        return (
            d
            * (V - U)
            / np.sqrt((c * c + m * (U * s) ** 2) * (c * c + m * (V * s) ** 2))
        )

    def _field(self, u, v, zeta):
        """The Cartesian field at X(u, v, zeta), Sengupta et al.'s equation A2."""
        _, xu, xv, xz, p, e = self._position(u, v, zeta)
        bu, bv, bz = self._covariant(u, v, zeta)
        h = e / p
        return ((bu - h * bv) * xu + (bv - h * bu) * xv + bz * xz) / (p * p - e * e)

    @property
    def _semiaxes(self):
        """The semiaxes in X and Y of the axis, an ellipse in the plane Z = 0."""
        s, c, d, _ = ellipj(self.u0 / 2.0, self.m_g)
        return c / s, d / s

    def axis(self, phi):
        a, b = self._semiaxes
        return a * b / np.hypot(b * np.cos(phi), a * np.sin(phi)), 0.0 * phi

    def _axis_zeta(self, phi):
        """The zeta of the axis point in the plane phi."""
        a, b = self._semiaxes
        m_t = 1.0 - self.m_g
        amplitude = np.arctan2(np.cos(phi) / a, np.sin(phi) / b)
        return 2.0 * ellipkinc(amplitude, m_t) - ellipk(m_t)

    def _coordinates(self, x, y, z):
        """(u, v, zeta) at Cartesian points, by Newton's method from the axis point in
        the same plane phi."""
        target = np.stack(np.broadcast_arrays(x, y, z)).astype(float)
        zeta = self._axis_zeta(np.arctan2(target[1], target[0]))
        q = np.stack([np.full_like(zeta, self.u0), np.full_like(zeta, self.u0), zeta])
        for _ in range(30):
            position, xu, xv, xz, _, _ = self._position(*q)
            jacobian = np.moveaxis(np.stack([xu, xv, xz], axis=-1), 0, -2)
            residual = np.moveaxis(position - target, 0, -1)[..., None]
            step = np.moveaxis(np.linalg.solve(jacobian, residual)[..., 0], -1, 0)
            q -= step
            if np.abs(step).max() < 1e-13:
                break
        return q

    def field(self, x, y, z):
        """The Cartesian field and psi - psi_0."""
        q = self._coordinates(x, y, z)
        return (*self._field(*q), self._label(*q) - self._psi0)

    def magnetic_field(self, r, phi, z):
        """The cylindrical components of the field, the external field when the boundary
        is free."""
        bx, by, bz = self._field(
            *self._coordinates(r * np.cos(phi), r * np.sin(phi), z)
        )
        return (
            bx * np.cos(phi) + by * np.sin(phi),
            by * np.cos(phi) - bx * np.sin(phi),
            bz,
        )

    @property
    def _rho_max(self):
        """The radius of the largest disc about the axis within the regular domain.

        In the (u, v) plane the domain is where |kappa (u - u0)| and |kappa (v - u0)|
        are below K(k^2), 0 < u + v < 4 K(m_g) and |u - v| < 4 K(m_g).
        """
        k_g = ellipk(self.m_g)
        return min(
            ellipk(self.k**2) / self._kappa,
            math.sqrt(2.0) * self.u0,
            math.sqrt(2.0) * (2.0 * k_g - self.u0),
            2.0 * math.sqrt(2.0) * k_g,
        )

    def _rho(self, psi, du, dv, zeta):
        """The distance from the axis along (du, dv) in the (u, v) plane at which the
        label, which increases along every such ray, reaches psi, by bisection."""
        lo = np.zeros(np.broadcast(psi, du, dv, zeta).shape)
        hi = np.full(lo.shape, 0.95 * self._rho_max)
        for _ in range(52):
            mid = (lo + hi) / 2.0
            inside = self._label(self.u0 + mid * du, self.u0 + mid * dv, zeta) < psi
            lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
        return (lo + hi) / 2.0

    def _label_at(self, r, phi, z):
        """The label above psi_0 at cylindrical points, infinite at points that Newton's
        method does not reach from the axis inside the disc of _rho_max."""
        x, y = r * np.cos(phi), r * np.sin(phi)
        q = self._coordinates(x, y, z)
        miss = ~(
            np.linalg.norm(self._position(*q)[0] - np.stack([x, y, z]), axis=0) < 1e-9
        )
        miss |= ~(np.hypot(q[0] - self.u0, q[1] - self.u0) < self._rho_max)
        return np.where(miss, np.inf, self._label(*q) - self._psi0)

    def section(self, psi, theta, phi):
        """R and Z of the surface psi in the plane phi at the angle theta about the axis
        in the coordinates that make the surfaces near the axis circles.

        These coordinates are H^(1/2) times the offset from the axis in (R, Z), H being
        the Hessian of psi at the axis from second differences, so the point lies on the
        ray from the axis along H^(-1/2) (cos theta, sin theta); bisection finds it,
        from twice the distance at which the quadratic form of H reaches psi.
        """
        theta, phi = np.broadcast_arrays(
            np.asarray(theta, float), np.asarray(phi, float)
        )
        r0, z0 = self.axis(phi)

        def label(dr, dz):
            return self._label_at(r0 + dr, phi, z0 + dz)

        h, centre = 1e-3, label(0.0, 0.0)
        h_rr = (label(h, 0.0) - 2.0 * centre + label(-h, 0.0)) / h**2
        h_zz = (label(0.0, h) - 2.0 * centre + label(0.0, -h)) / h**2
        h_rz = (label(h, h) - label(h, -h) - label(-h, h) + label(-h, -h)) / (4 * h * h)
        hessian = np.moveaxis(np.array([[h_rr, h_rz], [h_rz, h_zz]]), (0, 1), (-2, -1))
        w, v = np.linalg.eigh(hessian)
        ray = v @ (w[..., None] ** -0.5 * np.swapaxes(v, -1, -2))
        dr = ray[..., 0, 0] * np.cos(theta) + ray[..., 0, 1] * np.sin(theta)
        dz = ray[..., 1, 0] * np.cos(theta) + ray[..., 1, 1] * np.sin(theta)
        target = psi - self._psi0
        lo = np.zeros(theta.shape)
        hi = np.full(theta.shape, 2.0 * np.sqrt(2.0 * target))
        for _ in range(4):
            hi = np.where(label(hi * dr, hi * dz) < target, 1.5 * hi, hi)
        for _ in range(52):
            mid = (lo + hi) / 2.0
            inside = label(mid * dr, mid * dz) < target
            lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
        rho = (lo + hi) / 2.0
        return r0 + rho * dr, z0 + rho * dz

    def boundary(self, theta, phi):
        return self.section(self._psi0 + self.dpsi, theta, phi)

    def _rates(self, psi, n=256):
        """I' and J', the toroidal and poloidal fluxes over 2 pi differentiated in psi.

        I is the flux inside the surface psi, integrated around it in the section zeta
        = 0, where psi = V - U and the flux density in u and v is P (U + V); J is the
        flux through the ribbon v = u0 from the axis to the surface, integrated along
        its edge on the ribbon, where B_v = 0 (Sengupta et al.'s appendix A).
        """
        psi = np.atleast_1d(np.asarray(psi, float))[:, None]
        m, kappa = self.k**2, self._kappa
        du, dv = (
            np.cos(2.0 * np.pi * np.arange(n) / n),
            np.sin(2.0 * np.pi * np.arange(n) / n),
        )
        rho = self._rho(psi, du, dv, 0.0)
        sx, cx, dx, _ = ellipj(kappa * rho * du, m)
        sy, cy, dy, _ = ellipj(kappa * rho * dv, m)
        p = self._position(self.u0 + rho * du, self.u0 + rho * dv, 0.0 * rho)[4]
        # d psi / d rho, U = cd(x | m) and V = 1 / (k cd(y | m)) having derivatives
        # -(1 - m) sn / dn^2 and (1 - m) sn / (k cn^2)
        d_psi = kappa * (1.0 - m) * (du * sx / dx**2 + dv * sy / (self.k * cy**2))
        i_rate = np.mean(p * (cx / dx + dy / (self.k * cy)) * rho / d_psi, axis=1)
        k_t = ellipk(1.0 - self.m_g)
        zeta = 8.0 * k_t * np.arange(n) / n
        rho = self._rho(psi, 1.0, 0.0, zeta)
        u, v = self.u0 + rho, np.full(rho.shape, self.u0)
        sx, cx, dx, _ = ellipj(kappa * rho, m)
        s, c, d, _ = ellipj(kappa * zeta, 1.0 - m)
        # d psi / d u at v = u0, where V = 1 / k
        U, V = cx / dx, 1.0 / self.k
        qu, qv = c * c + m * (U * s) ** 2, c * c + m * (V * s) ** 2
        d_psi = d / np.sqrt(qu * qv) * (1.0 + (V - U) * m * U * s * s / qu)
        d_psi *= kappa * (1.0 - m) * sx / dx**2
        e = self._position(u, v, zeta)[5]
        j_rate = (
            -4.0
            * k_t
            / np.pi
            * np.mean(e * self._covariant(u, v, zeta)[0] / d_psi, axis=1)
        )
        return i_rate, j_rate

    @functools.cached_property
    def profiles(self):
        """The toroidal flux through the boundary, and psi and VMEC's iota in s.

        psi - psi_0 and iota are power series in the normalized toroidal flux s, from I'
        and J' at Chebyshev points in psi. Along a turn of zeta a field line turns J'/I'
        times about the axis in the sense from u towards v, which is VMEC's theta, while
        phi falls by 2 pi, so VMEC's iota is -J'/I'.
        """
        nodes, degree = 32, 12
        domain = [self._psi0, self._psi0 + self.dpsi]
        t = np.cos((np.arange(nodes) + 0.5) * np.pi / nodes)
        psi = self._psi0 + self.dpsi * (1.0 - t) / 2.0
        i_rate, j_rate = self._rates(psi)
        flux = Chebyshev.fit(psi, i_rate, nodes - 1, domain=domain).integ(
            lbnd=domain[0]
        )
        iota = Chebyshev.fit(psi, -j_rate / i_rate, nodes - 1, domain=domain)
        psi = (
            self._psi0
            + self.dpsi * (1.0 - np.cos(np.linspace(0.0, np.pi, 4 * nodes))) / 2.0
        )
        phi_edge = 2.0 * np.pi * float(flux(domain[1]))
        s = 2.0 * np.pi * flux(psi) / phi_edge
        return {
            "phi_edge": phi_edge,
            "psi": Polynomial.fit(s, psi - self._psi0, degree).convert(),
            "iota": Polynomial.fit(s, iota(psi), degree).convert(),
        }

    def psi_of_s(self, s):
        return self.profiles["psi"](s)

    def iota_of_s(self, s):
        return self.profiles["iota"](s)

    def vmec_input(self, ns, mpol, ntor):
        start = self._psi0 + self.dpsi / (2.0 if self.free_boundary else 1.0)
        fields = _geometry(
            functools.partial(self.section, start), self.axis, mpol, ntor, False
        )
        if self.free_boundary:
            fields |= {"lfreeb": True, "nzeta": 2 * ntor + 4, "nvacskip": 6}
        return vmecpp.VmecInput(
            nfp=NFP,
            mpol=mpol,
            ntor=ntor,
            ns_array=np.array([ns], dtype=np.int64),
            # phi falls along the field
            phiedge=-self.profiles["phi_edge"],
            pmass_type="power_series",
            am=np.zeros(1),
            ncurr=1,
            pcurr_type="power_series",
            ac=np.zeros(1),
            **fields,
        )


# The member the tests use; one near the resonance at iota = 7/2; configuration A of
# Landreman's supplement; the iota = 2 member of the supplement's DESC script; the first
# and the last with the parameter of Issan et al. that breaks stellarator symmetry; and
# an ellipsoidal member of Sengupta et al., inside a given boundary and in free
# boundary.
SHEARED = Sheared("sheared", eps=0.6, S=2.2, lam=2.97, k_b=0.2)
SHEARED_72 = Sheared("sheared-near-3.5", eps=1.0, S=1.75, lam=2.65, k_b=0.2)
SHEARED_A = Sheared("sheared-A", eps=1.08, S=3.0, lam=3.5, k_b=0.7)
IOTA2 = Iota2("iota2", eps=0.25, delta=1.0 / 64.0)
SHEARED_ASYM = Sheared("sheared-asym", eps=0.6, S=2.2, lam=2.97, k_b=0.2, tau1=0.15)
IOTA2_ASYM = Iota2("iota2-asym", eps=0.25, delta=1.0 / 64.0, tau0=0.5)
VACUUM = Vacuum(
    "vacuum", k=0.2, m_g=0.2, u0=2.0 * ellipkinc(math.pi / 4.0, 0.2), dpsi=0.1
)
VACUUM_FREE = dataclasses.replace(VACUUM, name="vacuum-free", free_boundary=True)
MEMBERS = {
    m.name: m
    for m in (
        SHEARED,
        SHEARED_ASYM,
        SHEARED_72,
        SHEARED_A,
        IOTA2,
        IOTA2_ASYM,
        VACUUM,
        VACUUM_FREE,
    )
}

REFERENCE_COMMIT = "c97fbcdd879206e0cfee6ad6e885ae0b13e215e8"
REFERENCE_TOLERANCE = 1.0e-13


def _load_reference(repository, relative_path, name):
    """A script of the reference repository as a module, with the matplotlib settings it
    makes on import undone."""
    import matplotlib as mpl  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        name, Path(repository) / relative_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with mpl.rc_context():
        spec.loader.exec_module(module)
    return module


def reference_deviations(repository):
    """The largest deviations of the formulas here from the functions of
    landreman/analytic_3d_equilibria at REFERENCE_COMMIT, checked out in repository.

    For "iota2" they cover the field and psi, the surfaces the boundary is taken from
    and the axis; for the sheared members, whose field the repository evaluates only
    within its DESC script, the confocal semiaxes, the cross-sections the boundary is
    taken from and the axis.
    """
    iota_2 = _load_reference(
        repository, "iota_2/analytic_3d_equilibrium_iota_2.py", "reference_iota_2"
    )
    sheared = _load_reference(
        repository,
        "sheared_iota/20260913-02_analytic_3D_equilibrium_sheared_iota_figures.py",
        "reference_sheared_iota",
    )
    rng = np.random.default_rng(0)
    phi = 2.0 * np.pi * rng.random(64)
    deviations = {}

    eps = IOTA2.eps
    psi = IOTA2.delta * rng.uniform(0.01, 1.0, 64)
    beta, t = 2.0 * np.pi * rng.random((2, 64))
    xyz = iota_2.embedding(
        -eps / 2.0 + np.sqrt(psi) * np.cos(beta), np.sqrt(psi) * np.sin(beta), t, eps
    )
    ours = np.stack(IOTA2.field(*np.moveaxis(xyz, -1, 0)), axis=-1)
    theirs = np.concatenate(
        (iota_2.field(xyz, eps), iota_2.flux(xyz, eps)[..., None]), axis=-1
    )
    deviations["iota2 field and psi"] = float(np.max(np.abs(ours - theirs)))
    r, z = IOTA2.section(psi, beta - 2.0 * phi, phi)
    u, v, _ = np.moveaxis(
        iota_2.inverse_embedding(
            np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1), eps
        ),
        -1,
        0,
    )
    deviations["iota2 surfaces"] = float(
        np.max(np.abs(u + eps / 2.0 + 1j * v - np.sqrt(psi) * np.exp(1j * beta)))
    )
    x, y, z_axis = np.moveaxis(iota_2.axis(phi, eps), -1, 0)
    r_axis, z_ours = IOTA2.axis(phi)
    deviations["iota2 axis"] = float(
        max(np.max(np.abs(np.hypot(x, y) - r_axis)), np.max(np.abs(z_axis - z_ours)))
    )

    chi = 2.0 * np.pi * np.arange(32) / 32
    for member in (SHEARED, SHEARED_72, SHEARED_A):
        case = sheared.Case(member.name, member.eps, member.S, member.k_b, member.lam)
        case.validate()
        sigma = member.S + np.arcsin(member.k_b) * np.linspace(-1.0, 1.0, 33)
        largest = float(
            np.max(
                np.abs(
                    np.subtract(member.semiaxes(sigma), sheared.semiaxes(sigma, case))
                )
            )
        )
        for angle in phi[:8]:
            largest = max(
                largest, abs(member.axis(angle)[0] - sheared.axis_radius(angle, case))
            )
            for k in member.k_b * np.array([1.0 / 3.0, 2.0 / 3.0, 1.0]):
                r, z = member.section(k, angle, chi)
                reference = sheared.cross_section(k, angle, chi, case)
                largest = max(
                    largest,
                    float(np.max(np.abs(np.stack((r, z), axis=-1) - reference))),
                )
        deviations[f"{member.name} geometry"] = largest
    return deviations


def ftol_for(ns):
    """The residual a run at ns is taken to: 1e-18 up to ns = 200, 1e-16 above."""
    return 1e-18 if ns <= 200 else 1e-16


def run(member, ns, mpol, ntor, niter=20000, delt=0.9, max_threads=1, grids=()):
    """VMEC++ on a member at one resolution, after the coarser radial grids given, if
    any, each grid run to ftol_for its ns, a free-boundary member in its own field and
    on one thread, since the vacuum threads would queue for that Python function; the
    output is returned whether or not the residual reached it."""
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
    if vmec_input.lfreeb:
        return vmecpp.run(
            vmec_input, member.magnetic_field, max_threads=1, verbose=False
        ).wout
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


def _cos_sin(wout, cos_name, sin_name, count):
    """The cos and sin coefficients of a quantity of the run, modes first, with zeros
    for the one a stellarator-symmetric run does not carry."""
    arrays = [getattr(wout, name) for name in (cos_name, sin_name)]
    shape = next(_modes_first(a, count).shape for a in arrays if a is not None)
    return [np.zeros(shape) if a is None else _modes_first(a, count) for a in arrays]


def _both(pair, xm, xn, theta, phi, d_theta=False, d_phi=False):
    """The series of a pair of cos and sin coefficient vectors, or its derivative."""
    return _series(pair[0], xm, xn, theta, phi, "c", d_theta, d_phi) + _series(
        pair[1], xm, xn, theta, phi, "s", d_theta, d_phi
    )


def angle_grid(nth=24, nph=24):
    theta = 2.0 * np.pi * (np.arange(nth) + 0.25) / nth
    phi = 2.0 * np.pi * (np.arange(nph) + 0.4) / (nph * NFP)
    return np.meshgrid(theta, phi, indexing="ij")


def surface_error(member, wout, grid=None, boundary_only=False):
    """Largest |psi - psi(s)| at the points of each full-grid surface, or of the
    boundary alone, relative to the boundary psi."""
    th, ph = grid if grid is not None else angle_grid()
    xm, xn = np.asarray(wout.xm, float), np.asarray(wout.xn, float)
    rr = _cos_sin(wout, "rmnc", "rmns", len(xm))
    zz = _cos_sin(wout, "zmnc", "zmns", len(xm))
    s = np.linspace(0.0, 1.0, wout.ns)
    worst = 0.0
    for j in [wout.ns - 1] if boundary_only else range(1, wout.ns):
        r = _both([c[:, j] for c in rr], xm, xn, th, ph)
        z = _both([c[:, j] for c in zz], xm, xn, th, ph)
        psi = member.field(r * np.cos(ph), r * np.sin(ph), z)[3]
        worst = max(worst, float(np.abs(psi - member.psi_of_s(s[j])).max()))
    return worst / float(member.psi_of_s(1.0))


def axis_error(member, wout, grid=None):
    """Largest distance of the run's axis from the exact axis, in metres."""
    th, ph = grid if grid is not None else angle_grid()
    xm, xn = np.asarray(wout.xm, float), np.asarray(wout.xn, float)
    r = _both(
        [c[:, 0] for c in _cos_sin(wout, "rmnc", "rmns", len(xm))], xm, xn, th, ph
    )
    z = _both(
        [c[:, 0] for c in _cos_sin(wout, "zmnc", "zmns", len(xm))], xm, xn, th, ph
    )
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
    rr = _cos_sin(wout, "rmnc", "rmns", len(xm))
    zz = _cos_sin(wout, "zmnc", "zmns", len(xm))
    bsupu = _cos_sin(wout, "bsupumnc", "bsupumns", len(xm_nyq))
    bsupv = _cos_sin(wout, "bsupvmnc", "bsupvmns", len(xm_nyq))
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

        rh, zh = [half(c) for c in rr], [half(c) for c in zz]
        r = _both(rh, xm, xn, th, ph)
        z = _both(zh, xm, xn, th, ph)
        ru = _both(rh, xm, xn, th, ph, d_theta=True)
        zu = _both(zh, xm, xn, th, ph, d_theta=True)
        rv = _both(rh, xm, xn, th, ph, d_phi=True)
        zv = _both(zh, xm, xn, th, ph, d_phi=True)
        bu = _both([c[:, j] for c in bsupu], xm_nyq, xn_nyq, th, ph)
        bv = _both([c[:, j] for c in bsupv], xm_nyq, xn_nyq, th, ph)
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


def iota_error(member, wout, s_min=0.0):
    """Largest |iota - iota_exact| on the full grid with s >= s_min, relative to the
    largest |iota_exact|."""
    s = np.linspace(0.0, 1.0, wout.ns)
    exact = member.iota_of_s(s)
    error = np.abs(np.asarray(wout.iotaf) - exact)[s >= s_min]
    return float(error.max() / np.abs(exact).max())


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
# The vacuum members prescribe the enclosed current, zero, and predict iota, and in free
# boundary the boundary.
VACUUM_MEASURES = {
    "psi": surface_error,
    "axis": axis_error,
    "B": field_error,
    "B, s >= 1/4": functools.partial(field_error, s_min=0.25),
    "iota": iota_error,
    "iota, s >= 1/4": functools.partial(iota_error, s_min=0.25),
}
FREE_BOUNDARY_MEASURES = VACUUM_MEASURES | {
    "boundary": functools.partial(surface_error, boundary_only=True)
}


def measures_of(member):
    if isinstance(member, Vacuum):
        return FREE_BOUNDARY_MEASURES if member.free_boundary else VACUUM_MEASURES
    return MEASURES


def measure(member, wout):
    out = {name: fn(member, wout) for name, fn in measures_of(member).items()}
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


def _vacuum_checks(results):
    """The convergence in ns of the vacuum members' surfaces, axis, field and iota, and
    of the free boundary."""
    out = []
    for name in ("vacuum", "vacuum-free"):
        entry = results.get(name)
        if entry is None or len(entry["radial"]) < 2:
            continue
        ns = [r["ns"] for r in entry["radial"]]
        keys = [key for key, _, _ in headline_of(MEMBERS[name])]
        fit = {k: fitted_order(ns, [r[k] for r in entry["radial"]]) for k in keys}
        out.append(
            (
                f"{name}: ns convergence at O(h)",
                "psi, the axis, B and iota for s >= 1/4"
                + (" and psi on the boundary" if "boundary" in keys else "")
                + " converge at fitted order >= 0.8",
                ", ".join(f"{k} {fit[k]:.2f}" for k in keys),
                all(fit[k] >= 0.8 for k in keys),
            )
        )
    return out


def checks(results, reference=None):
    """The checks, as (check, criterion, measured, passed): the agreement with the
    reference implementation when its deviations are given, the radial convergence of
    the vacuum members, and those on the scans of "sheared", which need a radial and an
    angular scan of it."""
    out = []
    if reference is not None:
        out.append(
            (
                "reference implementation",
                "the formulas deviate from those of landreman/analytic_3d_equilibria at "
                f"{REFERENCE_COMMIT[:7]} by less than {REFERENCE_TOLERANCE:.0e}",
                ", ".join(f"{k} {v:.1e}" for k, v in reference.items()),
                all(v < REFERENCE_TOLERANCE for v in reference.values()),
            )
        )
    out += _vacuum_checks(results)
    entry = results.get("sheared")
    if entry is None or not entry["angular"]:
        return out
    radial, angular = entry["radial"], entry["angular"]
    ns = [r["ns"] for r in radial]
    fit = {k: fitted_order(ns, [r[k] for r in radial]) for k in MEASURES}

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
            "B and jcurv, the radial derivative of the current, converge for s >= 1/4 "
            "at fitted order >= 1.5, and jcurv converges at a lower order than the "
            "current over the whole profile",
            f"B, s >= 1/4 {fit['B, s >= 1/4']:.2f}, jcurv, s >= 1/4 "
            f"{fit['jcurv, s >= 1/4']:.2f}; jcurv {fit['jcurv']:.2f} against the "
            f"current's {fit['current']:.2f}",
            fit["B, s >= 1/4"] >= 1.5
            and fit["jcurv, s >= 1/4"] >= 1.5
            and fit["jcurv"] < fit["current"],
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
        Scan("sheared-asym", QUICK, 12, 10),
        Scan("sheared-near-3.5", QUICK, 12, 10, niter=20000, multigrid=False),
        Scan("sheared-A", QUICK, 12, 10, niter=20000, multigrid=False),
        Scan("iota2", QUICK, 10, 8, niter=20000, multigrid=False),
        Scan("vacuum", QUICK, 12, 10),
        Scan("vacuum-free", QUICK[1:-1], 12, 10),
    ),
    "full": (
        Scan("sheared", FULL, 12, 10, angular_ns=400, angular_modes=MODES),
        Scan("sheared-asym", FULL, 12, 10),
        Scan("sheared-near-3.5", FULL, 12, 10, niter=20000, multigrid=False),
        Scan("sheared-A", FULL, 12, 10, niter=20000, multigrid=False),
        Scan("iota2", FULL, 10, 8, niter=20000, multigrid=False),
        Scan("iota2-asym", FULL, 10, 8, niter=20000, multigrid=False),
        Scan("vacuum", FULL, 12, 10),
        Scan("vacuum-free", FULL[:-1], 12, 10),
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
    if isinstance(member, Vacuum):
        row["iota profile"] = {
            "s": list(np.linspace(0, 1, ns)),
            "iota": list(np.asarray(wout.iotaf)),
        }
    else:
        row["current profile"] = {
            "s": list(0.5 * (np.linspace(0, 1, ns)[1:] + np.linspace(0, 1, ns)[:-1])),
            "buco": list(np.asarray(wout.buco)[1:]),
        }
    print(
        f"{member.name} ns {ns} mpol {mpol} ntor {ntor}: "
        + ", ".join(f"{k} {row[k]:.2e}" for k in (*measures_of(member), "fsq"))
        + f", {row['seconds']:.0f} s",
        flush=True,
    )
    return row


def run_suite(suite, members=None, out_dir=None, max_threads=None, reference=None):
    """Every scan of the suite, or of the members named; the results, their observed
    orders, the checks and plots go to out_dir when it is given."""
    results = {}
    for scan in SUITES[suite]:
        if members and scan.member not in members:
            continue
        member = MEMBERS[scan.member]
        entry: dict[str, Any] = {
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
            for key in measures_of(member)
        }
        results[scan.member] = entry
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{suite}.json").write_text(json.dumps(results, indent=1))
        (out_dir / f"{suite}.md").write_text(summary(results, suite, reference))
        plot(results, out_dir, suite)
    print(summary(results, suite, reference))
    return results


# The quantities the report shows, with the radial order each is expected to reach.
HEADLINE = (
    ("psi", r"$\psi$ (surfaces)", 1),
    ("axis", "axis [m]", 1),
    ("B, s >= 1/4", r"$B$, $s\geq 1/4$", 2),
    ("current, s >= 1/4", r"$I_{tor}$, $s\geq 1/4$", 2),
    ("jcurv, s >= 1/4", r"$dI/ds$, $s\geq 1/4$", 2),
)
VACUUM_HEADLINE = (
    ("psi", r"$\psi$ (surfaces)", 1),
    ("axis", "axis [m]", 1),
    ("B, s >= 1/4", r"$B$, $s\geq 1/4$", 1),
    ("iota, s >= 1/4", r"$\iota$, $s\geq 1/4$", 1),
)
FREE_BOUNDARY_HEADLINE = (*VACUUM_HEADLINE, ("boundary", r"$\psi$ (boundary)", 1))


def headline_of(member):
    if isinstance(member, Vacuum):
        return FREE_BOUNDARY_HEADLINE if member.free_boundary else VACUUM_HEADLINE
    return HEADLINE


def local_orders(ns_list, values):
    """The midpoints of consecutive h and the observed orders between them."""
    h = 1.0 / (np.asarray(ns_list, float) - 1.0)
    return np.sqrt(h[:-1] * h[1:]), np.asarray(orders(ns_list, values))


def _converged(row):
    return row["fsq"] <= 3.0 * ftol_for(row["ns"])


def _full_table(rows, measures, first, first_of):
    keys = [*measures, "fsq"]
    lines = [
        f"| {first} | " + " | ".join(keys) + " |",
        "|---" * (len(keys) + 1) + "|",
    ]
    lines += [
        f"| {first_of(r)} | " + " | ".join(f"{r[k]:.2e}" for k in keys) + " |"
        for r in rows
    ]
    return lines


def summary(results, suite, reference=None):
    """Per member a short table of the headline errors at the finest ns and their last
    observed orders, then the checks; the complete tables folded below."""
    lines = [f"## Exact equilibria, {suite} suite", ""]
    for name, entry in results.items():
        rows = entry["radial"]
        ns = [r["ns"] for r in rows]
        unconverged = [r["ns"] for r in rows if not _converged(r)]
        lines += [
            f"**{name}**, ns {ns[0]} to {ns[-1]}, mpol {rows[0]['mpol']}, "
            f"ntor {rows[0]['ntor']}"
            + (f", not converged at ns {unconverged}" if unconverged else ""),
            "",
            f"| quantity | error at ns {ns[-1]} | order, last pair | expected |",
            "|---|---|---|---|",
        ]
        for key, _, expected in headline_of(MEMBERS[name]):
            last = entry["orders"][key][-1]
            flag = "" if last >= expected - 0.3 else " ⚠"
            lines.append(
                f"| {key} | {rows[-1][key]:.1e} | {last:.2f}{flag} | {expected} |"
            )
        lines.append("")
    found = checks(results, reference)
    if found:
        lines += ["| check | measured | result |", "|---|---|---|"]
        lines += [
            f"| {name} | {measured} | {'pass' if ok else '**FAIL**'} |"
            for name, _, measured, ok in found
        ]
        lines.append("")
    lines += ["<details><summary>All deviations and criteria</summary>", ""]
    for name, entry in results.items():
        measures = measures_of(MEMBERS[name])
        lines += [f"### {name}", ""]
        lines += _full_table(entry["radial"], measures, "ns", lambda r: r["ns"])
        lines += [
            "| order | "
            + " | ".join(
                ", ".join(f"{o:.2f}" for o in entry["orders"][k]) for k in measures
            )
            + " | |",
            "",
        ]
        if entry["angular"]:
            lines += _full_table(
                entry["angular"],
                measures,
                "mpol, ntor",
                lambda r: f"{r['mpol']}, {r['ntor']}",
            )
            lines.append("")
    lines += [f"- **{name}**: {criterion}" for name, criterion, _, _ in found]
    lines += ["", "</details>", ""]
    return "\n".join(lines) + "\n"


def plot(results, out_dir, suite):
    """One figure per member: (a) the headline errors against h with their expected
    orders, (b) the observed order between consecutive ns, (c) the error in the
    enclosed current over s, for the vacuum members in iota, and (d) the errors against
    mpol with the radial floor at that ns. Runs that did not reach ftol are drawn
    hollow."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for name, entry in results.items():
        member = MEMBERS[name]
        headline = headline_of(member)
        rad, ang = entry["radial"], entry["angular"]
        ns = np.array([r["ns"] for r in rad])
        h = 1.0 / (ns - 1.0)
        conv = np.array([_converged(r) for r in rad])
        fig, axs = plt.subplots(2, 2, figsize=(9, 7))
        axs = axs.ravel()

        ax = axs[0]
        for (key, label, p), c in zip(headline, colors, strict=False):
            e = np.array([r[key] for r in rad])
            ax.loglog(h[conv], e[conv], "o-", color=c, label=label)
            ax.loglog(h[~conv], e[~conv], "o", mfc="none", color=c)
            ax.loglog(h, e[0] * (h / h[0]) ** p, ":", color=c, lw=1)
        ax.invert_xaxis()
        ax.set(xlabel="h = 1/(ns-1)", ylabel="deviation from the exact solution")
        ax.set_title("(a) radial convergence")
        ax.legend(fontsize=7, loc="lower left")

        ax = axs[1]
        for (key, _, _), c in zip(headline, colors, strict=False):
            hm, o = local_orders(ns, [r[key] for r in rad])
            ax.semilogx(hm, o, "o-", color=c)
        for p in (1, 2):
            ax.axhspan(p - 0.2, p + 0.2, color="0.9", zorder=0)
            ax.text(
                0.02,
                (p + 1) / 4,
                f"O(h$^{p}$)",
                transform=ax.transAxes,
                va="center",
                fontsize=7,
            )
        ax.invert_xaxis()
        ax.set(xlabel="h, midpoint of the pair", ylabel="observed order", ylim=(-1, 3))
        ax.set_title("(b) observed order between consecutive ns")

        ax = axs[2]
        vacuum = isinstance(member, Vacuum)
        for r in rad:
            if vacuum:
                s = np.asarray(r["iota profile"]["s"])
                exact = member.iota_of_s(s)
                value = np.asarray(r["iota profile"]["iota"])
            else:
                s = np.asarray(r["current profile"]["s"])
                exact = member.buco_of_s(s)
                value = np.asarray(r["current profile"]["buco"])
            ax.semilogy(
                s,
                np.abs(value - exact) / np.abs(exact).max(),
                ".-",
                ms=2,
                lw=0.8,
                label=f"ns {r['ns']}",
            )
        ax.axvline(0.25, color="k", lw=0.5, ls="--")
        quantity = "iota" if vacuum else "the enclosed current"
        ax.set(xlabel="s", ylabel=f"relative error in {quantity}")
        ax.set_title(f"(c) {'iota' if vacuum else 'enclosed current'} over s")
        ax.legend(fontsize=7)

        ax = axs[3]
        if ang:
            mpol = np.array([r["mpol"] for r in ang])
            for (key, label, _), c in zip(headline[:3], colors, strict=False):
                e = np.array([r[key] for r in ang])
                ax.semilogy(mpol, e, "o-", color=c, label=label)
                ax.axhline(e[-1], color=c, lw=0.6, ls="--")
                above = e > 3.0 * e[-1]
                if above.sum() >= 2:
                    rate = -np.polyfit(mpol[above], np.log(e[above]), 1)[0]
                    ax.text(
                        mpol[0],
                        e[0],
                        f" e$^{{-{rate:.2f}\\,mpol}}$",
                        color=c,
                        fontsize=7,
                        va="bottom",
                    )
            ax.set(xlabel="mpol, with ntor = mpol - 2")
            ax.set_title(f"(d) angular convergence at ns {ang[0]['ns']}")
            ax.legend(fontsize=7)
        else:
            ax.set_visible(False)

        fig.suptitle(
            f"{name}, {suite} suite, mpol {rad[0]['mpol']}, ntor {rad[0]['ntor']}; "
            "hollow: not converged to ftol",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{suite}_{name}.png", dpi=110)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
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
        help="exit with an error when a check fails",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="checkout of landreman/analytic_3d_equilibria at REFERENCE_COMMIT to "
        "compare the formulas with",
    )
    args = parser.parse_args()
    reference = None
    if args.reference is not None:
        reference = reference_deviations(args.reference)
    results = run_suite(args.suite, args.member, args.out, args.threads, reference)
    failed = [name for name, _, _, ok in checks(results, reference) if not ok]
    if args.check and failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
