# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Convergence of the half-grid rule's discrete solution, by manufactured solution.

`half_grid_consistency.py` measures the half-grid rule on exact half-point values:
the radial force it assembles at a node differs from the exact force by order h^2.
This file solves with it. On the annulus 1/4 < s < 1, with R and Z held at a
manufactured mapping at s = 1/4 and s = 1 and lambda and iota taken from the
mapping, the unknowns are the Fourier coefficients of R and Z on the interior
nodes, and the equations are the node rule's radial force r_s collocated at as
many angles per node as R has coefficients and the surface current component
r_u = -(d_u B_v - d_v B_u) B^v at the outer half point collocated at as many as Z
has. The right-hand side is the continuum value of each at the mapping, so the
mapping solves the continuum problem and the discrete solution's distance from it
is the discretization error.

Lambda held fixed is the problem on the quotient by the poloidal gauge. A
non-stellarator-symmetric mapping carries a second gauge direction, the rotation of
each surface's poloidal origin, which a fixed lambda leaves nearly free; the m = 1
cosine coefficients of Z are tied to the sine coefficients of R, VMEC's own
constraint RSC_1n = ZCC_1n, and r_u is collocated for the remaining cosine modes.

Stellarocq (https://github.com/CharlesCNorton/stellarocq, gen/mms_colloc.py)
establishes each discrete solution by the interval Newton test on the collocated
system (Colloc.colloc_correct) in a box of radius at most 5.2e-18 in each
coefficient, which encloses the error, and bounds the inverse of the system there
(Colloc.colloc_stability). tests/test_half_grid_convergence.py holds those numbers.

The pressure enters the radial force only as -mu0 dp/ds, a flux function that is the
same in the rule and in its source, so the discrete solution does not depend on it:
the high-beta mapping has the asymmetric one's error.

Usage:
    python half_grid_convergence.py [--ns 9 17 33]
"""

from __future__ import annotations

import argparse
import math

import manufactured_solution as mms
import numpy as np

MU0 = 4.0e-7 * math.pi


def cases():
    """The three mappings of half_grid_consistency restricted to m <= 1 and without
    lambda's m = 2 mode, so that five Fourier modes carry R, Z and lambda."""
    p = list(mms.FITTED_P)
    p[11] = 0.0
    sym = mms.build_case(p, m2=False)
    asym = mms.build_case(p, m2=False, asym=mms.ASYM)
    high = mms.build_case(
        p, base=dict(mms.FITTED_BASE, pres_scale=5 * 160000.0), m2=False, asym=mms.ASYM
    )
    return {"3d": sym, "asymmetric": asym, "high-beta": high}


def physics_modes(nfp):
    return [(0, 0), (0, nfp), (1, -nfp), (1, 0), (1, nfp)]


def coefficient_rows(case, svals, nfp, modes):
    """R cos, Z sin, lambda sin and their non-symmetric partners in the basis
    cos(m u - n v), n in units of the geometric angle."""
    c = mms.combined_coefficients(case, np.asarray(svals, float), 3, 1)
    order = mms.mode_order(3, 1)
    idx = {(m, n * nfp): i for i, (m, n) in enumerate(order)}
    return {
        key: np.array([[c[key][idx[mn], j] for mn in modes] for j in range(len(svals))])
        for key in ("rmnc", "zmns", "lmns", "rmns", "zmnc", "lmnc")
    }


def _select(points, funcs, count):
    A = np.array([[f(u, v) for f in funcs] for (u, v) in points])
    R = A.copy()
    chosen = []
    for _ in range(count):
        norms = np.linalg.norm(R, axis=1)
        for i in chosen:
            norms[i] = -1.0
        i = int(np.argmax(norms))
        chosen.append(i)
        q = R[i] / norms[i]
        R = R - np.outer(R @ q, q)
    return [points[i] for i in chosen]


def collocation(modes, nfp, lasym, gauge):
    cosf = [(lambda u, v, a=a, b=b: math.cos(a * u - b * v)) for a, b in modes]
    sinf = [
        (lambda u, v, a=a, b=b: math.sin(a * u - b * v))
        for a, b in modes
        if (a, b) != (0, 0)
    ]
    cosz = [
        (lambda u, v, a=a, b=b: math.cos(a * u - b * v))
        for a, b in modes
        if not (gauge and a == 1)
    ]
    mpol = max(m for m, _ in modes) + 1
    ntor = max(abs(n) for _, n in modes) // nfp
    nv = 2 * ntor + 3
    if lasym:
        nu = 2 * (mpol + 1)
        grid = [
            ((i + 0.5) * 2 * math.pi / nu, 2 * math.pi * k / (nfp * nv))
            for i in range(nu)
            for k in range(nv)
        ]
        return _select(grid, cosf + sinf, len(cosf + sinf)), _select(
            grid, sinf + cosz, len(sinf + cosz)
        )
    nu = mpol + 1
    grid = [
        ((i + 0.5) * math.pi / nu, 2 * math.pi * k / (nfp * nv))
        for i in range(nu)
        for k in range(nv)
    ]
    return _select(grid, cosf, len(cosf)), _select(grid, sinf, len(sinf))


class Continuum:
    """The continuum field of a mapping at complex radius and angles:
    sqrt(g) B^u = phip (iota - lambda_v), sqrt(g) B^v = phip (1 + lambda_u)."""

    def __init__(self, case):
        self.model = mms.Model(case)
        self.nfp = case.nfp
        self.phip = float(case.phip)
        self.iota = case.iota_coeff
        self.am = case.am
        self.pres_scale = case.pres_scale

    def _q(self, name, key, s, u, v):
        return self.model.fn[name + "|" + key](s, u, v * self.nfp) + 0.0 * (u + v)

    def cov(self, s, u, v):
        q = self._q
        R, Rs, Ru, Rv = (q("R", k, s, u, v) for k in ("", "s", "u", "v"))
        Zs, Zu, Zv = (q("Z", k, s, u, v) for k in ("s", "u", "v"))
        Lu, Lv = q("L", "u", s, u, v), q("L", "v", s, u, v)
        g = R * (Ru * Zs - Rs * Zu)
        chip = self.phip * sum(c * s**i for i, c in enumerate(self.iota))
        Bu = (chip - Lv) / g
        Bv = (self.phip + Lu) / g
        guu, guv = Ru**2 + Zu**2, Ru * Rv + Zu * Zv
        gvv = Rv**2 + Zv**2 + R**2
        gsu, gsv = Rs * Ru + Zs * Zu, Rs * Rv + Zs * Zv
        return {
            "Bu": Bu,
            "Bv": Bv,
            "B_u": guu * Bu + guv * Bv,
            "B_v": guv * Bu + gvv * Bv,
            "B_s": gsu * Bu + gsv * Bv,
        }

    def mu0_dpds(self, s):
        return (
            MU0
            * self.pres_scale
            * sum(i * c * s ** (i - 1) for i, c in enumerate(self.am) if i > 0)
        )

    def rs(self, s, u, v, h=1e-30):
        q = self.cov(complex(s), u, v)
        ds = self.cov(s + 1j * h, u, v)
        du = self.cov(complex(s), u + 1j * h, v)
        dv = self.cov(complex(s), u, v + 1j * h)

        def d(z, k):
            return np.imag(z[k]) / h

        return (
            (d(dv, "B_s") - d(ds, "B_v")) * np.real(q["Bv"])
            - (d(ds, "B_u") - d(du, "B_s")) * np.real(q["Bu"])
            - self.mu0_dpds(s)
        )

    def ru(self, s, u, v, h=1e-30):
        q = self.cov(complex(s), u, v)
        du = self.cov(complex(s), u + 1j * h, v)
        dv = self.cov(complex(s), u, v + 1j * h)
        js = np.imag(du["B_v"]) / h - np.imag(dv["B_u"]) / h
        return -js * np.real(q["Bv"])


class Problem:
    """The collocated half-grid system on the annulus smin < s < 1."""

    def __init__(self, case, ns, smin=0.25):
        self.lasym = bool(case.lasym)
        self.gauge = self.lasym
        nfp = case.nfp
        self.modes = physics_modes(nfp)
        self.K = len(self.modes)
        self.mm = np.array([m for m, _ in self.modes], float)
        self.nn = np.array([n for _, n in self.modes], float)
        self.sf = np.linspace(0.0, 1.0, ns)
        self.sh = 0.5 * (self.sf[1:] + self.sf[:-1])
        full = coefficient_rows(case, self.sf, nfp, self.modes)
        half = coefficient_rows(case, self.sh, nfp, self.modes)
        self.coef = {
            "R": full["rmnc"],
            "Z": full["zmns"],
            "Ra": full["rmns"],
            "Za": full["zmnc"],
        }
        self.lam, self.lama = half["lmns"], half["lmnc"]
        self.iota_h = np.array(
            [sum(c * s**i for i, c in enumerate(case.iota_coeff)) for s in self.sh]
        )
        self.phip = float(case.phip)
        self.am = [case.pres_scale * c for c in case.am]
        self.rows = [j for j in range(2, ns - 1) if self.sf[j] > smin + 1e-12]
        self.blocks = ["R", "Z"] + (["Ra", "Za"] if self.lasym else [])
        self.tied = [k for k, (m, _) in enumerate(self.modes) if self.gauge and m == 1]
        self.unknowns = [
            (j, b, k)
            for j in self.rows
            for b in self.blocks
            for k, mn in enumerate(self.modes)
            if not (b in ("Z", "Ra") and mn == (0, 0))
            and not (b == "Za" and k in self.tied)
        ]
        self.n = len(self.unknowns)
        self.xstar = np.array([self.coef[b][j, k] for j, b, k in self.unknowns])
        ps, pu = collocation(self.modes, nfp, self.lasym, self.gauge)
        cont = Continuum(case)
        self.points = []
        for j in self.rows:
            us = np.array([u for u, _ in ps])
            vs = np.array([v for _, v in ps])
            for (u, v), f in zip(ps, cont.rs(self.sf[j], us, vs), strict=True):
                self.points.append((j, 0, u, v, float(f)))
            us = np.array([u for u, _ in pu])
            vs = np.array([v for _, v in pu])
            for (u, v), f in zip(pu, cont.ru(self.sh[j], us, vs), strict=True):
                self.points.append((j, 1, u, v, float(f)))

    def _with(self, x):
        c = {b: self.coef[b].astype(complex) for b in self.coef}
        for (j, b, k), val in zip(self.unknowns, x, strict=True):
            c[b][j, k] = val
            if b == "Ra" and k in self.tied:
                c["Za"][j, k] = val
        return c

    def _half(self, c, blk, ra, rb, sa, sb, sh):
        ya, yb = c[blk][ra], c[blk][rb]
        odd = (self.mm % 2) == 1
        ev, ed = 0.5 * (ya + yb), (yb - ya) / (sb - sa)
        qa, qb = ya / math.sqrt(sa), yb / math.sqrt(sb)
        ov = math.sqrt(sh) * 0.5 * (qa + qb)
        od = math.sqrt(sh) * (qb - qa) / (sb - sa) + ov / (2.0 * sh)
        return np.where(odd, ov, ev), np.where(odd, od, ed)

    @staticmethod
    def _series(val, ds, cosk, sink, m, n, even):
        k0, k1 = (cosk, sink) if even else (sink, cosk)
        su, sv = (-m, n) if even else (m, -n)
        return {
            "0": val @ k0,
            "s": ds @ k0,
            "u": (su * val) @ k1,
            "v": (sv * val) @ k1,
            "su": (su * ds) @ k1,
            "sv": (sv * ds) @ k1,
            "uu": (-m * m * val) @ k0,
            "uv": (m * n * val) @ k0,
            "vv": (-n * n * val) @ k0,
        }

    @staticmethod
    def _lseries(lam, cosk, sink, m, n, even):
        k0, k1 = (cosk, sink) if even else (sink, cosk)
        su, sv = (-m, n) if even else (m, -n)
        return {
            "u": (su * lam) @ k1,
            "v": (sv * lam) @ k1,
            "uu": (-m * m * lam) @ k0,
            "uv": (m * n * lam) @ k0,
            "vv": (-n * n * lam) @ k0,
        }

    def _half_point(self, c, j, side, u, v):
        ra, rb = (j - 1, j) if side == 0 else (j, j + 1)
        hrow = j - 1 if side == 0 else j
        sa, sb, sh = self.sf[ra], self.sf[rb], self.sh[hrow]
        arg = self.mm * u - self.nn * v
        cosk, sink = np.cos(arg), np.sin(arg)
        m, n = self.mm, self.nn
        R = self._series(
            *self._half(c, "R", ra, rb, sa, sb, sh), cosk, sink, m, n, True
        )
        Z = self._series(
            *self._half(c, "Z", ra, rb, sa, sb, sh), cosk, sink, m, n, False
        )
        L = self._lseries(self.lam[hrow], cosk, sink, m, n, False)
        if self.lasym:
            Ra = self._series(
                *self._half(c, "Ra", ra, rb, sa, sb, sh), cosk, sink, m, n, False
            )
            Za = self._series(
                *self._half(c, "Za", ra, rb, sa, sb, sh), cosk, sink, m, n, True
            )
            La = self._lseries(self.lama[hrow], cosk, sink, m, n, True)
            R = {k: R[k] + Ra[k] for k in R}
            Z = {k: Z[k] + Za[k] for k in Z}
            L = {k: L[k] + La[k] for k in L}
        tau = R["u"] * Z["s"] - R["s"] * Z["u"]
        g = R["0"] * tau
        tau_u = (
            R["uu"] * Z["s"] + R["u"] * Z["su"] - (R["su"] * Z["u"] + R["s"] * Z["uu"])
        )
        tau_v = (
            R["uv"] * Z["s"] + R["u"] * Z["sv"] - (R["sv"] * Z["u"] + R["s"] * Z["uv"])
        )
        g_u = R["u"] * tau + R["0"] * tau_u
        g_v = R["v"] * tau + R["0"] * tau_v
        guu = R["u"] ** 2 + Z["u"] ** 2
        guv = R["u"] * R["v"] + Z["u"] * Z["v"]
        gvv = R["v"] ** 2 + Z["v"] ** 2 + R["0"] ** 2
        gsu = R["s"] * R["u"] + Z["s"] * Z["u"]
        gsv = R["s"] * R["v"] + Z["s"] * Z["v"]
        gsu_u = (
            R["su"] * R["u"] + R["s"] * R["uu"] + Z["su"] * Z["u"] + Z["s"] * Z["uu"]
        )
        gsu_v = (
            R["sv"] * R["u"] + R["s"] * R["uv"] + Z["sv"] * Z["u"] + Z["s"] * Z["uv"]
        )
        gsv_u = (
            R["su"] * R["v"] + R["s"] * R["uv"] + Z["su"] * Z["v"] + Z["s"] * Z["uv"]
        )
        gsv_v = (
            R["sv"] * R["v"] + R["s"] * R["vv"] + Z["sv"] * Z["v"] + Z["s"] * Z["vv"]
        )
        guu_v = 2 * (R["u"] * R["uv"] + Z["u"] * Z["uv"])
        guv_u = (
            R["uu"] * R["v"] + R["u"] * R["uv"] + Z["uu"] * Z["v"] + Z["u"] * Z["uv"]
        )
        guv_v = (
            R["uv"] * R["v"] + R["u"] * R["vv"] + Z["uv"] * Z["v"] + Z["u"] * Z["vv"]
        )
        gvv_u = 2 * (R["v"] * R["uv"] + Z["v"] * Z["uv"] + R["0"] * R["u"])
        bu = self.iota_h[hrow] - L["v"]
        bv = 1.0 + L["u"]
        Bu, Bv = self.phip * bu / g, self.phip * bv / g
        g2 = g * g
        Bu_u = self.phip * (-L["uv"] * g - bu * g_u) / g2
        Bv_u = self.phip * (L["uu"] * g - bv * g_u) / g2
        Bu_v = self.phip * (-L["vv"] * g - bu * g_v) / g2
        Bv_v = self.phip * (L["uv"] * g - bv * g_v) / g2
        return {
            "Bu": Bu,
            "Bv": Bv,
            "B_u": guu * Bu + guv * Bv,
            "B_v": guv * Bu + gvv * Bv,
            "B_s_u": gsu_u * Bu + gsu * Bu_u + (gsv_u * Bv + gsv * Bv_u),
            "B_s_v": gsu_v * Bu + gsu * Bu_v + (gsv_v * Bv + gsv * Bv_v),
            "Js": (guv_u * Bu + guv * Bu_u + (gvv_u * Bv + gvv * Bv_u))
            - (guu_v * Bu + guu * Bu_v + (guv_v * Bv + guv * Bv_v)),
        }

    def residual(self, x):
        c = self._with(x)
        out = np.zeros(self.n, dtype=complex)
        for i, (j, comp, u, v, f) in enumerate(self.points):
            qp = self._half_point(c, j, 1, u, v)
            if comp == 0:
                qm = self._half_point(c, j, 0, u, v)
                inv_h = 1.0 / (self.sh[j] - self.sh[j - 1])

                def avg(k, qm=qm, qp=qp):
                    return 0.5 * (qm[k] + qp[k])

                def dif(k, qm=qm, qp=qp, inv_h=inv_h):
                    return (qp[k] - qm[k]) * inv_h

                mu0pp = MU0 * sum(
                    i * a * self.sf[j] ** (i - 1)
                    for i, a in enumerate(self.am)
                    if i > 0
                )
                r = (
                    (avg("B_s_v") - dif("B_v")) * avg("Bv")
                    - (dif("B_u") - avg("B_s_u")) * avg("Bu")
                    - mu0pp
                )
            else:
                r = -(qp["Js"] * qp["Bv"])
            out[i] = r - f
        return out

    def jacobian(self, x, h=1e-30):
        J = np.zeros((self.n, self.n))
        for k in range(self.n):
            xx = x.astype(complex)
            xx[k] += 1j * h
            J[:, k] = np.imag(self.residual(xx)) / h
        return J

    def solve(self, steps=8):
        x = self.xstar.copy()
        J = self.jacobian(x)
        for _ in range(steps):
            step = np.linalg.solve(J, -np.real(self.residual(x)))
            x = x + step
            if np.abs(step).max() < 1e-15 * np.abs(x).max():
                break
            J = self.jacobian(x)
        return x, J


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", type=int, nargs="+", default=[9, 17, 33])
    a = ap.parse_args()
    for name, case in cases().items():
        prev = None
        for ns in a.ns:
            pb = Problem(case, ns)
            x, J = pb.solve()
            err = float(np.abs(x - pb.xstar).max())
            inv = float(np.abs(np.linalg.inv(J)).sum(axis=1).max())
            fall = "" if prev is None else f"  fall x{prev / err:.3f}"
            print(
                f"{name:10s} ns={ns:3d} unknowns {pb.n:4d}  max |x_h - x*| {err:.6e}  "
                f"||J^-1|| {inv:.4e}{fall}"
            )
            prev = err


if __name__ == "__main__":
    main()
