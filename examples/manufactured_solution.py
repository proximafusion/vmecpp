# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Method of manufactured solutions for the VMEC++ discretization.

An analytic mapping (R, Z, lambda)(s, theta, zeta) with a helical axis and a
rotating-ellipse cross-section is not an equilibrium, so its continuum ideal-MHD
force is not zero.  Installing the negative of that force as a source term
(`VmecModel.set_force_source`) makes the mapping the exact solution of the
modified problem, and every difference between the solver and the mapping is
then a discretization error rather than a disagreement with another code.

What this script measures:

  * the discrete MHD energy against the continuum energy of the same mapping,
  * the discrete spectral force against the continuum ideal-MHD force,
  * the converged discrete state against the mapping itself,
  * how all three behave as the radial and the angular resolution are refined.

The mapping is restricted to the poloidal harmonics m <= 2, with the m = 2
amplitudes exactly proportional to s.  xmpq[m] = m (m - 1) then vanishes for
m = 0 and m = 1, and rzConIntoVolume sets the spectral-condensation baseline to
the boundary value times s, so the constraint force is identically zero at the
mapping.  The mapping also satisfies RSS_{1n} = ZCS_{1n}, VMEC's m = 1
poloidal-origin gauge condition, so the frozen combination of the m = 1
coefficients is zero.  It is therefore an admissible state of the solver as
well as a solution of the continuum problem.

The file is in four parts, which can be read independently:

  1. Case and Model: the analytic mapping, the solver's own energy density read
     off its kernels, and the continuum Euler-Lagrange force sympy derives from
     it.  Nothing here knows about the solver.
  2. The mapping as a VMEC++ state: the mode ordering and the projection of the
     continuum force onto the solver's normalized basis.  The basis conventions
     themselves live in C++ (FourierGeometry::InitFromState).
  3. Free boundary: the analytic vacuum field, the mgrid that holds it, and
     what the last radial node needs beyond the volume source.
  4. The studies, one per table, and main.

Requires sympy and, for --study fit, scipy.  Neither is a dependency of the
solver.

The defaults are a short run at the coarsest resolutions that still separate
first order from second.  The tables in the pull request come from the same
script with

    --mpol 5 --ntor 3 --ntheta 24 --nzeta 24 --ns 25 51 101 201
    --projection 96 --angular 16 18 20 22 24 26 --angular-reference 96

Usage:
    python manufactured_solution.py [--study STUDY] [--ncurr 1] [--lasym]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import sympy as sp

import vmecpp
from vmecpp.cpp import _vmecpp

# ----------------------------------------------------------------------------
# The analytic mapping and its continuum ideal-MHD force
# ----------------------------------------------------------------------------

MU_0 = 4.0e-7 * np.pi

S, U, W = sp.symbols("s u w", real=True)

# The twelve pointwise arguments of the energy density.
Q_NAMES = ["R", "Rs", "Ru", "Rv", "Z", "Zs", "Zu", "Zv", "L", "Ls", "Lu", "Lv"]

# Which analytic field each local argument is.
Q_FIELD = {
    "R": ("R", ""),
    "Rs": ("R", "s"),
    "Ru": ("R", "u"),
    "Rv": ("R", "v"),
    "Z": ("Z", ""),
    "Zs": ("Z", "s"),
    "Zu": ("Z", "u"),
    "Zv": ("Z", "v"),
    "L": ("L", ""),
    "Ls": ("L", "s"),
    "Lu": ("L", "u"),
    "Lv": ("L", "v"),
}

DERIV_KEYS = ["", "s", "u", "v", "ss", "su", "sv", "uu", "uv", "vv"]


def _compose(a, b):
    """Sort-merge two derivative suffixes, e.g. ('s', 'u') -> 'su'."""
    return "".join(sorted(a + b, key="suv".index))


class Case:
    """A manufactured mapping together with the radial profiles it lives on."""

    def __init__(
        self,
        nfp=3,
        r00=(1.0, 0.06),
        rhel=(-0.10, 0.02),
        zhel=(0.10, -0.02),
        amaj=(0.30, 0.05),
        aell=(0.08, -0.012),
        lam=None,
        iota=(0.42, 0.13),
        am=(1.0, -1.0),
        pres_scale=180.0,
        phiedge=-0.035,
        sign_jacobian=-1,
        exact_axis=False,
        m2=None,
        asym=None,
    ):
        # exact_axis: make every odd-m amplitude exactly proportional to
        # sqrt(s) and the m = 0 lambda constant in s, so that the solver's
        # constant extrapolation of the odd-parity arrays onto the magnetic
        # axis is exact rather than first-order accurate.
        self.nfp = nfp
        self.iota_coeff = tuple(iota)
        self.am = tuple(am)
        self.pres_scale = pres_scale
        self.phiedge = phiedge
        self.sign_jacobian = sign_jacobian

        rho = sp.sqrt(S)

        def poly(c):
            return sum(ci * S**i for i, ci in enumerate(c))

        # Rotating ellipse: M(w) = Rot(w/2) diag(a, b) Rot(-w/2), symmetric, so
        # (a + b)/2 = amaj(s) sits at n = 0 and (a - b)/2 = aell(s) at n = 1.
        if exact_axis:
            amaj, aell = (amaj[0],), (aell[0],)
        amp = poly(amaj)
        ell = poly(aell)

        self.RCC = {
            (0, 0): poly(r00),
            (0, 1): poly(rhel),
            (1, 0): rho * amp,
            (1, 1): rho * ell,
        }
        self.RSS = {(1, 1): rho * ell}
        self.ZSC = {(1, 0): rho * amp, (1, 1): -rho * ell}
        self.ZCS = {(0, 1): poly(zhel), (1, 1): rho * ell}

        # m = 2 shaping, exactly proportional to s.  rzConIntoVolume sets the
        # spectral-condensation baseline to the boundary value times s, and
        # xmpq vanishes for m = 0 and m = 1, so a linear-in-s m = 2 amplitude
        # leaves the constraint force identically zero at the mapping while
        # giving the equilibrium the shaping it needs.
        if m2 is not None:
            for (kind, n), v in m2.items():
                d = {
                    "rcc": self.RCC,
                    "rss": self.RSS,
                    "zsc": self.ZSC,
                    "zcs": self.ZCS,
                }[kind]
                d[(2, n)] = v * S

        # Non-stellarator-symmetric content.  R gains a sine series and Z and
        # lambda a cosine one; the radial factor follows the same rule as the
        # symmetric part, sqrt(s) for odd m and s for m = 2, so the axis
        # extrapolation stays exact and the constraint force stays zero.
        self.RSC, self.RCS = {}, {}
        self.ZCC, self.ZSS = {}, {}
        self.LCC, self.LSS = {}, {}
        self.lasym = bool(asym)
        for (kind, m, n), coeff in (asym or {}).items():
            c = coeff if isinstance(coeff, tuple) else (coeff,)
            factor = rho if m % 2 else (S if m == 2 else sp.Integer(1))
            expr = poly(c) * factor
            if kind.startswith("l"):
                expr = expr * self.phip
            {
                "rsc": self.RSC,
                "rcs": self.RCS,
                "zcc": self.ZCC,
                "zss": self.ZSS,
                "lcc": self.LCC,
                "lss": self.LSS,
            }[kind][(m, n)] = expr
        # the second m = 1 gauge condition, RSC_{1n} = ZCC_{1n}
        for (m, n), expr in list(self.RSC.items()):
            if m == 1:
                self.ZCC[(1, n)] = expr

        if lam is None:
            lam = {
                ("sc", 1, 0): (0.11, -0.04),
                ("sc", 1, 1): (0.05, 0.02),
                ("cs", 1, 1): (-0.06, 0.015),
                ("cs", 0, 1): (0.09, 0.03),
            }
        # L = phip * lambda; the m = 1 parts carry rho, the m = 0 part carries s
        # because VMEC holds lambda(axis) = 0.
        self.LSC, self.LCS = {}, {}
        for (kind, m, n), coeff in lam.items():
            c = (coeff[0],) if exact_axis else coeff
            expr = poly(c) * (rho if m == 1 else (1 if exact_axis else S))
            (self.LSC if kind == "sc" else self.LCS)[(m, n)] = self.phip * expr

    # ------------------------------------------------------------- profiles
    @property
    def phip(self):
        """The toroidal flux derivative dPhi/ds, constant for the default aphi."""
        return self.sign_jacobian * self.phiedge / (2.0 * sp.pi)

    @property
    def lamscale(self):
        """The lambda scaling sqrt(rmsPhiP * deltaS), which is |phip| here."""
        return abs(float(self.phip))

    def iota(self, x):
        return sum(c * x**i for i, c in enumerate(self.iota_coeff))

    def chip(self, x):
        return self.phip * self.iota(x)

    def pressure(self, x):
        # RadialProfiles::evalMassProfile scales by MU_0 * pres_scale
        return MU_0 * self.pres_scale * sum(c * x**i for i, c in enumerate(self.am))

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _series(cc, ss, sc, cs):
        out = sp.Integer(0)
        for (m, n), e in cc.items():
            out += e * sp.cos(m * U) * sp.cos(n * W)
        for (m, n), e in ss.items():
            out += e * sp.sin(m * U) * sp.sin(n * W)
        for (m, n), e in sc.items():
            out += e * sp.sin(m * U) * sp.cos(n * W)
        for (m, n), e in cs.items():
            out += e * sp.cos(m * U) * sp.sin(n * W)
        return sp.expand(out)

    def expressions(self):
        R = self._series(self.RCC, self.RSS, self.RSC, self.RCS)
        Z = self._series(self.ZCC, self.ZSS, self.ZSC, self.ZCS)
        L = self._series(self.LCC, self.LSS, self.LSC, self.LCS)
        return R, Z, L

    def mode_table(self):
        """(kind, m, n) -> unweighted coefficient expression in s."""
        table = {}
        for kind, d in (
            ("rcc", self.RCC),
            ("rss", self.RSS),
            ("rsc", self.RSC),
            ("rcs", self.RCS),
            ("zsc", self.ZSC),
            ("zcs", self.ZCS),
            ("zcc", self.ZCC),
            ("zss", self.ZSS),
            ("lsc", self.LSC),
            ("lcs", self.LCS),
            ("lcc", self.LCC),
            ("lss", self.LSS),
        ):
            for mn, e in d.items():
                table[(kind, *mn)] = e
        return table


def energy_density():
    """Local symbols, profile symbols and the energy density built from them."""
    q = {name: sp.Symbol(name, real=True) for name in Q_NAMES}
    phip, chip, pres = sp.symbols("phip chip pres", real=True)
    tau = q["Ru"] * q["Zs"] - q["Rs"] * q["Zu"]
    gsq = q["R"] * tau
    guu = q["Ru"] ** 2 + q["Zu"] ** 2
    guv = q["Ru"] * q["Rv"] + q["Zu"] * q["Zv"]
    gvv = q["R"] ** 2 + q["Rv"] ** 2 + q["Zv"] ** 2
    # the solver's real-space lv carries a minus sign (dft_toroidal.cc), so
    # B^u = (chip - dL/dphi) / sqrt(g)
    bsupu = (chip - q["Lv"]) / gsq
    bsupv = (phip + q["Lu"]) / gsq
    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + gvv * bsupv
    pmag = sp.Rational(1, 2) * (bsupu * bsubu + bsupv * bsubv)
    dens = -gsq * (pmag - pres)
    aux = {
        "tau": tau,
        "gsq": gsq,
        "guu": guu,
        "guv": guv,
        "gvv": gvv,
        "bsupu": bsupu,
        "bsupv": bsupv,
        "pmag": pmag,
    }
    return q, (phip, chip, pres), dens, aux


_DENSITY_OPS: dict[str, object] = {}


def _density_ops():
    """Lambdified derivatives of the energy density; independent of the mapping."""
    if _DENSITY_OPS:
        return _DENSITY_OPS
    q, (ph, ch, pr), dens, aux = energy_density()
    args = [q[n] for n in Q_NAMES] + [ph, ch, pr]
    grad = {}
    for n in Q_NAMES:
        e = sp.diff(dens, q[n])
        grad[n] = None if e == 0 else sp.lambdify(args, e, "numpy")
    hess = {}
    for a in Q_NAMES:
        for b in Q_NAMES:
            e = sp.diff(dens, q[a], q[b])
            hess[(a, b)] = None if e == 0 else sp.lambdify(args, e, "numpy")
    grad_prof = {}
    for n in Q_NAMES:
        for k, p in enumerate((ph, ch, pr)):
            e = sp.diff(dens, q[n], p)
            grad_prof[(n, k)] = None if e == 0 else sp.lambdify(args, e, "numpy")
    _DENSITY_OPS.update(
        dens=sp.lambdify(args, dens, "numpy"),
        grad=grad,
        hess=hess,
        grad_prof=grad_prof,
        aux={k: sp.lambdify(args, v, "numpy") for k, v in aux.items()},
    )
    return _DENSITY_OPS


class Model:
    """Numerical evaluation of the mapping, its energy and its continuum force."""

    def __init__(self, case: Case):
        self.case = case
        nfp = case.nfp
        R, Z, L = case.expressions()

        def deriv(expr, key):
            e = expr
            for ch in key:
                if ch == "s":
                    e = sp.diff(e, S)
                elif ch == "u":
                    e = sp.diff(e, U)
                else:
                    e = nfp * sp.diff(e, W)
            return sp.expand(e)

        self.fn = {}
        for name, expr in (("R", R), ("Z", Z), ("L", L)):
            for key in DERIV_KEYS:
                self.fn[name + "|" + key] = sp.lambdify(
                    (S, U, W), deriv(expr, key), "numpy"
                )

        ops = _density_ops()
        self.dens = ops["dens"]
        self.grad = ops["grad"]
        self.hess = ops["hess"]
        self.grad_prof = ops["grad_prof"]
        self.aux = ops["aux"]
        x = sp.Symbol("x")
        self._dchip = sp.lambdify(x, sp.diff(case.chip(x), x), "numpy")
        self._dpres = sp.lambdify(x, sp.diff(case.pressure(x), x), "numpy")

    # ---------------------------------------------------------------- fields
    def _ev(self, name, key, s, u, w):
        return self.fn[name + "|" + key](s, u, w) * np.ones_like(u * w)

    def q_at(self, s, u, w):
        return {n: self._ev(*Q_FIELD[n], s, u, w) for n in Q_NAMES}

    def dq_at(self, s, u, w, wrt):
        out = {}
        for n in Q_NAMES:
            field, key = Q_FIELD[n]
            out[n] = self._ev(field, _compose(key, wrt), s, u, w)
        return out

    def profiles(self, s):
        c = self.case
        return (float(c.phip), float(c.chip(s)), float(c.pressure(s)))

    def dprofiles_ds(self, s):
        return (0.0, float(self._dchip(s)), float(self._dpres(s)))

    # ---------------------------------------------------------------- energy
    def density(self, s, u, w):
        q = self.q_at(s, u, w)
        return self.dens(*[q[n] for n in Q_NAMES], *self.profiles(s))

    def aux_at(self, name, s, u, w):
        q = self.q_at(s, u, w)
        return self.aux[name](*[q[n] for n in Q_NAMES], *self.profiles(s))

    # ----------------------------------------------------------------- force
    def force_density(self, s, u, w):
        """(f_R, f_Z, f_L) = -delta W / delta (field), densities in (s, u, phi)."""
        q = self.q_at(s, u, w)
        args = [q[n] for n in Q_NAMES] + list(self.profiles(s))
        dq = {d: self.dq_at(s, u, w, d) for d in ("s", "u", "v")}
        dprof = self.dprofiles_ds(s)
        one = np.ones_like(u * w)

        def d_total(name, wrt):
            acc = np.zeros_like(one)
            for m in Q_NAMES:
                fn = self.hess[(name, m)]
                if fn is None:
                    continue
                acc = acc + fn(*args) * dq[wrt][m]
            if wrt == "s":
                for k in range(3):
                    fn = self.grad_prof[(name, k)]
                    if fn is None or dprof[k] == 0.0:
                        continue
                    acc = acc + fn(*args) * dprof[k]
            return acc

        out = {}
        for field in ("R", "Z", "L"):
            gf = self.grad[field]
            el = (gf(*args) * one) if gf is not None else np.zeros_like(one)
            for wrt in ("s", "u", "v"):
                el = el - d_total(field + wrt, wrt)
            out[field] = -el
        return out


# ----------------------------------------------------------------------------
# The mapping as a VMEC++ state, and the continuum force as a source
# ----------------------------------------------------------------------------

SQRT2 = np.sqrt(2.0)

# The parities the solver carries, in FourierForces order.
SYM_SPANS = ["rcc", "rss", "zsc", "zcs", "lsc", "lcs"]
ASYM_SPANS = [
    "rcc",
    "rss",
    "rsc",
    "rcs",
    "zsc",
    "zcs",
    "zcc",
    "zss",
    "lsc",
    "lcs",
    "lcc",
    "lss",
]

# Which field each parity belongs to, and which trigonometric product it is.
BASIS = {
    "rcc": ("R", "cc"),
    "rss": ("R", "ss"),
    "rsc": ("R", "sc"),
    "rcs": ("R", "cs"),
    "zsc": ("Z", "sc"),
    "zcs": ("Z", "cs"),
    "zcc": ("Z", "cc"),
    "zss": ("Z", "ss"),
    "lsc": ("L", "sc"),
    "lcs": ("L", "cs"),
    "lcc": ("L", "cc"),
    "lss": ("L", "ss"),
}

# The two m = 1 gauge pairs: the solver stores the sum and the difference of
# each, and freezes the difference.
M1_PAIRS = [("rss", "zcs"), ("rsc", "zcc")]


def spans(lasym):
    return ASYM_SPANS if lasym else SYM_SPANS


def scale(m, n):
    """The mscale * nscale weight of the solver's normalized basis."""
    return (1.0 if m == 0 else SQRT2) * (1.0 if n == 0 else SQRT2)


def empty(ns, mpol, ntor, lasym=False):
    return {k: np.zeros((ns, mpol, ntor + 1)) for k in spans(lasym)}


def flatten(d, lasym=False):
    return np.concatenate([d[k].reshape(-1) for k in spans(lasym)])


def unflatten(flat, ns, mpol, ntor, lasym=False):
    n = ns * mpol * (ntor + 1)
    return {
        k: flat[i * n : (i + 1) * n].reshape(ns, mpol, ntor + 1)
        for i, k in enumerate(spans(lasym))
    }


# ------------------------------------------------- the mapping as a state


def mode_order(mpol, ntor):
    """The standard VMEC mode ordering: m = 0 carries n >= 0 only."""
    out = [(0, n) for n in range(ntor + 1)]
    for m in range(1, mpol):
        out += [(m, n) for n in range(-ntor, ntor + 1)]
    return out


def product_coefficients(case, sgrid, mpol, ntor):
    """The mapping's unweighted product-basis coefficients on a radial grid."""
    out = {k: np.zeros((mpol, ntor + 1, len(sgrid))) for k in ASYM_SPANS}
    for (kind, m, n), expr in case.mode_table().items():
        if m >= mpol or n > ntor:
            msg = f"mode ({kind},{m},{n}) is outside mpol={mpol}, ntor={ntor}"
            raise ValueError(msg)
        f = sp.lambdify(S, expr, "numpy")
        vals = np.array([float(f(s)) for s in sgrid], dtype=float)
        if BASIS[kind][0] == "L":
            vals = vals / float(case.phip)  # L = phip * lambda
        out[kind][m, n] = vals
    return out


def combined_coefficients(case, sgrid, mpol, ntor):
    """The mapping in the basis the wout file uses: R = sum rmnc cos(m u - n v)
    [+ rmns sin(...)], Z and lambda likewise, each [mnmax, ns].

    Only the trigonometric identity is done here; the solver's own basis
    normalization, m = 1 gauge rotation and lambda scaling are applied by
    VmecModel.set_state_from_fourier.
    """
    p = product_coefficients(case, sgrid, mpol, ntor)
    modes = mode_order(mpol, ntor)
    out = {
        k: np.zeros((len(modes), len(sgrid)))
        for k in ("rmnc", "zmns", "lmns", "rmns", "zmnc", "lmnc")
    }
    for i, (m, n) in enumerate(modes):
        q = abs(n)
        if m == 0:
            # one mode per n, not a +-n pair
            out["rmnc"][i] = p["rcc"][0, q]
            out["zmns"][i] = -p["zcs"][0, q]
            out["lmns"][i] = -p["lcs"][0, q]
            out["rmns"][i] = -p["rcs"][0, q]
            out["zmnc"][i] = p["zcc"][0, q]
            out["lmnc"][i] = p["lcc"][0, q]
            continue
        if n == 0:
            out["rmnc"][i] = p["rcc"][m, 0]
            out["zmns"][i] = p["zsc"][m, 0]
            out["lmns"][i] = p["lsc"][m, 0]
            out["rmns"][i] = p["rsc"][m, 0]
            out["zmnc"][i] = p["zcc"][m, 0]
            out["lmnc"][i] = p["lcc"][m, 0]
            continue
        sgn = 1.0 if n > 0 else -1.0
        out["rmnc"][i] = 0.5 * (p["rcc"][m, q] + sgn * p["rss"][m, q])
        out["zmns"][i] = 0.5 * (p["zsc"][m, q] - sgn * p["zcs"][m, q])
        out["lmns"][i] = 0.5 * (p["lsc"][m, q] - sgn * p["lcs"][m, q])
        out["rmns"][i] = 0.5 * (p["rsc"][m, q] - sgn * p["rcs"][m, q])
        out["zmnc"][i] = 0.5 * (p["zcc"][m, q] + sgn * p["zss"][m, q])
        out["lmnc"][i] = 0.5 * (p["lcc"][m, q] + sgn * p["lss"][m, q])
    return out


def install_state(model, case, ns, mpol, ntor, lasym=False):
    """Put the mapping into the solver's state and return it as it went in."""
    sgrid = np.linspace(0.0, 1.0, ns)
    c = combined_coefficients(case, sgrid, mpol, ntor)
    if lasym:
        model.set_state_from_fourier(
            c["rmnc"], c["zmns"], c["lmns"], c["rmns"], c["zmnc"], c["lmnc"]
        )
    else:
        model.set_state_from_fourier(c["rmnc"], c["zmns"], c["lmns"])
    return c


# --------------------------------------------------------------------- force


def _parity_projections(dens, mpol, ntor):
    """Exact trigonometric projections of a periodic field via one FFT.

    Returns cc/ss/sc/cs, entry [m, n] being the mean of dens * cos(m u) cos(n w) and so
    on over the (u, w) torus.
    """
    nu, nw = dens.shape
    a = np.fft.fft2(dens) / (nu * nw)
    m = np.arange(mpol)
    n = np.arange(ntor + 1)
    c = a[np.ix_(m, n)]  # mean of f exp(-i(mu + nw))
    d = a[np.ix_(m, (-n) % nw)]  # mean of f exp(-i(mu - nw))
    return {
        "cc": 0.5 * (c.real + d.real),
        "ss": 0.5 * (d.real - c.real),
        "sc": 0.5 * (-c.imag - d.imag),
        "cs": 0.5 * (-c.imag + d.imag),
    }


def project_force(
    model,
    case,
    sgrid,
    mpol,
    ntor,
    nu=96,
    nw=96,
    axis_eps=1.0e-9,
    endpoint_half=True,
    lasym=False,
):
    """The continuum force projected onto the solver's normalized basis.

    F[m, n, j] = mscale nscale / (4 pi^2) int du dw f(s_j, u, w) basis(u, w),
    with the lambda force taken with respect to the stored coefficients, so
    multiplied by lamscale.

    The force density is finite at the magnetic axis but is written in terms of
    sqrt(s), so s = 0 is evaluated at `axis_eps`.  The first and last radial
    nodes carry half-size control volumes in the discrete energy, so with
    `endpoint_half` their entries are halved, which is what the solver's own
    force at those nodes converges to.
    """
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    w = np.linspace(0.0, 2.0 * np.pi, nw, endpoint=False)
    UU, WW = np.meshgrid(u, w, indexing="ij")
    ns = len(sgrid)
    out = empty(ns, mpol, ntor, lasym)
    sc_mn = np.array([[scale(m, n) for n in range(ntor + 1)] for m in range(mpol)])
    for j, s in enumerate(sgrid):
        f = model.force_density(max(s, axis_eps), UU, WW)
        proj = {}
        for field in ("R", "Z", "L"):
            dens = f[field] * (case.lamscale if field == "L" else 1.0)
            proj[field] = _parity_projections(dens, mpol, ntor)
        for kind in spans(lasym):
            field, parity = BASIS[kind]
            out[kind][j] = sc_mn * proj[field][parity]
    if endpoint_half and ns > 1:
        for kind in spans(lasym):
            out[kind][0] *= 0.5
            out[kind][ns - 1] *= 0.5
    return out


def hat_force_to_decomposed(hat, mpol, zero_z_m1=True):
    """What the solver applies to the force after decomposeInto: the m = 1 gauge
    rotation, and the freezing of its difference component."""
    dec = {k: v.copy() for k, v in hat.items()}
    if mpol > 1:
        for r_key, z_key in M1_PAIRS:
            if r_key not in dec:
                continue
            r1 = hat[r_key][:, 1, :]
            z1 = hat[z_key][:, 1, :]
            dec[r_key][:, 1, :] = (r1 + z1) / SQRT2
            dec[z_key][:, 1, :] = 0.0 if zero_z_m1 else (r1 - z1) / SQRT2
    return dec


# ----------------------------------------------------------------------------
# Free boundary
# ----------------------------------------------------------------------------

# A vacuum field whose cylindrical components are bilinear in (R, Z),
#
#     B_R = a Z,    B_phi = b + c R,    B_Z = 0,
#
# is reproduced exactly by MGridProvider::interpolate, which is bilinear in
# (R, Z) at a fixed toroidal plane.  free_boundary_method = "only_coils" takes
# the vacuum field straight from the mgrid with no Laplace solve, so the vacuum
# magnetic pressure the solver works with is the analytic |B|^2 / 2 to machine
# precision and the free-boundary problem is as manufacturable as the fixed one.
# only_coils requires zero pressure and zero net input current.


def vacuum_pressure(vac, r, z):
    """The magnetic pressure |B|^2 / 2 of the bilinear vacuum field."""
    a, b, c = vac
    return 0.5 * ((a * z) ** 2 + (b + c * r) ** 2)


def _boundary_fields(model, nu, nw):
    """The mapping and its first derivatives on the boundary surface."""
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    w = np.linspace(0.0, 2.0 * np.pi, nw, endpoint=False)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    one = np.ones_like(uu)
    keys = ("R|", "R|u", "R|v", "Z|", "Z|u", "Z|v")
    return uu, ww, {k: model.fn[k](1.0, uu, ww) * one for k in keys}


def tune_vacuum_field(model, case, c=0.0, nu=64, nw=64):
    """Choose a vacuum field the solver will run this plasma against.

    Two conditions are enforced before the first vacuum iteration is accepted:
    the vacuum-side R B_phi must carry the sign of the plasma's, and the
    vacuum-side poloidal field must reproduce the plasma's net toroidal current
    to within one per cent of R B_phi.  Both surface averages are linear in the
    field's coefficients, so both are solved rather than searched: a carries the
    current and b matches R B_phi.  c is free shaping and defaults to zero.
    """
    uu, ww, f = _boundary_fields(model, nu, nw)
    one = np.ones_like(uu)
    g = {
        k: model.aux_at(k, 1.0, uu, ww) * one
        for k in ("guu", "guv", "gvv", "bsupu", "bsupv")
    }
    sgn = case.sign_jacobian
    b_sub_u = g["guu"] * g["bsupu"] + g["guv"] * g["bsupv"]
    b_sub_v = g["guv"] * g["bsupu"] + g["gvv"] * g["bsupv"]
    ctor = 2.0 * np.pi * sgn * float(np.mean(b_sub_u))
    rbtor = float(np.mean(b_sub_v))
    # B_vac . e_u = a Z R_u, and the solver scales that average by sign * 2 pi
    a = ctor / (2.0 * np.pi * sgn * float(np.mean(f["Z|"] * f["R|u"])))
    # B_vac . e_v = a Z R_v + b R + c R^2
    b = (
        rbtor
        - a * float(np.mean(f["Z|"] * f["R|v"]))
        - c * float(np.mean(f["R|"] ** 2))
    ) / float(np.mean(f["R|"]))
    return (a, b, c), {"ctor": ctor, "rbtor": rbtor}


def boundary_box(model, margin=0.3, nu=64, nw=64):
    """A cylindrical grid box around the plasma, with room for it to move."""
    _, _, f = _boundary_fields(model, nu, nw)
    r, z = f["R|"], f["Z|"]
    dr = margin * (r.max() - r.min())
    dz = margin * max(z.max() - z.min(), r.max() - r.min())
    return (r.min() - dr, r.max() + dr, z.min() - dz, z.max() + dz)


def write_mgrid(path, vac, nfp, nzeta, box, nr=48, nz=48):
    """Write the bilinear field as an mgrid file the solver can load."""
    import netCDF4  # noqa: PLC0415

    rmin, rmax, zmin, zmax = box
    rr, zz = np.meshgrid(
        np.linspace(rmin, rmax, nr), np.linspace(zmin, zmax, nz), indexing="xy"
    )
    a, b, c = vac
    fields = {
        "br_001": np.broadcast_to(a * zz, (nzeta, nz, nr)).copy(),
        "bp_001": np.broadcast_to(b + c * rr, (nzeta, nz, nr)).copy(),
        "bz_001": np.zeros((nzeta, nz, nr)),
    }
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 30)
        ds.createDimension("external_coil_groups", 1)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 1)
        ds.createDimension("rad", nr)
        ds.createDimension("zee", nz)
        ds.createDimension("phi", nzeta)
        for name, value in (
            ("ir", nr),
            ("jz", nz),
            ("kp", nzeta),
            ("nfp", nfp),
            ("nextcur", 1),
        ):
            ds.createVariable(name, "i4")[...] = value
        for name, value in (
            ("rmin", rmin),
            ("rmax", rmax),
            ("zmin", zmin),
            ("zmax", zmax),
        ):
            ds.createVariable(name, "f8")[...] = value
        ds.createVariable("mgrid_mode", "S1", ("dim_00001",))[:] = np.array(
            ["R"], dtype="S1"
        )
        ds.createVariable("coil_group", "S1", ("external_coil_groups", "stringsize"))[
            :
        ] = np.array([list("manufactured".ljust(30))], dtype="S1")
        ds.createVariable("raw_coil_cur", "f8", ("external_coils",))[:] = np.array(
            [1.0]
        )
        for tag, data in fields.items():
            ds.createVariable(tag, "f8", ("phi", "zee", "rad"))[:] = data
    return path


def lcfs_source(model, vac, mpol, ntor, ns, nu=96, nw=96):
    """What the source at the last radial node needs beyond the volume density.

    Only the half-grid cell inside the boundary exists there, so the kernel's
    radial difference is one-sided and the local terms are averaged over one
    cell rather than two.  Writing w for the energy density, the node's force is

        (1 / ds) dw/dx_s + (1 / 2) EL(w) + edge

    to first order in ds, against EL(w) at an interior node, and the free-
    boundary edge term of assembleTotalForces is z_u R p_vac / ds in R and
    -r_u R p_vac / ds in Z.  project_force already halves the endpoint, so what
    is missing is the first and the third, and their sum is the pressure jump

        (1 / ds) R (p_total - p_vac) (z_u, -r_u)

    which is what a free-boundary equilibrium drives to zero.
    """
    uu, ww, f = _boundary_fields(model, nu, nw)
    q = model.q_at(1.0, uu, ww)
    args = [q[n] for n in Q_NAMES] + list(model.profiles(1.0))
    one = np.ones_like(uu)
    p_vac = vacuum_pressure(vac, f["R|"], f["Z|"])
    ds = 1.0 / (ns - 1.0)
    dens = {
        "R": -(model.grad["Rs"](*args) * one + f["Z|u"] * f["R|"] * p_vac) / ds,
        "Z": -(model.grad["Zs"](*args) * one - f["R|u"] * f["R|"] * p_vac) / ds,
    }
    out = empty(ns, mpol, ntor)
    sc_mn = np.array([[scale(m, n) for n in range(ntor + 1)] for m in range(mpol)])
    proj = {field: _parity_projections(d, mpol, ntor) for field, d in dens.items()}
    for kind in spans(False):
        field, parity = BASIS[kind]
        if field in proj:
            out[kind][ns - 1] = sc_mn * proj[field][parity]
    return out


# ----------------------------------------------------------------------------
# The mapping the studies use
# ----------------------------------------------------------------------------

# A mapping that is merely admissible is not enough to drive the solver with.
# VMEC's preconditioner models the Hessian of an equilibrium-like state, so the
# source-augmented problem is only well conditioned when the mapping is close to
# force balance to begin with.  These coefficients were obtained by minimizing
# the mapping's own continuum force over its free parameters (`fit_parameters`
# below, reproduced by --study fit); they leave it within FITTED_FSQ of force
# balance.  The fit only conditions the problem: whatever mapping it lands on,
# the source is that mapping's exact continuum force, so the mapping is still
# the exact solution of the modified problem.
#
# The result is an aspect ratio of 18 at beta(0) = 2.0 per cent.
FITTED_P = [
    -0.0003316553290643698,
    -8.064775221197626e-06,
    0.0027074846406597137,
    6.8764080081576135e-06,
    -0.0002719683084603821,
    -0.002402448448679986,
    -0.0001722358502662977,
    0.0004495768921404468,
    -0.05185022109871347,
    -0.02929268739381755,
    0.013742877877949287,
    -1.8309477685077984e-07,
]
FITTED_FSQ = 2.1e-07

FITTED_BASE = {
    "nfp": 3,
    "r00": (1.0, 0.0, 0.0),
    "rhel": (-0.005,),
    "zhel": (0.005,),
    "amaj": (0.05,),
    "aell": (0.005,),
    "iota": (0.42, 0.13),
    "am": (1.0, -1.0),
    "pres_scale": 160000.0,
    "exact_axis": True,
}


# The non-stellarator-symmetric content used by --lasym: an asymmetric shift of
# the axis and a tilt of the ellipse.  Only m = 0 and m = 1 appear, so xmpq
# vanishes on it and the constraint force stays zero, and the m = 1 pair obeys
# RSC_{1n} = ZCC_{1n}, which Case enforces.
ASYM = {
    ("rcs", 0, 1): 0.004,
    ("zcc", 0, 1): 0.003,
    ("rsc", 1, 0): 0.006,
    ("rsc", 1, 1): 0.002,
}


# The free-boundary mapping.  only_coils requires zero pressure and zero net
# input current, and the free-boundary path decays the spectral-condensation
# baseline by 0.9 an iteration once the vacuum is on, so the mapping is
# restricted to m <= 1, where the constraint force is identically zero however
# far that baseline decays.  These coefficients are the same fit as FITTED_P,
# run against the zero-pressure density (--study fit --freeb).
FREEB_P = [
    4.503194764934e-04,
    -2.960561972701e-07,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -3.177003067965e-02,
    -5.102561557074e-03,
    8.564742390844e-03,
    -1.311420568630e-06,
]


def freeb_base():
    """FITTED_BASE at the zero pressure only_coils requires."""
    base = dict(FITTED_BASE)
    base["pres_scale"] = 0.0
    base["am"] = (0.0,)
    return base


def build_case(p, base=None, asym=None, m2=True):
    """Assemble a Case from the flat parameter vector used by the fit.

    With m2 false the mapping carries no m = 2 shaping at all, which the free-
    boundary path needs: it decays the spectral-condensation baseline by 0.9 an
    iteration, and only a mapping whose constraint force is identically zero,
    rather than merely equal to the baseline, survives that.
    """
    base = FITTED_BASE if base is None else base
    kw = dict(base)
    kw["r00"] = (base["r00"][0], p[0], p[1])
    kw["m2"] = (
        {
            ("rcc", 0): p[2],
            ("rcc", 1): p[3],
            ("rss", 1): p[4],
            ("zsc", 0): p[5],
            ("zsc", 1): p[6],
            ("zcs", 1): p[7],
        }
        if m2
        else None
    )
    # No m = 0 lambda.  FourierGeometry::extrapolateTowardsAxis copies that
    # component from the first interior surface onto the magnetic axis, and the
    # force transform does not transpose the copy, so the solver fixes it by an
    # implicit axis condition that an analytic lambda cannot be made to satisfy.
    kw["lam"] = {
        ("sc", 1, 0): (p[8],),
        ("sc", 1, 1): (p[9],),
        ("cs", 1, 1): (p[10],),
        ("sc", 2, 1): (p[11],),
    }
    kw["asym"] = asym
    return Case(**kw)


def fit_parameters(
    base=None, mpol=5, ntor=3, nsfit=8, nu=48, nw=48, maxiter=40, m2=True
):
    """Minimize the continuum force over the mapping's free parameters."""
    from scipy.optimize import least_squares  # noqa: PLC0415

    base = FITTED_BASE if base is None else base
    x, _ = np.polynomial.legendre.leggauss(nsfit)
    rho = 0.5 * (x + 1.0)
    srho = rho * rho
    p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0])

    case0 = build_case(p0, base, m2=m2)
    fa0 = project_force(
        Model(case0), case0, srho, mpol, ntor, nu=nu, nw=nw, endpoint_half=False
    )
    scale = max(float(np.max(np.abs(fa0[k]))) for k in spans(False))

    def residual(p):
        case = build_case(p, base, m2=m2)
        fa = project_force(
            Model(case), case, srho, mpol, ntor, nu=nu, nw=nw, endpoint_half=False
        )
        return np.concatenate([fa[k].reshape(-1) for k in spans(False)]) / scale

    sol = least_squares(
        residual,
        p0,
        method="lm",
        max_nfev=maxiter * (len(p0) + 1),
        xtol=1e-12,
        ftol=1e-12,
    )
    return sol.x


# ----------------------------------------------------------------------------
# Driving the solver
# ----------------------------------------------------------------------------


def _indata(
    case,
    ns,
    mpol,
    ntor,
    ntheta,
    nzeta,
    ftol=1e-16,
    niter=40000,
    delt=0.9,
    lasym=False,
    mgrid=None,
    tcon0=1.0,
    enable_force_source=True,
    current=None,
):
    """A VMEC++ input carrying the mapping's boundary, axis and profiles.

    The boundary is the mapping at s = 1 and the axis is the mapping at s = 0, both
    taken from the same combined-basis conversion the state uses.
    """
    edge = combined_coefficients(case, np.array([1.0]), mpol, ntor)
    axis = combined_coefficients(case, np.array([0.0]), mpol, ntor)
    modes = mode_order(mpol, ntor)

    def entries(key):
        return [
            {"n": int(n), "m": int(m), "value": float(edge[key][i, 0])}
            for i, (m, n) in enumerate(modes)
            if edge[key][i, 0] != 0.0
        ]

    def axis_array(key):
        out = np.zeros(ntor + 1)
        for i, (m, n) in enumerate(modes):
            if m == 0:
                out[n] = axis[key][i, 0]
        return out.tolist()

    d = {
        "lasym": lasym,
        "nfp": case.nfp,
        "mpol": mpol,
        "ntor": ntor,
        "ntheta": ntheta,
        "nzeta": nzeta,
        "ns_array": [ns],
        "ftol_array": [ftol],
        "niter_array": [niter],
        "delt": delt,
        "tcon0": tcon0,
        "aphi": [1.0],
        "phiedge": case.phiedge,
        "nstep": 200,
        "pmass_type": "power_series",
        "am": list(case.am),
        "pres_scale": case.pres_scale,
        "gamma": 0.0,
        "spres_ped": 1.0,
        "ncurr": 0,
        "piota_type": "power_series",
        "ai": list(case.iota_coeff),
        "pcurr_type": "power_series",
        "ac": [0.0],
        "curtor": 0.0,
        "ac_aux_s": [],
        "ac_aux_f": [],
        "bloat": 1.0,
        "lfreeb": False,
        "mgrid_file": "NONE",
        "nvacskip": 1,
        "lforbal": False,
        # the studies install a source, which a run has to ask for
        "enable_force_source": enable_force_source,
        "raxis_c": axis_array("rmnc"),
        "zaxis_s": axis_array("zmns"),
        "rbc": entries("rmnc"),
        "zbs": entries("zmns"),
    }
    if lasym:
        d["raxis_s"] = axis_array("rmns")
        d["zaxis_c"] = axis_array("zmnc")
        d["rbs"] = entries("rmns")
        d["zbc"] = entries("zmnc")
    if current is not None:
        knots, values, curtor = current
        d["ncurr"] = 1
        d["pcurr_type"] = "cubic_spline_i"
        d["ac_aux_s"] = [float(x) for x in knots]
        d["ac_aux_f"] = [float(x) for x in values]
        d["curtor"] = float(curtor)
    if mgrid is not None:
        d["lfreeb"] = True
        d["mgrid_file"] = str(mgrid)
        d["extcur"] = [1.0]
        d["free_boundary_method"] = "only_coils"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(d, fh)
        path = fh.name
    try:
        return vmecpp.VmecInput.from_file(path)._to_cpp_vmecindata()
    finally:
        Path(path).unlink(missing_ok=True)


def current_constraint(model, case, nknots=65, nu=64, nw=64):
    """The mapping's own enclosed toroidal current, as an ncurr = 1 input.

    With ncurr = 1 the solver does not read the iota profile: it solves
    chi' from the constraint currH(s) = <B_u>(s) on every half-grid surface
    (ideal_mhd_model, the ncurr == 1 branch of computeBContra), where currH is
    the prescribed profile scaled to curtor. Prescribing the mapping's own
    <B_u> therefore returns the mapping's own iota, and the continuum force is
    unchanged; what changes is which profile the discrete problem holds fixed.

    Returned as (knots, values, curtor) for a "cubic_spline_i" current profile,
    which VMEC evaluates as the enclosed current directly rather than as its
    derivative.
    """
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    w = np.linspace(0.0, 2.0 * np.pi, nw, endpoint=False)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    one = np.ones_like(uu)
    knots = np.linspace(0.0, 1.0, nknots)
    values = []
    for s in knots:
        g = {
            k: model.aux_at(k, max(float(s), 1.0e-12), uu, ww) * one
            for k in ("guu", "guv", "bsupu", "bsupv")
        }
        values.append(float(np.mean(g["guu"] * g["bsupu"] + g["guv"] * g["bsupv"])))
    values = np.array(values)
    curtor = 2.0 * np.pi * values[-1] / (MU_0 * case.sign_jacobian)
    return knots, values, curtor


def _model_at(case, ns, mpol, ntor, ntheta, nzeta, lasym=False, **kw):
    return _vmecpp.VmecModel.create(
        _indata(case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym, **kw), ns
    )


def masked_source(fa, ns, lasym=False, lfreeb=False):
    """Zero the source where the solver computes no force, so that the frozen
    parts of the state keep the values the mapping gives them: the fixed
    boundary, the m >= 1 coefficients on the axis, and lambda on the axis.

    A free-boundary run solves for the boundary, so its geometry there is kept.
    """
    src = {k: v.copy() for k, v in fa.items()}
    for k in spans(lasym):
        if BASIS[k][0] == "L":
            src[k][0] = 0.0
        else:
            if not lfreeb:
                src[k][ns - 1] = 0.0
            src[k][0, 1:, :] = 0.0
    return src


def solved_mask(ns, mpol, ntor, lasym=False, lfreeb=False):
    """True where the state is determined by the modified force balance."""
    m = {k: np.ones((ns, mpol, ntor + 1), dtype=bool) for k in spans(lasym)}
    for k in spans(lasym):
        if BASIS[k][0] == "L":
            m[k][0] = False
        else:
            m[k][ns - 1] = not lfreeb
            m[k][0, 1:, :] = False
    if mpol > 1:
        for _, z_key in M1_PAIRS:
            if z_key in m:
                m[z_key][:, 1, :] = False  # the frozen m = 1 gauge combination
    return m


class GaugeFixed:
    """Hold the m = 1 poloidal-origin gauge fixed on every force evaluation.

    The continuum force is exactly orthogonal to that gauge direction, so leaving it
    free makes the modified problem singular along it.  The solver fixes it only once
    fsqz < 1e-6, which a source-augmented run need not reach.
    """

    def __init__(self, model):
        object.__setattr__(self, "_m", model)

    def evaluate(self, iter1, iter2, precondition=True, always_fix_m1_gauge=True):  # noqa: ARG002
        return self._m.evaluate(iter1, iter2, precondition, True)

    def __getattr__(self, name):
        return getattr(self._m, name)

    def __setattr__(self, name, value):
        setattr(self._m, name, value)


# ----------------------------------------------------------------------------
# Studies
# ----------------------------------------------------------------------------


def continuum_energy(model, nrho=200, nu=64, nw=64):
    """W = int_0^1 ds <density>, integrated in rho = sqrt(s)."""
    x, wq = np.polynomial.legendre.leggauss(nrho)
    rho, wq = 0.5 * (x + 1.0), 0.5 * wq
    u = np.linspace(0.0, 2 * np.pi, nu, endpoint=False)
    w = np.linspace(0.0, 2 * np.pi, nw, endpoint=False)
    UU, WW = np.meshgrid(u, w, indexing="ij")
    return float(
        sum(
            q * 2.0 * r * np.mean(model.density(r * r, UU, WW))
            for r, q in zip(rho, wq, strict=True)
        )
    )


def discrete_force(
    case, ns, mpol, ntor, ntheta, nzeta, source=None, lasym=False, current=None
):
    m = _model_at(case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym, current=current)
    if source is not None:
        m.set_force_source(source)
    install_state(m, case, ns, mpol, ntor, lasym)
    m.evaluate(1, 1, False, True)
    return unflatten(m.get_forces(), ns, mpol, ntor, lasym), m


def study_energy(
    case, model, mpol, ntor, ntheta, nzeta, ns_list, lasym=False, current=None
):
    w_exact = continuum_energy(model)
    print(f"continuum energy of the mapping: {w_exact:.12e}")
    print(
        f"{'ns':>6} {'solver mhd_energy':>21} {'relative difference':>21} {'order':>7}"
    )
    prev = None
    for ns in ns_list:
        m = _model_at(case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym, current=current)
        install_state(m, case, ns, mpol, ntor, lasym)
        m.evaluate(1, 1, False, True)
        rel = abs(m.mhd_energy / w_exact - 1.0)
        o = "" if prev is None else f"{np.log(prev / rel) / np.log(2.0):7.2f}"
        print(f"{ns:6d} {m.mhd_energy:21.13e} {rel:21.3e} {o:>7}")
        prev = rel


def _force_by_parity(fv, fa, ns, mpol, smin, lasym):
    """Relative L2 deviation, split into R/Z, lambda m >= 1, lambda m = 0.

    An L2 norm over the retained surfaces and modes rather than a maximum: the
    maximum is set by whichever single coefficient happens to be worst at a
    given resolution, and its ratio between two resolutions is correspondingly
    noisy.
    """
    sgrid = np.linspace(0.0, 1.0, ns)
    sel = slice(2, ns - 2)
    keep = sgrid[sel] >= smin
    num = [0.0, 0.0, 0.0]
    den = 0.0
    for k in spans(lasym):
        # the solver's force is + dW/dx; the projected continuum force is - dW/dx
        d = (fv[k][sel] + fa[k][sel])[keep]
        a = fa[k][sel][keep]
        den += float(np.sum(a * a))
        for m in range(mpol):
            i = 0 if BASIS[k][0] != "L" else (1 if m else 2)
            num[i] += float(np.sum(d[:, m, :] ** 2))
    return [float(np.sqrt(x / den)) for x in num]


def study_force(
    case,
    model,
    mpol,
    ntor,
    ntheta,
    nzeta,
    ns_list,
    smin=0.1,
    nu=96,
    nw=96,
    lasym=False,
    current=None,
):
    print(f"relative L2 deviation over s >= {smin}")
    print(
        f"{'ns':>6} {'R, Z':>12} {'order':>7} {'lambda m>=1':>12} {'order':>7} "
        f"{'lambda m=0':>12} {'order':>7}"
    )
    prev = None
    for ns in ns_list:
        fv, _ = discrete_force(
            case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym, current=current
        )
        sgrid = np.linspace(0.0, 1.0, ns)
        fa = hat_force_to_decomposed(
            project_force(model, case, sgrid, mpol, ntor, nu=nu, nw=nw, lasym=lasym),
            mpol,
        )
        v = _force_by_parity(fv, fa, ns, mpol, smin, lasym)
        cols = []
        for i, x in enumerate(v):
            o = "" if prev is None else f"{np.log(prev[i] / x) / np.log(2.0):7.2f}"
            cols.append(f"{x:12.3e} {o:>7}")
        print(f"{ns:6d} " + " ".join(cols))
        prev = v


def study_source(
    case,
    model,
    mpol,
    ntor,
    ntheta,
    nzeta,
    ns_list,
    nu=96,
    nw=96,
    lasym=False,
    current=None,
):
    """Install the source and report what force is left at the mapping."""
    print(f"{'ns':>6} {'fsq, no source':>16} {'fsq, with source':>18} {'ratio':>11}")
    for ns in ns_list:
        sgrid = np.linspace(0.0, 1.0, ns)
        fa = project_force(model, case, sgrid, mpol, ntor, nu=nu, nw=nw, lasym=lasym)
        src = flatten(masked_source(fa, ns, lasym), lasym)
        _, m0 = discrete_force(
            case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym, current=current
        )
        _, m1 = discrete_force(
            case,
            ns,
            mpol,
            ntor,
            ntheta,
            nzeta,
            source=src,
            lasym=lasym,
            current=current,
        )
        a = m0.fsqr + m0.fsqz + m0.fsql
        b = m1.fsqr + m1.fsqz + m1.fsql
        print(f"{ns:6d} {a:16.3e} {b:18.3e} {a / b:11.1f}")


def state_error(
    m, case, ns, mpol, ntor, ntheta, nzeta, smin, edge, lasym, current=None
):
    """Relative L2 distance from the mapping, in the solver's own basis.

    The mapping is installed into a second model and both states are read as they are
    held internally, so the comparison needs no second implementation of the basis
    conventions and both sides carry the m = 1 gauge and the sqrt(s) scaling
    identically.
    """
    reference = _model_at(
        case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym, current=current
    )
    install_state(reference, case, ns, mpol, ntor, lasym)
    want = unflatten(reference.get_state(), ns, mpol, ntor, lasym)
    got = unflatten(m.get_state(), ns, mpol, ntor, lasym)
    sgrid = np.linspace(0.0, 1.0, ns)
    keep = (sgrid >= smin) & (np.arange(ns) < ns - edge)
    keep[0] = False  # the axis and the boundary are not solved for
    num = [0.0, 0.0, 0.0]
    den = 0.0
    for k in spans(lasym):
        d = got[k][keep] - want[k][keep]
        if BASIS[k][0] != "L":
            den += float(np.sum(want[k][keep] ** 2))
        for mm in range(mpol):
            i = 0 if BASIS[k][0] != "L" else (1 if mm else 2)
            num[i] += float(np.sum(d[:, mm, :] ** 2))
    return [float(np.sqrt(x / den)) for x in num]


def study_solve(
    case,
    model,
    mpol,
    ntor,
    ntheta,
    nzeta,
    ns_list,
    smin=0.1,
    edge=3,
    nu=96,
    nw=96,
    ftol=1e-16,
    niter=40000,
    lasym=False,
    current=None,
):
    """Solve the modified problem and measure the distance to the mapping."""
    from vmecpp import _iteration  # noqa: PLC0415

    print(f"bulk = surfaces with s >= {smin} and at least {edge} inside the boundary")
    print(
        f"{'ns':>6} {'fsq':>11} {'R, Z':>12} {'order':>7} {'lambda m>=1':>12} "
        f"{'order':>7} {'lambda m=0':>12} {'order':>7}"
    )
    prev = None
    for ns in ns_list:
        m = _model_at(
            case,
            ns,
            mpol,
            ntor,
            ntheta,
            nzeta,
            lasym=lasym,
            ftol=ftol,
            niter=niter,
            current=current,
        )
        sgrid = np.linspace(0.0, 1.0, ns)
        fa = project_force(model, case, sgrid, mpol, ntor, nu=nu, nw=nw, lasym=lasym)
        m.set_force_source(flatten(masked_source(fa, ns, lasym), lasym))
        install_state(m, case, ns, mpol, ntor, lasym)
        _iteration.solve_equilibrium(GaugeFixed(m), verbose=False)

        v = state_error(
            m, case, ns, mpol, ntor, ntheta, nzeta, smin, edge, lasym, current
        )
        cols = []
        for i, x in enumerate(v):
            o = "" if prev is None else f"{np.log(prev[i] / x) / np.log(2.0):7.2f}"
            cols.append(f"{x:12.3e} {o:>7}")
        print(f"{ns:6d} {m.fsqr + m.fsqz + m.fsql:11.3e} " + " ".join(cols))
        prev = v


def freeb_setup(nzeta, path):
    """The free-boundary mapping, its vacuum field and the mgrid holding it."""
    case = build_case(FREEB_P, freeb_base(), m2=False)
    model = Model(case)
    vac, surface = tune_vacuum_field(model, case)
    write_mgrid(path, vac, case.nfp, nzeta, boundary_box(model))
    return case, model, vac, surface


def freeb_source(model, case, vac, mpol, ntor, ns, nu=96, nw=96):
    """The volume source, plus what the last radial node needs on top of it."""
    sgrid = np.linspace(0.0, 1.0, ns)
    fa = project_force(model, case, sgrid, mpol, ntor, nu=nu, nw=nw)
    edge = lcfs_source(model, vac, mpol, ntor, ns, nu=nu, nw=nw)
    for k in spans(False):
        fa[k][ns - 1] = fa[k][ns - 1] + edge[k][ns - 1]
    return fa


def _freeb_force(case, ns, mpol, ntor, ntheta, nzeta, mgrid, source=None):
    """One force evaluation at the mapping, with the vacuum contribution live.

    The vacuum pressure state advances kOff -> kInitializing -> kInitialized -> kActive
    over the first iterations, so the state is re-installed and the model re-evaluated
    until the edge term is switched on.
    """
    m = _model_at(case, ns, mpol, ntor, ntheta, nzeta, mgrid=mgrid)
    if source is not None:
        m.set_force_source(source)
    for iteration in range(1, 6):
        install_state(m, case, ns, mpol, ntor)
        m.evaluate(1, iteration, False, True)
    return unflatten(m.get_forces(), ns, mpol, ntor), m


def study_freeb(mpol, ntor, ntheta, nzeta, ns_list, smin=0.1, nu=96, nw=96):
    """The free-boundary force and what the source leaves of it.

    The last radial node is where the free boundary differs: its force is
    one-sided, it carries the vacuum edge term, and it is a solved degree of
    freedom rather than a frozen one.  It is reported apart from the volume.
    """
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as fh:
        mgrid = fh.name
    try:
        case, model, vac, surface = freeb_setup(nzeta, mgrid)
        print(
            f"vacuum field B_R = {vac[0]:.4f} Z, B_phi = {vac[1]:.4f}, B_Z = 0; "
            f"plasma R B_phi = {surface['rbtor']:.4f}, "
            f"net toroidal current = {surface['ctor']:.4e}"
        )
        print(
            f"{'ns':>6} {'volume':>12} {'order':>7} {'last node':>12} {'order':>7} "
            f"{'fsq, source':>12} {'ratio':>10}"
        )
        prev = None
        for ns in ns_list:
            fa = freeb_source(model, case, vac, mpol, ntor, ns, nu=nu, nw=nw)
            fv, _ = _freeb_force(case, ns, mpol, ntor, ntheta, nzeta, mgrid)
            dec = hat_force_to_decomposed(fa, mpol)
            volume = max(_force_by_parity(fv, dec, ns, mpol, smin, False))
            edge_num = sum(
                float(np.sum((fv[k][ns - 1] + dec[k][ns - 1]) ** 2))
                for k in spans(False)
            )
            edge_den = sum(float(np.sum(dec[k][ns - 1] ** 2)) for k in spans(False))
            edge = float(np.sqrt(edge_num / edge_den))
            src = flatten(masked_source(fa, ns, lfreeb=True))
            _, bare = _freeb_force(case, ns, mpol, ntor, ntheta, nzeta, mgrid)
            _, with_source = _freeb_force(
                case, ns, mpol, ntor, ntheta, nzeta, mgrid, source=src
            )
            a = bare.fsqr + bare.fsqz + bare.fsql
            b = with_source.fsqr + with_source.fsqz + with_source.fsql
            cols = []
            for i, x in enumerate((volume, edge)):
                o = "" if prev is None else f"{np.log(prev[i] / x) / np.log(2.0):7.2f}"
                cols.append(f"{x:12.3e} {o:>7}")
            print(f"{ns:6d} " + " ".join(cols) + f" {b:12.3e} {a / b:10.1f}")
            prev = (volume, edge)

    finally:
        Path(mgrid).unlink(missing_ok=True)


def study_angular(mpol, ntor, ns, grids, fine=96):
    """Angular truncation, isolated from the radial error by comparing grids at one
    radial resolution.

    This uses the unfitted mapping.  The fitted one is close to force balance, so its
    force is near zero and a relative measure of it says nothing.
    """
    case = Case()
    ref, _ = discrete_force(case, ns, mpol, ntor, fine, fine)
    scale = max(float(np.max(np.abs(ref[k]))) for k in spans(False))
    print(f"ns = {ns}, reference angular grid {fine} x {fine}")
    print(f"{'ntheta = nzeta':>15} {'relative difference':>21} {'factor':>8}")
    prev = None
    for g in grids:
        fv, _ = discrete_force(case, ns, mpol, ntor, g, g)
        d = max(float(np.max(np.abs(fv[k] - ref[k]))) for k in spans(False)) / scale
        r = "" if prev is None else f"{prev / d:8.2f}"
        print(f"{g:15d} {d:21.3e} {r:>8}")
        prev = d


def main():
    ap = argparse.ArgumentParser(
        description="Method of manufactured solutions for the VMEC++ discretization."
    )
    ap.add_argument(
        "--study",
        default="all",
        choices=[
            "energy",
            "force",
            "source",
            "solve",
            "angular",
            "freeb",
            "fit",
            "all",
        ],
    )
    ap.add_argument("--mpol", type=int, default=4)
    ap.add_argument("--ntor", type=int, default=2)
    ap.add_argument("--ntheta", type=int, default=18)
    ap.add_argument("--nzeta", type=int, default=16)
    ap.add_argument("--ns", type=int, nargs="+", default=[13, 25, 49])
    ap.add_argument(
        "--projection",
        type=int,
        default=32,
        help="angular resolution at which the continuum force is projected",
    )
    ap.add_argument(
        "--angular",
        type=int,
        nargs="+",
        default=[16, 18, 20],
        help="angular grids compared in the truncation study",
    )
    ap.add_argument(
        "--angular-reference",
        type=int,
        default=48,
        help="angular grid the truncation study measures against",
    )
    ap.add_argument(
        "--ncurr",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 prescribes the mapping's own enclosed toroidal current instead "
        "of its iota profile, so the solver derives iota from the constraint",
    )
    ap.add_argument(
        "--freeb",
        action="store_true",
        help="with --study fit, fit the free-boundary mapping instead",
    )
    ap.add_argument(
        "--lasym",
        action="store_true",
        help="run the mapping through the non-stellarator-symmetric "
        "code path, with asymmetric content in the mapping",
    )
    ap.add_argument(
        "--lasym-degenerate",
        action="store_true",
        help="run the symmetric mapping through the asymmetric code "
        "path, which must reproduce the symmetric result",
    )
    args = ap.parse_args()

    if args.study == "fit":
        p = fit_parameters(freeb_base() if args.freeb else None, m2=not args.freeb)
        np.set_printoptions(precision=9)
        print("fitted parameters:", repr(p))
        return

    lasym = args.lasym or args.lasym_degenerate
    case = build_case(FITTED_P, asym=ASYM if args.lasym else None)
    model = Model(case)
    print(
        f"nfp = {case.nfp}, mpol = {args.mpol}, ntor = {args.ntor}, "
        f"grid = {args.ntheta} x {args.nzeta}, lasym = {lasym}, "
        f"ncurr = {args.ncurr}"
        f"{', degenerate' if args.lasym_degenerate else ''}\n"
    )

    common = (case, model, args.mpol, args.ntor, args.ntheta, args.nzeta, args.ns)
    proj = {"nu": args.projection, "nw": args.projection}
    current = current_constraint(model, case) if args.ncurr == 1 else None
    kw = {"lasym": lasym, "current": current}
    if args.study in ("energy", "all"):
        print("Discrete MHD energy against the continuum energy of the mapping")
        study_energy(*common, **kw)
        print()
    if args.study in ("force", "all"):
        print("Discrete spectral force against the continuum ideal-MHD force")
        study_force(*common, **kw, **proj)
        print()
    if args.study in ("source", "all"):
        print("Force left at the mapping once the source is installed")
        study_source(*common, **kw, **proj)
        print()
    if args.study in ("solve", "all"):
        print("Converged discrete state against the mapping")
        study_solve(*common, **kw, **proj)
        print()
    if args.study in ("angular", "all"):
        print("Angular truncation of the discrete force at fixed ns, unfitted mapping")
        study_angular(
            args.mpol, args.ntor, args.ns[-1], args.angular, fine=args.angular_reference
        )
        print()
    if args.study in ("freeb", "all"):
        print("Free boundary: the discrete force against the continuum one")
        study_freeb(args.mpol, args.ntor, args.ntheta, args.nzeta, args.ns, **proj)


if __name__ == "__main__":
    main()
