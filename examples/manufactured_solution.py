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

Requires sympy and, for --study fit, scipy.  Neither is a dependency of the
solver.

Usage:
    python manufactured_solution.py [--study energy|force|source|solve|angular|fit|all]
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
        """dPhi/ds, constant for the default aphi = [1]."""
        return self.sign_jacobian * self.phiedge / (2.0 * sp.pi)

    @property
    def lamscale(self):
        """sqrt(rmsPhiP * deltaS) = |phip| for the default aphi."""
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

    Returns cc/ss/sc/cs, entry [m, n] being the mean of dens * cos(m u) cos(n w)
    and so on over the (u, w) torus.
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


def build_case(p, base=None, asym=None):
    """Assemble a Case from the flat parameter vector used by the fit."""
    base = FITTED_BASE if base is None else base
    kw = dict(base)
    kw["r00"] = (base["r00"][0], p[0], p[1])
    kw["m2"] = {
        ("rcc", 0): p[2],
        ("rcc", 1): p[3],
        ("rss", 1): p[4],
        ("zsc", 0): p[5],
        ("zsc", 1): p[6],
        ("zcs", 1): p[7],
    }
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


def fit_parameters(base=None, mpol=5, ntor=3, nsfit=8, nu=48, nw=48, maxiter=40):
    """Minimize the continuum force over the mapping's free parameters."""
    from scipy.optimize import least_squares  # noqa: PLC0415

    base = FITTED_BASE if base is None else base
    x, _ = np.polynomial.legendre.leggauss(nsfit)
    rho = 0.5 * (x + 1.0)
    srho = rho * rho
    p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0])

    case0 = build_case(p0, base)
    fa0 = project_force(
        Model(case0), case0, srho, mpol, ntor, nu=nu, nw=nw, endpoint_half=False
    )
    scale = max(float(np.max(np.abs(fa0[k]))) for k in GEOM_SPANS)

    def residual(p):
        case = build_case(p, base)
        fa = project_force(
            Model(case), case, srho, mpol, ntor, nu=nu, nw=nw, endpoint_half=False
        )
        return np.concatenate([fa[k].reshape(-1) for k in GEOM_SPANS]) / scale

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
    case, ns, mpol, ntor, ntheta, nzeta, ftol=1e-16, niter=40000, delt=0.9, lasym=False
):
    """A VMEC++ input carrying the mapping's boundary, axis and profiles.

    The boundary is the mapping at s = 1 and the axis is the mapping at s = 0,
    both taken from the same combined-basis conversion the state uses.
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
        "tcon0": 1.0,
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
        "bloat": 1.0,
        "lfreeb": False,
        "mgrid_file": "NONE",
        "nvacskip": 1,
        "lforbal": False,
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
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(d, fh)
        path = fh.name
    try:
        return vmecpp.VmecInput.from_file(path)._to_cpp_vmecindata()
    finally:
        Path(path).unlink(missing_ok=True)


def _model_at(case, ns, mpol, ntor, ntheta, nzeta, lasym=False, **kw):
    return _vmecpp.VmecModel.create(
        _indata(case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym, **kw), ns
    )


def masked_source(fa, ns, lasym=False):
    """Zero the source where the solver computes no force, so that the frozen
    parts of the state keep the values the mapping gives them: the fixed
    boundary, the m >= 1 coefficients on the axis, and lambda on the axis."""
    src = {k: v.copy() for k, v in fa.items()}
    for k in spans(lasym):
        if BASIS[k][0] == "L":
            src[k][0] = 0.0
        else:
            src[k][ns - 1] = 0.0
            src[k][0, 1:, :] = 0.0
    return src


def solved_mask(ns, mpol, ntor, lasym=False):
    """True where the state is determined by the modified force balance."""
    m = {k: np.ones((ns, mpol, ntor + 1), dtype=bool) for k in spans(lasym)}
    for k in spans(lasym):
        if BASIS[k][0] == "L":
            m[k][0] = False
        else:
            m[k][ns - 1] = False
            m[k][0, 1:, :] = False
    if mpol > 1:
        for _, z_key in M1_PAIRS:
            if z_key in m:
                m[z_key][:, 1, :] = False  # the frozen m = 1 gauge combination
    return m


class GaugeFixed:
    """Hold the m = 1 poloidal-origin gauge fixed on every force evaluation.

    The continuum force is exactly orthogonal to that gauge direction, so
    leaving it free makes the modified problem singular along it.  The solver
    fixes it only once fsqz < 1e-6, which a source-augmented run need not reach.
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


def discrete_force(case, ns, mpol, ntor, ntheta, nzeta, source=None, lasym=False):
    m = _model_at(case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym)
    if source is not None:
        m.set_force_source(source)
    install_state(m, case, ns, mpol, ntor, lasym)
    m.evaluate(1, 1, False, True)
    return unflatten(m.get_forces(), ns, mpol, ntor, lasym), m


def study_energy(case, model, mpol, ntor, ntheta, nzeta, ns_list, lasym=False):
    w_exact = continuum_energy(model)
    print(f"continuum energy of the mapping: {w_exact:.12e}")
    print(
        f"{'ns':>6} {'solver mhd_energy':>21} {'relative difference':>21} {'order':>7}"
    )
    prev = None
    for ns in ns_list:
        m = _model_at(case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym)
        install_state(m, case, ns, mpol, ntor, lasym)
        m.evaluate(1, 1, False, True)
        rel = abs(m.mhd_energy / w_exact - 1.0)
        o = "" if prev is None else f"{np.log(prev / rel) / np.log(2.0):7.2f}"
        print(f"{ns:6d} {m.mhd_energy:21.13e} {rel:21.3e} {o:>7}")
        prev = rel


def _force_by_parity(fv, fa, ns, mpol, smin, lasym):
    """Worst relative deviation, split into R/Z, lambda m >= 1, lambda m = 0."""
    sgrid = np.linspace(0.0, 1.0, ns)
    sel = slice(2, ns - 2)
    keep = sgrid[sel] >= smin
    scale = max(float(np.max(np.abs(fa[k][sel]))) for k in spans(lasym))
    out = [0.0, 0.0, 0.0]
    for k in spans(lasym):
        # the solver's force is + dW/dx; the projected continuum force is - dW/dx
        d = np.abs(fv[k][sel] + fa[k][sel])[keep]
        for m in range(mpol):
            v = float(np.max(d[:, m, :])) / scale
            i = 0 if BASIS[k][0] != "L" else (1 if m else 2)
            out[i] = max(out[i], v)
    return out


def study_force(
    case, model, mpol, ntor, ntheta, nzeta, ns_list, smin=0.1, nu=96, nw=96, lasym=False
):
    print(f"worst relative deviation over s >= {smin}")
    print(
        f"{'ns':>6} {'R, Z':>12} {'order':>7} {'lambda m>=1':>12} {'order':>7} "
        f"{'lambda m=0':>12} {'order':>7}"
    )
    prev = None
    for ns in ns_list:
        fv, _ = discrete_force(case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym)
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
    case, model, mpol, ntor, ntheta, nzeta, ns_list, nu=96, nw=96, lasym=False
):
    """Install the source and report what force is left at the mapping."""
    print(f"{'ns':>6} {'fsq, no source':>16} {'fsq, with source':>18} {'ratio':>11}")
    for ns in ns_list:
        sgrid = np.linspace(0.0, 1.0, ns)
        fa = project_force(model, case, sgrid, mpol, ntor, nu=nu, nw=nw, lasym=lasym)
        src = flatten(masked_source(fa, ns, lasym), lasym)
        _, m0 = discrete_force(case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym)
        _, m1 = discrete_force(
            case, ns, mpol, ntor, ntheta, nzeta, source=src, lasym=lasym
        )
        a = m0.fsqr + m0.fsqz + m0.fsql
        b = m1.fsqr + m1.fsqz + m1.fsql
        print(f"{ns:6d} {a:16.3e} {b:18.3e} {a / b:11.1f}")


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
            case, ns, mpol, ntor, ntheta, nzeta, lasym=lasym, ftol=ftol, niter=niter
        )
        sgrid = np.linspace(0.0, 1.0, ns)
        fa = project_force(model, case, sgrid, mpol, ntor, nu=nu, nw=nw, lasym=lasym)
        m.set_force_source(flatten(masked_source(fa, ns, lasym), lasym))
        install_state(m, case, ns, mpol, ntor, lasym)
        _iteration.solve_equilibrium(GaugeFixed(m), verbose=False)

        got = m.get_state_as_fourier()
        want = combined_coefficients(case, sgrid, mpol, ntor)
        keys = ["rmnc", "zmns", "lmns"] + (["rmns", "zmnc", "lmnc"] if lasym else [])
        modes = mode_order(mpol, ntor)
        keep = (sgrid >= smin) & (np.arange(ns) < ns - edge)
        # the boundary and the axis are not solved for
        keep[0] = False
        scale = max(float(np.max(np.abs(want[k]))) for k in ("rmnc", "zmns"))
        v = [0.0, 0.0, 0.0]
        for idx, key in enumerate(keys):
            d = np.abs(got[idx][:, keep] - want[key][:, keep]) / scale
            for i, (mm, _n) in enumerate(modes):
                j = 0 if not key.startswith("l") else (1 if mm else 2)
                v[j] = max(v[j], float(np.max(d[i])))
        cols = []
        for i, x in enumerate(v):
            o = "" if prev is None else f"{np.log(prev[i] / x) / np.log(2.0):7.2f}"
            cols.append(f"{x:12.3e} {o:>7}")
        print(f"{ns:6d} {m.fsqr + m.fsqz + m.fsql:11.3e} " + " ".join(cols))
        prev = v


def study_angular(mpol, ntor, ns, grids, fine=96):
    """Angular truncation, isolated from the radial error by comparing grids at
    one radial resolution.

    This uses the unfitted mapping.  The fitted one is close to force balance,
    so its force is near zero and a relative measure of it says nothing.
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
        choices=["energy", "force", "source", "solve", "angular", "fit", "all"],
    )
    ap.add_argument("--mpol", type=int, default=5)
    ap.add_argument("--ntor", type=int, default=3)
    ap.add_argument("--ntheta", type=int, default=24)
    ap.add_argument("--nzeta", type=int, default=24)
    ap.add_argument("--ns", type=int, nargs="+", default=[25, 51, 101, 201])
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
        p = fit_parameters()
        np.set_printoptions(precision=9)
        print("fitted parameters:", repr(p))
        return

    lasym = args.lasym or args.lasym_degenerate
    case = build_case(FITTED_P, asym=ASYM if args.lasym else None)
    model = Model(case)
    print(
        f"nfp = {case.nfp}, mpol = {args.mpol}, ntor = {args.ntor}, "
        f"grid = {args.ntheta} x {args.nzeta}, lasym = {lasym}"
        f"{', degenerate' if args.lasym_degenerate else ''}\n"
    )

    common = (case, model, args.mpol, args.ntor, args.ntheta, args.nzeta, args.ns)
    if args.study in ("energy", "all"):
        print("Discrete MHD energy against the continuum energy of the mapping")
        study_energy(*common, lasym=lasym)
        print()
    if args.study in ("force", "all"):
        print("Discrete spectral force against the continuum ideal-MHD force")
        study_force(*common, lasym=lasym)
        print()
    if args.study in ("source", "all"):
        print("Force left at the mapping once the source is installed")
        study_source(*common, lasym=lasym)
        print()
    if args.study in ("solve", "all"):
        print("Converged discrete state against the mapping")
        study_solve(*common, lasym=lasym)
        print()
    if args.study in ("angular", "all"):
        print("Angular truncation of the discrete force at fixed ns, unfitted mapping")
        study_angular(args.mpol, args.ntor, args.ns[-1], [16, 18, 20, 22, 24, 26])


if __name__ == "__main__":
    main()
