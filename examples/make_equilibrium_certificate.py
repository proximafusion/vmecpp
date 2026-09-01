"""Emit a Stellarocq certificate from a VMEC wout file.

The checker that validates the certificate is built from
https://github.com/CharlesCNorton/stellarocq; see
docs/proof_carrying_equilibria.md.

The certificate states, for a set of evaluation points, that the mu0-scaled
ideal-MHD force residual of the equilibrium reconstructed from the wout
coefficients (by the rule fixed in theories/Physics.v) lies within the
claimed per-component bounds.  Every numeric input is an IEEE double from
the wout, emitted exactly as a dyadic rational m*2^e; the checker encloses
the true real arithmetic with proven-sound interval arithmetic, so a VALID
verdict is a theorem about these exact inputs.

Environment layout per point (must match theories/Physics.v):
  0 s | 1 u | 2 v | 3 phip | 4..7 sfull | 8..11 shalf | 12..15 iota
  16..36 am | 37+0..4K-1 R | +4K Z | +8K L      (K = mnmax)

Usage:  python make_cert.py wout_X.nc cert_X.txt  [--bands 6] [--nu 8] [--nv 4]
"""

import argparse
import pathlib

import netCDF4
import numpy as np

MU0 = 4e-7 * np.pi


def dyadic(x):
    """Exact (mantissa, exponent) with x = m * 2**e, for a finite double."""
    num, den = float(x).as_integer_ratio()
    e = 0
    d = den
    while d > 1:
        d >>= 1
        e -= 1
    m = num
    while m != 0 and m % 2 == 0:
        m //= 2
        e += 1
    if m == 0:
        e = 0
    return m, e


class Wout:
    """The wout fields the certificate needs."""

    def __init__(self, path):
        """Load the fields and reject unsupported pressure types."""
        d = netCDF4.Dataset(path)
        d.set_auto_mask(False)
        v = d.variables

        def g(k):
            return np.asarray(v[k][:], dtype=float)

        self.ns = int(v["ns"][:])
        self.xm = g("xm").astype(int)
        self.xn = g("xn").astype(int)
        self.rmnc = g("rmnc")
        self.zmns = g("zmns")
        self.lmns = g("lmns")
        self.iotas = g("iotas")
        self.phips = g("phips")
        self.am = g("am")
        ptype = v["pmass_type"][:].tobytes().decode().replace("\x00", "").strip()
        if ptype != "power_series":
            msg = f"v1 certifies power_series pressure only, got {ptype!r}"
            raise SystemExit(msg)
        d.close()
        self.h = 1.0 / (self.ns - 1)
        self.s_full = np.arange(self.ns) * self.h
        self.s_half = (
            np.arange(1, self.ns) - 0.5
        ) * self.h  # nodes of lmns[1:], iotas[1:]


def band_tables(w, i):
    """Stencils for an evaluation point in full-grid band [s_i, s_{i+1}].

    Full grid stencil: nodes i-1..i+2 of s_full (coefficients rmnc/zmns).
    Half grid: the eval point s* = midpoint of the band lies exactly on
    half node j = i (0-based into lmns[1:]); stencil j-1..j+2.
    """
    sf = w.s_full[i - 1 : i + 3]
    sh = w.s_half[i - 1 : i + 3]
    R = w.rmnc[i - 1 : i + 3, :]
    Z = w.zmns[i - 1 : i + 3, :]
    L = w.lmns[1:][i - 1 : i + 3, :]
    IO = w.iotas[1:][i - 1 : i + 3]
    return sf, sh, R, Z, L, IO


# ----- float reference evaluation (same rule), for choosing eps -------------


def hermite_rho(sn, y, s):
    """Value, d/ds, d2/ds2 of the Hermite-in-rho profile at s."""
    r = np.sqrt(sn)
    rr = np.sqrt(s)
    h = r[2] - r[1]
    t = (rr - r[1]) / h
    d1 = (y[2] - y[0]) / (r[2] - r[0])
    d2 = (y[3] - y[1]) / (r[3] - r[1])
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    val = h00 * y[1] + h10 * h * d1 + h01 * y[2] + h11 * h * d2
    dh00 = (6 * t**2 - 6 * t) / h
    dh10 = 3 * t**2 - 4 * t + 1
    dh01 = (-6 * t**2 + 6 * t) / h
    dh11 = 3 * t**2 - 2 * t
    drho = dh00 * y[1] + dh10 * d1 + dh01 * y[2] + dh11 * d2
    d2h00 = (12 * t - 6) / h**2
    d2h10 = (6 * t - 4) / h
    d2h01 = (-12 * t + 6) / h**2
    d2h11 = (6 * t - 2) / h
    d2rho = d2h00 * y[1] + d2h10 * d1 + d2h01 * y[2] + d2h11 * d2
    ds = drho / (2 * rr)
    dss = d2rho / (4 * s) - drho / (4 * s * rr)
    return val, ds, dss


def residual_ref(w, band, s, u, vv, phip):
    """Float reference of the residual at one point, for choosing bounds."""
    sf, sh, Rn, Zn, Ln, IOn = band
    m = w.xm
    n = w.xn
    ang = m * u - n * vv
    c = np.cos(ang)
    sn_ = np.sin(ang)
    rv, rd, rdd = hermite_rho(sf, Rn, s)
    zv, zd, zdd = hermite_rho(sf, Zn, s)
    lv, ld, _ = hermite_rho(sh, Ln, s)
    iot, iotp, _ = hermite_rho(sh, IOn.reshape(4, 1), s)
    iot = float(iot[0])
    iotp = float(iotp[0])

    def S(cf, k):
        return float(np.dot(cf, k))

    R = S(rv, c)
    Z_s = S(zd, sn_)
    lam_u = S(lv, m * c)
    R_s = S(rd, c)
    Z_ss = S(zdd, sn_)
    lam_v = S(lv, -n * c)
    R_ss = S(rdd, c)
    Z_u = S(zv, m * c)
    lam_su = S(ld, m * c)
    R_u = S(rv, -m * sn_)
    Z_v = S(zv, -n * c)
    lam_sv = S(ld, -n * c)
    R_v = S(rv, n * sn_)
    Z_su = S(zd, m * c)
    lam_uu = S(lv, -m * m * sn_)
    R_su = S(rd, -m * sn_)
    Z_sv = S(zd, -n * c)
    lam_uv = S(lv, m * n * sn_)
    R_sv = S(rd, n * sn_)
    Z_uu = S(zv, -m * m * sn_)
    lam_vv = S(lv, -n * n * sn_)
    R_uu = S(rv, -m * m * c)
    Z_uv = S(zv, m * n * sn_)
    R_uv = S(rv, m * n * c)
    Z_vv = S(zv, -n * n * sn_)
    R_vv = S(rv, -n * n * c)
    tau = R_u * Z_s - R_s * Z_u
    sqrtg = R * tau
    tau_s = R_su * Z_s + R_u * Z_ss - R_ss * Z_u - R_s * Z_su
    tau_u = R_uu * Z_s + R_u * Z_su - R_su * Z_u - R_s * Z_uu
    tau_v = R_uv * Z_s + R_u * Z_sv - R_sv * Z_u - R_s * Z_uv
    g_s = R_s * tau + R * tau_s
    g_u = R_u * tau + R * tau_u
    g_v = R_v * tau + R * tau_v
    guu = R_u**2 + Z_u**2
    guv = R_u * R_v + Z_u * Z_v
    gvv = R_v**2 + Z_v**2 + R**2
    gsu = R_s * R_u + Z_s * Z_u
    gsv = R_s * R_v + Z_s * Z_v
    guu_s = 2 * (R_u * R_su + Z_u * Z_su)
    guv_s = R_su * R_v + R_u * R_sv + Z_su * Z_v + Z_u * Z_sv
    gvv_s = 2 * (R_v * R_sv + Z_v * Z_sv + R * R_s)
    gsu_u = R_su * R_u + R_s * R_uu + Z_su * Z_u + Z_s * Z_uu
    gsu_v = R_sv * R_u + R_s * R_uv + Z_sv * Z_u + Z_s * Z_uv
    gsv_u = R_su * R_v + R_s * R_uv + Z_su * Z_v + Z_s * Z_uv
    gsv_v = R_sv * R_v + R_s * R_vv + Z_sv * Z_v + Z_s * Z_vv
    guu_v = 2 * (R_u * R_uv + Z_u * Z_uv)
    guv_u = R_uu * R_v + R_u * R_uv + Z_uu * Z_v + Z_u * Z_uv
    guv_v = R_uv * R_v + R_u * R_vv + Z_uv * Z_v + Z_u * Z_vv
    gvv_u = 2 * (R_v * R_uv + Z_v * Z_uv + R * R_u)
    bu = iot - lam_v
    bv = 1.0 + lam_u
    Bu = phip * bu / sqrtg
    Bv = phip * bv / sqrtg
    bu_s = iotp - lam_sv
    bv_s = lam_su
    Bu_s = phip * (bu_s * sqrtg - bu * g_s) / sqrtg**2
    Bv_s = phip * (bv_s * sqrtg - bv * g_s) / sqrtg**2
    Bu_u = phip * (-lam_uv * sqrtg - bu * g_u) / sqrtg**2
    Bv_u = phip * (lam_uu * sqrtg - bv * g_u) / sqrtg**2
    Bu_v = phip * (-lam_vv * sqrtg - bu * g_v) / sqrtg**2
    Bv_v = phip * (lam_uv * sqrtg - bv * g_v) / sqrtg**2
    B_u_s = guu_s * Bu + guu * Bu_s + guv_s * Bv + guv * Bv_s
    B_v_s = guv_s * Bu + guv * Bu_s + gvv_s * Bv + gvv * Bv_s
    B_s_u = gsu_u * Bu + gsu * Bu_u + gsv_u * Bv + gsv * Bv_u
    B_s_v = gsu_v * Bu + gsu * Bu_v + gsv_v * Bv + gsv * Bv_v
    B_u_v = guu_v * Bu + guu * Bu_v + guv_v * Bv + guv * Bv_v
    B_v_u = guv_u * Bu + guv * Bu_u + gvv_u * Bv + gvv * Bv_u
    pp = sum(k * a * s ** (k - 1) for k, a in enumerate(w.am) if k > 0)
    rs = (B_s_v - B_v_s) * Bv - (B_u_s - B_s_u) * Bu - MU0 * pp
    mu0Js = B_v_u - B_u_v
    ru = -mu0Js * Bv
    rv_ = mu0Js * Bu
    B2 = Bu * (guu * Bu + guv * Bv) + Bv * (guv * Bu + gvv * Bv)
    return rs, ru, rv_, B2


def main():
    """Read the wout, choose the bounds, write the certificate."""
    ap = argparse.ArgumentParser()
    ap.add_argument("wout")
    ap.add_argument("out")
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--nu", type=int, default=8)
    ap.add_argument("--nv", type=int, default=4)
    ap.add_argument("--slack", type=float, default=1.5)
    a = ap.parse_args()

    w = Wout(a.wout)
    K = len(w.xm)
    phip = float(w.phips[1])
    nfp = 1
    if (w.xn != 0).any():
        nfp = int(np.gcd.reduce(np.abs(w.xn[w.xn != 0])))
    three_d = (w.xn != 0).any()
    nv = a.nv if three_d else 1

    # evaluation bands: interior full-grid bands, evenly spread
    lo, hi = 2, w.ns - 4
    idx = np.unique(np.linspace(lo, hi, a.bands).astype(int))
    # angles: exact doubles
    us = [float(2 * np.pi * k / a.nu) for k in range(a.nu)]
    vs = [float(2 * np.pi * k / (nfp * nv)) for k in range(nv)]
    angles = [(u, v) for u in us for v in vs]

    # choose eps from the float reference
    worst = np.zeros(3)
    scale = 0.0
    per_band = []
    for i in idx:
        bt = band_tables(w, i)
        s = float(w.s_half[i])  # midpoint of band [s_i, s_{i+1}]
        mx = np.zeros(3)
        for u, v in angles:
            rs, ru, rv_, B2 = residual_ref(w, bt, s, u, v, phip)
            mx = np.maximum(mx, np.abs([rs, ru, rv_]))
            scale = max(scale, abs(B2))
        per_band.append((i, s, mx))
        worst = np.maximum(worst, mx)
    # Floor: interval evaluation carries genuine rounding width, so a claimed
    # bound must sit above it. 1e-10 of the field-energy scale is far above
    # the enclosure width of ~1e5 double operations and far below any
    # physical residual of interest.
    eps = np.maximum(worst * a.slack, 1e-10 * scale)

    lines = []
    P = lines.append
    P("STELLAROCQ-CERT 1")
    P("PREC 80")
    P(f"MODES {K}")
    for m, n in zip(w.xm, w.xn, strict=True):
        P(f"{m} {n}")
    P("PHIP {} {}".format(*dyadic(phip)))
    P("AM 21")
    for j in range(21):
        am_j = w.am[j] if j < len(w.am) else 0.0
        P("{} {}".format(*dyadic(am_j)))
    for tag, e in zip(("EPS_S", "EPS_U", "EPS_V"), eps, strict=True):
        P("{} {} {}".format(tag, *dyadic(e)))
    P(f"NANGLES {len(angles)}")
    for u, v in angles:
        P("{} {} {} {}".format(*dyadic(u), *dyadic(v)))
    P(f"NBANDS {len(idx)}")
    for i in idx:
        sf, sh, Rn, Zn, Ln, IOn = band_tables(w, i)
        P("BAND")
        P("S {} {}".format(*dyadic(w.s_half[i])))
        P("SFULL " + " ".join("{} {}".format(*dyadic(x)) for x in sf))
        P("SHALF " + " ".join("{} {}".format(*dyadic(x)) for x in sh))
        P("IOTA " + " ".join("{} {}".format(*dyadic(x)) for x in IOn))
        for tag, M in (("RNODES", Rn), ("ZNODES", Zn), ("LNODES", Ln)):
            P(tag)
            for row in M:  # 4 stencil rows
                P(" ".join("{} {}".format(*dyadic(x)) for x in row))
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")

    print(
        f"wrote {a.out}: {len(idx)} bands x {len(angles)} angles = "
        f"{len(idx) * len(angles)} points, K={K}"
    )
    print(
        f"claimed bounds (mu0-scaled, Pa*mu0):  "
        f"r_s {eps[0]:.3e}  r_u {eps[1]:.3e}  r_v {eps[2]:.3e}"
    )
    print(
        f"reference B^2 scale {scale:.3e}  ->  normalized r_s bound "
        f"{eps[0] / scale:.3e}"
    )
    for i, s, mx in per_band:
        print(f"  band {i:3d}  s={s:.4f}  |r|max = {mx[0]:.3e} {mx[1]:.3e} {mx[2]:.3e}")


if __name__ == "__main__":
    main()
