"""Emit a Stellarocq certificate from a VMEC wout file.

The checker that validates the certificate is built from
https://github.com/CharlesCNorton/stellarocq; see
docs/proof_carrying_equilibria.md.

The certificate states, for a set of full-grid nodes and angles, that the
mu0-scaled ideal-MHD force residual of the equilibrium reconstructed from the
wout coefficients by VMEC's own half-grid rule (fixed in theories/Physics.v of
Stellarocq) lies within the claimed per-component bounds: r_s at the node from
the centered differences of its two half points, r_u and r_v at the outer half
point.  Every numeric input is an IEEE double from the wout, emitted exactly as
a dyadic rational m*2^e; the checker encloses the true real arithmetic with
proven-sound interval arithmetic, so a VALID verdict is a theorem about these
exact inputs.

Environment layout per point (must match theories/Physics.v):
  0 s_j | 1 u | 2 v | 3 phip | 4..6 s_{j-1} s_j s_{j+1} | 7..8 s_{j-1/2} s_{j+1/2}
  9..10 iota(h-) iota(h+) | 11..31 am | 32+0..3K-1 R (rows j-1, j, j+1)
  +3K Z | +6K lambda (rows h-, h+)      (K = mnmax)
  32+8K..   scratch slots the checker fills with shared subexpressions

Usage:  python make_equilibrium_certificate.py [wout_X.nc [cert_X.txt]] [--nodes 6] [--nu 8] [--nv 4]
Without arguments it certifies the shipped wout_solovev.nc into cert_solovev.txt.
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
        self.rmnc = g("rmnc")  # (ns, mnmax), full grid
        self.zmns = g("zmns")
        self.lmns = g("lmns")  # (ns, mnmax), half grid: row j is s_{j-1/2}
        self.iotas = g("iotas")  # half grid, same indexing
        self.phips = g("phips")
        self.am = g("am")
        ptype = v["pmass_type"][:].tobytes().decode().replace("\x00", "").strip()
        if ptype != "power_series":
            msg = f"v1 certifies power_series pressure only, got {ptype!r}"
            raise SystemExit(msg)
        d.close()
        self.h = 1.0 / (self.ns - 1)
        self.s_full = np.arange(self.ns) * self.h
        self.s_half = (np.arange(self.ns) - 0.5) * self.h  # row j of lmns, iotas


# ----- float reference of the same rule, for choosing eps -----------------


def half_coefs(w, j_in, j_out, s_h, coefs):
    """C(h) and c'(h) of every mode between full nodes j_in and j_out, by VMEC's parity-
    aware rule."""
    ya, yb = coefs[j_in], coefs[j_out]
    s_a, s_b = w.s_full[j_in], w.s_full[j_out]
    odd = (w.xm % 2) == 1
    c = 0.5 * (ya + yb)
    cs = (yb - ya) / (s_b - s_a)
    qa, qb = ya / np.sqrt(s_a), yb / np.sqrt(s_b)
    c_odd = np.sqrt(s_h) * 0.5 * (qa + qb)
    cs_odd = np.sqrt(s_h) * (qb - qa) / (s_b - s_a) + c_odd / (2.0 * s_h)
    return np.where(odd, c_odd, c), np.where(odd, cs_odd, cs)


def half_point(w, j_in, j_out, row_l, u, vv, phip):
    """B^u, B^v, B_u, B_v, d_u B_s, d_v B_s and mu0 sqrtg J^s at the half point."""
    m, n = w.xm, w.xn
    s_h = w.s_half[row_l]
    cR, cRs = half_coefs(w, j_in, j_out, s_h, w.rmnc)
    cZ, cZs = half_coefs(w, j_in, j_out, s_h, w.zmns)
    cL = w.lmns[row_l]
    iota = float(w.iotas[row_l])
    ang = m * u - n * vv
    c = np.cos(ang)
    sn = np.sin(ang)

    def S(cf, k):
        return float(np.dot(cf, k))

    R = S(cR, c)
    R_s = S(cRs, c)
    R_u = S(cR, -m * sn)
    R_v = S(cR, n * sn)
    R_su = S(cRs, -m * sn)
    R_sv = S(cRs, n * sn)
    R_uu = S(cR, -m * m * c)
    R_uv = S(cR, m * n * c)
    R_vv = S(cR, -n * n * c)
    Z_s = S(cZs, sn)
    Z_u = S(cZ, m * c)
    Z_v = S(cZ, -n * c)
    Z_su = S(cZs, m * c)
    Z_sv = S(cZs, -n * c)
    Z_uu = S(cZ, -m * m * sn)
    Z_uv = S(cZ, m * n * sn)
    Z_vv = S(cZ, -n * n * sn)
    L_u = S(cL, m * c)
    L_v = S(cL, -n * c)
    L_uu = S(cL, -m * m * sn)
    L_uv = S(cL, m * n * sn)
    L_vv = S(cL, -n * n * sn)
    tau = R_u * Z_s - R_s * Z_u
    sqrtg = R * tau
    tau_u = R_uu * Z_s + R_u * Z_su - R_su * Z_u - R_s * Z_uu
    tau_v = R_uv * Z_s + R_u * Z_sv - R_sv * Z_u - R_s * Z_uv
    g_u = R_u * tau + R * tau_u
    g_v = R_v * tau + R * tau_v
    guu = R_u**2 + Z_u**2
    guv = R_u * R_v + Z_u * Z_v
    gvv = R_v**2 + Z_v**2 + R**2
    gsu = R_s * R_u + Z_s * Z_u
    gsv = R_s * R_v + Z_s * Z_v
    gsu_u = R_su * R_u + R_s * R_uu + Z_su * Z_u + Z_s * Z_uu
    gsu_v = R_sv * R_u + R_s * R_uv + Z_sv * Z_u + Z_s * Z_uv
    gsv_u = R_su * R_v + R_s * R_uv + Z_su * Z_v + Z_s * Z_uv
    gsv_v = R_sv * R_v + R_s * R_vv + Z_sv * Z_v + Z_s * Z_vv
    guu_v = 2 * (R_u * R_uv + Z_u * Z_uv)
    guv_u = R_uu * R_v + R_u * R_uv + Z_uu * Z_v + Z_u * Z_uv
    guv_v = R_uv * R_v + R_u * R_vv + Z_uv * Z_v + Z_u * Z_vv
    gvv_u = 2 * (R_v * R_uv + Z_v * Z_uv + R * R_u)
    bu = iota - L_v
    bv = 1.0 + L_u
    Bu = phip * bu / sqrtg
    Bv = phip * bv / sqrtg
    Bu_u = phip * (-L_uv * sqrtg - bu * g_u) / sqrtg**2
    Bv_u = phip * (L_uu * sqrtg - bv * g_u) / sqrtg**2
    Bu_v = phip * (-L_vv * sqrtg - bu * g_v) / sqrtg**2
    Bv_v = phip * (L_uv * sqrtg - bv * g_v) / sqrtg**2
    B_u = guu * Bu + guv * Bv
    B_v = guv * Bu + gvv * Bv
    B_s_u = gsu_u * Bu + gsu * Bu_u + gsv_u * Bv + gsv * Bv_u
    B_s_v = gsu_v * Bu + gsu * Bu_v + gsv_v * Bv + gsv * Bv_v
    B_u_v = guu_v * Bu + guu * Bu_v + guv_v * Bv + guv * Bv_v
    B_v_u = guv_u * Bu + guv * Bu_u + gvv_u * Bv + gvv * Bv_u
    mu0Js = B_v_u - B_u_v
    B2 = Bu * B_u + Bv * B_v
    return {
        "Bu": Bu,
        "Bv": Bv,
        "B_u": B_u,
        "B_v": B_v,
        "B_s_u": B_s_u,
        "B_s_v": B_s_v,
        "mu0Js": mu0Js,
        "B2": B2,
    }


def residual_ref(w, j, u, vv, phip):
    """Float reference of the residual at node j, for choosing bounds."""
    qm = half_point(w, j - 1, j, j, u, vv, phip)  # h- = row j of the half grid
    qp = half_point(w, j, j + 1, j + 1, u, vv, phip)  # h+ = row j+1
    h = w.s_half[j + 1] - w.s_half[j]
    avg = lambda k: 0.5 * (qm[k] + qp[k])  # noqa: E731
    dif = lambda k: (qp[k] - qm[k]) / h  # noqa: E731
    s = w.s_full[j]
    pp = sum(k * a * s ** (k - 1) for k, a in enumerate(w.am) if k > 0)
    rs = (
        (avg("B_s_v") - dif("B_v")) * avg("Bv")
        - (dif("B_u") - avg("B_s_u")) * avg("Bu")
        - MU0 * pp
    )
    ru = -qp["mu0Js"] * qp["Bv"]
    rv_ = qp["mu0Js"] * qp["Bu"]
    return rs, ru, rv_, max(qm["B2"], qp["B2"])


def main():
    """Read the wout, choose the bounds, write the certificate."""
    ap = argparse.ArgumentParser()
    repo = pathlib.Path(__file__).resolve().parent.parent
    default_wout = (
        repo / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data" / "wout_solovev.nc"
    )
    ap.add_argument("wout", nargs="?", default=str(default_wout))
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument("--nu", type=int, default=8)
    ap.add_argument("--nv", type=int, default=4)
    ap.add_argument("--slack", type=float, default=1.5)
    ap.add_argument(
        "--prec",
        type=int,
        default=53,
        help="working precision of the interval arithmetic in bits",
    )
    a = ap.parse_args()
    if a.out is None:
        a.out = f"cert_{pathlib.Path(a.wout).stem.removeprefix('wout_')}.txt"

    w = Wout(a.wout)
    K = len(w.xm)
    phip = float(w.phips[1])
    nfp = 1
    if (w.xn != 0).any():
        nfp = int(np.gcd.reduce(np.abs(w.xn[w.xn != 0])))
    three_d = (w.xn != 0).any()
    nv = a.nv if three_d else 1

    # certified nodes: interior full-grid nodes with both neighbors off the
    # axis, evenly spread
    lo, hi = 2, w.ns - 2
    idx = np.unique(np.linspace(lo, hi, a.nodes).astype(int))
    # angles: exact doubles
    us = [float(2 * np.pi * k / a.nu) for k in range(a.nu)]
    vs = [float(2 * np.pi * k / (nfp * nv)) for k in range(nv)]
    angles = [(u, v) for u in us for v in vs]

    # choose eps from the float reference
    worst = np.zeros(3)
    scale = 0.0
    per_node = []
    for j in idx:
        mx = np.zeros(3)
        for u, v in angles:
            rs, ru, rv_, B2 = residual_ref(w, j, u, v, phip)
            mx = np.maximum(mx, np.abs([rs, ru, rv_]))
            scale = max(scale, abs(B2))
        per_node.append((j, w.s_full[j], mx))
        worst = np.maximum(worst, mx)
    # Floor: interval evaluation carries genuine rounding width, so a claimed
    # bound must sit above it. 1e-10 of the field-energy scale is far above
    # the enclosure width of the evaluation and far below any physical residual
    # of interest.
    eps = np.maximum(worst * a.slack, 1e-10 * scale)

    lines = []
    P = lines.append
    P("STELLAROCQ-CERT 2")
    P(f"PREC {a.prec}")
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
    P(f"NNODES {len(idx)}")
    for j in idx:
        P("NODE")
        P("S {} {}".format(*dyadic(w.s_full[j])))
        P(
            "SNODES "
            + " ".join("{} {}".format(*dyadic(x)) for x in w.s_full[j - 1 : j + 2])
        )
        P("SHALF " + " ".join("{} {}".format(*dyadic(x)) for x in w.s_half[j : j + 2]))
        P("IOTA " + " ".join("{} {}".format(*dyadic(x)) for x in w.iotas[j : j + 2]))
        for tag, M in (
            ("RNODES", w.rmnc[j - 1 : j + 2]),
            ("ZNODES", w.zmns[j - 1 : j + 2]),
            ("LHALF", w.lmns[j : j + 2]),
        ):
            P(tag)
            for row in M:
                P(" ".join("{} {}".format(*dyadic(x)) for x in row))
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")

    print(
        f"wrote {a.out}: {len(idx)} nodes x {len(angles)} angles = "
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
    for j, s, mx in per_node:
        print(f"  node {j:3d}  s={s:.4f}  |r|max = {mx[0]:.3e} {mx[1]:.3e} {mx[2]:.3e}")


if __name__ == "__main__":
    main()
