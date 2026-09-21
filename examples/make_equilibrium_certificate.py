"""Emit a Stellarocq certificate from a VMEC wout file.

The checker that validates the certificate is built from
https://github.com/CharlesCNorton/stellarocq; see
docs/proof_carrying_equilibria.md. The file written here follows FORMAT.md of
that repository, point certificates at version 6 and cell certificates at
version 7.

The certificate states, for a set of full-grid nodes and angles, that the
mu0-scaled ideal-MHD force residual of the equilibrium reconstructed from the
wout coefficients by VMEC's half-grid rule (fixed in theories/Physics.v of
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
  +8K   the antisymmetric R, Z and lambda blocks, when lasym
  then  scratch slots the checker fills with shared subexpressions

Both symmetry classes are read. The pressure may be a power series or a
two-power profile; gen/make_cert.py of Stellarocq reads the other
parameterizations.

With --cells the certificate instead claims its bounds over cells of angles, so
that a VALID verdict covers the continuum between the sampled angles and not
only the samples. The bounds of a cell certificate are written by the checker
itself:

  python make_equilibrium_certificate.py wout_X.nc cell_X.txt --cells --nu 8192
  stellarocq-check --tighten cell_X.txt cert_X.txt
  stellarocq-check cert_X.txt

Usage:  python make_equilibrium_certificate.py [wout_X.nc [cert_X.txt]] [--nodes 6] [--nu 8] [--nv 4]
Without arguments it certifies the shipped wout_solovev.nc into cert_solovev.txt.
"""

import argparse
import pathlib

import netCDF4
import numpy as np

MU0 = 4e-7 * np.pi

CERT_MAGIC = "STELLAROCQ-CERT 6"
CCERT_MAGIC = "STELLAROCQ-CCERT 7"

# Slots of the am block that scale with PRES_SCALE, per closed form.
AMPLITUDE_SLOTS = {"POWER": range(21), "TWOPOWER": (0,)}


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


# ----- pressure ------------------------------------------------------------


def classify_pressure(ptype, am):
    """The PROFILE line for a VMEC pmass_type, or a refusal."""
    if ptype in ("power_series", ""):
        return "POWER"
    if ptype == "two_power":
        exponents = []
        for x in am[1:3]:
            if float(x) != int(x) or float(x) < 0:
                msg = f"two_power needs nonnegative integral exponents, got {float(x)}"
                raise SystemExit(msg)
            exponents.append(int(x))
        return "TWOPOWER {} {}".format(*exponents)
    msg = (
        f"pressure parameterization {ptype!r} is not read by this example; "
        "see gen/make_cert.py of Stellarocq"
    )
    raise SystemExit(msg)


def _amj(am, j):
    """Slot j of the am array, which a wout may store short of 21 entries."""
    return float(am[j]) if j < len(am) else 0.0


def pvalue_ref(profile, am, s):
    """P(s) of the certified profile, in floating point, before PRES_SCALE."""
    parts = profile.split()
    if parts[0] == "POWER":
        return sum(_amj(am, j) * s**j for j in range(21))
    p, q = (int(x) for x in parts[1:])
    return _amj(am, 0) * (1.0 - s**p) ** q


def pprime_ref(profile, am, s):
    """Dp/ds of the certified profile, mirroring pprime of theories/Physics.v."""
    parts = profile.split()
    if parts[0] == "POWER":
        return sum(j * _amj(am, j) * s ** (j - 1) for j in range(1, 21))
    p, q = (int(x) for x in parts[1:])
    return -_amj(am, 0) * p * q * s ** max(p - 1, 0) * (1.0 - s**p) ** max(q - 1, 0)


def calibrate_pressure(w):
    """Put PRES_SCALE back into the coefficients, from the wout's own pressure.

    VMEC evaluates the profile at the half points, multiplies by PRES_SCALE and
    stores the result as `pres`, and does not store the scale. The ratio of
    `pres` at the first half point to the profile there is the scale, and the
    coefficients linear in it are multiplied by it, so that the certificate is
    about the pressure the equilibrium balances.
    """
    s1 = float(w.s_half[1])
    raw = pvalue_ref(w.profile, w.am, s1)
    if raw == 0.0:
        return 1.0
    scale = float(w.pres_half[1]) / raw
    if abs(scale - 1.0) > 1e-12:
        am = np.array(w.am, dtype=float)
        for j in AMPLITUDE_SLOTS[w.profile.split()[0]]:
            if j < len(am):
                am[j] *= scale
        w.am = am
    return scale


class Wout:
    """The wout fields the certificate needs."""

    def __init__(self, path):
        """Load the fields and classify the pressure parameterization."""
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
        self.pres_half = g("pres")
        self.lasym = "lasym__logical__" in v and bool(int(v["lasym__logical__"][:]))
        if self.lasym:
            self.rmns = g("rmns")
            self.zmnc = g("zmnc")
            self.lmnc = g("lmnc")
        ptype = v["pmass_type"][:].tobytes().decode().replace("\x00", "").strip()
        self.profile = classify_pressure(ptype, self.am)
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
    """B^u, B^v, B_u, B_v, d_u B_s, d_v B_s and mu0 sqrtg J^s at the half point.

    Carries both parities: R is a cosine series plus, when lasym, a sine
    series; Z and lambda are sine series plus cosine series.
    """
    m, n = w.xm, w.xn
    s_h = w.s_half[row_l]
    iota = float(w.iotas[row_l])
    ang = m * u - n * vv
    c = np.cos(ang)
    sn = np.sin(ang)

    def ser(cf, even):
        """Value and the five angular derivatives of one parity's series."""
        k0, k1, su, sv = (c, sn, -m, n) if even else (sn, c, m, -n)
        kernels = (k0, su * k1, sv * k1, -m * m * k0, m * n * k0, -n * n * k0)
        return np.array([float(np.dot(cf, k)) for k in kernels])

    def series(sym, anti, even):
        """A series and its radial derivative, each with both parities."""
        cf, cfs = half_coefs(w, j_in, j_out, s_h, sym)
        val, val_s = ser(cf, even), ser(cfs, even)
        if w.lasym:
            ca, cas = half_coefs(w, j_in, j_out, s_h, anti)
            val, val_s = val + ser(ca, not even), val_s + ser(cas, not even)
        return val, val_s

    (R, R_u, R_v, R_uu, R_uv, R_vv), (R_s, R_su, R_sv, *_) = series(
        w.rmnc, w.rmns if w.lasym else None, True
    )
    (_, Z_u, Z_v, Z_uu, Z_uv, Z_vv), (Z_s, Z_su, Z_sv, *_) = series(
        w.zmns, w.zmnc if w.lasym else None, False
    )
    lam = ser(w.lmns[row_l], False)
    if w.lasym:
        lam = lam + ser(w.lmnc[row_l], True)
    _, L_u, L_v, L_uu, L_uv, L_vv = lam

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
    pp = pprime_ref(w.profile, w.am, w.s_full[j])
    rs = (
        (avg("B_s_v") - dif("B_v")) * avg("Bv")
        - (dif("B_u") - avg("B_s_u")) * avg("Bu")
        - MU0 * pp
    )
    ru = -qp["mu0Js"] * qp["Bv"]
    rv_ = qp["mu0Js"] * qp["Bu"]
    return rs, ru, rv_, max(qm["B2"], qp["B2"])


# ----- the certificate file ------------------------------------------------

ANGLE_EXP = -50


def dyadic_at(x, e=ANGLE_EXP):
    """X on the fixed dyadic grid of step 2^e, as (mantissa, e).

    The angles have to share a fine exponent: the checker varies the mantissa
    of the angle slot, so one mantissa unit is 2^e radians, and taking whatever
    exponent the double happens to carry makes the unit meaningless (u = 0 has
    exponent 0, one unit of which is a radian).
    """
    return round(float(x) / 2.0**e), e


def tile(width, n, e, scale=1.0):
    """Centres and half-width of n abutting cells of total span >= width.

    Everything is in units of 2**e: cell k is centred at (2k+1)d with half-width
    d, so consecutive cells share an endpoint exactly and the run covers
    [0, 2nd], which is the tiling theories/Cover.v of Stellarocq reasons about.
    `scale` shrinks the half-width without moving the centres, which leaves
    gaps between the cells.
    """
    d = int(np.ceil(width / (2.0 * n) / 2.0**e))
    return [(2 * k + 1) * d for k in range(n)], max(1, round(d * scale))


def pairs(xs):
    """The dyadic pairs of a run of doubles, on one line."""
    return " ".join("{} {}".format(*dyadic(x)) for x in xs)


def header(a, w, magic, phip):
    """The lines a certificate opens with, up to the pressure coefficients."""
    lines = [
        magic,
        f"PREC {a.prec}",
        f"LASYM {1 if w.lasym else 0}",
        f"PROFILE {w.profile}",
        "SLOTS 1 2",
        "OUTPUT residual",
        f"MODES {len(w.xm)}",
    ]
    lines += [f"{m} {n}" for m, n in zip(w.xm, w.xn, strict=True)]
    lines += ["PHIP " + pairs([phip]), "AM 21"]
    lines += [pairs([_amj(w.am, j)]) for j in range(21)]
    return lines


def node_block(w, j):
    """The lines of node j: its radii, iota and coefficient rows."""
    lines = [
        "NODE",
        "S " + pairs([w.s_full[j]]),
        "SNODES " + pairs(w.s_full[j - 1 : j + 2]),
        "SHALF " + pairs(w.s_half[j : j + 2]),
        "IOTA " + pairs(w.iotas[j : j + 2]),
    ]
    blocks = [
        ("RNODES", w.rmnc[j - 1 : j + 2]),
        ("ZNODES", w.zmns[j - 1 : j + 2]),
        ("LHALF", w.lmns[j : j + 2]),
    ]
    if w.lasym:
        blocks += [
            ("RNODES_A", w.rmns[j - 1 : j + 2]),
            ("ZNODES_A", w.zmnc[j - 1 : j + 2]),
            ("LHALF_A", w.lmnc[j : j + 2]),
        ]
    for tag, rows in blocks:
        lines.append(tag)
        lines += [pairs(row) for row in rows]
    return lines


def write_ccert(a, w, phip, idx, nu, vs, three_d):
    """Write a cell certificate: every angle of every cell is covered.

    The poloidal cells tile [0, 2 pi) exactly. An axisymmetric equilibrium has
    every n zero, so the v derivative of the residual encloses to zero and one
    cell covers the whole toroidal angle. A three-dimensional one needs the v
    extent resolved as finely as the u extent, which squares the cell count, so
    its cells are u segments at the toroidal angles vs instead. The half-widths
    are carried in units of the mantissa of the centre angle, which is what the
    checker varies, and the bounds are left to "stellarocq-check --tighten".
    """
    ums, du = tile(2.0 * np.pi, nu, ANGLE_EXP, a.wscale)
    if three_d:
        vms, dv = [dyadic_at(v)[0] for v in vs], 0
    else:
        vms, dv = tile(2.0 * np.pi, 1, ANGLE_EXP, a.wscale)
    angles = [(mu, mv) for mu in ums for mv in vms]

    lines = header(a, w, CCERT_MAGIC, phip)
    lines.append(f"NANGLES {len(angles)}")
    lines += [f"{mu} {ANGLE_EXP} {mv} {ANGLE_EXP} {du} {dv}" for mu, mv in angles]
    lines.append(f"NNODES {len(idx)}")
    for j in idx:
        lines += node_block(w, j)
        lines.append(f"CELLS {len(angles)}")
        # The bounds are placeholders that "stellarocq-check --tighten" replaces
        # with the enclosures the checker computes, because the width of an
        # interval enclosure of a cancelling expression is a property of the
        # arithmetic and cannot be predicted from a float sample of the function.
        lines += ["1 0 1 0 1 0 4 0"] * (3 * len(angles))
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
    print(
        f"wrote {a.out}: {len(idx)} nodes x {len(angles)} cells = "
        f"{len(idx) * len(angles)} cells, K={len(w.xm)}"
    )
    cover = (
        f"the cells tile u in [0, 2 pi) at each of {len(vs)} toroidal angles"
        if three_d
        else "the cells tile the whole angular torus"
    )
    unit = 2.0**ANGLE_EXP
    print(f"cell half-widths: u {du * unit:.4e} rad, v {dv * unit:.4e} rad; {cover}")
    print("run 'stellarocq-check --tighten' on it to set the bounds, then check it")


def main():
    """Read the wout, choose the bounds, write the certificate."""
    ap = argparse.ArgumentParser()
    repo = pathlib.Path(__file__).resolve().parent.parent
    default_wout = (
        repo / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data" / "wout_solovev.nc"
    )
    ap.add_argument("wout", nargs="?", default=str(default_wout))
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument(
        "--cells",
        action="store_true",
        help="emit a cell certificate: the bound holds at every angle of each "
        "cell, not only at its centre",
    )
    ap.add_argument(
        "--wscale",
        type=float,
        default=1.0,
        help="scale the cell half-widths. 1 makes the cells tile the angular "
        "torus exactly; smaller leaves gaps and is for diagnosis only.",
    )
    ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument(
        "--node",
        type=int,
        default=None,
        help="certify this single full-grid node instead of a spread of them",
    )
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
    pres_scale = calibrate_pressure(w)
    if abs(pres_scale - 1.0) > 1e-12:
        print(f"pressure scaled by {pres_scale:.9f}, read off the wout's own pres")
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
    if a.node is not None:
        if not lo <= a.node <= hi:
            msg = f"--node must lie in [{lo}, {hi}]"
            raise SystemExit(msg)
        idx = np.array([a.node])
    else:
        idx = np.unique(np.linspace(lo, hi, a.nodes).astype(int))
    # angles: exact doubles
    us = [float(2 * np.pi * k / a.nu) for k in range(a.nu)]
    vs = [float(2 * np.pi * k / (nfp * nv)) for k in range(nv)]
    angles = [(u, v) for u in us for v in vs]

    if a.cells:
        # the field-energy scale the bounds are read against, from a coarse
        # subset of the angles: the cell bounds themselves come from the
        # checker, so the reference is needed for scale only
        coarse = angles[:: max(1, len(angles) // 64)]
        scale = max(
            abs(residual_ref(w, j, u, v, phip)[3]) for j in idx for u, v in coarse
        )
        write_ccert(a, w, phip, idx, a.nu, vs, three_d)
        print(f"reference B^2 scale {scale:.3e}")
        return

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

    lines = header(a, w, CERT_MAGIC, phip)
    lines += [
        f"{tag} " + pairs([e])
        for tag, e in zip(("EPS_S", "EPS_U", "EPS_V"), eps, strict=True)
    ]
    lines.append(f"NANGLES {len(angles)}")
    lines += [pairs([u, v]) for u, v in angles]
    lines.append(f"NNODES {len(idx)}")
    for j in idx:
        lines += node_block(w, j)
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
