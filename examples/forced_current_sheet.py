# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The forced current sheet of a rippled circular tokamak.

A circular tokamak whose boundary carries a (1,1) ripple drives, through
toroidal coupling, a (2,1) sideband at the surface where iota = 1/2. A field
with nested flux surfaces cannot balance that resonant force: the ideal
response is a current sheet on the rational surface. VMEC++ still converges at
every radial resolution, since its grid smooths the sheet, and what shows the
obstruction is the radial force residual of the converged state reconstructed
from its own half-grid quantities: its (2,1) harmonic on the rational surface
stays put under radial refinement, while every non-resonant harmonic falls
fourfold per doubling of ns, which is the order of the discretization.

`node_residual` is that reconstruction: VMEC's parity-aware half-grid rule
for R and Z, the half-grid lambda and iota, and the radial residual at a
full-grid node from the averages and centred differences of its two half
points,

    r_s = (d_v B_s - d_s B_v) B^v - (d_s B_u - d_u B_s) B^u - mu0 dp/ds,

the same rule as theories/Physics.v of Stellarocq
(https://github.com/CharlesCNorton/stellarocq), which certifies these
harmonics with interval arithmetic. `harmonic` is the equispaced sum of
r_s cos(m u - n v) over the torus with the weights (2 pi / nu)(2 pi / nv).

    python examples/forced_current_sheet.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vmecpp

MU0 = 4.0e-7 * np.pi

RIPPLE = (1, 1, 0.03)
IOTA = (0.9, -0.64)
MPOL = 8
NTOR = 4
S_RATIONAL = 0.625


def rippled_tokamak(ns: int, ftol: float = 1.0e-15):
    """The circular tokamak of the test data with a boundary ripple and an iota profile
    that puts iota = 1/2 at s = 0.625, ramped through coarser grids."""
    base = (
        Path(__file__).parent.parent
        / "src"
        / "vmecpp"
        / "cpp"
        / "vmecpp"
        / "test_data"
        / "circular_tokamak.json"
    )
    vi = vmecpp.VmecInput.from_file(base)
    vi.mpol, vi.ntor = MPOL, NTOR
    vi.ai = np.array(IOTA)
    vi.pmass_type = "power_series"
    vi.am = np.array([0.0, 0.0])
    vi.pres_scale = 1.0
    rbc = np.zeros((MPOL, 2 * NTOR + 1))
    zbs = np.zeros((MPOL, 2 * NTOR + 1))
    rbc[0, NTOR] = 6.0
    rbc[1, NTOR] = 2.0
    zbs[1, NTOR] = 2.0
    m, n, amp = RIPPLE
    rbc[m, NTOR + n] = amp
    zbs[m, NTOR + n] = amp
    vi.rbc, vi.zbs = rbc, zbs
    vi.raxis_c = np.zeros(NTOR + 1)
    vi.raxis_c[0] = 6.0
    vi.zaxis_s = np.zeros(NTOR + 1)
    steps = [x for x in (17, 33, 65, 129, 257, 513) if x < ns] + [ns]
    vi.ns_array = np.array(steps)
    vi.ftol_array = np.full(len(steps), ftol)
    vi.niter_array = np.full(len(steps), 60000)
    vi.nstep = 1000
    return vmecpp.run(vi, verbose=False).wout


def _pressure_slope(wout, s: float) -> float:
    """Dp/ds of a power-series pressure at s, in pascal, with the scale VMEC applied
    recovered from the stored half-grid pressure."""
    if str(wout.pmass_type).strip() not in ("power_series", ""):
        msg = f"pmass_type {wout.pmass_type!r} is not handled here"
        raise ValueError(msg)
    am = np.asarray(wout.am, dtype=float)
    h = 1.0 / (wout.ns - 1)
    s1 = 0.5 * h
    raw = float(np.polynomial.polynomial.polyval(s1, am))
    scale = float(wout.pres[1]) / raw if raw != 0.0 else 1.0
    return scale * float(
        np.polynomial.polynomial.polyval(s, np.polynomial.polynomial.polyder(am))
    )


def _half_coefs(c_in, c_out, s_a, s_b, s_h, odd):
    """VMEC's half-point value and radial derivative of every mode."""
    even_val = 0.5 * (c_in + c_out)
    even_der = (c_out - c_in) / (s_b - s_a)
    qa, qb = c_in / np.sqrt(s_a), c_out / np.sqrt(s_b)
    odd_val = np.sqrt(s_h) * 0.5 * (qa + qb)
    odd_der = np.sqrt(s_h) * (qb - qa) / (s_b - s_a) + odd_val / (2.0 * s_h)
    return np.where(odd, odd_val, even_val), np.where(odd, odd_der, even_der)


def _half_point(wout, j_in, j_out, row_l, cosk, sink, m, n, phip):
    """The field quantities of one half point on the whole grid of angles."""
    ns = wout.ns
    h = 1.0 / (ns - 1)
    s_a, s_b = j_in * h, j_out * h
    s_h = (row_l - 0.5) * h
    odd = (m % 2) == 1
    rmnc, zmns, lmns = (
        np.asarray(x, dtype=float) for x in (wout.rmnc, wout.zmns, wout.lmns)
    )
    cR, cRs = _half_coefs(rmnc[:, j_in], rmnc[:, j_out], s_a, s_b, s_h, odd)
    cZ, cZs = _half_coefs(zmns[:, j_in], zmns[:, j_out], s_a, s_b, s_h, odd)
    cL = lmns[:, row_l]
    iota = float(wout.iotas[row_l])
    mm = m[:, None, None]
    nn = n[:, None, None]

    def cos_series(c):
        c = c[:, None, None]
        return (
            np.sum(c * cosk, 0),
            np.sum(-mm * c * sink, 0),
            np.sum(nn * c * sink, 0),
            np.sum(-mm * mm * c * cosk, 0),
            np.sum(mm * nn * c * cosk, 0),
            np.sum(-nn * nn * c * cosk, 0),
        )

    def sin_series(c):
        c = c[:, None, None]
        return (
            np.sum(c * sink, 0),
            np.sum(mm * c * cosk, 0),
            np.sum(-nn * c * cosk, 0),
            np.sum(-mm * mm * c * sink, 0),
            np.sum(mm * nn * c * sink, 0),
            np.sum(-nn * nn * c * sink, 0),
        )

    R, R_u, R_v, R_uu, R_uv, R_vv = cos_series(cR)
    R_s, R_su, R_sv = cos_series(cRs)[:3]
    _, Z_u, Z_v, Z_uu, Z_uv, Z_vv = sin_series(cZ)
    Z_s, Z_su, Z_sv = sin_series(cZs)[:3]
    _, L_u, L_v, L_uu, L_uv, L_vv = sin_series(cL)
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
    bu = iota - L_v
    bv = 1.0 + L_u
    Bu = phip * bu / sqrtg
    Bv = phip * bv / sqrtg
    Bu_u = phip * (-L_uv * sqrtg - bu * g_u) / sqrtg**2
    Bv_u = phip * (L_uu * sqrtg - bv * g_u) / sqrtg**2
    Bu_v = phip * (-L_vv * sqrtg - bu * g_v) / sqrtg**2
    Bv_v = phip * (L_uv * sqrtg - bv * g_v) / sqrtg**2
    return {
        "Bu": Bu,
        "Bv": Bv,
        "B_u": guu * Bu + guv * Bv,
        "B_v": guv * Bu + gvv * Bv,
        "B_s_u": gsu_u * Bu + gsu * Bu_u + gsv_u * Bv + gsv * Bv_u,
        "B_s_v": gsu_v * Bu + gsu * Bu_v + gsv_v * Bv + gsv * Bv_v,
    }


def node_residual(wout, j: int, nu: int = 64, nv: int = 32) -> np.ndarray:
    """r_s at full-grid node j on the grid u = 2 pi a / nu, v = 2 pi b / nv."""
    if wout.lasym:
        msg = "the reconstruction here is the stellarator-symmetric one"
        raise ValueError(msg)
    m = np.asarray(wout.xm, dtype=float)
    n = np.asarray(wout.xn, dtype=float)
    u = 2.0 * np.pi * np.arange(nu) / nu
    v = 2.0 * np.pi * np.arange(nv) / nv
    ang = m[:, None, None] * u[None, :, None] - n[:, None, None] * v[None, None, :]
    cosk, sink = np.cos(ang), np.sin(ang)
    phip = float(wout.phips[1])
    qm = _half_point(wout, j - 1, j, j, cosk, sink, m, n, phip)
    qp = _half_point(wout, j, j + 1, j + 1, cosk, sink, m, n, phip)
    h = 1.0 / (wout.ns - 1)

    def avg(k):
        return 0.5 * (qm[k] + qp[k])

    def dif(k):
        return (qp[k] - qm[k]) / h

    pp = _pressure_slope(wout, j * h)
    return (
        (avg("B_s_v") - dif("B_v")) * avg("Bv")
        - (dif("B_u") - avg("B_s_u")) * avg("Bu")
        - MU0 * pp
    )


def harmonic(rs: np.ndarray, m: int, n: int) -> float:
    """The equispaced sum of rs cos(m u - n v) over the torus."""
    nu, nv = rs.shape
    u = 2.0 * np.pi * np.arange(nu) / nu
    v = 2.0 * np.pi * np.arange(nv) / nv
    k = np.cos(m * u[:, None] - n * v[None, :])
    return float(np.sum(rs * k)) * (2.0 * np.pi / nu) * (2.0 * np.pi / nv)


def main():
    modes = [(2, 1), (2, 0), (1, 0), (3, 1), (1, 1)]
    print(f"rippled circular tokamak, ripple {RIPPLE}, iota = {IOTA[0]} {IOTA[1]:+} s")
    print(f"harmonics of r_s on s = {S_RATIONAL}, where iota = 1/2")
    prev = None
    for ns in (65, 129, 257):
        wout = rippled_tokamak(ns)
        j = round((ns - 1) * S_RATIONAL)
        rs = node_residual(wout, j)
        vals = {mn: harmonic(rs, *mn) for mn in modes}
        cells = []
        for mn in modes:
            ratio = ""
            if prev is not None and vals[mn] != 0.0:
                ratio = f" x{abs(prev[mn] / vals[mn]):4.1f}"
            cells.append(f"{mn}: {vals[mn]:+.4e}{ratio}")
        print(f"ns={ns:4d}  " + "  ".join(cells))
        prev = vals


if __name__ == "__main__":
    main()
