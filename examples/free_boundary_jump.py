# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Free-boundary balance on the boundary of the CTH-like case.

At a free-boundary equilibrium the total pressure p + B^2/2 is continuous across
the plasma boundary, where outside it is the magnetic pressure B_vac^2/2 of the
full vacuum field. With `return_vacuum_field` set, VMEC++ returns NESTOR's
vacuum field on its boundary grid in `threed1_free_boundary`, beside its own
extrapolated edge pressure, so the jump can be read from the output.

`edge_pressure` is the plasma side as Stellarocq
(https://github.com/CharlesCNorton/stellarocq) certifies it: VMEC's
parity-aware half-grid reconstruction of the two outermost half points, their
total pressure, and its linear extrapolation to s = 1, which on VMEC's grid is
the 3/2, -1/2 rule VMEC++ applies to its own edge pressure. Against it, the jump
at each point of NESTOR's grid is enclosed by Stellarocq's point certificates
at every point of that grid, over the segment between NESTOR's vacuum pressure
and the one BIEST's virtual casing gives for the same boundary and coils.

    python examples/free_boundary_jump.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vmecpp

MU0 = 4.0e-7 * np.pi

TEST_DATA_DIR = (
    Path(__file__).parent.parent / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
)


def cth_like(ns: int, ftol: float = 1.0e-14):
    """The CTH-like free-boundary case at radial resolution ns, ramped through coarser
    grids, with NESTOR's boundary vacuum field returned."""
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vi.return_vacuum_field = True
    steps = [x for x in (9, 17, 25, 33, 49, 65, 97, 129, 193) if x < ns] + [ns]
    vi.ns_array = np.array(steps)
    vi.ftol_array = np.full(len(steps), ftol)
    vi.niter_array = np.full(len(steps), 20000)
    # the coils' field on the grid of mgrid_cth_like.nc, which it reproduces
    params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", params
    )
    return vmecpp.run(vi, response, verbose=False)


def _half_coefs(c_in, c_out, s_a, s_b, s_h, odd):
    """VMEC's half-point value and radial derivative of every mode."""
    even_val = 0.5 * (c_in + c_out)
    even_der = (c_out - c_in) / (s_b - s_a)
    qa, qb = c_in / np.sqrt(s_a), c_out / np.sqrt(s_b)
    odd_val = np.sqrt(s_h) * 0.5 * (qa + qb)
    odd_der = np.sqrt(s_h) * (qb - qa) / (s_b - s_a) + odd_val / (2.0 * s_h)
    return np.where(odd, odd_val, even_val), np.where(odd, odd_der, even_der)


def _half_total_pressure(wout, j_in, j_out, row, cosk, sink, m, n):
    """P + B^2/2, mu0-scaled, at the half point between nodes j_in and j_out (half-grid
    row `row`) on a grid of angles."""
    h = 1.0 / (wout.ns - 1)
    s_a, s_b, s_h = j_in * h, j_out * h, (row - 0.5) * h
    odd = (m % 2) == 1
    rmnc, zmns, lmns = (
        np.asarray(x, dtype=float) for x in (wout.rmnc, wout.zmns, wout.lmns)
    )
    cR, cRs = _half_coefs(rmnc[:, j_in], rmnc[:, j_out], s_a, s_b, s_h, odd)
    cZ, cZs = _half_coefs(zmns[:, j_in], zmns[:, j_out], s_a, s_b, s_h, odd)
    cL = lmns[:, row]

    def ser(c, k):
        return np.sum(c[:, None, None] * k, 0)

    R, R_u, R_v, R_s = (
        ser(cR, cosk),
        ser(-m * cR, sink),
        ser(n * cR, sink),
        ser(cRs, cosk),
    )
    Z_u, Z_v, Z_s = ser(m * cZ, cosk), ser(-n * cZ, cosk), ser(cZs, sink)
    L_u, L_v = ser(m * cL, cosk), ser(-n * cL, cosk)
    sqrtg = R * (R_u * Z_s - R_s * Z_u)
    guu = R_u**2 + Z_u**2
    guv = R_u * R_v + Z_u * Z_v
    gvv = R_v**2 + Z_v**2 + R**2
    bu = float(wout.iotas[row]) - L_v
    bv = 1.0 + L_u
    phip = float(wout.phips[1])
    b2 = phip**2 * (bu * (guu * bu + guv * bv) + bv * (guv * bu + gvv * bv)) / sqrtg**2
    return 0.5 * b2 + MU0 * float(wout.pres[row])


def edge_pressure(wout, theta, zeta) -> np.ndarray:
    """The plasma-side total pressure at s = 1 on the angles theta x zeta, as an array
    indexed [zeta, theta]."""
    if wout.lasym:
        msg = "the reconstruction here is the stellarator-symmetric one"
        raise ValueError(msg)
    m = np.asarray(wout.xm, dtype=float)
    n = np.asarray(wout.xn, dtype=float)
    ang = (
        m[:, None, None] * np.asarray(theta)[None, None, :]
        - n[:, None, None] * np.asarray(zeta)[None, :, None]
    )
    cosk, sink = np.cos(ang), np.sin(ang)
    ns = wout.ns
    tm = _half_total_pressure(wout, ns - 3, ns - 2, ns - 2, cosk, sink, m, n)
    tp = _half_total_pressure(wout, ns - 2, ns - 1, ns - 1, cosk, sink, m, n)
    h = 1.0 / (ns - 1)
    return tp + (1.0 - (ns - 1.5) * h) * (tp - tm) / h


def boundary_grid(fb):
    """NESTOR's boundary angles: the poloidal half range of a symmetric run and the
    toroidal angles of one field period."""
    nth = fb.bsqvacf.shape[1]
    theta = 2.0 * np.pi * np.arange(nth) / (2 * (nth - 1))
    zeta = np.asarray(fb.phib)[:, 0]
    return theta, zeta


def vacuum_interpolant(fb, nfp: int):
    """NESTOR's vacuum pressure between its grid points: the grid over the poloidal half
    range extended by stellarator symmetry, P(-u, -v) = P(u, v), and the trigonometric
    interpolant of the full grid, the Nyquist frequency of an even grid carried as a
    cosine.

    It equals the grid values at the grid.
    """
    P = np.asarray(fb.bsqvacf, dtype=float)
    nzeta, nth = P.shape
    nu = 2 * (nth - 1)
    full = np.zeros((nzeta, nu))
    full[:, :nth] = P
    for col in range(nth, nu):
        full[:, col] = P[(-np.arange(nzeta)) % nzeta, nu - col]
    C = np.fft.fft2(full) / (nzeta * nu)

    def freqs(N):
        f = np.fft.fftfreq(N, 1.0 / N)
        w = np.ones(N)
        if N % 2 == 0:
            w[N // 2] = 0.5  # the Nyquist term, split between +N/2 and -N/2
        return f, w

    fk, wk = freqs(nzeta)
    fl, wl = freqs(nu)

    def at(theta, zeta):
        th = np.asarray(theta, dtype=float)[None, :]
        ze = np.asarray(zeta, dtype=float)[:, None]
        out = np.zeros((ze.shape[0], th.shape[1]))
        for a in range(nzeta):
            for b in range(nu):
                terms = [(fk[a], fl[b])]
                if wk[a] == 0.5:
                    terms.append((-fk[a], fl[b]))
                if wl[b] == 0.5:
                    terms = terms + [(kz, -ku) for kz, ku in terms]
                scale = (1.0 if wk[a] == 1.0 else 0.5) * (1.0 if wl[b] == 1.0 else 0.5)
                for kz, ku in terms:
                    out += scale * np.real(
                        C[a, b] * np.exp(1j * (ku * th + kz * nfp * ze))
                    )
        return out

    return at


def jump_between(out, refine: int = 4) -> np.ndarray:
    """The jump against NESTOR's interpolated vacuum pressure on a grid refine times
    finer than NESTOR's in each angle, over the full poloidal range."""
    fb = out.threed1_free_boundary
    nzeta, nth = np.asarray(fb.bsqvacf).shape
    nfp = int(out.wout.nfp)
    theta = 2.0 * np.pi * np.arange(refine * 2 * (nth - 1)) / (refine * 2 * (nth - 1))
    zeta = 2.0 * np.pi * np.arange(refine * nzeta) / (refine * nzeta * nfp)
    return edge_pressure(out.wout, theta, zeta) - vacuum_interpolant(fb, nfp)(
        theta, zeta
    )


def jump(out) -> dict:
    """The jump on NESTOR's grid: against the reconstruction and against VMEC++'s own
    extrapolated edge pressure."""
    fb = out.threed1_free_boundary
    theta, zeta = boundary_grid(fb)
    te = edge_pressure(out.wout, theta, zeta)
    return {
        "reconstruction": te - np.asarray(fb.bsqvacf),
        "vmecpp": np.asarray(fb.bsqmhdf) - np.asarray(fb.bsqvacf),
        "edge_difference": te - np.asarray(fb.bsqmhdf),
    }


def main():
    prev = None
    for ns in (25, 49, 97):
        j = jump(cth_like(ns))
        big = float(np.abs(j["reconstruction"]).max())
        fall = "" if prev is None else f"  (fell by {prev - big:.2e})"
        print(
            f"ns={ns:4d}  max |jump| {big:.6e}{fall}  VMEC++'s own {np.abs(j['vmecpp']).max():.6e}"
            f"  edge difference {np.abs(j['edge_difference']).max():.1e}"
        )
        prev = big


if __name__ == "__main__":
    main()
