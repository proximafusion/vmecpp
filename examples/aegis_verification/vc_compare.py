# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Validate the AEGIS on-surface virtual-casing quadrature against BIEST (the golden,
near-machine-precision reference) in isolation, on the 3D cth_like LCFS.

Both solvers evaluate the identical operator B_ext = BiotSavart(n x B) +
Laplace(n.B) + B/2; they differ only in quadrature (BIEST: high-order singular;
AEGIS: punctured-trapezoidal principal value). Feeding both the same field on the
same grid isolates the quadrature error.

Two cases:
  [1] Analytic interior dipole: its field on the LCFS has a known exterior limit
      (the field itself, since the source is strictly interior), so |VC(B)-B|
      measures each solver's absolute quadrature error on the real 3D geometry.
  [2] Real equilibrium plasma field B_plasma from a converged cth_like solve:
      no closed form, so BIEST is the reference and |B_aegis-B_biest|/|B_biest|
      is AEGIS's error on the physical field.

Reuses examples/aegis_virtual_casing.py for the LCFS geometry and coil/axis field.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = next(
    (p for p in _HERE.parents if (p / "src/vmecpp/cpp/vmecpp/test_data").is_dir()),
    _HERE.parents[2],
)
TD = _ROOT / "src/vmecpp/cpp/vmecpp/test_data"
EX = _ROOT / "examples"
DRIVER = _HERE.parent / "biest_driver"  # compiled from biest_driver.cpp here
WORK = _HERE.parent
MU0 = 4e-7 * np.pi

sys.path.insert(0, str(EX))

_s1, _s2 = os.dup(1), os.dup(2)
_dn = os.open(os.devnull, os.O_WRONLY)
os.dup2(_dn, 1)
os.dup2(_dn, 2)
import aegis_virtual_casing as A  # noqa: E402

import vmecpp  # noqa: E402

os.dup2(_s1, 1)
os.dup2(_s2, 2)
os.close(_dn)


class silence:
    def __enter__(self):
        self.a, self.b = os.dup(1), os.dup(2)
        self.dn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.dn, 1)
        os.dup2(self.dn, 2)

    def __exit__(self, *a):
        os.dup2(self.a, 1)
        os.dup2(self.b, 2)
        os.close(self.dn)
        os.close(self.a)
        os.close(self.b)


def aegis_singsub(X, Bp, nhat, dA):
    """AEGIS's OnSurfaceSingSub in numpy, matching aegis.cc exactly: punctured
    principal-value Biot-Savart sum + the analytic jump 0.5*(sigma*n + K x n).

    X,Bp,nhat: (N,3); dA: (N,). Returns B_ext (N,3).
    """
    n = len(X)
    K = np.cross(nhat, Bp)
    sigma = np.sum(nhat * Bp, axis=1)
    Bext = np.empty((n, 3))
    chunk = 256
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = X[s:e, None, :] - X[None, :, :]  # (c,N,3)
        d2 = np.sum(d * d, axis=2)  # (c,N)
        mask = d2 < 1e-24
        d2m = np.where(mask, 1.0, d2)
        inv = np.where(mask, 0.0, 1.0 / (np.sqrt(d2m) * d2m))
        term = np.cross(K[None, :, :], d) + sigma[None, :, None] * d
        pv = np.sum(term * (inv * dA[None, :])[:, :, None], axis=1) / (4 * np.pi)
        n0 = nhat[s:e]
        jump = 0.5 * (sigma[s:e, None] * n0 + np.cross(K[s:e], n0))
        Bext[s:e] = pv + jump
    return Bext


def run_biest(X, Bp, Nt, Np, tag):
    """Write the grid+field, run the BIEST driver, read back B_ext.

    X,Bp indexed flat in t-major (iv), p-minor (iu) order = the driver's read order.
    """
    if not DRIVER.exists():
        return None  # BIEST driver not built; the analytic check in [1] stands alone
    infile = WORK / f"grid_{tag}.txt"
    outfile = WORK / f"bext_{tag}.txt"
    with open(infile, "w") as fh:
        fh.write(f"{Nt} {Np}\n")
        fh.writelines(
            f"{X[k, 0]:.16e} {X[k, 1]:.16e} {X[k, 2]:.16e} "
            f"{Bp[k, 0]:.16e} {Bp[k, 1]:.16e} {Bp[k, 2]:.16e}\n"
            for k in range(len(X))
        )
    r = subprocess.run(
        [str(DRIVER), str(infile), str(outfile)],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        print("BIEST driver failed:", r.stderr[-400:], flush=True)
        return None
    return np.loadtxt(outfile)


def flat_tmajor(a):
    """(nu, nv, ...) with [iu, iv] -> flat index iv*nu + iu (t-major, p-minor)."""
    return np.ascontiguousarray(np.swapaxes(a, 0, 1)).reshape(-1, *a.shape[2:])


def relerr(Ba, Bb):
    """RMS relative error over the surface: ||Ba-Bb||_2 / ||Bb||_2."""
    num = np.sqrt(np.mean(np.sum((Ba - Bb) ** 2, axis=1)))
    den = np.sqrt(np.mean(np.sum(Bb**2, axis=1)))
    return num / den


def build_grid(w, mgrid, extcur, nu, nv):
    lcfs, external, _ = A.build(w, mgrid, extcur, nu=nu, nv=nv)
    X = flat_tmajor(lcfs.X)
    nhat = flat_tmajor(lcfs.nhat)
    dA = flat_tmajor(lcfs.dA.reshape(nu, nv, 1))[:, 0]
    Bcoil = flat_tmajor(external(lcfs.X))
    Bplasma = flat_tmajor(lcfs.B) - Bcoil
    return X, nhat, dA, Bcoil, Bplasma


def dipole_field(X, r0, m):
    dd = X - r0[None, :]
    rr = np.linalg.norm(dd, axis=1)
    return (MU0 / (4 * np.pi)) * (
        3 * (dd * (dd @ m)[:, None]) / rr[:, None] ** 5 - m[None, :] / rr[:, None] ** 3
    )


def main():
    mpol = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    inp = vmecpp.VmecInput.from_file(str(TD / "cth_like_free_bdy.json"))
    mgrid = str(TD / "mgrid_cth_like.nc")
    inp.mgrid_file = mgrid
    inp.mpol = mpol
    inp.ntor = 4
    inp.ns_array = np.array([25], dtype=np.int64)
    inp.ftol_array = np.array([1e-11])
    inp.niter_array = np.array([8000], dtype=np.int64)
    extcur = np.asarray(inp.extcur, float)
    with silence():
        w = vmecpp.run(inp, max_threads=1, verbose=False).wout
    rax = float(np.asarray(w.raxis_cc).ravel()[0])
    print(f"# cth_like mpol={mpol} axis R~{rax:.4f}", flush=True)

    # An interior magnetic dipole placed on the magnetic axis at phi=0; its
    # field on the LCFS has the known exterior limit B_dip itself (interior
    # source), so |VC(B_dip) - B_dip| is the absolute quadrature error. The axis
    # R at phi=0 is sum_n raxis_cc[n] (not the n=0 mean raxis_cc[0]); using the
    # mean would misplace the dipole toward the LCFS and make the field
    # near-singular there.
    raxis_cc = np.asarray(w.raxis_cc).ravel()
    R0 = float(raxis_cc.sum())
    r0 = np.array([R0, 0.0, 0.0])
    m = np.array([0.3, 0.1, 1.0])
    # sanity: distance from the dipole to the nearest LCFS point (should be a
    # good fraction of the minor radius, not near zero).
    Xchk, _, _, _, _ = build_grid(w, mgrid, extcur, 32, 128)
    print(
        f"# dipole at R0={R0:.4f}; min dist to LCFS="
        f"{np.linalg.norm(Xchk - r0[None, :], axis=1).min():.4f}",
        flush=True,
    )

    # [1] AEGIS absolute error vs the analytic dipole, swept over resolution
    # including the coupling's coarse poloidal grid (nu ~ 16). No BIEST needed.
    print("# [1] AEGIS vs analytic dipole (abs quadrature error):", flush=True)
    for nu, nv in [(16, 128), (24, 128), (32, 128), (48, 128), (64, 160)]:
        X, nhat, dA, _, _ = build_grid(w, mgrid, extcur, nu, nv)
        Bdip = dipole_field(X, r0, m)
        Ba = aegis_singsub(X, Bdip, nhat, dA)
        print(
            f"    nu={nu:3d} nv={nv:3d} N={len(X):5d}  AEGIS rms_err={relerr(Ba, Bdip):.2e}",
            flush=True,
        )

    # [2] Confirm BIEST is golden vs the same (correctly placed) analytic dipole.
    print("# [2] BIEST vs analytic dipole (confirms golden reference):", flush=True)
    for nu, nv in [(48, 128)]:
        X, *_ = build_grid(w, mgrid, extcur, nu, nv)
        Bdip = dipole_field(X, r0, m)
        Bb = run_biest(X, Bdip, nv, nu, f"dip_{nu}_{nv}")
        if Bb is not None:
            ma = np.sqrt(np.mean(np.sum(Bb**2, 1)))
            md = np.sqrt(np.mean(np.sum(Bdip**2, 1)))
            cos = np.sum(Bb * Bdip) / (np.linalg.norm(Bb) * np.linalg.norm(Bdip))
            print(
                f"    nu={nu:3d} nv={nv:3d}  BIEST rms_err={relerr(Bb, Bdip):.2e}"
                f"  |Bb|/|Bdip|={ma / md:.3f}  cos={cos:.3f}",
                flush=True,
            )

    # [3] Real equilibrium plasma field: AEGIS vs BIEST (golden). Reported both
    # as a field error and as the vacuum-pressure |B_ext|^2/2 error that drives
    # the coupling (coil field added back).
    print("# [3] Real plasma field AEGIS vs BIEST (golden):", flush=True)
    for nu, nv in [(48, 128)]:
        X, nhat, dA, Bcoil, Bplasma = build_grid(w, mgrid, extcur, nu, nv)
        Ba = aegis_singsub(X, Bplasma, nhat, dA)
        Bb = run_biest(X, Bplasma, nv, nu, f"real_{nu}_{nv}")
        if Bb is not None:
            ma = np.sqrt(np.mean(np.sum(Ba**2, 1)))
            mb = np.sqrt(np.mean(np.sum(Bb**2, 1)))
            mp = np.sqrt(np.mean(np.sum(Bplasma**2, 1)))
            cos = np.sum(Ba * Bb) / (np.linalg.norm(Ba) * np.linalg.norm(Bb))
            pa = 0.5 * np.sum((Ba + Bcoil) ** 2, axis=1)
            pb = 0.5 * np.sum((Bb + Bcoil) ** 2, axis=1)
            dp = np.abs(pa - pb).mean() / np.abs(pb).mean()
            print(
                f"    nu={nu:3d} nv={nv:3d}  B_ext rms_err={relerr(Ba, Bb):.2e}  "
                f"cos={cos:.3f} |Ba|={ma:.3e} |Bb|={mb:.3e} |Bplasma_in|={mp:.3e}  "
                f"vac-pressure rel={dp:.2e}",
                flush=True,
            )
    print("# VC_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
