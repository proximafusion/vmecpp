# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""On-surface accuracy of AEGIS versus BIEST on a balanced grid (issue #628).

The physical surface current n x (B_out - B_in) at the LCFS is masked on the
axisymmetric coupling because the pressure-jump magnitude excess hides in the
residual normal-field error n.B_ext rather than the tangential jump, and n.B_ext
floors near 1.7e-3 (see surface_current.py, floor_diag.py, ntor_aspect.py). This
asks whether AEGIS's punctured-trapezoidal principal-value quadrature is itself
the limit, by comparing it against BIEST's high-order singular quadrature (the
golden reference of vc_compare.py) on the identical operator B_ext =
BiotSavart(n x B) + Laplace(n.B) + B/2, on a converged free-boundary Solovev
equilibrium sampled on a balanced isotropic grid (nu = nv):

  - AEGIS: punctured-trapezoidal principal-value quadrature + analytic jump.
  - BIEST: high-order singular quadrature.

On a balanced grid AEGIS converges to BIEST's floor (n.B_ext ~ 1.5e-4, the clean
equilibrium's own normal-field level) as the grid refines, matching it to within
a factor of order one. So AEGIS's quadrature is not the limit; the coupling's
1.7e-3 floor is the aspect ratio of its axisymmetric source grid (poloidal
nThetaEff ~ 16 against toroidal 256), which ntor_aspect.py isolates. Requires
biest_driver built against a BIEST checkout alongside this file.
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
DRIVER = _HERE.parent / "biest_driver"
WORK = _HERE.parent
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


def flat_tmajor(a):
    return np.ascontiguousarray(np.swapaxes(a, 0, 1)).reshape(-1, *a.shape[2:])


def aegis_singsub(X, Bp, nhat, dA):
    n = len(X)
    K = np.cross(nhat, Bp)
    sigma = np.sum(nhat * Bp, axis=1)
    Bext = np.empty((n, 3))
    chunk = 256
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = X[s:e, None, :] - X[None, :, :]
        d2 = np.sum(d * d, axis=2)
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
    if not DRIVER.exists():
        return None
    infile = WORK / f"acc_grid_{tag}.txt"
    out_general = WORK / f"acc_gen_{tag}.txt"
    out_golden = WORK / f"acc_gold_{tag}.txt"
    with open(infile, "w") as fh:
        fh.write(f"{Nt} {Np}\n")
        fh.writelines(
            f"{X[k, 0]:.16e} {X[k, 1]:.16e} {X[k, 2]:.16e} "
            f"{Bp[k, 0]:.16e} {Bp[k, 1]:.16e} {Bp[k, 2]:.16e}\n"
            for k in range(len(X))
        )
    r = subprocess.run(
        [str(DRIVER), str(infile), str(out_general), str(out_golden)],
        capture_output=True,
        text=True,
        check=False,
    )
    ok = r.returncode == 0
    out = np.loadtxt(out_golden) if ok else None
    for f in (infile, out_general, out_golden):
        f.unlink(missing_ok=True)
    return out


def n_dot(Bext, nhat):
    return float(
        (np.abs(np.sum(nhat * Bext, axis=1)) / np.linalg.norm(Bext, axis=1)).mean()
    )


def main():
    inp = vmecpp.VmecInput.from_file(str(TD / "solovev_free_bdy.json"))
    mgrid = str(TD / "mgrid_solovev.nc")
    inp.mgrid_file = mgrid
    inp.mpol = 12
    inp.ntor = 0
    inp.ns_array = np.array([51], dtype=np.int64)
    inp.ftol_array = np.array([1e-11])
    inp.niter_array = np.array([8000], dtype=np.int64)
    extcur = np.asarray(inp.extcur, float)
    _o1, _o2 = os.dup(1), os.dup(2)
    _d = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_d, 1)
    os.dup2(_d, 2)
    w = vmecpp.run(inp, max_threads=1, verbose=False).wout
    os.dup2(_o1, 1)
    os.dup2(_o2, 2)
    os.close(_d)

    print(
        "# free-boundary Solovev mpol=12 ns=51 converged; on-surface n.B_ext/|B_ext|",
        flush=True,
    )
    print("#  nu   nv     N   AEGIS n.B     BIEST n.B    ratio", flush=True)
    for nu, nv in [(32, 32), (48, 48), (64, 64), (96, 96)]:
        lcfs, external, _ = A.build(w, mgrid, extcur, nu=nu, nv=nv)
        X = flat_tmajor(lcfs.X)
        nhat = flat_tmajor(lcfs.nhat)
        dA = flat_tmajor(lcfs.dA.reshape(nu, nv, 1))[:, 0]
        Bin = flat_tmajor(lcfs.B)
        Bcoil = flat_tmajor(external(lcfs.X))
        Bplasma = Bin - Bcoil
        Ba = aegis_singsub(X, Bplasma, nhat, dA) + Bcoil
        an = n_dot(Ba, nhat)
        Bb = run_biest(X, Bplasma, nv, nu, f"n{nu}")
        if Bb is not None:
            bn = n_dot(Bb + Bcoil, nhat)
            print(
                f"  {nu:3d}  {nv:3d}  {len(X):5d}   {an:.4e}    {bn:.4e}   {an / bn:5.1f}",
                flush=True,
            )
        else:
            print(
                f"  {nu:3d}  {nv:3d}  {len(X):5d}   {an:.4e}    (no driver)", flush=True
            )
    print("# ONSURF_ACC_DONE", flush=True)


if __name__ == "__main__":
    main()
