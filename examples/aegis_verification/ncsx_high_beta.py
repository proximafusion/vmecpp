# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Adjudicate the free-boundary vacuum solvers on a 3D high-beta NCSX case.

VMEC's free-boundary force only sees the scalar vacuum pressure |B_ext|^2/2, so at
high beta different vacuum solvers (NESTOR, AEGIS) can drive the boundary to
different magnetic axes while each reports a small delbsq against its OWN field.
delbsq therefore does not by itself say which equilibrium is correct.

This walks a pressure ladder up to the target (hot-restarting each step), then
scores each converged boundary against the golden BIEST exterior field: the
pressure-balance residual (|B_ext|^2 - |B_in|^2)/2 and the tangency residual
n.B/|B| on the LCFS (both should vanish at a true equilibrium, where p(edge)=0).
The self-consistent, golden-tangent boundary is the correct one.

usage: ncsx_high_beta.py [nestor|aegis] [target_pres_scale] [mpol]

Needs the NCSX mgrid (run ncsx_mgrid.py first), examples/aegis_virtual_casing.py,
and the compiled biest_driver (from biest_driver.cpp) next to this script for the
golden columns; without the driver the pbal_biest / nB_biest columns are NaN.
"""

import json
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
EX = _ROOT / "examples"
DRIVER = _HERE.parent / "biest_driver"  # compiled from biest_driver.cpp here
WORK = _HERE.parent
MGRID = str(_HERE.parent / "mgrid_ncsx.nc")
SOLVER = sys.argv[1] if len(sys.argv) > 1 else "aegis"
TARGET = float(sys.argv[2]) if len(sys.argv) > 2 else 60000.0
MPOL = int(sys.argv[3]) if len(sys.argv) > 3 else 6
phiedge = 0.30
if SOLVER == "aegis":
    os.environ["VMECPP_AEGIS"] = "1"
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
    for s in range(0, n, 256):
        e = min(s + 256, n)
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
    infile = WORK / f"adj_{tag}.txt"
    og = WORK / f"adjg_{tag}.txt"
    ogd = WORK / f"adjgold_{tag}.txt"
    with open(infile, "w") as fh:
        fh.write(f"{Nt} {Np}\n")
        fh.writelines(
            f"{X[k, 0]:.16e} {X[k, 1]:.16e} {X[k, 2]:.16e} "
            f"{Bp[k, 0]:.16e} {Bp[k, 1]:.16e} {Bp[k, 2]:.16e}\n"
            for k in range(len(X))
        )
    r = subprocess.run([str(DRIVER), str(infile), str(og), str(ogd)],
                       capture_output=True, text=True, check=False)
    out = np.loadtxt(ogd) if r.returncode == 0 else None
    for f in (infile, og, ogd):
        f.unlink(missing_ok=True)
    return out


def mk(ps, multigrid):
    inp = {
        "lasym": False, "nfp": 3, "mpol": MPOL, "ntor": 6, "ntheta": 0, "nzeta": 48,
        "ns_array": [15, 25] if multigrid else [25],
        "ftol_array": [1e-8, 1e-10] if multigrid else [1e-10],
        "niter_array": [2000, 6000] if multigrid else [8000],
        "delt": 0.7, "tcon0": 1.0, "aphi": [1.0], "phiedge": phiedge, "nstep": 200,
        "pmass_type": "power_series", "am": [1.0, -1.0], "pres_scale": ps,
        "gamma": 0.0, "spres_ped": 1.0, "ncurr": 1, "pcurr_type": "power_series",
        "ac": [0.0], "curtor": 0.0, "bloat": 1.0, "lfreeb": True,
        "mgrid_file": MGRID, "extcur": [1.0], "nvacskip": 6, "lforbal": False,
        "raxis_c": [1.44, 0, 0, 0, 0, 0, 0], "zaxis_s": [0, 0, 0, 0, 0, 0, 0],
        "rbc": [{"n": 0, "m": 0, "value": 1.44}, {"n": 0, "m": 1, "value": 0.20}],
        "zbs": [{"n": 0, "m": 1, "value": 0.20}],
    }
    jp = WORK / f"ncsx_adj_{int(ps)}.json"
    jp.write_text(json.dumps(inp))
    v = vmecpp.VmecInput.from_file(str(jp))
    v.mgrid_file = MGRID
    return v


def run(v, restart):
    sv1, sv2 = os.dup(1), os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(dn, 2)
    try:
        out = vmecpp.run(v, max_threads=1, verbose=False, restart_from=restart)
    finally:
        os.dup2(sv1, 1)
        os.dup2(sv2, 2)
        os.close(dn)
        os.close(sv1)
        os.close(sv2)
    return out


def evaluate(w):
    beta = float(np.asarray(w.betatotal).ravel()[0])
    rax = float(np.asarray(w.raxis_cc).ravel()[0])
    db = np.asarray(w.delbsq)
    db = db[db > 0]
    delbsq = float(db[-1]) if db.size else float("nan")
    nu, nv = 48, 96
    lcfs, external, _ = A.build(w, MGRID, np.array([1.0]), nu=nu, nv=nv)
    X = flat_tmajor(lcfs.X)
    nhat = flat_tmajor(lcfs.nhat)
    dA = flat_tmajor(lcfs.dA.reshape(nu, nv, 1))[:, 0]
    Bin = flat_tmajor(lcfs.B)
    Bcoil = flat_tmajor(external(lcfs.X))
    Bplasma = Bin - Bcoil
    Ba = aegis_singsub(X, Bplasma, nhat, dA) + Bcoil
    Bb = run_biest(X, Bplasma, nv, nu, SOLVER)
    p_in = 0.5 * np.sum(Bin**2, 1)
    den = p_in.mean()
    res_a = float(np.abs(0.5 * np.sum(Ba**2, 1) - p_in).mean() / den)
    nrmA = float((np.abs(np.sum(nhat * Ba, 1)) / np.linalg.norm(Ba, axis=1)).mean())
    if Bb is not None:
        Bb = Bb + Bcoil
        res_b = float(np.abs(0.5 * np.sum(Bb**2, 1) - p_in).mean() / den)
        cosab = float(np.sum(Ba * Bb) / (np.linalg.norm(Ba) * np.linalg.norm(Bb)))
        nrmB = float((np.abs(np.sum(nhat * Bb, 1)) / np.linalg.norm(Bb, axis=1)).mean())
    else:
        res_b, cosab, nrmB = float("nan"), float("nan"), float("nan")
    print(f"RES[{SOLVER}] beta={beta:.4e} raxis={rax:.4f} delbsq={delbsq:.3e} "
          f"pbal_aegis={res_a:.4e} pbal_biest={res_b:.4e} "
          f"nB_aegis={nrmA:.4e} nB_biest={nrmB:.4e} cos={cosab:.5f}", flush=True)


ladder = [p for p in [40000, 50000, 60000, 70000, 80000] if p <= TARGET]
prev = None
for i, ps in enumerate(ladder):
    prev = run(mk(ps, multigrid=(i == 0)), restart=prev)
    evaluate(prev.wout)
print("# ADJ_DONE", flush=True)
