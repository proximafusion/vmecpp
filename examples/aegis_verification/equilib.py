#!/usr/bin/env python
"""Generic single-case equilibrium invariant reporter for the AEGIS verification. Runs
one free- or fixed-boundary VMEC++ case and prints beta, magnetic-axis major radius,
iota at axis and edge, MHD energy wb, and delbsq. Used to compare AEGIS against NESTOR
on the geometries where NESTOR is an accurate reference (the axisymmetric Solovev
tokamak) and to anchor the 3D cth_like results.

Usage: equilib.py <config> <mode> <pscale> <mpol> <ns>
  config: cth_free | cth_fixed | solovev_free
  mode:   nestor | aegis
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

config = sys.argv[1] if len(sys.argv) > 1 else "solovev_free"
mode = sys.argv[2] if len(sys.argv) > 2 else "nestor"
pscale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
mpol = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ns = int(sys.argv[5]) if len(sys.argv) > 5 else 0
if mode == "aegis":
    os.environ["VMECPP_AEGIS"] = "1"

_s1, _s2 = os.dup(1), os.dup(2)
_dn = os.open(os.devnull, os.O_WRONLY)
os.dup2(_dn, 1)
os.dup2(_dn, 2)
import vmecpp  # noqa: E402

os.dup2(_s1, 1)
os.dup2(_s2, 2)
os.close(_dn)

TD = next(
    (
        p / "src/vmecpp/cpp/vmecpp/test_data"
        for p in Path(__file__).resolve().parents
        if (p / "src/vmecpp/cpp/vmecpp/test_data").is_dir()
    ),
    Path("src/vmecpp/cpp/vmecpp/test_data"),
)
CFG = {
    "cth_free": ("cth_like_free_bdy.json", "mgrid_cth_like.nc", 4, 25),
    "cth_fixed": ("cth_like_fixed_bdy.json", None, 4, 25),
    "solovev_free": ("solovev_free_bdy.json", "mgrid_solovev.nc", 0, 51),
}


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


def sc(w, *names, idx=0):
    for n in names:
        v = getattr(w, n, None)
        if v is not None:
            a = np.asarray(v).ravel()
            if a.size:
                return float(a[idx])
    return float("nan")


jf, mg, ntor, ns_def = CFG[config]
inp = vmecpp.VmecInput.from_file(str(TD / jf))
if mg is not None:
    inp.mgrid_file = str(TD / mg)
if mpol > 0:
    inp.mpol = mpol
inp.ntor = ntor
if pscale > 0:
    inp.pres_scale = pscale
if ns <= 0:
    ns = ns_def
# Single-grid solve: keep ns_array, ftol_array, niter_array the same length (1).
inp.ns_array = np.array([ns], dtype=np.int64)
inp.ftol_array = np.array([1e-11])
inp.niter_array = np.array([8000], dtype=np.int64)
try:
    with silence():
        out = vmecpp.run(inp, max_threads=1, verbose=False)
    w = out.wout
    db = np.asarray(w.delbsq)
    db = db[db > 0] if np.asarray(w.delbsq).size else np.array([np.nan])
    beta = sc(w, "betatotal", "betatot")
    rax = sc(w, "raxis_cc", "raxis_c")
    wb = sc(w, "wb")
    iot = np.asarray(getattr(w, "iotaf", [np.nan])).ravel()
    conv = "Y" if db.size and len(db) < 8000 else "N"
    print(
        f"{config:12s} {mode:6s} ps={pscale:<8g} mpol={int(getattr(inp, 'mpol', 0)):2d} "
        f"beta={beta:.4e} raxis={rax:.6f} iota_ax={iot[0]:8.5f} "
        f"iota_ed={iot[-1]:8.5f} wb={wb:.6e} delbsq={db[-1]:.3e} "
        f"iters={len(db)} conv={conv}",
        flush=True,
    )
except Exception as e:
    print(
        f"{config:12s} {mode:6s} ps={pscale:<8g} FAILED "
        f"{type(e).__name__}: {str(e)[:70]}",
        flush=True,
    )
