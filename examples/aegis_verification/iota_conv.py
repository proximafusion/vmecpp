#!/usr/bin/env python
"""Resolution convergence of the AEGIS-vs-NESTOR equilibrium difference (iota, magnetic
axis): does it close as mpol increases (two independent vacuum methods converging to a
common equilibrium) or persist (a fixed method difference)? A lower delbsq alone does
not establish correctness, since each solver's delbsq is formed with its own exterior
field; independent-method agreement on the physical invariants is the test.

cth_like free boundary at fixed low beta (isolates the vacuum-field method from high-
beta physics), mpol swept at fixed ns. Reports iota at axis and edge, the magnetic-axis
major radius, wb, and delbsq. One mode per process; each solve is run with stdout/stderr
redirected to /dev/null so a non-convergence dump cannot flood the log.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "nestor"
pscale = float(sys.argv[2]) if len(sys.argv) > 2 else 432.29
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 25
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


class silence:
    """Redirect the C-level fd 1 and fd 2 to /dev/null for the duration; keeps a non-
    convergence dump out of the captured log."""

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


MPOLS = [5, 6, 8, 10, 12]
print(f"# mode={mode} pscale={pscale} ns={ns} cth_like free-bdy", flush=True)
print(
    "# mpol  beta       raxis        iota_ax   iota_ed   wb           "
    "delbsq    iters conv",
    flush=True,
)
for mpol in MPOLS:
    try:
        inp = vmecpp.VmecInput.from_file(str(TD / "cth_like_free_bdy.json"))
        inp.mgrid_file = str(TD / "mgrid_cth_like.nc")
        inp.mpol = mpol
        inp.ntor = 4
        inp.pres_scale = pscale
        inp.ns_array = np.array([ns], dtype=np.int64)
        inp.ftol_array = np.array([1e-11])
        inp.niter_array = np.array([8000], dtype=np.int64)
        t0 = time.time()
        with silence():
            out = vmecpp.run(inp, max_threads=1, verbose=False)
        w = out.wout
        db = np.asarray(w.delbsq)
        db = db[db > 0]
        beta = sc(w, "betatotal", "betatot")
        rax = sc(w, "raxis_cc", "raxis_c")
        wb = sc(w, "wb")
        iot = np.asarray(getattr(w, "iotaf", [np.nan])).ravel()
        iax, ied = float(iot[0]), float(iot[-1])
        conv = "Y" if len(db) < 8000 else "N"
        print(
            f"{mpol:4d}  {beta:.4e} {rax:.6f} {iax:8.5f} {ied:8.5f} "
            f"{wb:.5e} {db[-1]:.3e} {len(db):5d} {conv}  t={time.time() - t0:.0f}s",
            flush=True,
        )
    except Exception as e:
        print(f"{mpol:4d}  FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
print("# IOTACONV_DONE", flush=True)
