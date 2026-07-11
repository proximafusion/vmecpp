# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Map the AEGIS-versus-NESTOR delbsq crossover against toroidal resolution ntor.

cth_like carries a coil grid up to ntor=4; running the free-boundary solve with
the boundary truncated to ntor = 1..4 (same nfp=5 geometry and mgrid) sweeps the
toroidal Fourier content of the vacuum field while holding everything else fixed.
With the axisymmetric Solovev point (ntor=0) this maps whether AEGIS overtakes
NESTOR as NESTOR's Fourier vacuum representation becomes incomplete. One mode per
process (kAegis static).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "nestor"
mpol = int(sys.argv[2]) if len(sys.argv) > 2 else 8
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
    Path.home() / "vmecpp_629/src/vmecpp/cpp/vmecpp/test_data",
)


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


print(f"# mode={mode} mpol={mpol} ns={ns} cth_like, ntor sweep", flush=True)
print("# ntor  beta       raxis      delbsq     iters conv", flush=True)
for ntor in [1, 2, 3, 4]:
    try:
        inp = vmecpp.VmecInput.from_file(str(TD / "cth_like_free_bdy.json"))
        inp.mgrid_file = str(TD / "mgrid_cth_like.nc")
        # Truncate the boundary and axis to the target ntor (the coil grid keeps
        # its full toroidal content). rbc/zbs are (mpol, 2*ntor0+1) with n=0 at
        # the center column; raxis_c/zaxis_s are (ntor0+1,).
        rbc = np.asarray(inp.rbc)
        zbs = np.asarray(inp.zbs)
        rax = np.asarray(inp.raxis_c)
        zax = np.asarray(inp.zaxis_s)
        c = (rbc.shape[1] - 1) // 2
        inp.rbc = np.ascontiguousarray(rbc[:, c - ntor : c + ntor + 1])
        inp.zbs = np.ascontiguousarray(zbs[:, c - ntor : c + ntor + 1])
        inp.raxis_c = np.ascontiguousarray(rax[: ntor + 1])
        inp.zaxis_s = np.ascontiguousarray(zax[: ntor + 1])
        inp.mpol = mpol
        inp.ntor = ntor
        inp.ns_array = np.array([ns], dtype=np.int64)
        inp.ftol_array = np.array([1e-11])
        inp.niter_array = np.array([8000], dtype=np.int64)
        t0 = time.time()
        with silence():
            out = vmecpp.run(inp, max_threads=1, verbose=False)
        w = out.wout
        db = np.asarray(w.delbsq)
        db = db[db > 0]
        beta = float(np.asarray(getattr(w, "betatotal", [np.nan])).ravel()[0])
        rax = float(
            np.asarray(getattr(w, "raxis_cc", getattr(w, "raxis_c", [np.nan]))).ravel()[
                0
            ]
        )
        conv = "Y" if len(db) < 8000 else "N"
        print(
            f"{ntor:4d}  {beta:.4e} {rax:.5f} {db[-1]:.3e} {len(db):5d} {conv}  "
            f"t={time.time() - t0:.0f}s",
            flush=True,
        )
    except Exception as e:
        print(f"{ntor:4d}  FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
print("# NTOR_DONE", flush=True)
