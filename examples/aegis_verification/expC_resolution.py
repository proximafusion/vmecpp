# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Exp C: delbsq vs resolution at HIGH beta on cth_like (3D), AEGIS vs NESTOR.

Confirms the AEGIS-beats-NESTOR-at-high-resolution result (issue #628 metric)
holds at high beta, not only at the default low beta. Analog of the DESC
finite-beta test (arXiv:2412.05680): at beta = 2% the DESC free boundary has
2-3x lower boundary-condition residual than VMEC. Sweep mpol at a fixed elevated
pres_scale; report converged delbsq. One mode per process (kAegis static).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "nestor"
PSCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 4320.0
if mode == "aegis":
    os.environ["VMECPP_AEGIS"] = "1"

# The scikit-build-core editable install re-runs cmake --install on every
# import and floods stdout/stderr; silence it so the result lines stay clean.
_s1, _s2 = os.dup(1), os.dup(2)
_dn = os.open(os.devnull, os.O_WRONLY)
os.dup2(_dn, 1)
os.dup2(_dn, 2)
import vmecpp  # noqa: E402

os.dup2(_s1, 1)
os.dup2(_s2, 2)
os.close(_dn)
os.close(_s1)
os.close(_s2)


class silence:
    """Redirect fd 1 and fd 2 to /dev/null so the free-boundary per-iteration status
    output does not flood the result log."""

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


TD = next(
    (
        p / "src/vmecpp/cpp/vmecpp/test_data"
        for p in Path(__file__).resolve().parents
        if (p / "src/vmecpp/cpp/vmecpp/test_data").is_dir()
    ),
    Path("src/vmecpp/cpp/vmecpp/test_data"),
)

# (mpol, ns) pairs; ns grows with mpol to keep the radial grid resolved.
GRID = [(5, 25), (7, 35), (9, 45), (12, 55)]
print(f"# mode={mode} pscale={PSCALE} cth_like free-bdy", flush=True)
print("# mpol ns   beta       raxis      delbsq     iters conv", flush=True)
for mpol, ns in GRID:
    try:
        inp = vmecpp.VmecInput.from_file(str(TD / "cth_like_free_bdy.json"))
        inp.mgrid_file = str(TD / "mgrid_cth_like.nc")
        inp.mpol = mpol
        inp.ntor = 4
        inp.pres_scale = PSCALE
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
            f"{mpol:4d} {ns:4d} {beta:.4e} {rax:.5f} {db[-1]:.3e} "
            f"{len(db):5d} {conv}  t={time.time() - t0:.0f}s",
            flush=True,
        )
    except Exception as e:
        print(f"{mpol:4d} {ns:4d} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
print("# EXPC_DONE", flush=True)
