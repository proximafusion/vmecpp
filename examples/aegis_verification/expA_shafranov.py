# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Exp A: 3D high-beta Shafranov sweep on cth_like (nfp=5, ntor=4), AEGIS vs NESTOR.

Raises pres_scale to grow beta and the Shafranov shift (the hard case named in
issue #628: the virtual-casing contribution is largest at high beta). Reports the
magnetic-axis major radius (Shafranov shift), beta, MHD energy wb, iota at axis
and edge, plasma volume, and delbsq. Correctness test: AEGIS must agree with
NESTOR on the physical invariants (axis, beta, wb, iota) at each beta while
driving delbsq lower at high resolution. Agreement refutes 'wrong but converges';
lower delbsq is the metric the issue names. One mode per process (kAegis static).
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

TD = next(
    (
        p / "src/vmecpp/cpp/vmecpp/test_data"
        for p in Path(__file__).resolve().parents
        if (p / "src/vmecpp/cpp/vmecpp/test_data").is_dir()
    ),
    Path("src/vmecpp/cpp/vmecpp/test_data"),
)


def sc(w, *names, idx=0):
    for n in names:
        v = getattr(w, n, None)
        if v is not None:
            a = np.asarray(v).ravel()
            if a.size:
                return float(a[idx])
    return float("nan")


PS = [432.29, 1080.0, 2160.0, 4320.0, 6480.0, 8640.0]  # 1x .. 20x baseline
print(f"# mode={mode} mpol={mpol} ns={ns} cth_like free-bdy", flush=True)
print(
    "# pscale     beta       raxis        wb           iota_ax  iota_ed  "
    "vol         delbsq    iters conv",
    flush=True,
)
for ps in PS:
    try:
        inp = vmecpp.VmecInput.from_file(str(TD / "cth_like_free_bdy.json"))
        inp.mgrid_file = str(TD / "mgrid_cth_like.nc")
        inp.mpol = mpol
        inp.ntor = 4
        inp.pres_scale = ps
        inp.ns_array = np.array([ns], dtype=np.int64)
        inp.ftol_array = np.array([1e-11])
        inp.niter_array = np.array([8000], dtype=np.int64)
        t0 = time.time()
        out = vmecpp.run(inp, max_threads=1, verbose=False)
        w = out.wout
        db = np.asarray(w.delbsq)
        db = db[db > 0]
        beta = sc(w, "betatotal", "betatot")
        rax = sc(w, "raxis_cc", "raxis_c")
        wb = sc(w, "wb")
        iot = np.asarray(getattr(w, "iotaf", [np.nan])).ravel()
        iax, ied = float(iot[0]), float(iot[-1])
        vol = sc(w, "volume_p", "volume", "vol")
        conv = "Y" if len(db) < 8000 else "N"
        print(
            f"{ps:9.1f} {beta:.4e} {rax:.6f} {wb:.5e} {iax:8.4f} {ied:8.4f} "
            f"{vol:.4e} {db[-1]:.3e} {len(db):5d} {conv}  t={time.time() - t0:.0f}s",
            flush=True,
        )
    except Exception as e:
        print(f"{ps:9.1f} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True)
print("# EXPA_DONE", flush=True)
