# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Coupling on-surface floor versus the axisymmetric toroidal sample count.

The isotropic-grid study (onsurface_accuracy.py) showed AEGIS's punctured- trapezoidal
principal-value quadrature converges to the equilibrium normal-field floor on a balanced
grid, but degrades on an anisotropic one. The axisymmetric coupling upsamples the
toroidal direction to VMECPP_AEGIS_NTOR (default 256) while leaving the poloidal grid at
nThetaEff (~16 at mpol 12), a ~16x aspect ratio. This sweeps VMECPP_AEGIS_NTOR on the
free-boundary Solovev tokamak and reads the converged physical surface current
(VMECPP_AEGIS_DIAG) and delbsq. If the floor is minimized near an isotropic toroidal
count rather than at 256, the coupling's on-surface floor is an aspect-ratio effect that
balancing the grid removes. One toroidal count per process (static getenv).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

NTOR = sys.argv[1] if len(sys.argv) > 1 else "256"
os.environ["VMECPP_AEGIS"] = "1"
os.environ["VMECPP_AEGIS_DIAG"] = "1"
os.environ["VMECPP_AEGIS_NTOR"] = NTOR

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
DIAG_RE = re.compile(r"phys surf current \|n x \(Bout-Bin\)\|/\|B\|=([0-9.eE+-]+)")

inp = vmecpp.VmecInput.from_file(str(TD / "solovev_free_bdy.json"))
inp.mgrid_file = str(TD / "mgrid_solovev.nc")
inp.ntor = 0
inp.mpol = 12
inp.pres_scale = 1000.0
inp.ns_array = np.array([51], dtype=np.int64)
inp.ftol_array = np.array([1e-11])
inp.niter_array = np.array([8000], dtype=np.int64)

errpath = Path("/tmp") / f"ntor_aspect_{os.getpid()}.txt"
fd_err = os.open(str(errpath), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
sv1, sv2 = os.dup(1), os.dup(2)
dn = os.open(os.devnull, os.O_WRONLY)
os.dup2(dn, 1)
os.dup2(fd_err, 2)
try:
    out = vmecpp.run(inp, max_threads=1, verbose=False)
finally:
    os.dup2(sv1, 1)
    os.dup2(sv2, 2)
    os.close(fd_err)
    os.close(dn)
    os.close(sv1)
    os.close(sv2)

w = out.wout
db = np.asarray(w.delbsq)
db = db[db > 0]
diag = DIAG_RE.findall(errpath.read_text(errors="ignore"))
surf = float(diag[-1]) if diag else float("nan")
errpath.unlink(missing_ok=True)
print(
    f"NTOR={int(NTOR):4d}  surf_current={surf:.4e}  delbsq={db[-1]:.3e}  iters={len(db)}",
    flush=True,
)
