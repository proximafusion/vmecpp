# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Boundary sheet current versus edge pressure.

A physical sheet current at the LCFS would make the exterior field magnitude
exceed the interior by the boundary pressure jump, |B_out|^2/2 - |B_in|^2/2 =
p(edge). Free-boundary VMEC instead enters the edge pressure as a scalar force
term (outsideEdgePressure = vacuum_magnetic_pressure + edgePressure), so the
converged vacuum field magnitude matches the interior and no field-discontinuity
sheet current develops, whatever the edge pressure. This runs the free-boundary
Solovev tokamak with a zero and a finite edge-pressure pedestal (am[0] shifted so
p(edge) = pres_scale * d) under both NESTOR and AEGIS and reads the ratio
(|B_out|^2/2 - extrap|B_in|^2/2) / p(edge) from VMECPP_FB_DIAG. It stays near zero
for both solvers, so AEGIS introduces no spurious sheet current and matches
NESTOR. The always-nonzero virtual-casing current K = n x B_plasma is a separate
equivalent-source construct, not this physical jump.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

solver = sys.argv[1] if len(sys.argv) > 1 else "nestor"
os.environ["VMECPP_FB_DIAG"] = "1"
if solver == "aegis":
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
D3 = re.compile(r"ratio=([0-9.eE+-]+) <vacP>=([0-9.eE+-]+)")


def run_case(ped):
    inp = vmecpp.VmecInput.from_file(str(TD / "solovev_free_bdy.json"))
    inp.mgrid_file = str(TD / "mgrid_solovev.nc")
    inp.ntor = 0
    inp.mpol = 12
    inp.pres_scale = 1000.0
    inp.am = np.array([0.125 + ped, -0.125])
    inp.ns_array = np.array([51], dtype=np.int64)
    inp.ftol_array = np.array([1e-11])
    inp.niter_array = np.array([8000], dtype=np.int64)
    errpath = Path("/tmp") / f"ej_{os.getpid()}.txt"
    fd = os.open(str(errpath), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sv1, sv2 = os.dup(1), os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(fd, 2)
    try:
        out = vmecpp.run(inp, max_threads=1, verbose=False)
    finally:
        os.dup2(sv1, 1)
        os.dup2(sv2, 2)
        os.close(fd)
        os.close(dn)
        os.close(sv1)
        os.close(sv2)
    beta = float(np.asarray(out.wout.betatotal).ravel()[0])
    m = D3.findall(errpath.read_text(errors="ignore"))
    errpath.unlink(missing_ok=True)
    ratio, vac = (
        (float(m[-1][0]), float(m[-1][1])) if m else (float("nan"), float("nan"))
    )
    return beta, ratio, vac


print(
    f"# {solver} free-boundary Solovev, |B_out|^2/2 - extrap|B_in|^2/2 vs p(edge)",
    flush=True,
)
print("# p(edge)   beta        vacP        jump/p(edge)", flush=True)
for ped in [0.0, 0.125, 0.25]:
    beta, ratio, vac = run_case(ped)
    rstr = f"{ratio:.4f}" if ped > 0 else "   ---"
    print(f"{1000.0 * ped:8.1f}  {beta:.4e}  {vac:.4e}  {rstr}", flush=True)
print("# EDGE_JUMP_DONE", flush=True)
