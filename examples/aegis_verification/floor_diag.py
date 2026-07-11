# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Root-cause the AEGIS on-surface physical-surface-current floor.

At a zero-edge-pressure free-boundary equilibrium the physical surface current
n x (B_out - B_in) must vanish, but the AEGIS diagnostic floors at ~1e-3. Two
candidates set that floor and separate cleanly by which grid resolves them:

  - poloidal source aliasing in the virtual-casing integral: the plasma surface
    current is sampled on the equilibrium poloidal grid (nThetaEff, set by mpol),
    so modes above its Nyquist alias into the quadrature. Drops with mpol.
  - radial extrapolation of the interior field bsupu/bsupv to the LCFS
    (1.5*last - 0.5*prev half-grid). Its error is O(ds^2) and drops with ns,
    independent of mpol.

Sweeps mpol at fixed ns, then ns at fixed mpol, on the zero-edge-pressure
free-boundary Solovev tokamak, and reports the converged physical surface
current (from VMECPP_AEGIS_DIAG) and delbsq. Whichever grid lowers the floor
identifies the dominant error.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

os.environ["VMECPP_AEGIS"] = "1"
os.environ["VMECPP_AEGIS_DIAG"] = "1"

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


def run_case(mpol, ns):
    inp = vmecpp.VmecInput.from_file(str(TD / "solovev_free_bdy.json"))
    inp.mgrid_file = str(TD / "mgrid_solovev.nc")
    inp.ntor = 0
    inp.mpol = mpol
    inp.pres_scale = 1000.0  # zero edge pressure (am = [0.125, -0.125] unchanged)
    inp.ns_array = np.array([ns], dtype=np.int64)
    inp.ftol_array = np.array([1e-11])
    inp.niter_array = np.array([8000], dtype=np.int64)

    errpath = Path("/tmp") / f"floor_diag_{os.getpid()}.txt"
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
    return surf, (db[-1] if db.size else float("nan")), len(db)


print("# free-boundary Solovev, zero edge pressure; AEGIS on-surface floor", flush=True)
print("# mpol   ns   surf_current   delbsq     iters", flush=True)
print("# -- vary mpol at fixed ns=51 (tests poloidal aliasing) --", flush=True)
for mpol in [6, 8, 12, 16, 20]:
    try:
        surf, db, it = run_case(mpol, 51)
        print(f"  {mpol:4d}  {51:4d}  {surf:.4e}   {db:.3e}  {it:5d}", flush=True)
    except Exception as e:
        print(
            f"  {mpol:4d}  {51:4d}  FAILED {type(e).__name__}: {str(e)[:45]}",
            flush=True,
        )
print("# -- vary ns at fixed mpol=12 (tests radial extrapolation) --", flush=True)
for ns in [25, 51, 101, 151]:
    try:
        surf, db, it = run_case(12, ns)
        print(f"  {12:4d}  {ns:4d}  {surf:.4e}   {db:.3e}  {it:5d}", flush=True)
    except Exception as e:
        print(
            f"  {12:4d}  {ns:4d}  FAILED {type(e).__name__}: {str(e)[:45]}", flush=True
        )
print("# FLOOR_DIAG_DONE", flush=True)
