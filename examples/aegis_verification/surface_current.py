# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Physical surface current versus the boundary pressure jump (issue #628).

The tangential-field jump n x (B_out - B_in) at the LCFS is the physical surface
current, distinct from the virtual-casing current K = n x B_plasma, which is
nonzero at every surface point regardless of the equilibrium. A free-boundary
equilibrium fixes the physical current through boundary total-pressure balance,

    |B_out|^2 - |B_in|^2 = 2 mu0 p(edge),

so the surface current |n x (B_out - B_in)| / |B| ~ mu0 p(edge) / |B|^2 vanishes
as the edge pressure p(edge) -> 0 and grows linearly with it. This is the check
raised on the issue: a physical surface current must appear only when the
total-pressure jump at the boundary is nonzero, not from the virtual casing.

The axisymmetric free-boundary Solovev tokamak is swept over a uniform edge
pressure pedestal by shifting the mass profile am = [0.125 + d, -0.125], which
raises p(edge) = pres_scale * d while leaving the interior pressure gradient
dp/ds = -pres_scale * 0.125 unchanged. The converged physical surface current is
read from the AEGIS diagnostic (VMECPP_AEGIS_DIAG) and compared against the
pressure-balance prediction mu0 p(edge) / <|B|^2> formed from the same run's
edge pressure and surface-averaged field.

Result: p(edge) = 0 gives the on-surface quadrature floor (~1.7e-3), and the
current is at that floor with no physical sheet current, as required. Raising
p(edge) increases the measured current only weakly above the floor (~1.7e-3 to
~2.1e-3 up to beta ~ 4%), far below the prediction, because the pressure-driven
magnitude excess |B_out|^2 - |B_in|^2 = 2 mu0 p(edge) hides in AEGIS's residual
normal-field error rather than the tangential jump. So the physical current is
correctly zero at p(edge) = 0 but is not cleanly separable from the floor on the
axisymmetric coupling grid. floor_diag.py traces the floor to the poloidal
on-surface quadrature, ntor_aspect.py to the grid aspect ratio, and
onsurface_accuracy.py shows a balanced grid brings AEGIS to the golden-reference
level, removing the mask.
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
MU0 = 4e-7 * np.pi
PRES_SCALE = 1000.0
NS = 51
DIAG_RE = re.compile(r"phys surf current \|n x \(Bout-Bin\)\|/\|B\|=([0-9.eE+-]+)")


def edge_b2(w):
    """Surface-averaged |B|^2 on the LCFS from bmnc at s=1 (Parseval over the cosine
    series; ntor=0, so the modes are cos(m*theta))."""
    bmnc = np.asarray(w.bmnc)[:, -1]
    xm = np.asarray(w.xm_nyq)
    weight = np.where(xm == 0, 1.0, 0.5)
    return float(np.sum(weight * bmnc**2))


def run_case(pedestal):
    inp = vmecpp.VmecInput.from_file(str(TD / "solovev_free_bdy.json"))
    inp.mgrid_file = str(TD / "mgrid_solovev.nc")
    inp.ntor = 0
    inp.pres_scale = PRES_SCALE
    inp.am = np.array([0.125 + pedestal, -0.125])
    inp.ns_array = np.array([NS], dtype=np.int64)
    inp.ftol_array = np.array([1e-11])
    inp.niter_array = np.array([8000], dtype=np.int64)

    # Capture the C-level stderr (the per-iteration AEGIS diagnostic) to a file;
    # the last line is the converged value. Drop stdout (iteration prints).
    errpath = Path("/tmp") / f"aegis_diag_{os.getpid()}.txt"
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
    beta = float(np.asarray(w.betatotal).ravel()[0])
    db = np.asarray(w.delbsq)
    db = db[db > 0]
    p_edge = PRES_SCALE * pedestal  # p(edge) = pres_scale * (am0 + am1), am1 = -0.125
    b2 = edge_b2(w)
    predicted = MU0 * p_edge / b2  # mu0 p(edge) / |B|^2 from pressure balance
    diag = DIAG_RE.findall(errpath.read_text(errors="ignore"))
    surf = float(diag[-1]) if diag else float("nan")
    errpath.unlink(missing_ok=True)
    return beta, p_edge, b2, predicted, surf, len(db)


print(f"# free-boundary Solovev, pres_scale={PRES_SCALE:g}, ns={NS}", flush=True)
print(
    "# p(edge)   beta        <|B|^2>    predicted    measured     meas/pred  iters",
    flush=True,
)
for ped in [0.0, 0.05, 0.125, 0.25, 0.5]:
    try:
        beta, p_edge, b2, pred, surf, iters = run_case(ped)
        ratio = surf / pred if pred > 0 else float("nan")
        print(
            f"{p_edge:8.1f}  {beta:.4e}  {b2:.3e}  {pred:.4e}   {surf:.4e}   "
            f"{ratio:8.2f}  {iters:5d}",
            flush=True,
        )
    except Exception as e:
        print(
            f"{PRES_SCALE * ped:8.1f}  FAILED {type(e).__name__}: {str(e)[:55]}",
            flush=True,
        )
print("# SURFCUR_DONE", flush=True)
