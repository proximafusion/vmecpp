# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Derive the AEGIS edge-damping schedule from the measured operator spectrum.

The damping A counteracts the near-null edge mode of the free-boundary force
Jacobian. That Jacobian is available matrix-free as the Hessian-vector product of
VMEC's augmented functional (unpreconditioned, so independent of A). At a
converged axisymmetric Solovev equilibrium with AEGIS active, this estimates the
smallest eigenvalue lambda_min of that Jacobian (shift-and-invert-free power
iteration on sigma*I - H) as beta rises. If the minimum damping needed to
converge scales as 1/lambda_min, the beta ramp is the measured near-null scaling,
not a fitted constant.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ["VMECPP_AEGIS"] = "1"
# The HVP is the unpreconditioned force Jacobian, so it is independent of the edge
# damping A; the default (beta-scaled) schedule is left in place only so the
# convergence runs at high beta reach an equilibrium to probe.

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore

TD = next(
    (
        p / "src/vmecpp/cpp/vmecpp/test_data"
        for p in Path(__file__).resolve().parents
        if (p / "src/vmecpp/cpp/vmecpp/test_data").is_dir()
    ),
    Path.home() / "vmecpp_629/src/vmecpp/cpp/vmecpp/test_data",
)
NS = 51


def hv(model, v):
    return np.asarray(model.hessian_vector_product(np.ascontiguousarray(v)), float)


def measure(pscale):
    inp = vmecpp.VmecInput.from_file(str(TD / "solovev_free_bdy.json"))
    inp.mgrid_file = str(TD / "mgrid_solovev.nc")
    inp.ntor = 0
    inp.pres_scale = pscale
    inp.ns_array = np.array([NS], dtype=np.int64)
    inp.ftol_array = np.array([1e-11])
    inp.niter_array = np.array([8000], dtype=np.int64)
    out = vmecpp.run(inp, max_threads=1, verbose=False)
    beta = float(np.asarray(out.wout.betatotal).ravel()[0])

    hrs = _vmecpp.HotRestartState(out.wout._to_cpp_wout(), inp._to_cpp_vmecindata())
    model = _vmecpp.VmecModel.create(inp._to_cpp_vmecindata(), NS, hrs)
    model.evaluate(2, 2, False)
    x = np.asarray(model.get_state(), float)
    n = x.size
    rng = np.random.default_rng(0)

    # Largest |eigenvalue| via power iteration -> shift.
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(30):
        h = hv(model, v)
        lam = float(v @ h)
        nv = np.linalg.norm(h)
        if nv == 0:
            break
        v = h / nv
    sigma = 1.2 * abs(lam)

    # Smallest eigenvalue via power iteration on (sigma*I - H).
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    for _ in range(80):
        w = sigma * v - hv(model, v)
        v = w / np.linalg.norm(w)
    h = hv(model, v)
    lam_min = float(v @ h)
    return beta, lam_min, sigma


print("# pscale  beta        lam_min      1/lam_min", flush=True)
for ps in [1000, 3000, 6000]:
    try:
        b, lmin, sig = measure(ps)
        inv = 1.0 / lmin if lmin != 0 else float("inf")
        print(f"{ps:7d}  {b:.4e}  {lmin:.4e}  {inv:.4e}", flush=True)
    except Exception as e:
        print(f"{ps:7d}  FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
print("# HVP_DONE", flush=True)
