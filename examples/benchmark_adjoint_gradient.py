# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Benchmark the boundary-shape gradient dJ/dx_B used by external optimizers.

Three ways to get the same gradient: finite difference over boundary DOFs (the
stock approach, re-solving VMEC per DOF), the implicit-function adjoint with the
finite-difference HVP, and the adjoint with the exact autodiff HVP. Reports force
evaluations, wall time, and agreement vs the FD reference. The exact-HVP path
needs an Enzyme-enabled build.

Run: python examples/benchmark_adjoint_gradient.py
"""

import time
from pathlib import Path

import numpy as np
import vmecpp_adjoint as va
from scipy.sparse.linalg import LinearOperator, gmres

DATA = Path(__file__).resolve().parent / "data"
CASES = [
    ("solovev", DATA / "solovev.json", 11),
    ("cth_like", DATA / "cth_like_fixed_bdy.json", 11),
]
OBJ = va.mhd_energy


def adjoint_grad(model, x_star, interior, boundary, exact):
    """va.boundary_gradient with a selectable HVP backend."""
    n = x_star.size
    dj = va.objective_state_gradient(model, x_star, OBJ)
    hvp = model.exact_hessian_vector_product if exact else model.hessian_vector_product
    model.set_state(np.ascontiguousarray(x_star))
    model.evaluate(2, 2, True)
    ni = interior.size

    def hii(vi):
        v = np.zeros(n)
        v[interior] = vi
        return np.asarray(hvp(np.ascontiguousarray(v)), float)[interior]

    def mii(bi):
        v = np.zeros(n)
        v[interior] = bi
        return np.asarray(model.apply_preconditioner(np.ascontiguousarray(v)), float)[
            interior
        ]

    lam, _ = gmres(
        LinearOperator((ni, ni), matvec=hii),
        dj[interior],
        M=LinearOperator((ni, ni), matvec=mii),
        rtol=1e-6,
        restart=100,
        maxiter=30,
    )
    emb = np.zeros(n)
    emb[interior] = lam
    model.set_state(np.ascontiguousarray(x_star))
    model.evaluate(2, 2, False)
    return dj[boundary] - np.asarray(hvp(np.ascontiguousarray(emb)), float)[boundary]


for case, path, ns in CASES:
    print(f"\n=== {case} (ns={ns}) ===")
    model = va.make_model(path, ns)
    interior, boundary = va.partition(model, ns)
    x0 = np.asarray(model.get_state(), float)
    x_star = va.solve_interior(model, x0, interior, boundary, x0[boundary])
    ndof = boundary.size
    dofs = list(range(min(ndof, 12)))

    model.reset_force_eval_count()
    t0 = time.perf_counter()
    g_fd = va.finite_difference_boundary_gradient(
        model, x_star, interior, boundary, OBJ, dofs
    )
    fd_evals, fd_t = model.force_eval_count, time.perf_counter() - t0
    gfd = np.array([g_fd[j] for j in dofs])
    full = fd_evals * ndof / len(dofs)

    print(f"  boundary DOFs = {ndof} (FD reference on {len(dofs)})")
    print(f"  {'method':28s} {'F-evals':>10s} {'time[s]':>8s} {'rel vs FD':>10s}")
    print(
        f"  {'FD over boundary (all, est)':28s} {int(full):10d} "
        f"{fd_t * ndof / len(dofs):8.2f} {'(ref)':>10s}"
    )
    for label, exact in [("adjoint, FD HVP", False), ("adjoint, exact HVP", True)]:
        try:
            model.reset_force_eval_count()
            t0 = time.perf_counter()
            g = adjoint_grad(model, x_star, interior, boundary, exact)
            ev, t = model.force_eval_count, time.perf_counter() - t0
            rel = np.linalg.norm(g[dofs] - gfd) / (np.linalg.norm(gfd) + 1e-300)
            print(f"  {label:28s} {ev:10d} {t:8.2f} {rel:10.1e}")
        except AttributeError:
            print(f"  {label:28s} (needs an Enzyme-enabled build)")
