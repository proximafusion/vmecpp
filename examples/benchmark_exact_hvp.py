# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Benchmark the exact autodiff Hessian-vector product as an internal solver.

Compares preconditioned JFNK, the finite-difference Newton-HVP, and the exact
autodiff Newton-HVP on the same VMEC++ residual, counting force evaluations
inside VMEC++. Requires an Enzyme-enabled build (VMECPP_ENABLE_ENZYME) for the
exact-HVP solver; the others run on any build.

Run: python -m examples.benchmark_exact_hvp  (or python examples/benchmark_exact_hvp.py)
"""

from pathlib import Path

import external_optimizers as eo

DATA = Path(__file__).resolve().parent / "data"
CASES = [
    ("solovev", DATA / "solovev.json", 11),
    ("cth_like", DATA / "cth_like_fixed_bdy.json", 11),
]

SOLVERS = [
    ("precond JFNK", eo.solve_newton_krylov_preconditioned),
    ("Newton FD-HVP + M^-1", eo.solve_newton_hvp),
    ("Newton exact-HVP + M^-1", eo.solve_newton_exact_hvp),
]

for case, path, ns in CASES:
    print(f"\n=== {case} (ns={ns}) ===")
    _, wref = eo.reference_equilibrium(path, ns)
    print(f"native solve: W = {wref:.8e}")
    print(
        f"{'optimizer':28s} {'F-evals':>8s} {'iters':>6s} {'time[s]':>8s} "
        f"{'||F||':>10s} {'dW':>10s}"
    )
    for label, solver in SOLVERS:
        try:
            r = solver(path, ns)[1]
            print(
                f"{label:28s} {r.force_evals:8d} {r.outer_iters:6d} "
                f"{r.seconds:8.2f} {r.residual_norm:10.1e} "
                f"{abs(r.energy - wref):10.1e}"
            )
        except AttributeError:
            print(f"{label:28s} (needs an Enzyme-enabled build)")
