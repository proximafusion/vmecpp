# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Benchmark exact autodiff Hessian-vector products as an internal solver.

Compare preconditioned JFNK, finite-difference Newton-HVP, and exact-autodiff
Newton-HVP on the same VMEC++ residual. Report force evaluations, iterations,
wall time, residual norm, and energy error. Requires an Enzyme-enabled build.

Run: python benchmarks/benchmark_exact_hvp.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import external_optimizers as eo  # type: ignore  # noqa: E402

DATA = ROOT / "examples" / "data"
CASES = [
    ("solovev", DATA / "solovev.json", 11),
    ("cth_like", DATA / "cth_like_fixed_bdy.json", 11),
]

SOLVERS = [
    ("precond JFNK", eo.solve_newton_krylov_preconditioned),
    ("Newton FD-HVP + M^-1", eo.solve_newton_hvp),
    ("Newton exact-HVP + M^-1", eo.solve_newton_exact_hvp),
]


def main():
    for case, path, ns in CASES:
        print(f"\n=== {case} (ns={ns}) ===")
        _, wref = eo.reference_equilibrium(path, ns)
        print(f"native solve: W = {wref:.8e}")
        print(
            f"{'optimizer':28s} {'F-evals':>8s} {'iters':>6s} {'time[s]':>8s} "
            f"{'||F||':>10s} {'dW':>10s}"
        )
        for label, solver in SOLVERS:
            result = solver(path, ns)[1]
            print(
                f"{label:28s} {result.force_evals:8d} {result.outer_iters:6d} "
                f"{result.seconds:8.2f} {result.residual_norm:10.1e} "
                f"{abs(result.energy - wref):10.1e}"
            )


if __name__ == "__main__":
    main()
