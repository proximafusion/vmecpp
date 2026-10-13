"""Experimental WP5: Newton history and a dense active-H solve comparison.

Start from the VMEC-converged WP0 small case. Write plot_01_newton_small.png
and named NPZ data to --output; convergence is measured, not assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmecpp
from vmecpp.autodiff_diagnostics import raw_force
from vmecpp.autodiff_newton import newton_polish

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from adjoint_verification.harness import (  # pyright: ignore[reportMissingImports]
    EvaluationSettings,
    dense_jacobian,
    small_input,
    solved_model,
)
from plot_style import STYLE, save_plot


def newton_data(max_steps: int, tol: float) -> dict[str, np.ndarray]:
    """Measure polishing and compare the final factor handle with exact dense H."""
    vmec_input = small_input()
    model = solved_model(vmec_input)
    initial_state = np.asarray(model.get_state()).copy()
    before = int(model.force_eval_count)
    result = newton_polish(model, max_steps, tol)
    after_polish = int(model.force_eval_count)
    force = raw_force(model)
    try:
        full_h = dense_jacobian(
            model,
            vmec_input=vmec_input,
            evaluation=EvaluationSettings(precondition=False),
        )
        active_h = full_h[np.ix_(result.active, result.active)]
        rhs = np.random.default_rng(0).standard_normal(result.active.size)
        block_solution = result.factorization.solve_transpose(rhs)
        dense_solution = np.linalg.solve(active_h, rhs)
        rhs_norm = float(np.linalg.norm(rhs))
        difference = float(np.linalg.norm(block_solution - dense_solution))
        relative_difference = difference / max(
            float(np.linalg.norm(dense_solution)), np.finfo(float).tiny
        )
        block_residual = float(np.linalg.norm(active_h @ block_solution - rhs)) / (
            rhs_norm
        )
        dense_residual = float(np.linalg.norm(active_h @ dense_solution - rhs)) / (
            rhs_norm
        )
    finally:
        model.set_state(np.ascontiguousarray(result.state))
        raw_force(model)
    return {
        "accepted_step": np.arange(result.residual_history.size),
        "residual_history": result.residual_history,
        "converged": np.asarray(result.converged),
        "tol": np.asarray(tol),
        "max_steps": np.asarray(max_steps),
        "active_force_norm": np.asarray(np.linalg.norm(force[result.active])),
        "full_force_norm": np.asarray(np.linalg.norm(force)),
        "force_eval_count_labels": np.asarray(
            ["before_polish", "after_polish", "after_dense_and_restore"]
        ),
        "force_eval_counts": np.asarray(
            [before, after_polish, int(model.force_eval_count)]
        ),
        "initial_state": initial_state,
        "final_state": result.state,
        "active": result.active,
        "active_h": active_h,
        "rhs": rhs,
        "block_solution": block_solution,
        "dense_solution": dense_solution,
        "solve_relative_difference": np.asarray(relative_difference),
        "block_solve_relative_residual": np.asarray(block_residual),
        "dense_solve_relative_residual": np.asarray(dense_residual),
        "input_json": np.asarray(vmec_input.to_json()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1.0e-14)
    args = parser.parse_args()
    if not vmecpp.has_exact_force_jacobian():
        message = "this experiment requires VMECPP_ENABLE_ENZYME=ON"
        raise RuntimeError(message)
    data = newton_data(args.max_steps, args.tol)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots()
        axes.semilogy(
            data["accepted_step"],
            data["residual_history"],
            "o-",
            label="active raw force",
        )
        axes.axhline(
            args.tol, color="black", linestyle="--", label="requested tolerance"
        )
        axes.set(
            xlabel="accepted Newton step",
            ylabel="Euclidean force norm",
            title=f"Experimental small case: converged={bool(data['converged'])}",
        )
        axes.legend()
        save_plot(figure, args.output, "plot_01_newton_small", **data)
        plt.close(figure)
    print(  # noqa: T201 - command-line experiment summary
        f"converged={bool(data['converged'])}; "
        f"active force={float(data['active_force_norm']):.6e}; "
        f"block/dense solve difference={float(data['solve_relative_difference']):.6e}; "
        f"block residual={float(data['block_solve_relative_residual']):.6e}; "
        f"force counts={data['force_eval_counts'].tolist()}"
    )


if __name__ == "__main__":
    main()
