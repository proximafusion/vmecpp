"""WP1: compare exact H with central differences of the raw force.

Run from any directory with Python and an Enzyme-enabled VMEC++ installation.
Writes plot_00_force_definition_small.png and its numerical data to --output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vmecpp import autodiff
from vmecpp.autodiff_diagnostics import raw_force
from vmecpp.cpp import _vmecpp  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from adjoint_verification.harness import (  # pyright: ignore[reportMissingImports]
    small_input,
    solved_model,
)
from plot_style import STYLE, save_plot


def force_definition_data(model) -> dict[str, np.ndarray]:
    """Measure full/interior/boundary-direction FD errors without changing state."""
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    interior, boundary = autodiff._interior_and_boundary(model)
    generator = np.random.default_rng(0)
    steps = np.logspace(-3, -7, 9)
    directions = []
    for indices in (np.arange(state.size), interior, boundary):
        direction = np.zeros(state.size)
        direction[indices] = generator.standard_normal(indices.size)
        directions.append(direction / np.linalg.norm(direction))
    errors = np.zeros((len(directions), len(steps)))
    try:
        force = raw_force(model)
        model.evaluate(2, 2, True, True)
        preconditioned = np.asarray(model.get_forces(), dtype=np.float64).copy()
        for row, direction in enumerate(directions):
            model.set_state(np.ascontiguousarray(state))
            raw_force(model)
            exact = np.asarray(model.exact_hessian_vector_product(direction)).copy()
            for column, step in enumerate(steps):
                model.set_state(np.ascontiguousarray(state + step * direction))
                plus = raw_force(model)
                model.set_state(np.ascontiguousarray(state - step * direction))
                minus = raw_force(model)
                difference = (plus - minus) / (2 * step)
                errors[row, column] = np.linalg.norm(exact - difference) / max(
                    float(np.linalg.norm(exact)), float(np.finfo(float).tiny)
                )
    finally:
        model.set_state(np.ascontiguousarray(state))
        raw_force(model)
    return {
        "steps": steps,
        "relative_errors": errors,
        "raw_force_norm": np.asarray(np.linalg.norm(force)),
        "preconditioned_force_norm": np.asarray(np.linalg.norm(preconditioned)),
        "preconditioning_relative_change": np.asarray(
            np.linalg.norm(preconditioned - force)
            / max(float(np.linalg.norm(force)), float(np.finfo(float).tiny))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not _vmecpp.VMECPP_ENABLE_ENZYME:
        error_message = "this plot requires VMECPP_ENABLE_ENZYME=ON"
        raise RuntimeError(error_message)
    data = force_definition_data(solved_model(small_input()))
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots()
        for row, label in enumerate(("full", "interior", "boundary")):
            axes.loglog(data["steps"], data["relative_errors"][row], "o-", label=label)
        axes.set(
            xlabel="central-difference step", ylabel="relative H / raw-force FD error"
        )
        axes.legend()
        save_plot(figure, args.output, "plot_00_force_definition_small", **data)
        plt.close(figure)


if __name__ == "__main__":
    main()
