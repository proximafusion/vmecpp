# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Plot committed radial finite-difference benchmark results."""

import csv
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "radial_fd_results.csv"
OUTPUT = ROOT / "docs" / "radial_fd_benchmark.png"
SCHEMES = ("baseline", "four_point")
LABELS = {"baseline": "Centered 2-point", "four_point": "4-point"}
PLT: Any = import_module("matplotlib.pyplot")


def main() -> None:
    with RESULTS.open(newline="") as results_file:
        rows = list(csv.DictReader(results_file))

    cases = list(dict.fromkeys(row["case"] for row in rows))
    figure, axes = PLT.subplots(1, 3, figsize=(13, 4.2))
    colors = ("#4c78a8", "#f58518")
    x = list(range(len(cases)))
    width = 0.34

    for scheme_index, scheme in enumerate(SCHEMES):
        selected = [
            next(r for r in rows if r["case"] == case and r["scheme"] == scheme)
            for case in cases
        ]
        offset = (scheme_index - 0.5) * width
        bar_positions = [value + offset for value in x]
        iterations = [int(row["iterations"]) for row in selected]
        per_iteration_ms = [
            1000 * float(row["time_per_iteration_s"]) for row in selected
        ]
        axes[0].bar(
            bar_positions,
            iterations,
            width,
            label=LABELS[scheme],
            color=colors[scheme_index],
        )
        axes[1].bar(
            bar_positions,
            per_iteration_ms,
            width,
            label=LABELS[scheme],
            color=colors[scheme_index],
        )
        for index, row in enumerate(selected):
            if row["converged"] == "false":
                axes[0].text(
                    bar_positions[index],
                    iterations[index],
                    "cap",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[0].set_title("Iterations to tolerance or cap")
    axes[0].set_ylabel("Force evaluations")
    axes[0].set_yscale("log")
    axes[1].set_title("Mean time per force evaluation")
    axes[1].set_ylabel("Milliseconds")
    for axis in axes[:2]:
        axis.set_xticks(x, [case.replace("_", "\n") for case in cases])
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.25)

    accuracy_rows = [row for row in rows if row["case"] == "exact_sheared_ns25"]
    field_errors = [float(row["field_error"]) for row in accuracy_rows]
    accuracy_bars = axes[2].bar(
        [LABELS[row["scheme"]] for row in accuracy_rows],
        field_errors,
        color=[colors[SCHEMES.index(row["scheme"])] for row in accuracy_rows],
    )
    axes[2].set_title("Exact-equilibrium relative field error")
    axes[2].set_ylabel("Maximum relative error")
    axes[2].set_yscale("log")
    axes[2].grid(axis="y", alpha=0.25)
    for bar, row in zip(accuracy_bars, accuracy_rows, strict=True):
        if row["converged"] == "false":
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "not converged",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.suptitle("Impact of a wider radial derivative stencil (single thread)")
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=180)
    PLT.close(figure)


if __name__ == "__main__":
    main()
