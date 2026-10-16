# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Compare centered and four-point radial derivatives on representative solves."""

import csv
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import vmecpp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
NP: Any = import_module("numpy")
EXACT_EQUILIBRIA: Any = import_module("exact_equilibria")
OUTPUT = ROOT / "benchmarks" / "radial_fd_results.csv"
FIELDS = (
    "case",
    "scheme",
    "elapsed_s",
    "iterations",
    "time_per_iteration_s",
    "converged",
    "ftol",
    "max_residual",
    "surface_error",
    "field_error",
    "axis_error",
    "current_error",
)


def measure(case: str, vmec_input: vmecpp.VmecInput, member: Any = None) -> list[dict]:
    rows = []
    for scheme, use_four_point in (("baseline", False), ("four_point", True)):
        vmec_input.use_fourth_order_radial_derivatives = use_four_point
        trace: list[vmecpp.SolverState] = []
        start = time.perf_counter()
        output = vmecpp.run(
            vmec_input,
            max_threads=1,
            verbose=False,
            iteration_callback=trace.append,
        )
        elapsed_s = time.perf_counter() - start
        wout = output.wout
        residual = max(float(wout.fsqr), float(wout.fsqz), float(wout.fsql))
        row = {
            "case": case,
            "scheme": scheme,
            "elapsed_s": f"{elapsed_s:.9g}",
            "iterations": len(trace),
            "time_per_iteration_s": f"{elapsed_s / len(trace):.9g}",
            "converged": str(residual <= float(vmec_input.ftol_array[-1])).lower(),
            "ftol": float(vmec_input.ftol_array[-1]),
            "max_residual": f"{residual:.9g}",
            "surface_error": "",
            "field_error": "",
            "axis_error": "",
            "current_error": "",
        }
        if member is not None:
            row.update(
                {
                    "surface_error": f"{EXACT_EQUILIBRIA.surface_error(member, wout):.9g}",
                    "field_error": f"{EXACT_EQUILIBRIA.field_error(member, wout):.9g}",
                    "axis_error": f"{EXACT_EQUILIBRIA.axis_error(member, wout):.9g}",
                    "current_error": f"{EXACT_EQUILIBRIA.current_error(member, wout):.9g}",
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    cma_input = vmecpp.VmecInput.from_file(
        ROOT / "src/vmecpp/cpp/vmecpp/test_data/cma.json"
    )
    cma_input.ftol_array[0] = 1e-8
    cma_input.ftol_array[1] = 1e-9
    rows = measure("cma", cma_input)

    member = EXACT_EQUILIBRIA.SHEARED
    exact_input = member.vmec_input(25, 12, 10).model_copy(
        update={
            "ns_array": NP.array([25], dtype=NP.int64),
            "ftol_array": NP.array([EXACT_EQUILIBRIA.ftol_for(25)]),
            "niter_array": NP.array([20000], dtype=NP.int64),
            "delt": 0.9,
            "return_outputs_even_if_not_converged": True,
        }
    )
    rows.extend(measure("exact_sheared_ns25", exact_input, member))

    with OUTPUT.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
