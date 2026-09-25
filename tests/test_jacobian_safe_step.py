# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The Jacobian step limit in the Python-driven solve."""

from pathlib import Path

import vmecpp

CONSTELLARATION_NFP5 = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vmecpp"
    / "cpp"
    / "vmecpp"
    / "test_data"
    / "constellaration_nfp5.json"
)


def test_python_loop_takes_the_steps_of_the_cpp_solve() -> None:
    """On a boundary whose default time steps make flux surfaces cross, the Python
    iteration fails without the step limit and converges with it, in the iterations of
    the C++ solve: backups taken at the end of a shortened step are shortened with it in
    both."""
    vmec_input = vmecpp.VmecInput.from_file(CONSTELLARATION_NFP5)
    _, unlimited = vmecpp.solve_multigrid(vmec_input)
    assert unlimited[-1].failed

    limited_input = vmec_input.model_copy(update={"jacobian_safe_step": True})
    _, limited = vmecpp.solve_multigrid(limited_input)
    assert limited[-1].converged
    output = vmecpp.run(limited_input, verbose=False, max_threads=1)
    assert limited[-1].num_iterations == output.wout.itfsq
