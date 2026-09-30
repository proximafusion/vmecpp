# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""``backup_evaluated_state``: a restart backup of the last evaluated state instead of
the state the time step advanced to."""

from pathlib import Path

import numpy as np

import vmecpp

TEST_DATA = Path(__file__).resolve().parents[1] / "src" / "vmecpp" / "cpp" / "vmecpp"
TEST_DATA = TEST_DATA / "test_data"


def _run(case: str, backup_evaluated_state: bool) -> vmecpp.VmecWOut:
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA / f"{case}.json").model_copy(
        update={"backup_evaluated_state": backup_evaluated_state}
    )
    return vmecpp.run(vmec_input, verbose=False).wout


def _restarts(wout: vmecpp.VmecWOut) -> int:
    return sum(reason != 1 for _, reason in wout.restart_reasons)


def test_a_run_without_restarts_is_unchanged():
    """The backup is only restored on a restart, so without one the option changes
    nothing."""
    default = _run("cth_like_fixed_bdy", False)
    evaluated = _run("cth_like_fixed_bdy", True)
    assert _restarts(default) == 0
    np.testing.assert_array_equal(evaluated.fsqt, default.fsqt)
    np.testing.assert_array_equal(evaluated.rmnc, default.rmnc)


def test_a_run_that_restarts_takes_the_evaluated_backup():
    default = _run("cma", False)
    evaluated = _run("cma", True)
    assert _restarts(default) > 0
    assert default.ier_flag == 0
    assert evaluated.ier_flag == 0
    assert not np.array_equal(evaluated.fsqt, default.fsqt)
