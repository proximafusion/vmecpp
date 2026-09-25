# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Abandoning a multigrid sequence whose early steps converge poorly.

``lgiveup`` ends the sequence at the first step that runs out of iterations with a
residual above ``fgiveup`` times its tolerance, instead of interpolating that state
onto the next grid and spending the remaining budget on it.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
CASE = TEST_DATA_DIR / "cth_like_fixed_bdy.json"


def _sequence_that_stalls_on_its_first_step() -> vmecpp.VmecInput:
    """Three grids, each with too few iterations to reach a tolerance of 1e-14."""
    vmec_input = vmecpp.VmecInput.from_file(CASE)
    vmec_input.ns_array = np.array([5, 11, 25])
    vmec_input.ftol_array = np.array([1.0e-14, 1.0e-14, 1.0e-14])
    vmec_input.niter_array = np.array([30, 30, 30])
    return vmec_input


def test_defaults_match_fortran():
    vmec_input = vmecpp.VmecInput.from_file(CASE)
    assert vmec_input.lgiveup is False
    assert vmec_input.fgiveup == 30.0


def test_sequence_runs_to_the_last_grid_when_unset():
    """Without the flag every grid of the schedule is attempted."""
    with pytest.raises(RuntimeError, match=r"ns = 25"):
        vmecpp.run(
            _sequence_that_stalls_on_its_first_step(), verbose=False, max_threads=1
        )


def test_sequence_stops_at_the_first_grid_that_converges_poorly():
    vmec_input = _sequence_that_stalls_on_its_first_step()
    vmec_input.lgiveup = True
    with pytest.raises(RuntimeError, match=r"ns = 5"):
        vmecpp.run(vmec_input, verbose=False, max_threads=1)


def test_a_step_within_fgiveup_of_its_tolerance_continues():
    """The threshold is what decides, so raising it past the residual carries on."""
    vmec_input = _sequence_that_stalls_on_its_first_step()
    vmec_input.lgiveup = True
    vmec_input.fgiveup = 1.0e30
    with pytest.raises(RuntimeError, match=r"ns = 25"):
        vmecpp.run(vmec_input, verbose=False, max_threads=1)


def test_a_converging_sequence_is_unaffected():
    """Every step meets its tolerance, so the flag has nothing to act on."""
    vmec_input = vmecpp.VmecInput.from_file(CASE)
    without = vmecpp.run(vmec_input, verbose=False, max_threads=1)

    vmec_input.lgiveup = True
    with_flag = vmecpp.run(vmec_input, verbose=False, max_threads=1)

    assert with_flag.wout.niter == without.wout.niter
    np.testing.assert_array_equal(with_flag.wout.rmnc, without.wout.rmnc)
    np.testing.assert_array_equal(with_flag.wout.zmns, without.wout.zmns)


def test_a_non_positive_fgiveup_is_rejected():
    vmec_input = vmecpp.VmecInput.from_file(CASE)
    vmec_input.lgiveup = True
    vmec_input.fgiveup = 0.0
    with pytest.raises(AttributeError, match="fgiveup"):
        vmecpp.run(vmec_input, verbose=False, max_threads=1)


def test_both_fields_survive_a_json_round_trip(tmp_path):
    vmec_input = vmecpp.VmecInput.from_file(CASE)
    vmec_input.lgiveup = True
    vmec_input.fgiveup = 12.5

    json_path = tmp_path / "with_giveup.json"
    json_path.write_text(vmec_input.to_json())
    reloaded = vmecpp.VmecInput.from_file(json_path)

    assert reloaded.lgiveup is True
    assert reloaded.fgiveup == 12.5
