# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A free-boundary input without its mgrid file, with
``fixed_boundary_without_mgrid``."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

TEST_DATA_DIR = (
    Path(__file__).parent.parent / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
)


@pytest.fixture(scope="module")
def free_boundary_input() -> vmecpp.VmecInput:
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")


@pytest.mark.parametrize("mgrid_file", ["does_not_exist.nc", "NONE"])
def test_runs_the_fixed_boundary_problem(free_boundary_input, mgrid_file):
    """Without an mgrid file the run is the fixed-boundary run of the same input."""
    missing = free_boundary_input.model_copy(
        update={"mgrid_file": mgrid_file, "fixed_boundary_without_mgrid": True}
    )
    fixed = free_boundary_input.model_copy(update={"lfreeb": False})
    wout = vmecpp.run(missing, max_threads=1, verbose=False).wout
    expected = vmecpp.run(fixed, max_threads=1, verbose=False).wout
    assert not wout.lfreeb
    np.testing.assert_array_equal(wout.rmnc, expected.rmnc)
    np.testing.assert_array_equal(wout.zmns, expected.zmns)
    np.testing.assert_array_equal(wout.iotaf, expected.iotaf)


def test_fails_without_the_flag(free_boundary_input):
    missing = free_boundary_input.model_copy(update={"mgrid_file": "does_not_exist.nc"})
    with pytest.raises(RuntimeError, match=r"does_not_exist\.nc"):
        vmecpp.run(missing, verbose=False)


def test_a_response_table_keeps_the_free_boundary(free_boundary_input):
    makegrid_params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    makegrid_params.number_of_r_grid_points = 31
    makegrid_params.number_of_phi_grid_points = 36
    makegrid_params.number_of_z_grid_points = 20
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", makegrid_params
    )
    vmec_input = free_boundary_input.model_copy(
        update={"mgrid_file": "does_not_exist.nc", "fixed_boundary_without_mgrid": True}
    )
    wout = vmecpp.run(vmec_input, magnetic_field=response, verbose=False).wout
    assert wout.lfreeb
    assert wout.volume == pytest.approx(0.306990, 1e-5, 1e-5)
