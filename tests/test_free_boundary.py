# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.fixture
def makegrid_params() -> vmecpp.MakegridParameters:
    return vmecpp.MakegridParameters(
        normalize_by_currents=True,
        assume_stellarator_symmetry=True,
        number_of_field_periods=2,
        r_grid_minimum=0.1,
        r_grid_maximum=1.0,
        number_of_r_grid_points=10,
        z_grid_minimum=-0.5,
        z_grid_maximum=0.5,
        number_of_z_grid_points=20,
        number_of_phi_grid_points=20,
    )


def test_run_free_boundary_from_response_table():
    makegrid_params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    # Lower the makegrid resolution
    makegrid_params.number_of_r_grid_points = 31
    makegrid_params.number_of_phi_grid_points = 36
    makegrid_params.number_of_z_grid_points = 20
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", makegrid_params
    )
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vmec_output = vmecpp.run(vmec_input, response)
    assert vmec_output.wout.volume == pytest.approx(0.307512, 1e-5, 1e-5)

    # Test hot-restart functionality
    hot_restart_output = vmecpp.run(
        vmec_input, magnetic_field=response, restart_from=vmec_output
    )
    # The change in initial guess should only result in a tiny change in output
    assert hot_restart_output.wout.volume != vmec_output.wout.volume
    assert hot_restart_output.wout.volume == pytest.approx(
        vmec_output.wout.volume, 1e-5, 1e-5
    )


def test_makegrid_parameters_conversion(makegrid_params):
    # Convert resolution parameters to C++ and back to Python
    cpp_params = makegrid_params._to_cpp_makegrid_parameters()
    py_params = vmecpp.MakegridParameters._from_cpp_makegrid_parameters(cpp_params)
    assert makegrid_params == py_params


def test_response_table_conversion(makegrid_params):
    # Create a dummy response table
    response_table = vmecpp.MagneticFieldResponseTable(
        parameters=makegrid_params,
        b_r=np.linspace(0, 1, 10 * 20 * 20).reshape((1, 10 * 20 * 20)),
        b_z=np.linspace(0, 1, 10 * 20 * 20).reshape((1, 10 * 20 * 20)),
        b_p=np.linspace(0, 1, 10 * 20 * 20).reshape((1, 10 * 20 * 20)),
    )

    # Convert to C++ and back to Python
    cpp_response_table = response_table._to_cpp_magnetic_field_response_table()
    py_response_table = (
        vmecpp.MagneticFieldResponseTable._from_cpp_magnetic_field_response_table(
            cpp_response_table
        )
    )

    np.testing.assert_allclose(response_table.b_r, py_response_table.b_r)
    np.testing.assert_allclose(response_table.b_z, py_response_table.b_z)
    np.testing.assert_allclose(response_table.b_p, py_response_table.b_p)


def test_response_table_round_trip(makegrid_params):
    """Check that converting C++ to Python and back returns the original C++ object
    instead of a new copy."""
    cpp_response1 = vmecpp._vmecpp.MagneticFieldResponseTable(
        parameters=makegrid_params._to_cpp_makegrid_parameters(),
        b_r=np.linspace(0, 1, 10).reshape((1, 10)),
        b_z=np.linspace(0, 1, 10).reshape((1, 10)),
        b_p=np.linspace(0, 1, 10).reshape((1, 10)),
    )
    py_response = (
        vmecpp.MagneticFieldResponseTable._from_cpp_magnetic_field_response_table(
            cpp_response1
        )
    )
    # Read-write arrays
    assert py_response.b_p.flags["WRITEABLE"]
    assert py_response.b_p.flags["C_CONTIGUOUS"]
    assert py_response.b_p.flags["ALIGNED"]

    cpp_response2 = py_response._to_cpp_magnetic_field_response_table()

    assert isinstance(cpp_response1, vmecpp._vmecpp.MagneticFieldResponseTable)
    assert cpp_response1 == cpp_response2

    np.testing.assert_equal(cpp_response1.b_r, cpp_response2.b_r)
    np.testing.assert_equal(cpp_response1.b_z, cpp_response2.b_z)
    np.testing.assert_equal(cpp_response1.b_p, cpp_response2.b_p)

    # Since they reference the same memory, changes should be visible in all objects
    cpp_response2.b_r[0, 0] = 5
    assert cpp_response1.b_r[0, 0] == 5
    np.testing.assert_equal(cpp_response1.b_r, cpp_response2.b_r)
    np.testing.assert_equal(py_response.b_r, cpp_response2.b_r)


def test_response_table_py_to_cpp_copy(makegrid_params):
    """When creating a MagneticFieldResponseTable C++ object from a Pydantic object, the
    arrays are copies and NOT linked, but copied."""
    response_table = vmecpp.MagneticFieldResponseTable(
        parameters=makegrid_params,
        b_r=np.linspace(0, 1, 10 * 20 * 20).reshape((1, 10 * 20 * 20)),
        b_z=np.linspace(0, 1, 10 * 20 * 20).reshape((1, 10 * 20 * 20)),
        b_p=np.linspace(0, 1, 10 * 20 * 20).reshape((1, 10 * 20 * 20)),
    )

    # Convert to C++, this will copy the data
    cpp_response_table = response_table._to_cpp_magnetic_field_response_table()
    assert cpp_response_table.b_p.flags["WRITEABLE"]
    assert cpp_response_table.b_p.flags["C_CONTIGUOUS"]
    assert cpp_response_table.b_p.flags["ALIGNED"]
    assert cpp_response_table.b_r.base is not response_table.b_r.base


def test_magnetic_field_response_table_loading(makegrid_params):
    invalid_coils_file = "path/to/invalid_coils_file"
    with pytest.raises(RuntimeError):
        vmecpp.MagneticFieldResponseTable.from_coils_file(
            invalid_coils_file, makegrid_params
        )
    # Test with a valid file
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR
        / ".."
        / "common"
        / "makegrid_lib"
        / "test_data"
        / "coils.test_symmetric_even",
        makegrid_params,
    )
    assert response.b_p.shape == response.b_r.shape
    assert response.b_z.shape == response.b_p.shape
    assert response.b_r[0, 0] == pytest.approx(-9.78847973e-06, abs=1e-8, rel=1e-6)
