# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest

import vmecpp


@pytest.fixture
def makegrid_params() -> vmecpp.MakegridParameters:
    return vmecpp.MakegridParameters(
        normalize_by_currents=True,
        assume_stellarator_symmetry=True,
        number_of_field_periods=5,
        r_grid_minimum=0.1,
        r_grid_maximum=1.0,
        number_of_r_grid_points=10,
        z_grid_minimum=-0.5,
        z_grid_maximum=0.5,
        number_of_z_grid_points=20,
        number_of_phi_grid_points=30,
    )


def test_makegrid_parameters_conversion(makegrid_params):
    # Convert to Python and back to C++
    cpp_params = makegrid_params._to_cpp_makegrid_parameters()
    py_params = vmecpp.MakegridParameters._from_cpp_makegrid_parameters(cpp_params)
    assert makegrid_params == py_params


def test_magnetic_field_response_table_loading(makegrid_params):
    coils_file = "path/to/invalid_coils_file"
    with pytest.raises(RuntimeError):
        vmecpp.MagneticFieldResponseTable.from_coils_file(coils_file, makegrid_params)
    # Test with a valid file
    coils_file = (
        Path(__file__).parent.parent
        / "src"
        / "vmecpp"
        / "cpp"
        / "vmecpp"
        / "common"
        / "makegrid_lib"
        / "test_data"
        / "coils.test_symmetric_even"
    )
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        coils_file, makegrid_params
    )
    assert response.b_p.shape == response.b_r.shape
    assert response.b_z.shape == response.b_p.shape
    assert response.b_r[0, 0] == pytest.approx(-9.78847973e-06, abs=1e-8, rel=1e-6)
