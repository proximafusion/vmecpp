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
    vmec_output = vmecpp.run(vmec_input, response, verbose=False)
    assert vmec_output.wout.volume == pytest.approx(0.306990, 1e-5, 1e-5)

    # Test hot-restart functionality. w/o perturbation, should converge immediately
    hot_restart_output = vmecpp.run(
        vmec_input, magnetic_field=response, restart_from=vmec_output
    )
    assert hot_restart_output.wout.itfsq >= 1
    # Max iterations for includeEdgeRZForces special case
    assert hot_restart_output.wout.itfsq <= 50
    assert hot_restart_output.wout.volume == pytest.approx(
        vmec_output.wout.volume, 1e-5, 1e-5
    )
    FLUX_INCREASE = 1.02
    response.b_r *= FLUX_INCREASE
    response.b_p *= FLUX_INCREASE
    response.b_z *= FLUX_INCREASE
    hot_restart_output_perturbed = vmecpp.run(
        vmec_input, magnetic_field=response, restart_from=vmec_output, verbose=False
    )
    assert hot_restart_output_perturbed.wout.itfsq > 1
    assert hot_restart_output_perturbed.wout.itfsq < vmec_output.wout.itfsq
    # The change in field strength should only result in a proportional change in volume
    assert hot_restart_output_perturbed.wout.volume * FLUX_INCREASE == pytest.approx(
        vmec_output.wout.volume, 1e-3, 1e-3
    )


def test_always_include_edge_forces():
    """With always_include_edge_forces a free-boundary run converges only once the force
    on its boundary meets ftol too, and a fixed-boundary run is unchanged."""
    makegrid_params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    makegrid_params.number_of_r_grid_points = 31
    makegrid_params.number_of_phi_grid_points = 36
    makegrid_params.number_of_z_grid_points = 20
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", makegrid_params
    )
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vmec_input.always_include_edge_forces = True
    wout = vmecpp.run(vmec_input, response, verbose=False).wout
    ftol = vmec_input.ftol_array[-1]
    assert wout.lcfs_force[-1] <= wout.fsqr + wout.fsqz <= 2.0 * ftol

    fixed_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    plain = vmecpp.run(fixed_input, verbose=False).wout
    fixed_input.always_include_edge_forces = True
    edge = vmecpp.run(fixed_input, verbose=False).wout
    assert edge.itfsq == plain.itfsq
    np.testing.assert_array_equal(edge.rmnc, plain.rmnc)


def test_vacuum_potential_cutoffs():
    """The vacuum potential's cutoffs default to the plasma's, must not fall below them,
    and a run with larger ones converges on the same grid."""
    makegrid_params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    response_table = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", makegrid_params
    )
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    assert vmec_input.vacuum_mpol == 0
    assert vmec_input.vacuum_ntor == 0

    assert isinstance(vmec_input.ntor, int)
    below = vmec_input.model_copy(deep=True)
    below.vacuum_ntor = vmec_input.ntor - 1
    with pytest.raises(Exception, match="vacuum_ntor"):
        vmecpp.run(below, response_table, verbose=False)

    expanded = vmec_input.model_copy(deep=True)
    expanded.vacuum_mpol = 8
    expanded.vacuum_ntor = 8
    output = vmecpp.run(expanded, response_table, verbose=False)
    assert output.wout.potvac is not None
    assert output.wout.potvac.shape == (2 * (8 + 2) * (2 * 8 + 1),)


def test_raise_invalid_nzeta():
    makegrid_params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    # Lower the makegrid resolution
    makegrid_params.number_of_r_grid_points = 3
    makegrid_params.number_of_phi_grid_points = 24
    makegrid_params.number_of_z_grid_points = 2
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", makegrid_params
    )
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    assert vmec_input.nzeta != makegrid_params.number_of_phi_grid_points
    with pytest.raises(ValueError, match=r"MGridProvider has \d"):
        vmecpp.run(vmec_input, response, verbose=False)


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


def test_invalid_path_magnetic_field_response_table(makegrid_params):
    invalid_coils_file = "path/to/invalid_coils_file"
    with pytest.raises(RuntimeError):
        vmecpp.MagneticFieldResponseTable.from_coils_file(
            invalid_coils_file, makegrid_params
        )


def test_magnetic_field_response_table_scaled_raw(makegrid_params):
    assert makegrid_params.normalize_by_currents
    coils_path = (
        TEST_DATA_DIR
        / ".."
        / "common"
        / "makegrid_lib"
        / "test_data"
        / "coils.test_symmetric_odd"
    )
    scaled_response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        coils_path, makegrid_params
    )
    makegrid_params.normalize_by_currents = False
    raw_response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        coils_path, makegrid_params
    )
    assert scaled_response.b_r.shape[0] == 2
    assert scaled_response.b_r.shape == raw_response.b_r.shape
    assert scaled_response.b_z.shape == raw_response.b_z.shape
    assert scaled_response.b_p.shape == raw_response.b_p.shape
    # The scaled response should be the raw response divided by the currents
    extcur = [96.0, -36.0]
    for i, single_current in enumerate(extcur):
        np.testing.assert_allclose(
            scaled_response.b_r[i] * single_current,
            raw_response.b_r[i],
            atol=1e-14,
            rtol=0.0,
        )


def _lagrange_field(table: vmecpp.MagneticFieldResponseTable, extcur):
    """MagneticFieldFunction with the interpolation that VMEC++ applies to a
    response table: a 4-node Lagrange stencil in R and Z on the toroidal grid."""
    p = table.parameters
    shape = (p.number_of_phi_grid_points, p.number_of_z_grid_points, -1)
    grids = [
        np.tensordot(extcur, getattr(table, name), axes=1).reshape(shape)
        for name in ("b_r", "b_p", "b_z")
    ]
    r_step = (p.r_grid_maximum - p.r_grid_minimum) / (p.number_of_r_grid_points - 1)
    z_step = (p.z_grid_maximum - p.z_grid_minimum) / (p.number_of_z_grid_points - 1)

    def weights(index, num_nodes):
        start = np.clip(np.floor(index).astype(int) - 1, 0, num_nodes - 4)
        u = (index - start)[:, None]
        nodes = np.arange(4)
        w = np.ones((index.size, 4))
        for j in range(4):
            w *= np.where(nodes == j, 1.0, (u - j) / np.where(nodes == j, 1, nodes - j))
        return start, w

    def field(r, phi, z):
        k = (
            np.rint(
                phi
                * p.number_of_field_periods
                * p.number_of_phi_grid_points
                / (2 * np.pi)
            ).astype(int)
            % p.number_of_phi_grid_points
        )
        r_start, r_w = weights(
            (r - p.r_grid_minimum) / r_step, p.number_of_r_grid_points
        )
        z_start, z_w = weights(
            (z - p.z_grid_minimum) / z_step, p.number_of_z_grid_points
        )
        b = []
        for grid in grids:
            values = np.zeros_like(r)
            for j in range(4):
                for i in range(4):
                    values += z_w[:, j] * r_w[:, i] * grid[k, z_start + j, r_start + i]
            b.append(values)
        return tuple(b)

    return field


@pytest.mark.parametrize("free_boundary_method", ["nestor", "only_coils"])
def test_run_free_boundary_from_callable(free_boundary_method):
    """A callable with the response table's interpolation reproduces its run."""
    makegrid_params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    makegrid_params.number_of_r_grid_points = 31
    makegrid_params.number_of_phi_grid_points = 36
    makegrid_params.number_of_z_grid_points = 20
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", makegrid_params
    )
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vmec_input.free_boundary_method = free_boundary_method
    if free_boundary_method == "only_coils":
        vmec_input.curtor = 0.0
        vmec_input.pres_scale = 0.0

    reference = vmecpp.run(vmec_input, response, max_threads=2, verbose=False).wout
    field = _lagrange_field(response, vmec_input.extcur)
    wout = vmecpp.run(vmec_input, field, max_threads=2, verbose=False).wout

    assert wout.itfsq == reference.itfsq
    np.testing.assert_allclose(wout.rmnc, reference.rmnc, rtol=0, atol=1e-11)
    np.testing.assert_allclose(wout.zmns, reference.zmns, rtol=0, atol=1e-11)


def test_magnetic_field_callable_errors():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")

    class FieldError(Exception):
        pass

    def raising_field(*_):
        raise FieldError

    with pytest.raises(FieldError):
        vmecpp.run(vmec_input, raising_field, verbose=False)

    def short_field(r, *_):
        return r[:-1], r[:-1], r[:-1]

    with pytest.raises(ValueError, match="magnetic_field must return arrays"):
        vmecpp.run(vmec_input, short_field, verbose=False)

    with pytest.raises(TypeError, match="magnetic_field must be"):
        vmecpp.run(vmec_input, 1.0, verbose=False)  # type: ignore[arg-type]
