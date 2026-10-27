# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
import pytest

import vmecpp

TEST_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "vmecpp"
    / "cpp"
    / "vmecpp"
    / "test_data"
)


@pytest.fixture(scope="module")
def magnetic_response():
    params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    params.number_of_r_grid_points = 31
    params.number_of_phi_grid_points = 36
    params.number_of_z_grid_points = 20
    return vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", params
    )


@pytest.mark.parametrize("max_threads", [1, 2])
def test_lcfs_residuals_and_additional_tolerance(
    magnetic_response, max_threads, tmp_path, capfd
):
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    assert vmec_input.ftol_edge == 0.0
    states = []
    legacy = vmecpp.run(
        vmec_input,
        magnetic_response,
        max_threads=max_threads,
        verbose=vmecpp.OutputMode.LEGACY,
        iteration_callback=states.append,
    )
    log = capfd.readouterr().out
    assert "FSQR_EDGE" in log
    assert "FSQZ_EDGE" in log
    wout = legacy.wout
    last = states[-1]
    assert last.fsqr_edge == wout.fsqr_edge > 0.0
    assert last.fsqz_edge == wout.fsqz_edge > 0.0
    assert last.ftol_edge == wout.ftol_edge == 0.0
    assert max(last.fsqr, last.fsqz, last.fsql) <= last.ftol
    assert max(last.fsqr_edge, last.fsqz_edge) > last.ftol

    stage_ends = {}
    for index, state in enumerate(states):
        stage_ends[state.multigrid_step] = index
    recorded = [
        s
        for index, s in enumerate(states)
        if s.restart_reason == 1 and index not in stage_ends.values()
    ]
    np.testing.assert_array_equal(
        [s.fsqr_edge for s in recorded], wout.force_residual_r_edge
    )
    np.testing.assert_array_equal(
        [s.fsqz_edge for s in recorded], wout.force_residual_z_edge
    )
    assert wout.force_residual_r_edge.shape == wout.force_residual_r.shape

    vmec_input.ftol_edge = max(wout.fsqr_edge, wout.fsqz_edge) / 10
    stricter = vmecpp.run(
        vmec_input, magnetic_response, max_threads=max_threads, verbose=False
    ).wout
    assert stricter.niter > wout.niter
    assert max(stricter.fsqr, stricter.fsqz, stricter.fsql) <= stricter.ftolv
    assert max(stricter.fsqr_edge, stricter.fsqz_edge) <= vmec_input.ftol_edge
    assert stricter.ftol_edge == vmec_input.ftol_edge

    restarted = vmecpp.run(
        vmec_input,
        magnetic_response,
        restart_from=legacy,
        max_threads=max_threads,
        verbose=False,
    ).wout
    assert max(restarted.fsqr_edge, restarted.fsqz_edge) <= vmec_input.ftol_edge

    path = tmp_path / "wout.nc"
    stricter.save(path)
    reloaded = vmecpp.VmecWOut.from_wout_file(path)
    for name in (
        "fsqr_edge",
        "fsqz_edge",
        "ftol_edge",
        "force_residual_r_edge",
        "force_residual_z_edge",
    ):
        np.testing.assert_array_equal(getattr(reloaded, name), getattr(stricter, name))

    vmec_input.niter_array = np.array([wout.niter])
    with pytest.raises(RuntimeError, match="ftol_edge"):
        vmecpp.run(
            vmec_input, magnetic_response, max_threads=max_threads, verbose=False
        )


def test_fixed_boundary_ignores_edge_tolerance():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    legacy = vmecpp.run(vmec_input, verbose=False, max_threads=1).wout
    vmec_input.ftol_edge = 1.0e-30
    states = []
    with_edge_tolerance = vmecpp.run(
        vmec_input, verbose=False, max_threads=1, iteration_callback=states.append
    ).wout
    assert with_edge_tolerance.niter == legacy.niter
    np.testing.assert_array_equal(with_edge_tolerance.rmnc, legacy.rmnc)
    np.testing.assert_array_equal(with_edge_tolerance.zmns, legacy.zmns)
    assert all(s.fsqr_edge == s.fsqz_edge == 0.0 for s in states)
    np.testing.assert_array_equal(with_edge_tolerance.force_residual_r_edge, 0.0)
    np.testing.assert_array_equal(with_edge_tolerance.force_residual_z_edge, 0.0)


@pytest.mark.parametrize("ftol_edge", [-1.0, np.nan, np.inf])
def test_invalid_edge_tolerance(ftol_edge):
    with pytest.raises(ValueError, match="ftol_edge"):
        vmecpp.VmecInput(ftol_edge=ftol_edge)

    cpp_input = vmecpp.VmecInput()._to_cpp_vmecindata()
    cpp_input.ftol_edge = ftol_edge
    with pytest.raises(ValueError, match="ftol_edge"):
        vmecpp._vmecpp.run(cpp_input, verbose=vmecpp._vmecpp.OutputMode.SILENT)


def test_edge_tolerance_json_round_trip(tmp_path):
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vmec_input.ftol_edge = 1.0e-12
    path = tmp_path / "input.json"
    path.write_text(vmec_input.to_json())
    assert vmecpp.VmecInput.from_file(path).ftol_edge == vmec_input.ftol_edge


def test_edge_diagnostics_hdf5_round_trip(magnetic_response, tmp_path):
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vmec_input.ftol_edge = 1.0e-10
    output = vmecpp._vmecpp.run(
        vmec_input._to_cpp_vmecindata(),
        magnetic_response._to_cpp_magnetic_field_response_table(),
        max_threads=1,
        verbose=vmecpp._vmecpp.OutputMode.SILENT,
    )
    path = tmp_path / "output.h5"
    output.save(path)
    reloaded = vmecpp._vmecpp.OutputQuantities.load(path)
    assert reloaded.indata.ftol_edge == vmec_input.ftol_edge
    for name in (
        "fsqr_edge",
        "fsqz_edge",
        "ftol_edge",
        "force_residual_r_edge",
        "force_residual_z_edge",
    ):
        np.testing.assert_array_equal(
            getattr(reloaded.wout, name), getattr(output.wout, name)
        )


def test_legacy_wout_has_no_edge_diagnostics():
    wout = vmecpp.VmecWOut.from_wout_file(TEST_DATA_DIR / "wout_cma.nc")
    assert np.isnan(wout.fsqr_edge)
    assert np.isnan(wout.fsqz_edge)
    assert wout.ftol_edge == 0.0
    assert wout.force_residual_r_edge.size == 0
    assert wout.force_residual_z_edge.size == 0
