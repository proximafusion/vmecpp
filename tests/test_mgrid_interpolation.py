# SPDX-FileCopyrightText: 2026 Proxima Fusion GmbH <info@proximafusion.com>
# SPDX-License-Identifier: MIT
"""Vacuum interpolation selection, compatibility and equilibrium accuracy."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

DATA = Path(__file__).parents[1] / "src/vmecpp/cpp/vmecpp/test_data"


@pytest.mark.parametrize("scheme", list(vmecpp.MGridInterpolation))
def test_interpolation_round_trip(scheme, tmp_path):
    inputs = vmecpp.VmecInput.from_file(DATA / "cth_like_free_bdy.json")
    inputs.mgrid_interpolation = scheme
    cpp = inputs._to_cpp_vmecindata()
    restored = vmecpp.VmecInput._from_cpp_vmecindata(cpp)
    assert restored.mgrid_interpolation == scheme
    path = tmp_path / "input.json"
    inputs.save(path)
    assert vmecpp.VmecInput.from_file(path).mgrid_interpolation == scheme


def test_interpolation_default_and_validation():
    assert vmecpp.VmecInput().mgrid_interpolation == vmecpp.MGridInterpolation.CUBIC
    assert (
        vmecpp.VmecInput.from_file(DATA / "cth_like_free_bdy.json").mgrid_interpolation
        == vmecpp.MGridInterpolation.CUBIC
    )
    with pytest.raises(ValueError, match="mgrid_interpolation"):
        vmecpp.VmecInput.model_validate({"mgrid_interpolation": "nearest"})


def test_historical_output_interpolation():
    inputs = vmecpp.VmecInput.from_file(DATA / "solovev.json")
    saved = vmecpp.run(inputs, max_threads=2, verbose=False).model_dump(mode="json")
    del saved["input"]["mgrid_interpolation"]
    restored = vmecpp.VmecOutput.model_validate(saved)
    assert restored.input.mgrid_interpolation == vmecpp.MGridInterpolation.LINEAR
    assert "mgrid_interpolation" not in saved["input"]


@pytest.mark.parametrize("family", ["cth_like", "solovev"])
def test_cubic_reduces_equilibrium_field_table_error(family):
    parameters = vmecpp.MakegridParameters.from_file(
        DATA / f"makegrid_parameters_{family}.json"
    )
    parameters.number_of_r_grid_points = 129
    parameters.number_of_z_grid_points = 129
    field = vmecpp.MagneticFieldResponseTable.from_coils_file(
        DATA / f"coils.{family}", parameters
    )
    inputs = vmecpp.VmecInput.from_file(DATA / f"{family}_free_bdy.json")
    inputs.ftol_array[-1] = 1e-12
    inputs.niter_array[:] = 12000
    reference = vmecpp.run(inputs, field, max_threads=2, verbose=False).wout
    coarse_parameters = parameters.model_copy(deep=True)
    coarse_parameters.number_of_r_grid_points = 33
    coarse_parameters.number_of_z_grid_points = 33
    arrays = {}
    for component in ["b_r", "b_p", "b_z"]:
        values = getattr(field, component)
        grid = values.reshape(-1, parameters.number_of_phi_grid_points, 129, 129)
        arrays[component] = np.ascontiguousarray(grid[:, :, ::4, ::4]).reshape(
            values.shape[0], -1
        )
    coarse = vmecpp.MagneticFieldResponseTable(parameters=coarse_parameters, **arrays)
    errors = {}
    for scheme in vmecpp.MGridInterpolation:
        inputs.mgrid_interpolation = scheme
        output = vmecpp.run(inputs, coarse, max_threads=2, verbose=False).wout
        assert max(output.fsqr, output.fsqz, output.fsql) <= inputs.ftol_array[-1]
        errors[scheme] = np.linalg.norm(output.rmnc - reference.rmnc) + np.linalg.norm(
            output.zmns - reference.zmns
        )
    assert errors[vmecpp.MGridInterpolation.LINEAR] > 1e-5
    assert (
        errors[vmecpp.MGridInterpolation.CUBIC]
        < 0.1 * errors[vmecpp.MGridInterpolation.LINEAR]
    )
