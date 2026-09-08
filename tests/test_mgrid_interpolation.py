# SPDX-FileCopyrightText: 2026 Proxima Fusion GmbH <info@proximafusion.com>
# SPDX-License-Identifier: MIT
"""Equilibrium convergence under vacuum field table refinement."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

DATA = Path(__file__).parents[1] / "src/vmecpp/cpp/vmecpp/test_data"


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
    errors = []
    for stride in [4, 2]:
        coarse_parameters = parameters.model_copy(deep=True)
        coarse_parameters.number_of_r_grid_points = 128 // stride + 1
        coarse_parameters.number_of_z_grid_points = 128 // stride + 1
        arrays = {}
        for component in ["b_r", "b_p", "b_z"]:
            values = getattr(field, component)
            grid = values.reshape(-1, parameters.number_of_phi_grid_points, 129, 129)
            arrays[component] = np.ascontiguousarray(
                grid[:, :, ::stride, ::stride]
            ).reshape(values.shape[0], -1)
        coarse = vmecpp.MagneticFieldResponseTable(
            parameters=coarse_parameters, **arrays
        )
        output = vmecpp.run(inputs, coarse, max_threads=2, verbose=False).wout
        assert max(output.fsqr, output.fsqz, output.fsql) <= inputs.ftol_array[-1]
        errors.append(
            np.linalg.norm(output.rmnc - reference.rmnc)
            + np.linalg.norm(output.zmns - reference.zmns)
        )
    # Halving the spacing gives fourth-order field interpolation. Check that
    # its effect on the converged Fourier geometry improves faster than linear.
    assert errors[1] < errors[0] / 4
