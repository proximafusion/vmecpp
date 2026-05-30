# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the boundary-optimization coupling (``vmecpp._optimization``).

``optimize_boundary`` runs the Python force-balance iteration as the equilibrium
solver inside a derivative-free optimization over the plasma boundary. The test
varies one boundary coefficient and checks that the optimizer recovers the
boundary whose converged equilibrium matches a target stored MHD energy.
"""

from pathlib import Path

import numpy as np

import vmecpp
from vmecpp._optimization import optimize_boundary

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def test_optimize_boundary_drives_energy_to_target():
    base = vmecpp.VmecInput.from_file(TEST_DATA / "solovev.json")
    # A loose single resolution keeps each equilibrium solve fast.
    base.ns_array = np.array([15], dtype=np.int64)
    base.ftol_array = np.array([1.0e-9])
    base.niter_array = np.array([2000], dtype=np.int64)

    m, col = 1, base.ntor + 0
    base_coeff = float(base.rbc[m, col])

    def set_params(vmec_input, x):
        modified = vmec_input.model_copy(deep=True)
        modified.rbc[m, col] = base_coeff + float(x[0])
        return modified

    # Target the stored energy of a known, slightly perturbed boundary.
    x_true = 0.03
    target_model, target_result = vmecpp.iterate(set_params(base, [x_true]), ns=15)
    assert target_result.converged
    energy_target = target_model.mhd_energy

    def objective(model, _result):
        return (model.mhd_energy - energy_target) ** 2

    start_model, _ = vmecpp.iterate(base, ns=15)
    j_start = (start_model.mhd_energy - energy_target) ** 2

    result = optimize_boundary(
        base,
        set_params,
        objective,
        x0=[0.0],
        ns=15,
        method="Nelder-Mead",
        options={"xatol": 1.0e-3, "fatol": 1.0e-18},
    )

    assert result.converged
    assert result.num_evaluations > 1
    # the optimizer drove the objective down and recovered the target boundary
    assert result.objective < 1.0e-2 * j_start
    assert abs(result.x[0] - x_true) < 1.0e-2
