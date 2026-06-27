# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""External optimizers reach the same equilibrium as the native solver.

The raw internal-basis force (gradient of VMEC's augmented functional) is the residual
F(x); F(x) = 0 at equilibrium. Both a native-style preconditioned descent and a
Jacobian-free Newton-Krylov solver drive it to zero and recover the native solver's
converged state and energy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from external_optimizers import (  # type: ignore
    make_model,
    reference_equilibrium,
    residual,
    solve_newton_krylov,
    solve_newton_krylov_preconditioned,
    solve_preconditioned_descent,
)

CMA = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vmecpp"
    / "cpp"
    / "vmecpp"
    / "test_data"
    / "cma.json"
)


@pytest.fixture(scope="module")
def reference():
    return reference_equilibrium()


@pytest.mark.parametrize(
    "solver",
    [
        solve_preconditioned_descent,
        solve_newton_krylov,
        solve_newton_krylov_preconditioned,
    ],
)
def test_optimizer_reaches_equilibrium(solver, reference):
    x_star, w_star = reference
    x, result = solver()
    # Force balance achieved.
    assert result.residual_norm < 1e-7
    # Same equilibrium as the native solver.
    assert abs(result.energy - w_star) < 1e-8
    assert np.linalg.norm(x - x_star) < 1e-5


def test_preconditioner_accelerates_newton_krylov():
    # VMEC's preconditioner is the inverse-Hessian approximation: using it as the
    # inner Krylov preconditioner cuts the force evaluations substantially.
    _, plain = solve_newton_krylov()
    _, precond = solve_newton_krylov_preconditioned()
    assert precond.force_evals < plain.force_evals


def test_cma_cold_start_exercises_non_axisymmetric_paths():
    # cma.json is a 3D stellarator (nfp=2, ntor=6) that ships no magnetic axis
    # (raxis/zaxis all zero), so the initial geometry has a singular Jacobian.
    # make_model reguesses the axis like the native solver, after which the raw
    # internal-basis force is well defined on the non-axisymmetric force chain.
    model = make_model(CMA, ns=25)
    x0 = np.asarray(model.get_state(), float)
    f0 = residual(model)(x0)
    assert np.all(np.isfinite(f0))
    assert np.linalg.norm(f0) > 0.0
