# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""External optimizers reach the same equilibrium as the native solver.

The raw internal-basis force (gradient of VMEC's augmented functional) is the
residual F(x); F(x) = 0 at equilibrium. Both a native-style preconditioned
descent and a Jacobian-free Newton-Krylov solver drive it to zero and recover
the native solver's converged state and energy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from external_optimizers import (
    reference_equilibrium,
    solve_newton_krylov,
    solve_preconditioned_descent,
)


@pytest.fixture(scope="module")
def reference():
    return reference_equilibrium()


@pytest.mark.parametrize("solver", [solve_preconditioned_descent, solve_newton_krylov])
def test_optimizer_reaches_equilibrium(solver, reference):
    x_star, w_star = reference
    x, result = solver()
    # Force balance achieved.
    assert result.residual_norm < 1e-7
    # Same equilibrium as the native solver.
    assert abs(result.energy - w_star) < 1e-8
    assert np.linalg.norm(x - x_star) < 1e-5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
