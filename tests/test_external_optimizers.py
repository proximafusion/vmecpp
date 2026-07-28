# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""External optimizers reach force balance on axisymmetric and 3D cases.

The Solov'ev equilibrium has a reproducible internal state, so the 2D test compares the
full state vector with the native solver. In 3D, the poloidal parameterization is not
unique and spectral condensation only regularizes that coordinate freedom. The 3D test
therefore compares the residual and energy instead of coordinate-dependent Fourier
coefficients.
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
    solve_newton_hvp,
    solve_newton_krylov,
    solve_newton_krylov_preconditioned,
    solve_preconditioned_descent,
    solve_vmecpp,
)

ROOT = Path(__file__).resolve().parents[1]
SOLOVEV = ROOT / "examples" / "data" / "solovev.json"
CTH_LIKE = ROOT / "examples" / "data" / "cth_like_fixed_bdy.json"
CMA = ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data" / "cma.json"

SOLVERS = (
    solve_preconditioned_descent,
    solve_newton_krylov,
    solve_newton_krylov_preconditioned,
    solve_newton_hvp,
)


def _solve_all(input_path):
    return {solver.__name__: solver(input_path, ns=11) for solver in SOLVERS}


@pytest.fixture(scope="module")
def reference_2d():
    return reference_equilibrium(SOLOVEV, ns=11)


@pytest.fixture(scope="module")
def reference_3d():
    return reference_equilibrium(CTH_LIKE, ns=11)


@pytest.fixture(scope="module")
def solutions_2d():
    return _solve_all(SOLOVEV)


@pytest.fixture(scope="module")
def solutions_3d():
    return _solve_all(CTH_LIKE)


def test_optimizers_reach_2d_equilibrium(reference_2d, solutions_2d):
    x_star, w_star = reference_2d
    for name, (x, result) in solutions_2d.items():
        assert result.residual_norm < 1e-7, name
        assert abs(result.energy - w_star) < 1e-8, name
        assert np.linalg.norm(x - x_star) < 1e-5, name


def test_optimizers_reach_3d_force_balance(reference_3d, solutions_3d):
    _, w_star = reference_3d
    for name, (_, result) in solutions_3d.items():
        assert result.residual_norm < 1e-7, name
        assert np.isclose(result.energy, w_star, rtol=1e-4, atol=0.0), name


def test_preconditioner_reduces_newton_krylov_force_evaluations(solutions_3d):
    plain = solutions_3d[solve_newton_krylov.__name__][1]
    preconditioned = solutions_3d[solve_newton_krylov_preconditioned.__name__][1]
    assert preconditioned.force_evals < plain.force_evals


def test_hvp_newton_uses_fewer_outer_iterations_than_descent(solutions_3d):
    newton = solutions_3d[solve_newton_hvp.__name__][1]
    descent = solutions_3d[solve_preconditioned_descent.__name__][1]
    assert newton.outer_iters < 20
    assert newton.outer_iters < descent.outer_iters


def test_native_metrics_match_external_metric_definitions(reference_3d):
    x_star, w_star = reference_3d
    x, result = solve_vmecpp(CTH_LIKE, ns=11)

    assert result.energy == w_star
    np.testing.assert_array_equal(x, x_star)
    assert result.residual_norm < 1e-8
    assert result.force_evals == result.outer_iters + 1
    assert result.outer_iters > 0


def test_newton_krylov_reports_outer_iterations(solutions_2d):
    result = solutions_2d[solve_newton_krylov.__name__][1]
    assert result.outer_iters > 0


def test_cma_cold_start_exercises_non_axisymmetric_paths():
    model = make_model(CMA, ns=25)
    x0 = np.asarray(model.get_state(), float)
    f0 = residual(model)(x0)
    assert np.all(np.isfinite(f0))
    assert np.linalg.norm(f0) > 0.0

    rng = np.random.default_rng(0)
    v = rng.standard_normal(x0.size)
    hv = np.asarray(model.hessian_vector_product(np.ascontiguousarray(v)), float)
    assert np.all(np.isfinite(hv))
    assert np.linalg.norm(hv) > 0.0
