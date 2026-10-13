"""Newton algebra and restoration checks; native tests are Enzyme smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff_newton
from vmecpp.autodiff_diagnostics import raw_force

from .harness import small_input, solved_model


class NonlinearModel:
    ns = 3
    mpol = 2
    ntor = 0
    lasym = False
    lfreeb = False
    lforbal = False
    always_fix_m1_gauge = True
    has_exact_force_jacobian = True
    mixing = np.asarray([[2.0, 1.0], [-1.0, 3.0]])

    def __init__(self, initial=(1.2, 1.3)):
        self.state = np.asarray([*initial, 7.0, 8.0, 9.0])
        self.forces = np.zeros(5)
        self.need_restart = False
        self.reject_below = None
        self.evaluated_state = None
        self.rejections = 0
        self.assembled_states = []

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state.copy()

    def geometry_state_vjp(self, coefficients, flux):
        assert not np.any(coefficients)
        assert flux.size == 0

    def evaluate(self, iteration, previous, full_update, residual_only):
        assert (iteration, previous, full_update, residual_only) == (2, 2, False, True)
        self.evaluated_state = self.state.copy()
        self.need_restart = (
            self.reject_below is not None and self.state[0] < self.reject_below
        )
        self.rejections += int(self.need_restart)
        self.forces[:] = 0.0
        self.forces[:2] = self.mixing @ (self.state[:2] ** 2 - 1.0)

    def get_forces(self):
        return self.forces

    def jacobian(self):
        return self.mixing @ np.diag(2.0 * self.state[:2])


class DenseFactors:
    def __init__(self, model):
        self.matrix = model.jacobian().T

    def factorize(self):
        pass

    def solve(self, rhs):
        return np.linalg.solve(self.matrix, rhs)

    def solve_transpose(self, rhs):
        return np.linalg.solve(self.matrix.T, rhs)


@pytest.fixture
def mock_operator(monkeypatch):
    monkeypatch.setattr(
        autodiff_newton.autodiff,
        "_interior_and_boundary",
        lambda _: (np.asarray([0, 1, 3, 4]), np.asarray([2])),
    )
    monkeypatch.setattr(
        autodiff_newton.autodiff, "_gauge_entries", lambda _: np.asarray([3])
    )

    def probe(_model, interior):
        np.testing.assert_array_equal(interior, [0, 1, 4])
        return np.asarray([0, 1])

    def assemble(model, active):
        np.testing.assert_array_equal(active, [0, 1])
        model.assembled_states.append(model.get_state())
        # Exact products overwrite the model force buffers.
        factors = DenseFactors(model)
        model.forces[:] = 100.0
        return factors

    monkeypatch.setattr(
        autodiff_newton.autodiff, "_structural_nullfree_interior", probe
    )
    monkeypatch.setattr(
        autodiff_newton.autodiff, "_assemble_block_tridiagonal", assemble
    )


@pytest.mark.usefixtures("mock_operator")
def test_nonsymmetric_newton_quadratic_decline_and_final_factors():
    model = NonlinearModel()
    initial = model.get_state()
    result = autodiff_newton.newton_polish(model, 6, 1.0e-12)
    assert result.converged
    norms = result.residual_history
    assert len(norms) >= 4
    assert np.all(np.diff(norms) < 0)
    assert np.all(norms[1:4] < norms[:3] ** 2)
    np.testing.assert_allclose(result.state[:2], 1.0, atol=1.0e-12)
    np.testing.assert_array_equal(result.state[2:], initial[2:])
    np.testing.assert_array_equal(model.get_state(), result.state)
    np.testing.assert_array_equal(model.evaluated_state, result.state)
    np.testing.assert_array_equal(model.assembled_states[-1], result.state)
    rhs = np.asarray([0.7, -0.2])
    np.testing.assert_allclose(
        model.jacobian() @ result.factorization.solve_transpose(rhs), rhs
    )
    np.testing.assert_allclose(
        model.jacobian().T @ result.factorization.solve(rhs), rhs
    )
    assert np.linalg.norm(raw_force(model)[result.active]) == norms[-1]


@pytest.mark.usefixtures("mock_operator")
def test_backtracking_accepts_smaller_step():
    model = NonlinearModel((0.1, 0.1))
    result = autodiff_newton.newton_polish(model, 1, 1.0e-12)
    assert not result.converged
    assert len(result.residual_history) == 2
    assert result.residual_history[-1] < result.residual_history[0]
    assert result.state[0] < 0.1 + (1.0 - 0.1**2) / 0.2


@pytest.mark.usefixtures("mock_operator")
def test_invalid_candidates_restore_last_state():
    model = NonlinearModel()
    model.reject_below = model.state[0]
    initial = model.get_state()
    result = autodiff_newton.newton_polish(model, 3, 1.0e-12)
    assert not result.converged
    assert model.rejections == 24
    assert len(result.residual_history) == 1
    np.testing.assert_array_equal(result.state, initial)
    np.testing.assert_array_equal(model.get_state(), initial)
    np.testing.assert_array_equal(model.evaluated_state, initial)
    assert not model.need_restart


@pytest.mark.usefixtures("mock_operator")
def test_assembly_failure_restores_raw_force(monkeypatch):
    model = NonlinearModel()
    initial = model.get_state()
    expected_force = raw_force(model)

    def fail(model, _active):
        model.forces[:] = np.nan
        message = "assembly failed"
        raise RuntimeError(message)

    monkeypatch.setattr(autodiff_newton.autodiff, "_assemble_block_tridiagonal", fail)
    with pytest.raises(RuntimeError, match="assembly failed"):
        autodiff_newton.newton_polish(model, 1, 1.0e-12)
    np.testing.assert_array_equal(model.get_state(), initial)
    np.testing.assert_array_equal(model.get_forces(), expected_force)


def test_force_evaluation_cannot_clear_prescribed_entries(monkeypatch):
    model = NonlinearModel()
    initial = model.get_state()
    evaluate = model.evaluate

    def clear_entry(*args):
        model.state[-1] = 0.0
        evaluate(*args)

    monkeypatch.setattr(model, "evaluate", clear_entry)
    with pytest.raises(RuntimeError, match="changed prescribed state entries"):
        autodiff_newton.newton_polish(model, 1, 1.0e-12)
    np.testing.assert_array_equal(model.get_state(), initial)


@pytest.mark.usefixtures("mock_operator")
@pytest.mark.parametrize(
    ("initial", "converged"), [((1.0, 1.0), True), ((1.2, 1.3), False)]
)
def test_zero_steps_returns_final_factors(initial, converged):
    model = NonlinearModel(initial)
    result = autodiff_newton.newton_polish(model, 0, 1.0e-12)
    assert result.converged == converged
    assert len(result.residual_history) == 1
    assert len(model.assembled_states) == 1


@pytest.mark.parametrize("max_steps", [-1, 1.5, np.nan, np.inf, True])
def test_invalid_step_count(max_steps):
    with pytest.raises(ValueError, match="max_steps"):
        autodiff_newton.newton_polish(NonlinearModel(), max_steps, 1.0e-12)


@pytest.mark.parametrize("tol", [0.0, -1.0, np.nan, np.inf])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        autodiff_newton.newton_polish(NonlinearModel(), 1, tol)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_state(value):
    model = NonlinearModel()
    model.state[2] = value
    with pytest.raises(ValueError, match="finite one-dimensional state"):
        autodiff_newton.newton_polish(model, 1, 1.0e-12)


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("lasym", True, ValueError),
        ("lfreeb", True, ValueError),
        ("always_fix_m1_gauge", False, ValueError),
        ("lforbal", True, RuntimeError),
        ("has_exact_force_jacobian", False, RuntimeError),
    ],
)
def test_unsupported_model(name, value, exception):
    model = NonlinearModel()
    initial = model.get_state()
    setattr(model, name, value)
    with pytest.raises(exception):
        autodiff_newton.newton_polish(model, 1, 1.0e-12)
    np.testing.assert_array_equal(model.get_state(), initial)


@pytest.mark.skipif(
    not vmecpp.has_exact_force_jacobian(), reason="requires VMECPP_ENABLE_ENZYME=ON"
)
def test_enzyme_newton_smoke():
    model = solved_model(small_input())
    initial = np.asarray(model.get_state()).copy()
    result = autodiff_newton.newton_polish(model, 1, 1.0e-14)
    assert np.all(np.isfinite(result.residual_history))
    assert np.all(np.diff(result.residual_history) < 0)
    assert result.converged == (result.residual_history[-1] <= 1.0e-14)
    prescribed = np.setdiff1d(np.arange(initial.size), result.active)
    np.testing.assert_array_equal(result.state[prescribed], initial[prescribed])
    np.testing.assert_array_equal(model.get_state(), result.state)
    rhs = np.ones(result.active.size)
    direction = np.zeros(initial.size)
    direction[result.active] = result.factorization.solve_transpose(rhs)
    product = np.asarray(model.exact_hessian_vector_product(direction)).copy()
    np.testing.assert_allclose(product[result.active], rhs, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        np.linalg.norm(raw_force(model)[result.active]), result.residual_history[-1]
    )
