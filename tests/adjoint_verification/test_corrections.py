"""Nonlinear, nonsymmetric Option B oracle and raw-buffer restoration tests."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import _adjoint_contracts, autodiff, autodiff_corrections, geometry
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
    gauge_weight = 0.4

    def __init__(self):
        self.state = np.asarray([1.2, 0.9, 0.7, 0.28, 0.0])
        self.forces = np.zeros(5)
        self.need_restart = False
        self.evaluated_state = None

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
        x, y, b, pinned, _ = self.state
        self.forces[:] = 0.0
        self.forces[:2] = self.mixing @ np.asarray(
            [
                x**2 + 0.2 * y * b + 0.3 * pinned**2 - b - 1.0,
                y**2 + 0.1 * x * pinned + 0.2 * b**2 - 1.0,
            ]
        )

    def get_forces(self):
        return self.forces

    def jacobian(self):
        x, y, b, pinned, _ = self.state
        result = np.zeros((5, 5))
        result[:2] = self.mixing @ np.asarray(
            [
                [2.0 * x, 0.2 * b, 0.2 * y - 1.0, 0.6 * pinned, 0.0],
                [0.1 * pinned, 2.0 * y, 0.4 * b, 0.1 * x, 0.0],
            ]
        )
        return result

    def exact_hessian_vector_product(self, vector):
        self.forces[:] = self.jacobian() @ vector
        return self.forces

    def exact_hessian_vector_product_transpose(self, vector):
        self.forces[:] = self.jacobian().T @ vector
        return self.forces

    def force_hessians(self):
        source = np.zeros((2, 5, 5))
        source[0, 0, 0] = 2.0
        source[0, 1, 2] = source[0, 2, 1] = 0.2
        source[0, 3, 3] = 0.6
        source[1, 1, 1] = 2.0
        source[1, 0, 3] = source[1, 3, 0] = 0.1
        source[1, 2, 2] = 0.4
        return np.einsum("ij,jkl->ikl", self.mixing, source)


OBJECTIVE_MATRIX = np.asarray(
    [
        [2.0, 0.3, 0.4, 0.2, 0.0],
        [0.3, 1.5, 0.2, -0.1, 0.0],
        [0.4, 0.2, 0.7, 0.0, 0.0],
        [0.2, -0.1, 0.0, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)


def objective(state):
    return 0.5 * state @ OBJECTIVE_MATRIX @ state + 0.1 * state[0] ** 3


def objective_gradient(state):
    result = OBJECTIVE_MATRIX @ state
    result[0] += 0.3 * state[0] ** 2
    return result


class GeometryModel(NonlinearModel):
    scales = np.linspace(0.1, 1.2, 12)

    def get_geometry(self):
        coefficients = {
            name: jnp.asarray(self.scales[index] * self.state).reshape(5, 1, 1)
            for index, name in enumerate(autodiff._GEOMETRY_COEFFICIENTS)
        }
        x, y, b, pinned, _ = self.state
        return geometry.Geometry(
            toroidal_flux=jnp.ones(3),
            poloidal_flux=jnp.asarray([x * y, b**2, pinned]),
            nfp=1,
            **coefficients,
        )

    def geometry_state_vjp(self, coefficients, flux):
        x, y, b, _, _ = self.state
        flux_jacobian = np.asarray(
            [[y, x, 0, 0, 0], [0, 0, 2 * b, 0, 0], [0, 0, 0, 1, 0]]
        )
        # Overwrite the residual like a native derivative.
        self.forces[:] = 123.0
        return self.scales @ coefficients.reshape(12, 5) + flux_jacobian.T @ flux


def geometry_objective(value):
    return (
        jnp.sum(value.r_cc**3) / 3
        + 0.25 * jnp.sum(value.lambda_cs**4)
        + 0.5 * jnp.sum(value.poloidal_flux**2)
    )


def analytic_geometry_objective_gradient(state):
    x, y, b, pinned, _ = state
    result = GeometryModel.scales[0] ** 3 * state**2
    result += GeometryModel.scales[9] ** 4 * state**3
    result += np.asarray([x * y**2, x**2 * y, 2 * b**3, pinned, 0.0])
    return result


def test_geometry_callback_recomputes_nonlinear_coefficients_and_poloidal_flux(
    monkeypatch,
):
    model = GeometryModel()
    monkeypatch.setattr(geometry, "from_cpp", lambda value: value)
    original = model.get_state()
    force = raw_force(model)
    callback = autodiff_corrections.geometry_objective_state_gradient(
        model, geometry_objective
    )
    for offset in (0.0, 0.03, -0.02):
        candidate = original + offset
        actual = callback(candidate)
        np.testing.assert_allclose(
            actual, analytic_geometry_objective_gradient(candidate), rtol=1.0e-13
        )
        np.testing.assert_array_equal(model.get_state(), original)
        np.testing.assert_array_equal(model.get_forces(), force)


def test_geometry_callback_failure_restores_state_and_force(monkeypatch):
    model = GeometryModel()
    monkeypatch.setattr(geometry, "from_cpp", lambda value: value)
    original = model.get_state()
    force = raw_force(model)
    callback = autodiff_corrections.geometry_objective_state_gradient(
        model, lambda _: jnp.asarray(np.nan)
    )
    with pytest.raises(ValueError, match="objective value must be finite"):
        callback(original + 0.01)
    np.testing.assert_array_equal(model.get_state(), original)
    np.testing.assert_array_equal(model.get_forces(), force)


def test_geometry_callback_rejects_float32_without_mutating_configuration():
    model = GeometryModel()
    original = model.get_state()
    force = raw_force(model)
    callback = autodiff_corrections.geometry_objective_state_gradient(
        model, geometry_objective
    )
    with jax.enable_x64(False):  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(ValueError, match="require jax_enable_x64=True"):
            autodiff_corrections.geometry_objective_state_gradient(
                model, geometry_objective
            )
        with pytest.raises(ValueError, match="require jax_enable_x64=True"):
            callback(original)
        assert not jax.config.x64_enabled
    np.testing.assert_array_equal(model.get_state(), original)
    np.testing.assert_array_equal(model.get_forces(), force)


def test_native_geometry_objective_callback_matches_state_directional_difference():
    # The coefficient-only map is analytic and does not require Enzyme.
    model = solved_model(small_input())
    state = np.asarray(model.get_state()).copy()
    force = raw_force(model)

    def nonlinear(value):
        return jnp.mean(value.r_cc**4) + 0.3 * jnp.sum(value.lambda_sc**2)

    callback = autodiff_corrections.geometry_objective_state_gradient(model, nonlinear)
    cotangent = callback(state)
    direction = np.zeros_like(state)
    span = autodiff._span_slices(model)["r_cc"]
    index = span.start + (model.ntor + 1) * model.mpol + (model.ntor + 1)
    direction[index] = 1.0
    values = []
    try:
        for sign in (1.0, -1.0):
            model.set_state(np.ascontiguousarray(state + sign * 1.0e-5 * direction))
            raw_force(model)
            values.append(float(nonlinear(geometry.from_cpp(model.get_geometry()))))
    finally:
        model.set_state(np.ascontiguousarray(state))
        raw_force(model)
    np.testing.assert_allclose(
        cotangent @ direction,
        (values[0] - values[1]) / 2.0e-5,
        rtol=1.0e-5,
        atol=1.0e-8,
    )
    np.testing.assert_array_equal(model.get_state(), state)
    np.testing.assert_allclose(model.get_forces(), force)


@pytest.fixture
def mock_operator(monkeypatch):
    monkeypatch.setattr(
        autodiff,
        "_interior_and_boundary",
        lambda _: (np.asarray([0, 1, 3, 4]), np.asarray([2])),
    )
    monkeypatch.setattr(autodiff, "_gauge_entries", lambda _: np.asarray([3]))

    def assemble(model, active):
        np.testing.assert_array_equal(active, [0, 1])
        matrix = autodiff._BlockTridiagonal(np.zeros(2, dtype=np.int64))
        matrix.diagonal[0] = model.jacobian()[:2, :2].T
        model.forces[:] = 100.0
        return matrix

    def pull_gauge(model, cotangent, gauge):
        np.testing.assert_array_equal(gauge, [3])
        result = cotangent.copy()
        result[2] += model.gauge_weight * result[3]
        result[3] = 0.0
        return result

    monkeypatch.setattr(autodiff, "_assemble_block_tridiagonal", assemble)
    monkeypatch.setattr(autodiff, "_pull_gauge_back_to_boundary", pull_gauge)
    monkeypatch.setattr(
        autodiff, "_boundary_from_state_vjp", lambda _, cotangent: cotangent[2:3]
    )


def oracle(model, *, gradient=None, curvature=None):
    state = model.get_state()
    force = raw_force(model)
    jacobian = model.jacobian()
    if gradient is None:
        gradient = objective_gradient(state)
    delta = np.zeros(5)
    delta[:2] = np.linalg.solve(jacobian[:2, :2], -force[:2])
    adjoint = np.zeros(5)
    adjoint[:2] = np.linalg.solve(jacobian[:2, :2].T, gradient[:2])
    if curvature is None:
        curvature = OBJECTIVE_MATRIX.copy()
        curvature[0, 0] += 0.6 * state[0]
    lxx = curvature - np.einsum("i,ijk->jk", adjoint[:2], model.force_hessians())
    adjoint_dot = np.zeros(5)
    adjoint_dot[:2] = np.linalg.solve(jacobian[:2, :2].T, (lxx @ delta)[:2])
    boundary_chain = np.asarray([0.0, 0.0, 1.0, model.gauge_weight, 0.0])
    base = boundary_chain @ (gradient - jacobian.T @ adjoint)
    corrected = base + boundary_chain @ (lxx @ delta - jacobian.T @ adjoint_dot)
    return corrected, -adjoint @ force, base, delta


@pytest.mark.usefixtures("mock_operator")
def test_nonlinear_nonsymmetric_oracle_and_contract():
    model = NonlinearModel()
    initial = model.get_state()
    expected, expected_value, base, _ = oracle(model)
    expected_force = raw_force(model)
    gradient, value, diagnostics = _adjoint_contracts.corrected_vjp(
        model,
        objective_gradient(initial),
        "B",
        objective_state_gradient=objective_gradient,
    )
    np.testing.assert_allclose(gradient, [expected], rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(value, expected_value, rtol=1.0e-14)
    assert abs(gradient[0] - base) > 0.01
    assert value == -diagnostics.adjoint_weighted_residual
    assert diagnostics.class_sizes == {
        "active": 2,
        "pinned": 1,
        "null": 1,
        "zero_column": 0,
    }
    assert 0 < diagnostics.block_rcond <= 1
    assert diagnostics.solve_residual < 1.0e-12
    assert diagnostics.newton_solve_residual is not None
    assert diagnostics.second_adjoint_solve_residual is not None
    assert diagnostics.newton_solve_residual < 1.0e-12
    assert diagnostics.second_adjoint_solve_residual < 1.0e-12
    np.testing.assert_array_equal(model.get_state(), initial)
    np.testing.assert_array_equal(model.evaluated_state, initial)
    np.testing.assert_array_equal(model.get_forces(), expected_force)


@pytest.mark.usefixtures("mock_operator")
def test_value_correction_is_first_order_objective_change():
    model = NonlinearModel()
    errors = []
    for displacement in (0.02, 0.01, 0.005):
        # Compare against the exactly known root at b=pinned=0.
        model.state[:] = [1.0 + displacement, 1.0 + 1.5 * displacement, 0.0, 0.0, 0.0]
        state = model.get_state()
        _, value, _ = autodiff_corrections.corrected_vjp(
            model,
            objective_gradient(state),
            "B",
            objective_state_gradient=objective_gradient,
        )
        root = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0])
        errors.append(abs(objective(root) - objective(state) - value))
    assert np.all(np.asarray(errors[1:]) < 0.3 * np.asarray(errors[:-1]))
    slopes = np.log(np.asarray(errors[1:]) / np.asarray(errors[:-1])) / np.log(0.5)
    print(f"WP7 value errors={errors}, slopes={slopes.tolist()}")


@pytest.mark.usefixtures("mock_operator")
def test_boundary_gradient_error_declines_quadratically():
    model = NonlinearModel()
    model.state[:] = [1.0, 1.0, 0.0, 0.0, 0.0]
    _, _, exact_gradient, _ = oracle(model)
    errors = []
    for displacement in (0.02, 0.01, 0.005):
        model.state[:] = [1.0 + displacement, 1.0 + 2 * displacement, 0.0, 0.0, 0.0]
        _, _, uncorrected, _ = oracle(model)
        gradient, _, _ = autodiff_corrections.corrected_vjp(
            model,
            objective_gradient(model.state),
            "B",
            objective_state_gradient=objective_gradient,
        )
        errors.append(abs(gradient[0] - exact_gradient))
        assert errors[-1] < abs(uncorrected - exact_gradient)
    assert np.all(np.asarray(errors[1:]) < 0.3 * np.asarray(errors[:-1]))
    slopes = np.log(np.asarray(errors[1:]) / np.asarray(errors[:-1])) / np.log(0.5)
    print(f"WP7 boundary-gradient errors={errors}, slopes={slopes.tolist()}")


@pytest.mark.usefixtures("mock_operator")
def test_objective_curvature_cannot_be_replaced_by_fixed_xbar():
    model = NonlinearModel()
    state = model.get_state()
    x_bar = objective_gradient(state)
    nonlinear, _, _ = autodiff_corrections.corrected_vjp(
        model, x_bar, "B", objective_state_gradient=objective_gradient
    )
    linear, _, _ = autodiff_corrections.corrected_vjp(
        model, x_bar, "B", objective_state_gradient=lambda _: x_bar
    )
    assert np.linalg.norm(nonlinear - linear) > 0.01


@pytest.mark.usefixtures("mock_operator")
def test_exponential_objective_directional_difference_has_quadratic_truncation():
    model = NonlinearModel()
    state = model.get_state()
    exponent_direction = np.asarray([1.1, -0.5, 0.8, 0.3, 0.0])

    def callback(candidate):
        return (
            objective_gradient(candidate)
            + np.exp(exponent_direction @ candidate) * exponent_direction
        )

    curvature = OBJECTIVE_MATRIX.copy()
    curvature[0, 0] += 0.6 * state[0]
    curvature += np.exp(exponent_direction @ state) * np.outer(
        exponent_direction, exponent_direction
    )
    expected, _, _, _ = oracle(model, gradient=callback(state), curvature=curvature)
    force = raw_force(model)
    errors = []
    for h in (0.5, 0.25, 0.125):
        gradient, _, _ = autodiff_corrections.corrected_vjp(
            model,
            callback(state),
            "B",
            objective_state_gradient=callback,
            directional_step=h,
        )
        errors.append(abs(gradient[0] - expected))
        np.testing.assert_array_equal(model.get_state(), state)
        np.testing.assert_array_equal(model.get_forces(), force)
    ratios = np.asarray(errors[1:]) / np.asarray(errors[:-1])
    assert np.all((ratios > 0.24) & (ratios < 0.26))
    slopes = np.log(ratios) / np.log(0.5)
    print(f"WP7 exponential-objective FD errors={errors}, slopes={slopes.tolist()}")


@pytest.mark.usefixtures("mock_operator")
@pytest.mark.parametrize("failure", ["callback", "assembly", "product", "candidate"])
def test_failures_restore_state_and_raw_force(monkeypatch, failure):
    model = NonlinearModel()
    state = model.get_state()
    expected_force = raw_force(model)
    callback: Callable[[np.ndarray], np.ndarray] = objective_gradient
    if failure == "callback":

        def failing_callback(candidate):
            if not np.array_equal(candidate, state):
                model.forces[:] = np.nan
                message = "callback failed"
                raise RuntimeError(message)
            return objective_gradient(candidate)

        callback = failing_callback
    elif failure == "assembly":

        def fail(model, _):
            model.forces[:] = np.nan
            message = "assembly failed"
            raise RuntimeError(message)

        monkeypatch.setattr(autodiff, "_assemble_block_tridiagonal", fail)
    elif failure == "product":
        monkeypatch.setattr(
            model,
            "exact_hessian_vector_product_transpose",
            lambda _: np.full(5, np.nan),
        )
    else:
        evaluate = model.evaluate

        def reject(*args):
            evaluate(*args)
            model.need_restart = not np.array_equal(model.state, state)

        monkeypatch.setattr(model, "evaluate", reject)
    with pytest.raises((RuntimeError, ValueError)):
        autodiff_corrections.corrected_vjp(
            model, objective_gradient(state), "B", objective_state_gradient=callback
        )
    np.testing.assert_array_equal(model.get_state(), state)
    np.testing.assert_array_equal(model.get_forces(), expected_force)
    assert not model.need_restart


@pytest.mark.usefixtures("mock_operator")
@pytest.mark.parametrize(
    "gradient",
    [lambda _: np.zeros(5), lambda _: np.ones(4), lambda _: np.full(5, np.nan)],
)
def test_invalid_objective_gradient_restores_model(gradient):
    model = NonlinearModel()
    state = model.get_state()
    expected_force = raw_force(model)
    with pytest.raises(ValueError, match="objective_state_gradient"):
        autodiff_corrections.corrected_vjp(
            model, objective_gradient(state), "B", objective_state_gradient=gradient
        )
    np.testing.assert_array_equal(model.get_state(), state)
    np.testing.assert_array_equal(model.get_forces(), expected_force)


@pytest.mark.parametrize("step", [0.0, -1.0, np.nan, np.inf])
def test_invalid_directional_step(step):
    model = NonlinearModel()
    with pytest.raises(ValueError, match="directional_step"):
        autodiff_corrections.corrected_vjp(
            model,
            objective_gradient(model.state),
            "B",
            objective_state_gradient=objective_gradient,
            directional_step=step,
        )


def test_missing_objective_callable_and_unsupported_methods():
    model = NonlinearModel()
    with pytest.raises(ValueError, match="requires objective_state_gradient"):
        autodiff_corrections.corrected_vjp(model, np.zeros(5), "B")
    with pytest.raises(NotImplementedError, match="Option A"):
        autodiff_corrections.corrected_vjp(model, np.zeros(5), "A")
    with pytest.raises(ValueError, match="method"):
        autodiff_corrections.corrected_vjp(model, np.zeros(5), "C")


@pytest.mark.usefixtures("mock_operator")
@pytest.mark.parametrize("solve", ["solve", "solve_transpose"])
def test_inaccurate_solves_are_rejected_and_restore_buffers(monkeypatch, solve):
    model = NonlinearModel()
    state = model.get_state()
    force = raw_force(model)
    monkeypatch.setattr(
        autodiff._BlockTridiagonal, solve, lambda _, rhs: np.zeros_like(rhs)
    )
    with pytest.raises(RuntimeError, match="linear solve failed"):
        autodiff_corrections.corrected_vjp(
            model,
            objective_gradient(state),
            "B",
            objective_state_gradient=objective_gradient,
        )
    np.testing.assert_array_equal(model.get_state(), state)
    np.testing.assert_array_equal(model.get_forces(), force)


@pytest.mark.usefixtures("mock_operator")
def test_second_adjoint_solve_is_checked(monkeypatch):
    model = NonlinearModel()
    state = model.get_state()
    force = raw_force(model)
    solve = autodiff._BlockTridiagonal.solve
    calls = 0

    def corrupt_second(matrix, rhs):
        nonlocal calls
        calls += 1
        return solve(matrix, rhs) if calls == 1 else np.zeros_like(rhs)

    monkeypatch.setattr(autodiff._BlockTridiagonal, "solve", corrupt_second)
    with pytest.raises(RuntimeError, match="linear solve failed"):
        autodiff_corrections.corrected_vjp(
            model,
            objective_gradient(state),
            "B",
            objective_state_gradient=objective_gradient,
        )
    assert calls == 2
    np.testing.assert_array_equal(model.get_state(), state)
    np.testing.assert_array_equal(model.get_forces(), force)


@pytest.mark.usefixtures("mock_operator")
def test_empty_active_operator_fails_clearly_and_restores_buffers(monkeypatch):
    model = NonlinearModel()
    state = model.get_state()
    force = raw_force(model)
    monkeypatch.setattr(
        autodiff,
        "_structural_nullfree_interior",
        lambda *_: np.zeros(0, dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="nonempty active force operator"):
        autodiff_corrections.corrected_vjp(
            model,
            objective_gradient(state),
            "B",
            objective_state_gradient=objective_gradient,
        )
    np.testing.assert_array_equal(model.get_state(), state)
    np.testing.assert_array_equal(model.get_forces(), force)


def test_empty_schur_blocks_fail_clearly():
    matrix = autodiff._BlockTridiagonal(np.asarray([0]))
    matrix.members = [np.zeros(0, dtype=np.int64)]
    with pytest.raises(RuntimeError, match="nonempty active force operator"):
        autodiff_corrections._block_rcond(matrix)


@pytest.mark.parametrize("target", ["state", "x_bar"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs(target, value):
    model = NonlinearModel()
    x_bar = objective_gradient(model.state)
    if target == "state":
        model.state[0] = value
    else:
        x_bar[0] = value
    with pytest.raises(ValueError, match="finite"):
        autodiff_corrections.corrected_vjp(
            model, x_bar, "B", objective_state_gradient=objective_gradient
        )


@pytest.mark.usefixtures("mock_operator")
def test_zero_residual_needs_no_directional_perturbation():
    model = NonlinearModel()
    model.state[:] = [1.0, 1.0, 0.0, 0.0, 0.0]
    expected, value, _, _ = oracle(model)
    evaluations = []

    def callback(state):
        evaluations.append(state.copy())
        return objective_gradient(state)

    gradient, correction, diagnostics = autodiff_corrections.corrected_vjp(
        model,
        objective_gradient(model.state),
        "B",
        objective_state_gradient=callback,
    )
    np.testing.assert_allclose(gradient, [expected])
    assert correction == value == 0.0
    assert diagnostics.force_residual == 0.0
    assert diagnostics.second_adjoint_solve_residual == 0.0
    assert len(evaluations) == 1


@pytest.mark.usefixtures("mock_operator")
@pytest.mark.parametrize("displacement", [1.0e-9, 1.0e-12, np.finfo(float).eps])
def test_tiny_newton_direction_keeps_nonlinear_curvature_without_gradient_floor(
    displacement,
):
    model = NonlinearModel()
    model.state[:] = [1.0 + displacement, 1.0 + 2 * displacement, 0.0, 0.0, 0.0]
    state = model.get_state()
    expected, _, _, delta = oracle(model)
    assert np.any(delta)
    # With the requested h, the smallest direction cannot move the stored state.
    if displacement == np.finfo(float).eps:
        np.testing.assert_array_equal(state + 1.0e-3 * delta, state)
    evaluations = []

    def callback(candidate):
        evaluations.append(candidate.copy())
        return objective_gradient(candidate)

    gradient, _, _ = autodiff_corrections.corrected_vjp(
        model,
        objective_gradient(state),
        "B",
        objective_state_gradient=callback,
        directional_step=1.0e-3,
    )
    assert len(evaluations) == 3
    floor = np.cbrt(np.finfo(float).eps) * np.max(np.abs(state[:2]))
    for candidate in evaluations[1:]:
        assert np.max(np.abs(candidate[:2] - state[:2])) >= 0.99 * floor
        np.testing.assert_array_equal(candidate[2:], state[2:])
    np.testing.assert_allclose(gradient, [expected], rtol=0, atol=2.0e-14)
    np.testing.assert_array_equal(model.get_state(), state)


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
    state = model.get_state()
    setattr(model, name, value)
    with pytest.raises(exception):
        autodiff_corrections.corrected_vjp(
            model,
            objective_gradient(state),
            "B",
            objective_state_gradient=objective_gradient,
        )
    np.testing.assert_array_equal(model.get_state(), state)


@pytest.mark.skipif(
    not vmecpp.has_exact_force_jacobian(), reason="requires VMECPP_ENABLE_ENZYME=ON"
)
def test_enzyme_correction_smoke():
    model = solved_model(small_input())
    state = np.asarray(model.get_state()).copy()
    expected_force = raw_force(model)
    gradient, value, diagnostics = autodiff_corrections.corrected_vjp(
        model, state, "B", objective_state_gradient=lambda candidate: candidate
    )
    assert np.all(np.isfinite(gradient))
    assert np.isfinite(value)
    assert diagnostics.solve_residual < 1.0e-6
    np.testing.assert_array_equal(model.get_state(), state)
    np.testing.assert_allclose(model.get_forces(), expected_force)
