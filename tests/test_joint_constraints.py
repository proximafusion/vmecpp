# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The constraints and bounds of vmecpp.joint against the conditions of a constrained
minimum, on least-squares problems of a step and on converged equilibria."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp.joint import (
    _STIFFNESS,
    CONSTRAINT_TOLERANCE,
    STAGE_DROP,
    JointProblem,
    QuasisymmetryProblem,
    Stage,
    _bounded_step,
    _Iterate,
    _moved_shifts,
    _on_shifts,
    _penalized,
    _Point,
    _ridge,
    _step_scale,
    boundary_modes,
    converged_input,
    optimize,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from joint_optimization import rotating_ellipse  # type: ignore

jax.config.update("jax_enable_x64", True)


class Three(JointProblem):
    """Three constraints for the least-squares problem of a step."""

    def residuals(self, solved, boundary):  # noqa: ARG002
        return jnp.zeros(1)

    def constraints(self, solved, boundary):  # noqa: ARG002
        return jnp.zeros(3)


def _three(**bounds) -> Three:
    return Three(rotating_ellipse(4, 4, 17), max_mode=1, **bounds)


def _step_problem(problem, constraints, seed=0):
    """A point of the problem with random residuals and derivatives and the given values
    of its constraints."""
    random = np.random.default_rng(seed)
    keys = boundary_modes(1, problem.ntor)
    jacobian = random.normal(size=(12 + problem.lower.size, len(keys)))
    raw = np.concatenate([random.normal(size=12), constraints])
    point = _Point(problem.boundary0, np.zeros(1), **_penalized(problem, raw, raw))
    return keys, jacobian, point


def _model(problem, point, jacobian, ridge, delta) -> float:
    """The merit of the linearized residuals and constraints after a step, with its
    damping."""
    penalized = problem._penalized(point.raw + jacobian @ delta)
    return float(penalized @ penalized + ridge @ delta**2)


def _model_slope(problem, point, jacobian, ridge, delta) -> np.ndarray:
    """The derivative of :func:`_model` along each coefficient by central
    differences."""
    slope = np.zeros(delta.size)
    for index in range(delta.size):
        step = np.zeros(delta.size)
        step[index] = 1.0e-6
        slope[index] = (
            _model(problem, point, jacobian, ridge, delta + step)
            - _model(problem, point, jacobian, ridge, delta - step)
        ) / 2.0e-6
    return slope


WIDE = {"lower": [-9.0, -9.0, -9.0], "upper": 9.0}

STEP_CASES = {
    "within every bound": (WIDE, [0.3, -0.2, 0.1]),
    "outside an upper bound": ({"upper": [0.1, 9.0, 9.0]}, [0.4, 0.0, 0.2]),
    "outside a lower bound": ({"lower": [-9.0, 0.5, -9.0]}, [0.0, 0.1, 0.0]),
    "an equality": (
        {"lower": [-9.0, -9.0, 0.3], "upper": [9.0, 9.0, 0.3]},
        [0.0, 0.0, 0.9],
    ),
    "a bound the step reaches": (
        {"lower": [-0.02, -0.02, -0.02], "upper": 0.02},
        [0.0, 0.01, -0.01],
    ),
}


@pytest.mark.parametrize("name", STEP_CASES)
@pytest.mark.parametrize("damping", [1.0e-3, 1.0])
def test_a_step_minimizes_the_linearized_merit(name: str, damping: float) -> None:
    """No coefficient has a slope of the merit of the linearized problem left, and the
    predicted drop is that of this merit."""
    bounds, constraints = STEP_CASES[name]
    problem = _three(weights=[1.0, 3.0, 0.5], **bounds)
    problem.shifts.values[:] = [0.05, -0.02, 0.0]
    keys, jacobian, point = _step_problem(problem, np.asarray(constraints))
    delta, predicted = _bounded_step(problem, point, keys, jacobian, damping)
    ridge = _ridge(problem, point, jacobian, damping)
    slope = _model_slope(problem, point, jacobian, ridge, delta)
    np.testing.assert_allclose(slope, 0.0, atol=1.0e-6 * np.abs(jacobian).max())
    np.testing.assert_allclose(
        predicted,
        point.merit - _model(problem, point, jacobian, ridge, delta) + ridge @ delta**2,
        rtol=1.0e-9,
    )
    assert predicted > 0.0


def test_a_step_without_active_constraints_is_the_damped_normal_step() -> None:
    problem = _three(**WIDE)
    keys, jacobian, point = _step_problem(problem, np.asarray([0.3, -0.2, 0.1]))
    delta, _ = _bounded_step(problem, point, keys, jacobian, 1.0e-2)
    rows = jacobian[:12]
    ridge = _ridge(problem, point, jacobian, 1.0e-2)
    np.testing.assert_allclose(
        delta,
        -np.linalg.solve(rows.T @ rows + np.diag(ridge), rows.T @ point.balanced[:12]),
        rtol=1.0e-9,
    )


def test_a_step_stays_within_the_bounds_of_the_coefficients() -> None:
    """A coefficient the free step would take past its bound ends on it, with the merit
    still falling towards the bound, and the others have no slope left."""
    free = _three(**WIDE)
    keys, jacobian, point = _step_problem(free, np.zeros(3))
    step, _ = _bounded_step(free, point, keys, jacobian, 1.0e-3)
    held = int(np.argmax(np.abs(step)))
    lower = np.full(free.boundary0.shape, -np.inf)
    upper = np.full(free.boundary0.shape, np.inf)
    side = upper if step[held] > 0.0 else lower
    side[keys[held]] = free.boundary0[keys[held]] + 0.25 * step[held]
    bounded = _three(**WIDE, boundary_lower=lower, boundary_upper=upper)
    delta, predicted = _bounded_step(bounded, point, keys, jacobian, 1.0e-3)
    np.testing.assert_allclose(delta[held], 0.25 * step[held], rtol=1.0e-12)
    ridge = _ridge(bounded, point, jacobian, 1.0e-3)
    slope = _model_slope(bounded, point, jacobian, ridge, delta)
    others = np.arange(len(keys)) != held
    np.testing.assert_allclose(slope[others], 0.0, atol=1.0e-6 * np.abs(jacobian).max())
    assert slope[held] * step[held] < 0.0
    assert 0.0 < predicted < _bounded_step(free, point, keys, jacobian, 1.0e-3)[1]


def test_a_constraint_is_the_weighted_distance_of_its_shifted_value() -> None:
    problem = _three(
        lower=[-1.0, 0.0, 2.0], upper=[1.0, np.inf, 2.0], weights=[1, 2, 3]
    )
    problem.shifts.values[:] = [0.5, -0.25, 0.1]
    raw = np.asarray([7.0, 0.75, 0.125, 2.5])
    np.testing.assert_allclose(
        problem._penalized(raw), [7.0, 0.25, 2.0 * -0.125, 3.0 * 0.6]
    )
    np.testing.assert_allclose(
        problem.violations(np.asarray([1.5, -0.5, 2.5])), [0.5, 0.5, 0.25]
    )
    np.testing.assert_allclose(problem.multipliers(), [1.0, -2.0, 1.8])


def test_the_scale_of_a_drop_leaves_out_the_shifts() -> None:
    """A constraint on its bound adds the square of its weighted shift to the merit and
    nothing to the scale that a drop is measured against."""
    problem = _three(upper=[1.0, 1.0, 1.0], weights=[1, 2, 3])
    problem.shifts.values[:] = [0.5, 0.0, 0.0]
    raw = np.asarray([7.0, 1.0, 1.25, 0.5])
    point = _Point(problem.boundary0, np.zeros(1), **_penalized(problem, raw, raw))
    np.testing.assert_allclose(point.size, 49.0 + (2.0 * 0.25) ** 2)
    np.testing.assert_allclose(point.merit, point.size + 0.5**2)
    problem.shifts.values[:] = 0.0
    moved = _on_shifts(problem, point)
    np.testing.assert_allclose(moved.merit, point.size)
    np.testing.assert_allclose(moved.size, point.size)


def test_a_shift_moves_to_the_distance_of_its_shifted_constraint() -> None:
    problem = _three(lower=[-1.0, 0.0, 2.0], upper=[1.0, np.inf, 2.0])
    values = np.asarray([1.25, 0.5, 1.75])
    # within the tolerance of 0.3 times the size of the bounds nothing moves
    assert problem.shift(values, 0.3) is None
    np.testing.assert_array_equal(problem.shifts.values, 0.0)
    slow = problem.shift(values, 1.0e-4)
    np.testing.assert_allclose(problem.shifts.values, [0.25, 0.0, -0.25])
    np.testing.assert_array_equal(slow, [False, False, False])
    # the first constraint has come to a fifth of its distance, the third to a half
    slow = problem.shift(np.asarray([1.05, 0.5, 1.875]), 1.0e-4)
    np.testing.assert_allclose(problem.shifts.values, [0.3, 0.0, -0.375])
    np.testing.assert_array_equal(slow, [False, False, True])
    before = problem.multipliers()
    problem.stiffen(slow)
    np.testing.assert_allclose(problem.shifts.weights, [1.0, 1.0, 10.0])
    np.testing.assert_allclose(problem.multipliers(), before)


def test_a_stalled_stage_stiffens_a_constraint_that_its_steps_do_not_resolve() -> None:
    """Of two constraints as far outside their bounds, the one whose weighted distance
    is smaller than the residuals takes ten times its weight at the same multiplier."""
    problem = _three(upper=[1.0, 1.0, 1.0], weights=[1.0, 100.0, 1.0])
    raw = np.asarray([7.0, 1.5, 1.5, 0.5])
    point = _Point(problem.boundary0, np.zeros(1), **_penalized(problem, raw, raw))
    iterate, end = _Iterate(point, 1.0e-12), Stage(1)
    assert _moved_shifts(problem, iterate, end, 1.0e-4)
    assert end.shift_updates == 1
    np.testing.assert_allclose(problem.shifts.weights, [10.0, 100.0, 1.0])
    np.testing.assert_allclose(problem.multipliers(), [1.0, 1.0e4, 0.0])
    np.testing.assert_allclose(iterate.point.balanced, [7.0, 5.05, 100.0, 0.0])
    # within the tolerance of the size of the bounds nothing moves
    assert not _moved_shifts(problem, iterate, end, 0.6)
    assert end.shift_updates == 1


def test_a_balanced_weight_gives_a_constraint_the_stiffness_of_the_residuals() -> None:
    """Each weight makes the curvature of its constraint along its gradient _STIFFNESS
    times that of the residuals, whatever the unit of the constraint, and leaves the
    multiplier as it is."""
    problem = _three(**WIDE)
    problem.shifts.values[:] = [0.5, 0.0, -0.25]
    before = problem.multipliers()
    _, jacobian, _ = _step_problem(problem, np.zeros(3))
    jacobian[13] *= 100.0
    jacobian[14] = 0.0
    assert problem.balance(jacobian)
    rows, gradients = jacobian[:12], jacobian[12:]
    normal = rows.T @ rows
    softness = np.einsum(
        "ij,ji->i",
        gradients,
        np.linalg.solve(normal + 1.0e-3 * np.diag(_step_scale(normal)), gradients.T),
    )
    weights = problem.shifts.weights
    np.testing.assert_allclose(weights[:2] ** 2 * softness[:2], _STIFFNESS)
    np.testing.assert_allclose(problem.multipliers(), before)
    # a constraint that does not depend on the coefficients keeps its weight
    assert weights[2] == 1.0
    scaled = _three(**WIDE)
    jacobian[13] /= 100.0
    scaled.balance(jacobian)
    np.testing.assert_allclose(scaled.shifts.weights[1], 100.0 * weights[1])
    given = _three(**WIDE, weights=[1.0, 2.0, 3.0])
    assert not given.balance(jacobian)
    np.testing.assert_array_equal(given.shifts.weights, [1.0, 2.0, 3.0])


def test_bounds_that_cross_or_differ_in_number_are_refused() -> None:
    with pytest.raises(ValueError, match="at or below"):
        _three(lower=[0.0, 1.0, 0.0], upper=[1.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="as many"):
        _three(lower=[0.0, 0.0], upper=[1.0, 1.0, 1.0])


def test_constraints_without_as_many_bounds_are_refused() -> None:
    problem = _three(lower=[0.0, 0.0])
    with pytest.raises(ValueError, match="3 constraints have 2 bounds"):
        problem.evaluate(problem.create_model(problem.boundary0), problem.boundary0)


def test_a_start_outside_the_bounds_of_its_coefficients_is_refused() -> None:
    vmec_input = rotating_ellipse(4, 4, 17)
    upper = np.full((2, 4, 9), np.inf)
    upper[0, 1, 4] = 0.1
    with pytest.raises(ValueError, match="within their bounds"):
        _three(boundary_upper=upper)
    # a coefficient that no stage frees may lie outside
    upper[0, 1, 4], upper[0, 3, 4] = np.inf, -1.0
    Three(vmec_input, max_mode=1, boundary_upper=upper)


class WithIota(QuasisymmetryProblem):
    """Quasisymmetry with its targets, the mean iota of 0.42 among them, and the mean
    iota as a constraint as well."""

    def constraints(self, solved, boundary):  # noqa: ARG002
        return jnp.atleast_1d(
            (solved.poloidal_flux[-1] - solved.poloidal_flux[0])
            / (solved.toroidal_flux[-1] - solved.toroidal_flux[0])
        )


BOUNDS = {
    "at 0.45": {"lower": 0.45, "upper": 0.45},
    "above 0.45": {"lower": 0.45},
    "below 0.4": {"upper": 0.4},
    "above 0.3": {"lower": 0.3},
}
ACTIVE = ["at 0.45", "above 0.45", "below 0.4"]


def _quasisymmetry(kind=QuasisymmetryProblem, **options):
    return kind(rotating_ellipse(4, 4, 17), ntheta=24, nphi=24, max_mode=2, **options)


@pytest.fixture(scope="module")
def runs():
    """The mean iota without bounds and with each of BOUNDS, and above 0.45 by the
    nested baseline."""
    results = {}
    for name, bounds in ({"free": None} | BOUNDS).items():
        problem = (
            _quasisymmetry() if bounds is None else _quasisymmetry(WithIota, **bounds)
        )
        results[name] = (problem, optimize(problem, (1, 2)))
    problem = _quasisymmetry(WithIota, **BOUNDS["above 0.45"])
    results["nested"] = (problem, optimize(problem, (1, 2), nested=True))
    return results


def _linearization(problem, result):
    """The derivative of the sum of the squares of the residuals of a result, the
    derivatives of its constraints, both with respect to the boundary coefficients of
    the last stage in force balance, and the drop relative to that sum which a damped
    step along a slope in the given coefficients predicts."""
    keys = boundary_modes(2, problem.ntor)
    model = problem.create_model(result.boundary)
    tangents = problem.state_tangents(model, result.state, keys)
    evaluation, jacobian = problem.residual_jacobian(
        model, result.state, result.boundary, keys, tangents
    )
    first = jacobian.shape[0] - problem.lower.size
    rows, residuals = jacobian[:first], evaluation.residuals[:first]
    normal = rows.T @ rows
    damped = normal + 1.0e-3 * np.diag(_step_scale(normal))

    def drop(slope, among=slice(None)):
        return (
            0.25
            * slope[among]
            @ np.linalg.solve(damped[among][:, among], slope[among])
            / (residuals @ residuals)
        )

    return keys, 2.0 * rows.T @ residuals, jacobian[first:], damped, drop


@pytest.mark.parametrize("name", [*ACTIVE, "nested"])
def test_a_constraint_ends_on_its_bound(runs, name) -> None:
    """Within the tolerance on the modes of the problem, and in a cold converged run of
    the final boundary.

    The stages before the last leave the shifts as they are.
    """
    problem, result = runs[name]
    bound = problem.lower[0] if np.isfinite(problem.lower[0]) else problem.upper[0]
    np.testing.assert_allclose(
        result.evaluation.constraints, bound, rtol=CONSTRAINT_TOLERANCE
    )
    assert result.evaluation.parts["violation"] <= CONSTRAINT_TOLERANCE
    wout = vmecpp.run(
        converged_input(problem, result.boundary, pin_gauge=True, ns=problem.ns),
        verbose=False,
        max_threads=1,
    ).wout
    np.testing.assert_allclose(np.mean(np.asarray(wout.iotas)[1:]), bound, rtol=1.0e-4)
    assert result.stages[0].shift_updates == 0
    assert result.stages[1].shift_updates >= 1


@pytest.mark.parametrize("name", [*ACTIVE, "nested"])
def test_the_multiplier_of_a_constraint_balances_the_residuals(runs, name) -> None:
    """The multiplier has the sign of its side, the targets pulling the mean iota
    towards 0.42, and lies within 15% of the one that fits the gradient of the
    residuals.

    A damped step along what is left of that gradient predicts a drop below that at
    which a stage stalls.
    """
    problem, result = runs[name]
    _, gradient, gradients, damped, drop = _linearization(problem, result)
    assert result.multipliers[0] * (0.42 - result.evaluation.constraints[0]) > 0.0
    solved = np.linalg.solve(damped, gradients.T)
    fitted = -np.linalg.solve(gradients @ solved, solved.T @ gradient)
    np.testing.assert_allclose(result.multipliers, fitted, rtol=0.15)
    assert drop(gradient) > 0.1
    assert drop(gradient + gradients.T @ result.multipliers) < STAGE_DROP


def test_a_constraint_that_a_run_leaves_behind_does_not_move_its_result(runs) -> None:
    """The start has a mean iota of 0.18, below the bound of 0.3 that the targets take
    it past.

    The run ends where the free one does, up to its path.
    """
    _, free = runs["free"]
    _, result = runs["above 0.3"]
    np.testing.assert_allclose(result.boundary, free.boundary, rtol=0.0, atol=1.0e-3)
    np.testing.assert_allclose(
        result.evaluation.merit, free.evaluation.merit, rtol=2.0e-3
    )
    np.testing.assert_array_equal(result.multipliers, 0.0)
    assert [stage.shift_updates for stage in result.stages] == [0, 0]
    assert result.evaluation.parts["violation"] == 0.0


def _bounded(runs, **options):
    """The free run with zbs(1, 1) kept above -0.06, which the free run passes."""
    _, free = runs["free"]
    problem = _quasisymmetry()
    held = (1, 1, problem.ntor + 1)
    assert free.boundary[held] < -0.07
    lower = np.full(problem.boundary0.shape, -np.inf)
    lower[held] = -0.06
    problem = _quasisymmetry(boundary_lower=lower)
    return problem, optimize(problem, (1, 2), **options), held


def test_a_bound_holds_a_coefficient(runs) -> None:
    """The coefficient ends on its bound with the residuals still falling beyond it, and
    a damped step along the other coefficients predicts no drop."""
    problem, result, held = _bounded(runs)
    assert result.boundary[held] == -0.06
    keys, gradient, _, _, drop = _linearization(problem, result)
    index = keys.index(held)
    assert gradient[index] > 0.0
    assert drop(gradient, np.arange(len(keys)) != index) < 2.0e-3
    assert drop(gradient) > 2.0 * drop(gradient, np.arange(len(keys)) != index)


def test_a_bound_holds_a_coefficient_on_the_modes_of_a_margin(runs) -> None:
    problem, result, held = _bounded(runs, margin=1)
    assert (result.stages[0].mpol, result.stages[0].ntor) == (2, 2)
    assert result.boundary[held] == -0.06
    assert result.boundary.shape == problem.boundary0.shape
