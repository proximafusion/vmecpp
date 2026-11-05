# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""vmecpp.joint against differences of the force and of the residuals, re-solved
equilibria and SIMSOPT."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore
from vmecpp.joint import (
    NESTED_TOLERANCE,
    JointProblem,
    QuasisymmetryProblem,
    Stage,
    _joint_point,
    _relaxed_further,
    _tightened,
    boundary_modes,
    converged_input,
    optimize,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from joint_optimization import rotating_ellipse  # type: ignore

jax.config.update("jax_enable_x64", True)

needs_exact_products = pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="the exact force Jacobian needs an Enzyme-enabled build",
)

PRESSURE = 6000.0
"""The pressure on axis of the finite-beta case: beta 0.66%, with the axis moved by
0.47 of the minor radius."""


@pytest.fixture(scope="module", params=[0.0, PRESSURE], ids=["vacuum", "pressure"])
def problem(request) -> JointProblem:
    vmec_input = rotating_ellipse(4, 4, 17)
    vmec_input.pres_scale = request.param
    vmec_input.am = np.asarray([1.0, -1.0, 0.0])
    return QuasisymmetryProblem(vmec_input, ntheta=24, nphi=24, max_mode=1)


@pytest.fixture(scope="module")
def keys(problem) -> list[tuple[int, int, int]]:
    return boundary_modes(1, problem.ntor)


@pytest.fixture(scope="module")
def solved(problem):
    return problem.relax(problem.boundary0, ftol=1.0e-14)


@pytest.fixture(scope="module")
def linearization(problem, keys, solved):
    state = np.array(solved.get_state())
    model = problem.create_model(problem.boundary0)
    tangents = problem.state_tangents(model, state, keys)
    evaluation, jacobian = problem.residual_jacobian(
        model, state, problem.boundary0, keys, tangents
    )
    return state, tangents, evaluation, jacobian


@pytest.fixture(scope="module")
def loose(problem, keys):
    """A state far from force balance and its tangents."""
    state = np.array(problem.relax(problem.boundary0, ftol=1.0e-5).get_state())
    model = problem.create_model(problem.boundary0)
    return state, problem.state_tangents(model, state, keys)


def _simsopt_quasisymmetry(problem, wout, tmp_path) -> float:
    path = tmp_path / "wout_joint.nc"
    wout.save(path)
    return QuasisymmetryRatioResidual(
        Vmec(str(path)),
        np.asarray(problem.surfaces),
        helicity_m=problem.helicity_m,
        helicity_n=problem.helicity_n,
    ).total()


def test_targets_and_quasisymmetry_match_the_wout_and_simsopt(
    problem, linearization, tmp_path
) -> None:
    """Aspect ratio, mean iota and QS error of a converged state against the wout."""
    evaluation = linearization[2]
    pinned = problem.vmec_input.model_copy(update={"ftol_array": np.asarray([1.0e-14])})
    wout = vmecpp.run(pinned, verbose=False, max_threads=1).wout
    np.testing.assert_allclose(evaluation.parts["aspect"], wout.aspect, rtol=1.0e-8)
    np.testing.assert_allclose(
        evaluation.parts["mean_iota"], np.mean(np.asarray(wout.iotas)[1:]), rtol=1.0e-8
    )
    np.testing.assert_allclose(
        evaluation.parts["quasisymmetry"],
        _simsopt_quasisymmetry(problem, wout, tmp_path),
        rtol=1.0e-2,
    )


@pytest.mark.parametrize("balance", ["converged", "loose"])
def test_state_tangents_keep_the_force_balance(
    problem, keys, linearization, loose, balance
) -> None:
    """H t vanishes on the interior, by the exact product and by a difference of the
    force, at the solved state and away from force balance."""
    state, tangents = (
        linearization[:2] if balance == "converged" else (loose[0], loose[1])
    )
    model = problem.create_model(problem.boundary0)
    for i in range(len(keys)):
        products = [(model.hessian_vector_product, 1.0e-5)]
        if problem.exact_products:
            products.append((model.exact_hessian_vector_product, 1.0e-10))
        for product, tolerance in products:
            model.set_state(np.ascontiguousarray(state))
            model.evaluate(2, 2, True)
            tangent = np.asarray(product(np.ascontiguousarray(tangents[:, i])))
            model.set_state(np.ascontiguousarray(state))
            model.evaluate(2, 2, True)
            reference = np.asarray(
                product(np.ascontiguousarray(problem.boundary_map[keys[i]]))
            )
            np.testing.assert_allclose(
                tangent[problem.interior],
                0.0,
                atol=tolerance * np.linalg.norm(reference[problem.interior]),
                equal_nan=False,
            )


@pytest.mark.parametrize("index", [0, 2, 5, 7])
def test_residual_jacobian_matches_converged_solves(
    problem, keys, linearization, index
) -> None:
    """The total derivative against central differences of re-converged equilibria."""
    state, tangents, _, jacobian = linearization
    step = 1.0e-5
    sides = []
    for sign in (1.0, -1.0):
        boundary = problem.boundary0.copy()
        boundary[keys[index]] += sign * step
        model = problem.relax(
            boundary, state + sign * step * tangents[:, index], ftol=1.0e-15
        )
        sides.append(problem.evaluate(model, boundary).residuals)
    difference = (sides[0] - sides[1]) / (2.0 * step)
    np.testing.assert_allclose(
        jacobian[:, index],
        difference,
        rtol=0.0,
        atol=1.0e-4 * np.linalg.norm(difference),
        equal_nan=False,
    )


@pytest.mark.parametrize("index", [0, 2, 5, 7])
def test_residual_jacobian_holds_at_an_unconverged_state(
    problem, keys, loose, index
) -> None:
    """The residual derivative along a tangent against a difference of the residuals of
    the two displaced states, away from force balance."""
    state, tangents = loose
    model = problem.create_model(problem.boundary0)
    _, jacobian = problem.residual_jacobian(
        model, state, problem.boundary0, keys, tangents
    )
    step = 1.0e-6 / np.abs(tangents[:, index]).max()
    sides = []
    for sign in (1.0, -1.0):
        boundary = problem.boundary0.copy()
        boundary[keys[index]] += sign * step
        displaced = problem.create_model(boundary)
        displaced.set_state(
            np.ascontiguousarray(state + sign * step * tangents[:, index])
        )
        sides.append(problem.evaluate(displaced, boundary).residuals)
    difference = (sides[0] - sides[1]) / (2.0 * step)
    np.testing.assert_allclose(
        jacobian[:, index],
        difference,
        rtol=0.0,
        atol=1.0e-6 * np.linalg.norm(difference),
        equal_nan=False,
    )


def test_balanced_residuals_match_the_converged_state(problem, solved) -> None:
    """The residuals of a state away from force balance, with the force residual removed
    to first order, against those of the converged state."""
    boundary = problem.boundary0
    state = np.array(problem.relax(boundary, ftol=1.0e-10).get_state())
    matrix = problem.force_jacobian(problem.create_model(boundary), state)
    values, balanced, _ = problem.balanced_residuals(boundary, state, matrix)
    reference = problem.evaluate(solved, boundary).residuals
    assert np.linalg.norm(balanced - reference) < 0.1 * np.linalg.norm(
        values - reference
    )


def test_a_step_satisfies_the_linearized_force_balance(problem, keys, loose) -> None:
    """F + H dx vanishes on the interior for dx = -H^-1 F + T db, away from force
    balance, by a difference of the force."""
    state, tangents = loose
    boundary = problem.boundary0
    matrix = problem.force_jacobian(problem.create_model(boundary), state)
    _, _, normal = problem.balanced_residuals(boundary, state, matrix)
    delta = 1.0e-3 * np.random.default_rng(2).standard_normal(len(keys))
    model = problem.create_model(boundary)
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, False)
    force = np.array(model.get_forces())[problem.interior]
    product = np.asarray(
        model.hessian_vector_product(np.ascontiguousarray(normal + tangents @ delta))
    )[problem.interior]
    np.testing.assert_allclose(
        force + product,
        0.0,
        atol=1.0e-2 * np.linalg.norm(force),
        equal_nan=False,
    )


def test_normal_steps_restore_force_balance(problem, keys, linearization) -> None:
    """Newton steps on the force from the state of a tangential step converge."""
    state, tangents, _, _ = linearization
    delta = 1.0e-3 * np.random.default_rng(3).standard_normal(len(keys))
    boundary = problem.boundary0.copy()
    for key, value in zip(keys, delta, strict=True):
        boundary[key] += value
    moved = state + tangents @ delta
    norms = []
    for _ in range(3):
        model = problem.create_model(boundary)
        model.set_state(np.ascontiguousarray(moved))
        model.evaluate(2, 2, False)
        norms.append(np.linalg.norm(np.array(model.get_forces())[problem.interior]))
        matrix = problem.force_jacobian(problem.create_model(boundary), moved)
        moved = moved + problem.balanced_residuals(boundary, moved, matrix)[2]
    assert norms[1] < 1.0e-2 * norms[0]
    assert norms[2] < 1.0e-2 * norms[1]


@needs_exact_products
def test_differenced_products_match_the_exact_ones(
    problem, keys, linearization
) -> None:
    """The structural set and the residual derivatives of a build without Enzyme."""
    state, _, _, jacobian = linearization
    differenced = QuasisymmetryProblem(
        problem.vmec_input, ntheta=24, nphi=24, max_mode=1, exact_products=False
    )
    np.testing.assert_array_equal(differenced.interior, problem.interior)
    model = differenced.create_model(differenced.boundary0)
    tangents = differenced.state_tangents(model, state, keys)
    _, approximate = differenced.residual_jacobian(
        model, state, differenced.boundary0, keys, tangents
    )
    np.testing.assert_allclose(
        approximate,
        jacobian,
        rtol=0.0,
        atol=1.0e-4 * np.abs(jacobian).max(),
        equal_nan=False,
    )


@needs_exact_products
def test_exact_derivatives_hold_at_an_unconverged_state(problem) -> None:
    """The exact product against a directional difference of the force, away from
    balance."""
    model = problem.relax(problem.boundary0, ftol=1.0e-5)
    state = np.array(model.get_state())
    direction = np.random.default_rng(0).standard_normal(state.size)
    direction[problem.prescribed] = 0.0
    direction /= np.linalg.norm(direction)
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    exact = np.asarray(
        model.exact_hessian_vector_product(np.ascontiguousarray(direction))
    )
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    difference = np.asarray(
        model.hessian_vector_product(np.ascontiguousarray(direction))
    )
    np.testing.assert_allclose(
        exact,
        difference,
        rtol=0.0,
        atol=1.0e-7 * np.linalg.norm(difference),
        equal_nan=False,
    )


@pytest.mark.parametrize("pressure", [0.0, PRESSURE], ids=["vacuum", "pressure"])
def test_force_is_the_energy_gradient_at_an_unconverged_state(pressure: float) -> None:
    """The raw force against a difference of the MHD energy at fixed flux functions.

    The displacement leaves the axis, the first surface, the boundary and the m = 1
    gauge.
    """
    ns = 17
    vmec_input = rotating_ellipse(4, 4, ns)
    vmec_input.tcon0 = 0.0
    vmec_input.ncurr = 0
    vmec_input.ai = np.asarray([0.2, 0.0, 0.0])
    vmec_input.pres_scale = pressure
    vmec_input.am = np.asarray([1.0, -1.0, 0.0])
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), ns)
    model.evaluate(2, 2, False)
    state = np.array(model.get_state())
    shape = (state.size // (ns * 4 * 5), ns, 4, 5)
    direction = np.random.default_rng(1).standard_normal(shape)
    direction[4:] = 0.0
    direction[:, [0, 1, ns - 1]] = 0.0
    direction[:, :, 1] = 0.0
    direction = direction.ravel() / np.linalg.norm(direction)
    force = np.asarray(model.get_forces()).copy()
    energies = []
    step = 1.0e-5
    for sign in (1.0, -1.0):
        model.set_state(np.ascontiguousarray(state + sign * step * direction))
        model.evaluate(2, 2, False)
        energies.append(model.mhd_energy)
    np.testing.assert_allclose(
        force @ direction,
        (energies[0] - energies[1]) / (2.0 * step) * (ns - 1),
        rtol=1.0e-4,
    )


def test_gradient_at_a_loose_state_points_along_the_converged_one(
    problem, keys, linearization
) -> None:
    """The merit gradient of a state far from force balance against the converged
    gradient."""
    _, _, evaluation, jacobian = linearization
    reference = jacobian.T @ evaluation.residuals
    model = problem.relax(problem.boundary0, ftol=1.0e-6)
    state = np.array(model.get_state())
    fresh = problem.create_model(problem.boundary0)
    tangents = problem.state_tangents(fresh, state, keys)
    loose, loose_jacobian = problem.residual_jacobian(
        fresh, state, problem.boundary0, keys, tangents
    )
    gradient = loose_jacobian.T @ loose.residuals
    cosine = (
        gradient @ reference / (np.linalg.norm(gradient) * np.linalg.norm(reference))
    )
    assert cosine > 0.99


def test_a_relaxation_rolls_back_to_its_starting_state(problem, linearization) -> None:
    """A tolerance already met takes no iterations, so the backup is still the first."""
    state = linearization[0]
    model = problem.relax(problem.boundary0, state, ftol=1.0e-5)
    model.restore_backup()
    np.testing.assert_array_equal(model.get_state(), state)


def test_threads_reproduce_one_thread() -> None:
    """Tangents and residual derivatives do not depend on the number of workers, at a
    boundary other than the one the extra models start from."""
    results = []
    for workers in (1, 3):
        problem = QuasisymmetryProblem(
            rotating_ellipse(4, 4, 17),
            ntheta=24,
            nphi=24,
            max_mode=1,
            workers=workers,
        )
        keys = boundary_modes(1, problem.ntor)
        boundary = problem.boundary0.copy()
        boundary[keys[2]] += 1.0e-2
        state = np.array(problem.relax(boundary, ftol=1.0e-8).get_state())
        model = problem.create_model(boundary)
        tangents = problem.state_tangents(model, state, keys)
        results.append(
            (
                state,
                tangents,
                problem.residual_jacobian(model, state, boundary, keys, tangents)[1],
            )
        )
    for one_thread, threads in zip(*results, strict=True):
        np.testing.assert_array_equal(one_thread, threads)


def test_single_precision_is_refused() -> None:
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            JointProblem(rotating_ellipse(4, 4, 17))
    finally:
        jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("update", "max_mode"),
    [({"lasym": True}, 1), ({"lfreeb": True}, 1), ({}, 4), ({"ntor": 2}, 3)],
    ids=["asymmetric", "free boundary", "max_mode at mpol", "max_mode above ntor"],
)
def test_inputs_outside_the_method_are_refused(update, max_mode: int) -> None:
    vmec_input = rotating_ellipse(4, 4, 17).model_copy(update=update)
    with pytest.raises(ValueError, match="needs"):
        JointProblem(vmec_input, max_mode=max_mode)


def test_a_multigrid_input_runs_on_its_last_resolution() -> None:
    single = rotating_ellipse(4, 4, 17)
    multigrid = single.model_copy(
        update={
            "ns_array": np.asarray([9, 17]),
            "ftol_array": np.asarray([1.0e-6, 1.0e-11]),
            "niter_array": np.asarray([1000, 30000]),
        }
    )
    states = []
    for vmec_input in (single, multigrid):
        problem = JointProblem(vmec_input, max_mode=1)
        model = problem.relax(problem.boundary0, ftol=1.0e-7)
        assert max(model.fsqr, model.fsqz, model.fsql) <= 1.0e-7
        states.append(np.array(model.get_state()))
    np.testing.assert_array_equal(states[0], states[1])


class MeanIota(JointProblem):
    """Mean iota against a target, alone."""

    def residuals(self, solved, boundary):  # noqa: ARG002
        mean_iota = (solved.poloidal_flux[-1] - solved.poloidal_flux[0]) / (
            solved.toroidal_flux[-1] - solved.toroidal_flux[0]
        )
        return jnp.atleast_1d(mean_iota - 0.3)


def test_another_objective_runs_on_the_same_driver() -> None:
    problem = MeanIota(rotating_ellipse(4, 4, 17), max_mode=1)
    result = optimize(problem, (1,))
    wout = vmecpp.run(
        converged_input(problem, result.boundary, pin_gauge=True, ns=problem.ns),
        verbose=False,
        max_threads=1,
    ).wout
    np.testing.assert_allclose(np.mean(np.asarray(wout.iotas)[1:]), 0.3, atol=1.0e-2)


class Constant(JointProblem):
    """Residuals with no derivative, so no trial step lowers the merit."""

    value = 1.0

    def residuals(self, solved, boundary):  # noqa: ARG002
        return jnp.full(1, self.value)


def test_a_stage_reports_how_it_ended() -> None:
    limited = optimize(
        QuasisymmetryProblem(rotating_ellipse(4, 4, 17), max_mode=1), (1,), max_steps=1
    )
    assert [stage.stopped for stage in limited.stages] == ["max_steps"]
    stuck = optimize(Constant(rotating_ellipse(4, 4, 17), max_mode=1), (1,))
    assert [stage.stopped for stage in stuck.stages] == ["no_step"]
    assert stuck.steps == []


def test_a_stage_above_max_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="above max_mode"):
        optimize(Constant(rotating_ellipse(4, 4, 17), max_mode=1), (1, 2))


def test_a_solve_that_raises_counts_its_force_evaluations(monkeypatch) -> None:
    class Raises:
        force_eval_count = 7

        def solve(self) -> None:
            raise RuntimeError

    problem = Constant(rotating_ellipse(4, 4, 17), max_mode=1)
    monkeypatch.setattr(problem, "create_model", lambda *_args: Raises())
    with pytest.raises(RuntimeError):
        problem.relax(problem.boundary0, ftol=1.0e-6)
    assert problem.costs.force_evaluations == 7


class RaisesOnEveryStep(MeanIota):
    """VMEC raises at every boundary other than the first."""

    def relax(self, boundary, state=None, ftol=None, niter=None):
        if not np.array_equal(boundary, self.boundary0):
            raise RuntimeError
        return super().relax(boundary, state, ftol, niter)


def test_steps_on_which_vmec_raises_end_the_stage() -> None:
    result = optimize(RaisesOnEveryStep(rotating_ellipse(4, 4, 17), max_mode=1), (1,))
    assert result.steps == []
    assert result.stages[0].stopped == "no_step"
    assert result.stages[0].failed_solves == 24
    assert result.stages[0].tightened == 1
    assert result.stages[0].rejected_steps == 0


class RaisesOnTheLastNormalStep(MeanIota):
    """VMEC raises once in the last relaxation, the first without an iteration limit."""

    raised = False

    def relax(self, boundary, state=None, ftol=None, niter=None):
        if state is not None and niter is None and not self.raised:
            self.raised = True
            raise RuntimeError
        return super().relax(boundary, state, ftol, niter)


def test_a_run_keeps_its_result_when_the_last_normal_step_fails() -> None:
    problem = RaisesOnTheLastNormalStep(rotating_ellipse(4, 4, 17), max_mode=1)
    result = optimize(problem, (1,), max_steps=1)
    assert problem.raised
    assert result.force_residual <= 3.0 * NESTED_TOLERANCE


class WorseOnEveryStep(MeanIota):
    """The residuals rise at every boundary other than the first."""

    def evaluate(self, model, boundary):
        evaluation = super().evaluate(model, boundary)
        if not np.array_equal(boundary, self.boundary0):
            evaluation.residuals = evaluation.residuals + 1.0
        return evaluation


def test_a_rise_of_the_residuals_refuses_a_step_before_its_force_jacobian() -> None:
    problem = WorseOnEveryStep(rotating_ellipse(4, 4, 17), max_mode=1)
    result = optimize(problem, (1,))
    assert result.steps == []
    assert result.stages[0].stopped == "no_step"
    assert result.stages[0].rejected_steps == 24
    assert problem.costs.assemblies == 2


def test_a_state_that_cannot_be_relaxed_further_stays(monkeypatch) -> None:
    problem = MeanIota(rotating_ellipse(4, 4, 17), max_mode=1)
    state = problem.relax(problem.boundary0, ftol=1.0e-5).get_state()
    point = _joint_point(problem, problem.boundary0, state)
    assert _relaxed_further(problem, point, 1.0e-6) is not None
    relaxations = []

    def raises(*_args, **_kwargs):
        relaxations.append(1)
        raise RuntimeError

    monkeypatch.setattr(problem, "relax", raises)
    assert _relaxed_further(problem, point, 1.0e-6) is None
    relaxations.clear()
    end = Stage(1)
    for _ in range(2):
        point, tolerance = _tightened(problem, point, 1.0e-5, 0.0, end)
        assert tolerance == 1.0e-5
    assert len(relaxations) == 1
    assert end.tightened == 0


def test_a_start_with_a_nearly_singular_force_jacobian_reaches_the_targets() -> None:
    """The normal step of the first state is ten times the major radius."""
    vmec_input = rotating_ellipse(5, 5, 25)
    vmec_input.rbc[1, 5] = vmec_input.zbs[1, 5] = 0.12
    vmec_input.rbc[1, 6], vmec_input.zbs[1, 6] = 0.02, -0.02
    vmec_input.pres_scale = 10000.0
    vmec_input.am = np.asarray([1.0, -1.0, 0.0])
    problem = QuasisymmetryProblem(vmec_input, max_mode=1)
    result = optimize(problem, (1,))
    parts = result.steps[-1].evaluation.parts
    np.testing.assert_allclose(parts["aspect"], problem.aspect, atol=1.0e-1)
    np.testing.assert_allclose(parts["mean_iota"], problem.mean_iota, atol=5.0e-2)


def test_the_merit_takes_the_normal_step_within_its_radius(
    problem, loose, monkeypatch
) -> None:
    """The change of the residuals scales with the part of the normal step within its
    radius, and the error estimate keeps the whole step."""
    boundary = problem.boundary0
    size = _joint_point(problem, boundary, loose[0]).normal_size
    changes, steps, errors = [], [], []
    for fraction in (0.2, 0.1):
        radius = fraction * size / np.abs(boundary[0]).max()
        monkeypatch.setattr("vmecpp.joint._NORMAL_RADIUS", radius)
        point = _joint_point(problem, boundary, loose[0])
        changes.append(point.balanced - point.residuals)
        steps.append(np.abs(point.normal_step()).max())
        errors.append(point.merit_error / np.linalg.norm(point.balanced))
    np.testing.assert_allclose(changes[1], 0.5 * changes[0], rtol=1.0e-9, atol=1.0e-14)
    np.testing.assert_allclose(steps, [0.2 * size, 0.1 * size])
    np.testing.assert_allclose(errors[1], errors[0])


def test_residuals_that_are_not_finite_raise() -> None:
    problem = Constant(rotating_ellipse(4, 4, 17), max_mode=1)
    problem.value = np.nan
    with pytest.raises(ValueError, match="not finite"):
        optimize(problem, (1,))


@pytest.fixture(scope="module")
def nested_and_joint():
    results = {}
    for name in ("nested", "joint"):
        problem = QuasisymmetryProblem(
            rotating_ellipse(4, 4, 17), ntheta=24, nphi=24, max_mode=2
        )
        results[name] = (problem, optimize(problem, (1, 2), nested=name == "nested"))
    return results


@pytest.mark.parametrize("name", ["nested", "joint"])
def test_both_methods_reach_the_targets_from_the_ellipse(
    nested_and_joint, name, tmp_path
) -> None:
    """A cold, converged run of the final boundary, read by SIMSOPT."""
    problem, result = nested_and_joint[name]
    wout = vmecpp.run(
        converged_input(problem, result.boundary, pin_gauge=True),
        verbose=False,
        max_threads=1,
    ).wout
    assert max(wout.fsqr, wout.fsqz, wout.fsql) <= wout.ftolv
    np.testing.assert_allclose(wout.aspect, problem.aspect, atol=5.0e-3)
    np.testing.assert_allclose(
        np.mean(np.asarray(wout.iotas)[1:]), problem.mean_iota, atol=5.0e-3
    )
    assert _simsopt_quasisymmetry(problem, wout, tmp_path) < 5.0e-3


@pytest.mark.parametrize("name", ["nested", "joint"])
def test_a_run_ends_in_force_balance_at_its_last_merit(nested_and_joint, name) -> None:
    _, result = nested_and_joint[name]
    assert result.force_residual <= 3.0 * NESTED_TOLERANCE
    np.testing.assert_allclose(
        result.evaluation.merit, result.steps[-1].evaluation.merit, rtol=5.0e-2
    )


def test_joint_uses_fewer_force_evaluations_for_the_same_error(
    nested_and_joint,
) -> None:
    nested_problem, nested = nested_and_joint["nested"]
    joint_problem, joint = nested_and_joint["joint"]
    assert (
        joint_problem.costs.force_evaluations
        < 0.7 * nested_problem.costs.force_evaluations
    )
    assert (
        joint.steps[-1].evaluation.parts["quasisymmetry"]
        < 2.0 * nested.steps[-1].evaluation.parts["quasisymmetry"]
    )
