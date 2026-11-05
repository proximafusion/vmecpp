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
    JointProblem,
    QuasisymmetryProblem,
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
        atol=(1.0e-4 if problem.exact_products else 1.0e-2)
        * np.linalg.norm(difference),
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
        atol=1.0e-3 * np.abs(jacobian).max(),
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


def test_threads_reproduce_one_thread(linearization) -> None:
    """Tangents and residual derivatives do not depend on the number of workers."""
    state = linearization[0]
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
        model = problem.create_model(problem.boundary0)
        tangents = problem.state_tangents(model, state, keys)
        results.append(
            (
                tangents,
                problem.residual_jacobian(
                    model, state, problem.boundary0, keys, tangents
                )[1],
            )
        )
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_single_precision_is_refused() -> None:
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            JointProblem(rotating_ellipse(4, 4, 17))
    finally:
        jax.config.update("jax_enable_x64", True)


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
    assert stuck.stages[0].rejected_steps == 12
    assert stuck.stages[0].tightened > 0
    assert stuck.steps == []


def test_residuals_that_are_not_finite_raise() -> None:
    problem = Constant(rotating_ellipse(4, 4, 17), max_mode=1)
    problem.value = np.nan
    with pytest.raises(ValueError, match="not finite"):
        optimize(problem, (1,))


@pytest.fixture(scope="module")
def nested_and_joint():
    results = {}
    for name, factor in (("nested", 0.0), ("joint", 0.3)):
        problem = QuasisymmetryProblem(
            rotating_ellipse(4, 4, 17), ntheta=24, nphi=24, max_mode=2
        )
        results[name] = (problem, optimize(problem, (1, 2), factor))
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
