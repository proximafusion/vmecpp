# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""examples/joint_mercier.py against the wout of VMEC++ and differences of re-converged
equilibria."""

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

import vmecpp
from vmecpp.joint import CONSTRAINT_TOLERANCE, boundary_modes, converged_input, optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from joint_mercier import MercierStableProblem, stability  # type: ignore
from joint_optimization import rotating_ellipse  # type: ignore

jax.config.update("jax_enable_x64", True)


def _input() -> vmecpp.VmecInput:
    """The rotating ellipse at a pressure that gives a beta of 0.7%."""
    vmec_input = rotating_ellipse(4, 4, 17)
    vmec_input.pres_scale = 6000.0
    vmec_input.am = np.asarray([1.0, -1.0, 0.0])
    return vmec_input


@pytest.fixture(scope="module")
def problem() -> MercierStableProblem:
    return MercierStableProblem(_input(), ntheta=24, nphi=24, max_mode=1)


def test_the_constraints_of_the_start_are_those_of_its_wout(problem) -> None:
    """``DMerc`` on the surfaces between the radii of the problem of a state in force
    balance against that of a cold run of its boundary."""
    model = problem.relax(problem.boundary0, ftol=1.0e-14)
    values = problem.evaluate(model, problem.boundary0).constraints
    cold = problem.vmec_input.model_copy(update={"ftol_array": np.asarray([1.0e-14])})
    wout = vmecpp.run(cold, verbose=False, max_threads=1).wout
    full = np.linspace(0.0, 1.0, wout.ns)
    inside = (full >= 0.1) & (full <= 0.9)
    assert values.size == inside.sum()
    np.testing.assert_allclose(values, np.asarray(wout.DMerc)[inside], rtol=1.0e-6)


def test_the_constraint_derivatives_match_converged_solves(problem) -> None:
    """The total derivatives of ``DMerc`` against central differences of re-converged
    equilibria."""
    keys = boundary_modes(1, problem.ntor)
    state = np.array(problem.relax(problem.boundary0, ftol=1.0e-14).get_state())
    model = problem.create_model(problem.boundary0)
    tangents = problem.state_tangents(model, state, keys)
    _, jacobian = problem.residual_jacobian(
        model, state, problem.boundary0, keys, tangents
    )
    rows = jacobian[jacobian.shape[0] - problem.lower.size :]
    step = 1.0e-6
    for index in (0, 3, 6):
        sides = []
        for sign in (1.0, -1.0):
            boundary = problem.boundary0.copy()
            boundary[keys[index]] += sign * step
            solved = problem.relax(
                boundary, state + sign * step * tangents[:, index], ftol=1.0e-15
            )
            sides.append(problem.evaluate(solved, boundary).constraints)
        difference = (sides[0] - sides[1]) / (2.0 * step)
        np.testing.assert_allclose(
            rows[:, index],
            difference,
            rtol=0.0,
            atol=1.0e-3 * np.linalg.norm(difference),
        )


@pytest.fixture(scope="module")
def runs():
    """The run without constraints, and the run within the Mercier criterion by both
    methods, each with the wout of a cold run of its result."""
    results = {}
    for name, nested, mercier in (
        ("free", False, None),
        ("nested", True, 0.0),
        ("joint", False, 0.0),
    ):
        problem = MercierStableProblem(_input(), max_mode=2, mercier=mercier)
        result = optimize(problem, (1, 2), nested=nested)
        wout = vmecpp.run(
            converged_input(problem, result.boundary, pin_gauge=True),
            verbose=False,
            max_threads=1,
        ).wout
        results[name] = (problem, result, wout)
    return results


def test_quasisymmetry_alone_ends_unstable(runs) -> None:
    """Without the constraints, a cold run of the result has ``DMerc`` below 0 on every
    surface between the radii."""
    problem, _, wout = runs["free"]
    found = stability(problem, wout)
    assert found["unstable"] == found["surfaces"]


@pytest.mark.parametrize("name", ["nested", "joint"])
def test_both_methods_end_within_the_mercier_criterion(runs, name) -> None:
    """The run ends with ``DMerc`` of a surface on its bound and none below it, a cold
    run of its result on more surfaces has none below 0, and its quasisymmetry residuals
    are larger than those of the run without the constraints."""
    problem, result, wout = runs[name]
    values = result.evaluation.constraints
    assert result.evaluation.parts["violation"] <= CONSTRAINT_TOLERANCE
    np.testing.assert_allclose(values.min(), 0.0, atol=CONSTRAINT_TOLERANCE)
    assert [stage.shift_updates > 0 for stage in result.stages] == [False, True]
    found = stability(problem, wout)
    assert found["unstable"] == 0
    np.testing.assert_allclose(wout.aspect, problem.aspect, atol=1.0e-3)
    free = runs["free"][1].evaluation.parts["quasisymmetry"]
    assert free < result.evaluation.parts["quasisymmetry"] < 20.0 * free
