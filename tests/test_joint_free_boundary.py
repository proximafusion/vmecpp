# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""examples/joint_free_boundary.py on the CTH-like case against the wout, the runs of
examples/free_boundary_root.py, differences of converged free boundaries and VMEC++'s
own free-boundary run."""

import sys
from pathlib import Path

import numpy as np
import pytest

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import free_boundary_root  # type: ignore
import joint_free_boundary  # type: ignore


@pytest.fixture(scope="module")
def case():
    """The free-boundary input, the response table of its coils and their currents."""
    vmec_input, response = joint_free_boundary.cth_like(15)
    return vmec_input, response, np.asarray(vmec_input.extcur, dtype=float)


@pytest.fixture(scope="module")
def problem(case):
    vmec_input, response, currents = case
    return joint_free_boundary.FreeBoundaryProblem(
        vmec_input.model_copy(update={"lfreeb": False}),
        aspect=5.5,
        mean_iota=1.15,
        field=joint_free_boundary.CoilCurrents(response),
        scales=np.abs(currents),
        modes=joint_free_boundary.shape_modes(vmec_input.mpol - 1, vmec_input.ntor),
    )


@pytest.fixture(scope="module")
def free_boundary(problem, case):
    """The converged free boundary at the currents of the input."""
    return joint_free_boundary.converge(
        problem, np.zeros(len(problem.modes)), case[2], None
    )


@pytest.fixture(scope="module")
def nested_and_joint(problem, free_boundary):
    """The linearizations of each method and its result on a converged free boundary."""
    results = {}
    for name in ("nested", "joint"):
        problem.linearizations = 0
        result = joint_free_boundary.optimize(
            problem, free_boundary, nested=name == "nested"
        )
        results[name] = (
            problem.linearizations,
            joint_free_boundary.converge(
                problem, result["shape"], result["controls"], result["state"]
            ),
        )
    return results


def _modes(problem, pressure) -> np.ndarray:
    """A pressure on the solver's boundary grid in the shape modes."""
    return problem.matching.projection @ (np.asarray(pressure).ravel() / problem.scale)


def _vmecpp_own(case, currents):
    """The wout of VMEC++'s own free-boundary run at the currents."""
    vmec_input, response, _ = case
    return vmecpp.run(
        vmec_input.model_copy(update={"extcur": np.asarray(currents)}),
        response,
        verbose=False,
        max_threads=1,
    ).wout


def test_the_plasma_side_is_the_edge_pressure_of_the_wout(problem) -> None:
    """The pressure modes of the state at the root of the force against the wout of a
    converged run inside the same boundary."""
    boundary = problem.reference
    state, _ = problem.root(boundary)
    model = problem.create_model(boundary)
    model.set_state(state)
    plasma = problem.split(problem.evaluate(model, boundary).residuals)[1]
    wout = vmecpp.run(
        problem.vmec_input.model_copy(
            update={
                "rbc": boundary[0],
                "zbs": boundary[1],
                "always_fix_m1_gauge": True,
                "ftol_array": np.array([1.0e-15]),
            }
        ),
        verbose=False,
        max_threads=1,
    ).wout
    expected = _modes(
        problem, free_boundary_root.edge_pressure(wout, *problem.matching.grid)
    )
    np.testing.assert_allclose(
        plasma, expected, rtol=0.0, atol=1.0e-6 * np.abs(expected).max()
    )


def test_the_vacuum_side_of_a_state_is_that_of_a_run_inside_its_boundary(
    problem, case
) -> None:
    """At a moved boundary, against the free-boundary iteration that
    examples/free_boundary_root.py restarts from a converged run."""
    _, response, currents = case
    shape = np.zeros(len(problem.modes))
    shape[problem.modes.index((1, 0, 0))] = 1.0e-2
    shape[problem.modes.index((2, 1, 0))] = -3.0e-3
    boundary = problem.boundary_of(shape)
    state, _ = problem.root(boundary)
    matching = free_boundary_root.Matching(
        problem.vmec_input.model_copy(update={"extcur": currents}),
        problem.modes,
        lambda _: response,
    )
    _, free = free_boundary_root.solve(matching, shape, [])
    expected = _modes(problem, free.threed1_free_boundary.bsqvaci)
    np.testing.assert_allclose(
        problem.vacuum_modes(state, currents),
        expected,
        rtol=0.0,
        atol=1.0e-6 * np.abs(expected).max(),
    )


def test_the_free_boundary_matches_vmecpps_own_run(
    problem, case, free_boundary
) -> None:
    """Newton on the jump from the input's boundary against VMEC++'s iteration."""
    np.testing.assert_allclose(free_boundary["jump"], 0.0, atol=1.0e-9)
    parts = problem.parts(
        np.concatenate([free_boundary["residuals"], free_boundary["jump"]])
    )
    wout = _vmecpp_own(case, case[2])
    np.testing.assert_allclose(parts["aspect"], wout.aspect, rtol=5.0e-4)
    np.testing.assert_allclose(
        parts["mean_iota"], np.mean(np.asarray(wout.iotas)[1:]), rtol=2.0e-3
    )


def test_every_shape_mode_of_the_free_boundary_is_stable(free_boundary) -> None:
    """The eigenvalues of the jump's Jacobian are negative where VMEC++'s iteration
    settles."""
    assert np.all(np.linalg.eigvals(free_boundary["jump_along_shape"]).real < 0.0)


@pytest.mark.parametrize("index", [0, 1])
def test_current_derivatives_match_converged_free_boundaries(
    problem, free_boundary, index
) -> None:
    """The derivatives of the objective residuals and of the shape with respect to a
    current, in units of its scale, against central differences of free boundaries
    converged at changed currents."""
    relative = 2.0e-3
    sides = []
    for sign in (1.0, -1.0):
        currents = free_boundary["controls"].copy()
        currents[index] += sign * relative * problem.scales[index]
        change = sign * relative * free_boundary["sensitivity"][:, index]
        sides.append(
            joint_free_boundary.converge(
                problem,
                free_boundary["shape"] + change,
                currents,
                problem.carried(
                    free_boundary["state"], change, free_boundary["tangents"]
                ),
            )
        )
    for derivative, name in (("reduced", "residuals"), ("sensitivity", "shape")):
        difference = (sides[0][name] - sides[1][name]) / (2.0 * relative)
        np.testing.assert_allclose(
            free_boundary[derivative][:, index],
            difference,
            rtol=0.0,
            atol=1.0e-4 * np.linalg.norm(difference),
            err_msg=name,
        )


def test_a_state_off_force_balance_takes_the_normal_step(
    problem, free_boundary
) -> None:
    """The objective residuals and the jump of a loosely relaxed state against those of
    the converged one.

    The normal step removes the force residual to first order.
    """
    shape, currents = free_boundary["shape"], free_boundary["controls"]
    boundary = problem.boundary_of(shape)
    loose = np.array(problem.relax(boundary, None, 1.0e-8).get_state())
    model = problem.create_model(boundary)
    model.set_state(np.ascontiguousarray(loose))
    residuals, plasma = problem.split(problem.evaluate(model, boundary).residuals)
    jump = plasma - problem.vacuum_modes(loose, currents)
    point = problem.linearize(shape, currents, loose, at_root=False)
    for name, plain in (("residuals", residuals), ("jump", jump)):
        assert np.linalg.norm(point[name] - free_boundary[name]) < 0.1 * (
            np.linalg.norm(plain - free_boundary[name])
        ), name


def test_a_tether_adds_the_distance_from_its_anchor_to_the_residuals(
    problem, free_boundary, monkeypatch
) -> None:
    controls = free_boundary["controls"]
    monkeypatch.setattr(problem, "tether", 1.0e-2)
    monkeypatch.setattr(problem, "anchor", controls - 2.0 * problem.scales)
    point = problem.linearize(
        free_boundary["shape"], controls, free_boundary["state"], at_root=True
    )
    count = controls.size
    np.testing.assert_allclose(point["balanced"][-count:], 2.0e-2)
    np.testing.assert_allclose(point["reduced"][-count:], 1.0e-2 * np.eye(count))
    np.testing.assert_allclose(
        point["balanced"][:-count], free_boundary["balanced"], atol=1.0e-9
    )
    np.testing.assert_allclose(
        point["merit"], free_boundary["merit"] + count * 4.0e-4, rtol=1.0e-9
    )


def test_shape_modes_outside_those_of_the_boundary_are_refused(case) -> None:
    vmec_input, response, currents = case
    with pytest.raises(ValueError, match="shape modes"):
        joint_free_boundary.FreeBoundaryProblem(
            vmec_input.model_copy(update={"lfreeb": False}),
            field=joint_free_boundary.CoilCurrents(response),
            scales=np.abs(currents),
            modes=joint_free_boundary.shape_modes(vmec_input.mpol, vmec_input.ntor),
        )


def test_both_methods_reach_the_same_currents(nested_and_joint) -> None:
    np.testing.assert_allclose(
        nested_and_joint["joint"][1]["controls"],
        nested_and_joint["nested"][1]["controls"],
        rtol=1.0e-5,
    )


def test_joint_takes_fewer_linearizations(nested_and_joint) -> None:
    assert nested_and_joint["joint"][0] < nested_and_joint["nested"][0]


@pytest.mark.parametrize("name", ["nested", "joint"])
def test_vmecpps_own_run_at_the_currents_found_meets_the_targets(
    problem, case, nested_and_joint, name
) -> None:
    wout = _vmecpp_own(case, nested_and_joint[name][1]["controls"])
    assert max(wout.fsqr, wout.fsqz, wout.fsql) <= wout.ftolv
    np.testing.assert_allclose(wout.aspect, problem.aspect, rtol=5.0e-4)
    np.testing.assert_allclose(
        np.mean(np.asarray(wout.iotas)[1:]), problem.mean_iota, rtol=2.0e-3
    )
