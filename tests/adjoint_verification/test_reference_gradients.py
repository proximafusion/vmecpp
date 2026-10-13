"""Native interim WP6 gradient checks and an analytic noise/slope oracle."""

import numpy as np
import pytest

from vmecpp import autodiff

from .conftest import requires_enzyme
from .harness import INPUT_NAMES, boundary_array
from .reference import (
    OBJECTIVES,
    ReferenceStudy,
    _comparison,
    _constant_iota_assessment,
    active_boundary,
    directions,
    native_restart,
    reference_input,
    solve_reference,
    taylor_assessment,
)


def test_taylor_analytic_oracle_and_noise_floor():
    steps = np.geomspace(0.1, 1.0e-8, 15)
    # J(x)=3x+2x^2 at x=0: the absolute first-order remainder is 2h^2.
    residual = 2 * steps**2
    floor = 1.0e-12
    noisy = np.maximum(residual, floor)
    result = taylor_assessment(steps, noisy, floor)
    assert result["passed"]
    assert result["fit_stop"] >= 4
    assert result["fit_stop"] < len(steps)
    np.testing.assert_allclose(result["fit_slope"], 2)
    np.testing.assert_allclose(result["slopes"][: result["fit_stop"] - 1], 2)
    assert not taylor_assessment(steps, steps, floor)["passed"]
    assert taylor_assessment(steps, np.zeros_like(steps), floor)["status"] == "noise"
    # A single good pair in a bad sequence is not a second-order range.
    misleading = residual.copy()
    misleading[:3] = steps[:3]
    assert not taylor_assessment(steps, misleading, floor)["passed"]


def test_relative_comparison_and_unresolved_noise():
    steps = np.asarray([0.01, 0.001, 0.0001])
    # J(x)=1e-4*x+2*x^2 at x=0 has central derivative exactly 1e-4.
    exact = 1.0e-4
    plus = exact * steps + 2 * steps**2
    minus = -exact * steps + 2 * steps**2
    fd = (plus - minus) / (2 * steps)
    comparison, matches = _comparison(fd, exact, 1.0e-16, steps)
    assert matches.all()
    assert not comparison["near_zero"].any()
    _, inaccurate = _comparison(fd + 1.0e-9, exact, 1.0e-16, steps)
    assert not inaccurate.any()
    _, additive_allowance = _comparison(np.full(3, 1.0e-4), 1.00005e-4, 0, steps)
    assert not additive_allowance.any()
    noisy, unresolved = _comparison(fd, exact, 1.0e-6, steps)
    assert not noisy["resolved"].any()
    assert not unresolved.any()
    near_zero, zeros = _comparison(np.zeros(3), 0, 1.0e-16, steps)
    assert near_zero["near_zero"].all()
    assert np.isnan(near_zero["relative_error"]).all()
    assert zeros.all()


def test_prescribed_iota_constant_is_not_arbitrary_noise_exemption():
    source = reference_input("cth_like_fixed_bdy_iota")
    gradient = np.zeros_like(boundary_array(source))
    values = np.full(5, 0.8)
    remainder = np.zeros(5)
    result = _constant_iota_assessment(
        source, "iota", gradient, 0.8, values, values, remainder
    )
    assert result is not None
    assert result["passed"]
    assert result["status"] == "constant"
    assert not result["slope_applicable"]
    assert np.isnan(result["fit_slope"])
    assert (
        _constant_iota_assessment(
            source, "volume", gradient, 0.8, values, values, remainder
        )
        is None
    )
    assert (
        _constant_iota_assessment(
            reference_input("small"), "iota", gradient, 0.8, values, values, remainder
        )
        is None
    )
    bad_gradient = gradient.copy()
    bad_gradient[0, 1, source.ntor] = 1.0e-5
    bad_derivative = _constant_iota_assessment(
        source, "iota", bad_gradient, 0.8, values, values, remainder
    )
    assert bad_derivative is not None
    assert not bad_derivative["passed"]
    bad_value = _constant_iota_assessment(
        source, "iota", gradient, 0.8, values + 1.0e-6, values, remainder
    )
    assert bad_value is not None
    assert not bad_value["passed"]
    failed_solve = _constant_iota_assessment(
        source, "iota", gradient, 0.8, values * np.nan, values, remainder
    )
    assert failed_solve is not None
    assert failed_solve["status"] == "failure"


def test_component_scan_polynomial_oracle_all_slots_and_strict_target():
    class PolynomialOracle(ReferenceStudy):
        def __init__(self):
            self.input = reference_input("small")
            self.boundary = boundary_array(self.input)
            self.weights = np.arange(1, self.boundary.size + 1, dtype=float).reshape(
                self.boundary.shape
            ) * active_boundary(self.input)
            self.objectives = {"polynomial": None}
            self.gradients = {"polynomial": self.weights * (2 * self.boundary + 1)}
            self.noise = {"polynomial": 0}
            self.visited = []

        def evaluate(self, boundary):
            changed = np.flatnonzero((boundary - self.boundary).ravel())
            assert len(changed) == 1
            self.visited.append(int(changed[0]))
            return np.asarray([np.sum(self.weights * (boundary**2 + boundary))])

    oracle = PolynomialOracle()
    steps = np.asarray([0.001, 0.0003, 0.0001])
    result = oracle.components(steps)[0]
    assert result["passed"]
    np.testing.assert_array_equal(
        np.bincount(oracle.visited, minlength=oracle.boundary.size),
        np.full(oracle.boundary.size, 2 * len(steps)),
    )
    ntor = oracle.input.ntor
    assert np.all(result["gradient_ad"][:, 0, :ntor] != 0)
    np.testing.assert_allclose(
        result["gradient_fd"][:, :, 0, :ntor],
        np.broadcast_to(result["gradient_ad"][:, 0, :ntor], (len(steps), 2, ntor)),
        rtol=1.0e-6,
        atol=0,
    )
    oracle.gradients["polynomial"] *= 1 + 2.0e-6
    inaccurate = oracle.components(steps)[0]
    assert not inaccurate["passed"]
    assert inaccurate["status"] == "failure"
    assert not inaccurate["passed_modes"][inaccurate["active"]].any()


def test_direction_normalization_and_inactive_modes():
    source = reference_input("small")
    mask = active_boundary(source)
    assert not mask[1, 0, source.ntor]
    assert mask[:, 0, : source.ntor].all()
    assert mask[:, 1:, : source.ntor].all()
    for name, direction in directions(source, seed=42).items():
        np.testing.assert_array_equal(direction, directions(source, seed=42)[name])
        np.testing.assert_allclose(np.linalg.norm(direction), 1)
        assert np.all(direction[~mask] == 0)
        assert np.all(direction[:, 0, : source.ntor] == 0)
    axisymmetric = reference_input("solovev")
    assert directions(axisymmetric)["highn"] is None


def test_native_reference_convergence_and_state_restart():
    source = reference_input("small")
    base = solve_reference(source)
    original = np.asarray(base.get_state()).copy()
    restart = native_restart(source)
    repeated = solve_reference(source, restart_from=restart)
    assert base.status == repeated.status == 11
    assert base.ftolv == repeated.ftolv == 1.0e-16
    np.testing.assert_allclose(repeated.get_state(), base.get_state(), atol=1.0e-7)
    boundary = boundary_array(source)
    boundary[0, 1, source.ntor + 1] += 1.0e-4
    warm = solve_reference(source, boundary, restart_from=restart)
    cold = solve_reference(source, boundary)
    # ftol bounds squared forces; this interim solve is not machine-precision.
    np.testing.assert_allclose(
        warm.get_state(), cold.get_state(), rtol=0, atol=100 * np.sqrt(base.ftolv)
    )
    np.testing.assert_array_equal(base.get_state(), original)
    gauge = autodiff._gauge_entries(cold)
    np.testing.assert_array_equal(
        np.asarray(warm.get_state())[gauge], np.asarray(cold.get_state())[gauge]
    )


@requires_enzyme
def test_reference_boundary_smoke():
    study = ReferenceStudy(reference_input("small"), ("boundary",))
    result = study.taylor(np.geomspace(1.0e-2, 1.0e-6, 9), names=("random",))
    assert result[0]["passed"], result[0]
    np.testing.assert_allclose(
        result[0]["central_fd"][:4],
        result[0]["directional_ad"],
        rtol=1.0e-6,
        atol=1.0e-8,
    )


@pytest.mark.slow
@requires_enzyme
@pytest.mark.parametrize("case", INPUT_NAMES)
def test_reference_taylor_shared_cases(case):
    study = ReferenceStudy(reference_input(case), OBJECTIVES)
    for result in study.taylor():
        if result["status"] == "inactive":
            assert case in ("solovev", "solovev_analytical", "circular_tokamak")
            continue
        assert result["passed"], (
            case,
            result["objective"],
            result["direction_name"],
            result["status"],
            result["fit_slope"],
            result["fit_stop"],
        )


@pytest.mark.slow
@requires_enzyme
@pytest.mark.parametrize("case", ["cth_like_fixed_bdy", "li383_low_res"])
def test_reference_components_all_signed_modes(case):
    study = ReferenceStudy(reference_input(case), OBJECTIVES)
    for result in study.components():
        assert result["passed"], (
            case,
            result["objective"],
            np.argwhere(~result["passed_modes"]),
        )
