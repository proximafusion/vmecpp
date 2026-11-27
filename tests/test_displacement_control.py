# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The example examples/displacement_control.py on a low-resolution spherical
tokamak."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import displacement_control as example  # type: ignore # noqa: E402

pytestmark = pytest.mark.slow

CASE = example.SPHERICAL_TOKAMAK._replace(resolution=(4, 4, 17))
NS = CASE.resolution[2]
# the two minima at this field and pressure, below the fold at eps 5.4e-3
PRESSURE, EPS = 5.35e4, 2.0e-3


def relative_distance(x, y, free) -> float:
    return np.linalg.norm((x - y)[free]) / np.linalg.norm(y[free])


def unbroken_saddle(pressure: float):
    """The saddle between the two helical cores at eps = 0."""
    model, x, x_turned, free = example.helical_cores(CASE, pressure)
    return example.Saddle(CASE, 0.0, model, x, x_turned, free).find()


def follow(pressure, seed, eps, x_previous):
    """The minimum at eps, from the one at eps = 0 with the boundary moved."""

    def guess(value):
        vmec_input = CASE.make(pressure, value, seed)
        return np.asarray(example.create_model(vmec_input).get_state())

    return example.solve(
        CASE.make(pressure, eps, seed), x_previous + guess(eps) - guess(0.0)
    )


def lowest_at_minimum(saddle) -> float:
    return example.spectrum(saddle.model, saddle.free, saddle.x0)[0]


def lowest_without_symmetry(vmec_input, x) -> np.ndarray:
    """The lowest eigenvalues at a symmetric state in the space of lasym = True."""
    nothing, seed = np.zeros_like(vmec_input.rbc), np.zeros_like(vmec_input.raxis_c)
    seed[1] = 0.1
    update = {"lasym": True, "rbs": nothing, "zbc": nothing}
    update |= {"raxis_s": seed, "zaxis_c": seed}
    model = example.create_model(vmec_input.model_copy(update=update))
    free = example.free_coefficients(model)
    state = np.zeros_like(model.get_state())
    size = x.size // 6
    # the blocks of the symmetric state among the twelve of the larger one
    for k, block in enumerate((0, 1, 4, 5, 8, 9)):
        state[block * size : (block + 1) * size] = x[k * size : (k + 1) * size]
    model.set_state(state)
    model.evaluate(2, 2, True)
    return example.spectrum(model, free, state)[:3]


def test_the_saddle_without_an_error_field_is_the_axisymmetric_equilibrium() -> None:
    """The saddle lies halfway between the two helical cores, with the energy of the
    solve from an axisymmetric guess."""
    saddle = unbroken_saddle(PRESSURE)
    axisymmetric, _ = example.solve(CASE.make(PRESSURE))

    assert saddle.residual < 1.0e-19
    np.testing.assert_allclose(saddle.fraction, 0.5, atol=1.0e-5)
    np.testing.assert_allclose(saddle.energy, axisymmetric.mhd_energy, rtol=1.0e-11)
    lowest, second = saddle.lowest_eigenvalues()
    assert lowest < 0.0 < second


def test_the_identities_of_a_pitchfork_hold_at_its_threshold() -> None:
    """Extrapolated linearly in the pressure to the threshold, the barrier is the
    critical eigenvalue at the saddle times tau^2 / 4, and the eigenvalue at the minimum
    is twice minus that at the saddle."""
    pressures = (5.235e4, 5.27e4)
    limit = example.stability_limit(CASE, pressures[::-1])
    identities = []
    for pressure in pressures:
        saddle = unbroken_saddle(pressure)
        lowest = saddle.lowest_eigenvalues()[0]
        identities.append(
            [
                saddle.barrier() * (NS - 1) / (-lowest * saddle.tau**2 / 4.0),
                lowest_at_minimum(saddle) / -lowest,
            ]
        )
    (near, far), (d_near, d_far) = np.array(identities), np.subtract(pressures, limit)
    at_threshold = near - (far - near) * d_near / (d_far - d_near)
    np.testing.assert_allclose(at_threshold, [1.0, 2.0], rtol=3.0e-2)


@pytest.fixture(scope="module")
def broken():
    """The saddle under an error field, the two minima and the free coefficients."""
    _, x, x_turned, free = example.helical_cores(CASE, PRESSURE)
    model, x_high = follow(PRESSURE, CASE.seed, EPS, x)
    _, x_low = follow(PRESSURE, -CASE.seed, EPS, x_turned)
    saddle = example.Saddle(CASE, EPS, model, x_high, x_low, free).find()
    return saddle, x_high, x_low, free


def test_the_saddle_under_an_error_field_has_one_unstable_direction(broken) -> None:
    saddle, *_ = broken

    assert saddle.residual < 1.0e-19
    lowest, second = saddle.lowest_eigenvalues()
    assert lowest < 0.0 < second


def test_the_saddle_under_an_error_field_joins_the_two_minima(broken) -> None:
    """A push along the line ends at either minimum, and the solver from an axisymmetric
    guess does not reach the saddle."""
    saddle, x_high, x_low, free = broken
    push = 0.2 * saddle.tau * saddle.along
    vmec_input = CASE.make(PRESSURE, EPS)
    _, towards_high = example.solve(vmec_input, saddle.state(saddle.x - push))
    _, towards_low = example.solve(vmec_input, saddle.state(saddle.x + push))
    _, from_guess = example.solve(vmec_input)

    assert relative_distance(towards_high, x_high, free) < 1.0e-5
    assert relative_distance(towards_low, x_low, free) < 1.0e-5
    assert relative_distance(from_guess, saddle.state(saddle.x), free) > 1.0e-2


def test_without_stellarator_symmetry_only_the_lower_state_is_a_minimum(broken) -> None:
    """With lasym = True the higher state has one negative eigenvalue and the saddle
    two."""
    saddle, x_high, x_low, _ = broken
    vmec_input = CASE.make(PRESSURE, EPS)

    for state, unstable in ((x_low, 0), (x_high, 1), (saddle.state(saddle.x), 2)):
        assert np.sum(lowest_without_symmetry(vmec_input, state) < 0.0) == unstable


def test_the_saddle_merges_with_the_higher_minimum_at_a_fold() -> None:
    """Three laws of the fold give one eps_c."""
    rows = example.ladder(CASE, PRESSURE, stop_fraction=0.1)
    near = example.near_fold(rows, 3)
    eps_c = example.fold_position(near, [row.tau**2 for row in near])
    for law in (
        [lowest_at_minimum(row) ** 2 for row in near],
        [row.barrier() ** (2.0 / 3.0) for row in near],
    ):
        np.testing.assert_allclose(example.fold_position(near, law), eps_c, rtol=1e-2)
