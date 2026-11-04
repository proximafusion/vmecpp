# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The example examples/displacement_control.py: saddle points of the spherical tokamak
with and without an error field, and the laws of its imperfect pitchfork."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import displacement_control as example  # type: ignore # noqa: E402

CASE = example.SPHERICAL_TOKAMAK
# the two minima at this field and pressure, well below the fold
PRESSURE, EPS = 3.8e4, 0.02


def relative_distance(x, y, free) -> float:
    return np.linalg.norm((x - y)[free]) / np.linalg.norm(y[free])


def unbroken_saddle(pressure: float):
    """The saddle between the mirror images at eps = 0."""
    model, x, x_mirror, free = example.mirror_minima(CASE, pressure)
    return example.saddle_between(CASE, 0.0, model, x, x_mirror, free)


def test_the_saddle_without_an_error_field_is_the_axisymmetric_equilibrium() -> None:
    """The saddle lies halfway between the mirror images and has the energy of the
    equilibrium the solver reaches from an axisymmetric guess."""
    saddle = unbroken_saddle(1.2e5)
    axisymmetric, _ = example.solve(CASE.make(1.2e5))

    assert saddle.residual < 1.0e-19
    np.testing.assert_allclose(saddle.fraction, 0.5, atol=1.0e-5)
    np.testing.assert_allclose(saddle.energy, axisymmetric.mhd_energy, rtol=1.0e-11)
    lowest, second = saddle.lowest_eigenvalues()
    assert lowest < 0.0 < second


def test_the_barrier_follows_the_law_of_a_pitchfork() -> None:
    """The barrier from the work of the force grows as the square of the distance from
    the threshold, which comes from the critical eigenvalue of the unbroken equilibrium.

    Its size and the eigenvalues at the two states are those of V(a) = -r a^2 / 2 + g
    a^4 / 4.
    """
    limit = example.stability_limit(CASE, [2.7e4, 2.6e4])
    barriers = {}
    for pressure in (2.6e4, 2.8e4):
        saddle = unbroken_saddle(pressure)
        barriers[pressure] = saddle.barrier()
        lowest = saddle.lowest_eigenvalues()[0]
        ns = saddle.landscape.model.ns
        np.testing.assert_allclose(
            barriers[pressure] * (ns - 1), -lowest * saddle.tau**2 / 4.0, rtol=5.0e-3
        )
        np.testing.assert_allclose(
            saddle.lowest_at_minimum() / -lowest, 2.0, rtol=2.0e-2
        )
    np.testing.assert_allclose(
        barriers[2.8e4] / barriers[2.6e4],
        ((2.8e4 - limit) / (2.6e4 - limit)) ** 2,
        rtol=2.0e-2,
    )


@pytest.fixture(scope="module")
def broken():
    """The saddle under an error field, the two minima and the free coefficients."""
    _, x, x_mirror, free = example.mirror_minima(CASE, PRESSURE)
    model, x_high = example.follow(CASE, PRESSURE, CASE.seed, EPS, x, 0.0)
    _, x_low = example.follow(CASE, PRESSURE, -CASE.seed, EPS, x_mirror, 0.0)
    saddle = example.saddle_between(CASE, EPS, model, x_high, x_low, free)
    return saddle, x_high, x_low, free


def test_the_saddle_under_an_error_field_has_one_unstable_direction(broken) -> None:
    saddle, *_ = broken

    assert saddle.residual < 1.0e-19
    lowest, second = saddle.lowest_eigenvalues()
    assert lowest < 0.0 < second


def test_the_saddle_under_an_error_field_joins_the_two_minima(broken) -> None:
    """A push along the line in either direction ends at a different minimum, and the
    solver from an axisymmetric guess does not reach the saddle."""
    saddle, x_high, x_low, free = broken
    push = 0.2 * saddle.tau * saddle.e
    state = saddle.landscape.state
    vmec_input = CASE.make(PRESSURE, EPS)
    _, towards_high = example.solve(vmec_input, state(saddle.y - push))
    _, towards_low = example.solve(vmec_input, state(saddle.y + push))
    _, from_guess = example.solve(vmec_input)

    assert relative_distance(towards_high, x_high, free) < 1.0e-5
    assert relative_distance(towards_low, x_low, free) < 1.0e-5
    assert relative_distance(from_guess, saddle.x, free) > 1.0e-2


def test_without_stellarator_symmetry_only_the_lower_state_is_a_minimum(broken) -> None:
    """In the larger space of lasym = True, the core can turn toroidally: the higher
    state gains one unstable direction and the saddle a second."""
    saddle, x_high, x_low, _ = broken
    vmec_input = CASE.make(PRESSURE, EPS)

    for state, unstable in ((x_low, 0), (x_high, 1), (saddle.x, 2)):
        lowest = example.lowest_without_symmetry(vmec_input, state)
        assert np.sum(lowest < 0.0) == unstable


def test_the_saddle_merges_with_the_higher_minimum_at_a_fold() -> None:
    """Three laws of the fold give one eps_c, and eps_c scales as the cube of the
    helical amplitude, which involves no estimate of the threshold."""
    critical = {}
    for pressure in (2.7e4, 2.8e4):
        model, x, _, _ = example.mirror_minima(CASE, pressure)
        amplitude = example.odd_amplitude(model, x)
        rows = example.ladder(CASE, pressure, 4.0 * amplitude**3, stop_fraction=0.1)
        near = [row for row in rows if row.fraction < example.NEAR_FOLD][-3:]
        eps = [row.eps for row in near]
        eps_c = example.fold_position(eps, [row.tau**2 for row in near])
        critical[pressure] = eps_c / amplitude**3
        from_eigenvalue = [row.lowest_at_minimum() ** 2 for row in near]
        from_barrier = [row.barrier() ** (2.0 / 3.0) for row in near]
        np.testing.assert_allclose(
            example.fold_position(eps, from_eigenvalue), eps_c, rtol=1.0e-2
        )
        np.testing.assert_allclose(
            example.fold_position(eps, from_barrier), eps_c, rtol=1.0e-2
        )
    np.testing.assert_allclose(critical[2.7e4], critical[2.8e4], rtol=2.0e-2)
