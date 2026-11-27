# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the helical-core Ising example."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import helical_core_ising as hc

# the coupling favours opposite spins and the second field the same ones
FRUSTRATED = hc.Boundary(coupling=5e-3, field_1=1e-4, field_2=-5e-5)


@pytest.fixture(scope="module")
def coefficients():
    energies = [hc.held_energy(*hold) for hold in hc.HOLDS]
    return hc.ising_coefficients(*energies)


def test_a_quarter_turn_maps_the_cosine_field_to_the_sine_field():
    cosine = hc.boundary_coefficients(hc.Boundary(0.0, 1e-4, 0.0))
    sine = hc.boundary_coefficients(hc.Boundary(0.0, 0.0, 1e-4))
    np.testing.assert_allclose(hc.turned(cosine, 90.0), sine, atol=1e-15)


def test_the_holds_are_symmetric_in_the_frames_of_their_diagonals():
    for boundary, diagonal, _ in hc.HOLDS:
        _, rbs, zbc, _ = hc.turned(hc.boundary_coefficients(boundary), diagonal)
        np.testing.assert_allclose(rbs, 0.0, atol=1e-15)
        np.testing.assert_allclose(zbc, 0.0, atol=1e-15)


def test_the_coupling_outweighs_the_second_field(coefficients):
    coupling, field_1, field_2 = hc.ising(FRUSTRATED, *coefficients)
    assert coupling < 0.0 < field_2 < field_1 < -coupling


def test_the_plasma_reaches_the_ground_state_against_the_weaker_field(coefficients):
    ground = hc.ground_state(*hc.ising(FRUSTRATED, *coefficients))
    assert ground == (1, -1)
    assert hc.spins(hc.anneal(FRUSTRATED)) == ground


@pytest.mark.slow
def test_the_held_states_follow_the_ising_energy(coefficients):
    # the four states the boundary holds without stellarator symmetry
    e = {}
    for orientation in (45.0, 135.0, 225.0, 315.0):
        vmec_input = hc.spherical_tokamak(
            hc.PRESSURE, hc.boundary_coefficients(FRUSTRATED), orientation, hc.HELD
        )
        vmec_input.ftol_array = np.array([1e-16])
        output = hc.solve(vmec_input)
        e[hc.spins(hc.core_of(output)[1])] = hc.energy(output)
    assert len(e) == 4
    held = (
        (e[1, -1] + e[-1, 1] - e[1, 1] - e[-1, -1]) / 4.0,
        (e[-1, 1] + e[-1, -1] - e[1, 1] - e[1, -1]) / 4.0,
        (e[1, -1] + e[-1, -1] - e[1, 1] - e[-1, 1]) / 4.0,
    )
    np.testing.assert_allclose(held, hc.ising(FRUSTRATED, *coefficients), rtol=0.1)
    assert min(e, key=e.get) == (1, -1)
