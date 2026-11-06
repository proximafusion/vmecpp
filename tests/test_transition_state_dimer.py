# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The example examples/transition_state_dimer.py: a dimer climbs from a helical-core
equilibrium to the transition state between it and its mirror image."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import transition_state_dimer as example  # type: ignore # noqa: E402


def test_the_dimer_reaches_the_axisymmetric_equilibrium() -> None:
    """The saddle between the two mirror-image helical-core equilibria is the
    axisymmetric equilibrium: the dimer reaches its energy, with the odd-n
    coefficients gone and one direction of negative curvature."""
    saddle = example.climb_from_helical_minimum()
    axisymmetric = example.solve(example.helical_core_input())
    x_saddle = saddle.landscape.state(saddle.y)

    assert saddle.curvature < 0.0
    assert example.negative_curvatures(saddle.landscape, saddle.y) == 1
    helical_amplitude = np.abs(saddle.x_min[saddle.odd]).max()
    assert np.abs(x_saddle[saddle.odd]).max() < 1.0e-4 * helical_amplitude
    np.testing.assert_allclose(saddle.energy, axisymmetric.mhd_energy, rtol=1.0e-11)
    assert saddle.energy > saddle.energy_min
