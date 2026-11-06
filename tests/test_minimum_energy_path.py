# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The example examples/minimum_energy_path.py: the string method between the two mirror
images of a helical-core equilibrium."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import minimum_energy_path as example  # type: ignore # noqa: E402


def test_the_path_climbs_to_the_axisymmetric_equilibrium() -> None:
    """The climbing image of the path between the mirror images is the axisymmetric
    equilibrium; the energy rises to it from both ends, and the two halves of the path
    carry the same energies."""
    path = example.path_between_mirror_images()
    axisymmetric = example.solve(example.helical_core_input())
    energies = path.energies
    top = path.top
    barrier = energies[top] - energies[0]

    assert top == (energies.size - 1) // 2
    np.testing.assert_allclose(energies[top], axisymmetric.mhd_energy, rtol=1.0e-11)
    assert np.all(np.diff(energies[: top + 1]) > 0.0)
    np.testing.assert_allclose(
        energies, energies[::-1], rtol=0.0, atol=1.0e-6 * barrier
    )
    x_top = path.landscape.state(path.chain[top])
    assert np.abs(x_top[path.odd]).max() < 1.0e-10
