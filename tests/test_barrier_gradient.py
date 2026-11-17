# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The example examples/barrier_gradient.py: the gradient of the energy barrier between
two equilibria from one adjoint solve per equilibrium."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import barrier_gradient as example  # type: ignore # noqa: E402


def test_the_barrier_gradient_matches_re_solved_equilibria() -> None:
    """For a profile parameter and a boundary coefficient, the gradient from the two
    adjoint solves agrees with a central difference of re-solved barriers."""
    parameters = example.BASE_PARAMETERS
    gradients = example.barrier_gradients(
        parameters, {"pres_scale": 1.0e2, "elongation": 1.0e-5}
    )
    for name, step in (("pres_scale", 1.0e3), ("elongation", 2.0e-3)):
        gradient, _ = gradients[name]
        re_solved = example.re_solved_barrier_gradient(parameters, name, step)
        np.testing.assert_allclose(gradient, re_solved, rtol=2.0e-4)
