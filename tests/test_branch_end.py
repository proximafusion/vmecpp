# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The example examples/branch_end.py: the helical-core branch of a spherical tokamak
ends at a pitchfork, and with a boundary ripple at a fold."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import branch_end as example  # type: ignore # noqa: E402


def test_the_helical_core_ends_at_a_pitchfork_and_with_a_ripple_at_a_fold() -> None:
    """Without the ripple, the critical eigenvalue vanishes linearly, at the pres_scale
    where the axisymmetric equilibrium loses its stability; with it, the branch the
    ripple opposes ends at a higher pres_scale, where the eigenvalue vanishes as a
    square root."""
    mu_pitchfork, b_pitchfork = example.fit_branch_end(*example.pitchfork_case())
    mu_fold, b_fold = example.fit_branch_end(*example.fold_case())
    limit = example.axisymmetric_stability_limit([2.8e4, 2.7e4])

    assert abs(b_pitchfork - 1.0) < 0.05
    np.testing.assert_allclose(mu_pitchfork, limit, rtol=5.0e-3)
    assert 0.45 < b_fold < 0.65
    assert mu_fold > mu_pitchfork
