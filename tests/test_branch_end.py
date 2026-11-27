# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the example examples/branch_end.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import branch_end as example  # type: ignore # noqa: E402


def test_the_fit_recovers_the_end_and_the_exponent_of_both_laws() -> None:
    lam = np.linspace(1.0e-6, 1.0e-5, 6)
    for b in (1.0, 0.5):
        mu = 2.5e4 + 600.0 * (lam / 1.0e-5) ** (1.0 / b) * (1.0 + 2.0e4 * lam)
        mu_end, fitted = example.fit_branch_end(mu, lam)
        np.testing.assert_allclose(mu_end, 2.5e4, rtol=1.0e-9)
        np.testing.assert_allclose(fitted, b, rtol=1.0e-6)


@pytest.mark.slow
def test_the_helical_core_ends_at_a_pitchfork_and_with_a_ripple_at_a_fold() -> None:
    """The critical eigenvalue vanishes linearly without the ripple, and as a square
    root at a higher pres_scale with it."""
    mu_pitchfork, b_pitchfork = example.fit_branch_end(*example.pitchfork_case())
    mu_fold, b_fold = example.fit_branch_end(*example.fold_case())
    limit = example.axisymmetric_stability_limit([2.6e4, 2.5e4])

    assert abs(b_pitchfork - 1.0) < 0.05
    np.testing.assert_allclose(mu_pitchfork, limit, rtol=1.0e-3)
    assert 0.45 < b_fold < 0.6
    assert mu_fold > mu_pitchfork
