# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The external-optimizer benchmark reports comparable solver metrics."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from external_optimizers import (  # type: ignore  # noqa: E402
    reference_equilibrium,
    solve_newton_krylov,
    solve_vmecpp,
)

SOLOVEV = ROOT / "examples" / "data" / "solovev.json"
CTH_LIKE = ROOT / "examples" / "data" / "cth_like_fixed_bdy.json"


def test_native_metrics_match_external_metric_definitions():
    x_star, w_star = reference_equilibrium(CTH_LIKE, ns=11)
    x, result = solve_vmecpp(CTH_LIKE, ns=11)

    assert result.energy == w_star
    np.testing.assert_array_equal(x, x_star)
    assert result.residual_norm < 1e-8
    assert result.force_evals == result.outer_iters + 1
    assert result.outer_iters > 0


def test_newton_krylov_reports_outer_iterations():
    _, result = solve_newton_krylov(SOLOVEV, ns=11)
    assert result.outer_iters > 0
