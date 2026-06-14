# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ exposes an analytic boundary gradient to SIMSOPT.

The VmecEnergy Optimizable's analytic dJ (the implicit-function adjoint) matches finite
differences of its objective, and computing it is much cheaper than the conventional
finite-difference boundary gradient (which re-solves the equilibrium per boundary degree
of freedom).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

pytest.importorskip("simsopt")

from simsopt_vmec_gradient import (
    VmecBoundaryProblem,
    gradient_cost,
    make_simsopt_optimizable,
)


def test_simsopt_optimizable_gradient_matches_fd():
    problem = VmecBoundaryProblem(ns=11)
    opt = make_simsopt_optimizable(problem)
    g = np.asarray(opt.dJ(), float)

    p0 = np.asarray(opt.local_full_x, float)
    h = 1e-5
    scale = max(np.linalg.norm(g), 1e-30)
    for j in (0, 2, 9):
        pp = p0.copy()
        pp[j] += h
        opt.local_full_x = pp
        jp = opt.J()
        pm = p0.copy()
        pm[j] -= h
        opt.local_full_x = pm
        jm = opt.J()
        opt.local_full_x = p0
        assert abs(g[j] - (jp - jm) / (2 * h)) < 1e-3 * scale


def test_adjoint_gradient_cheaper_than_finite_difference():
    analytic = gradient_cost(analytic=True)
    fd = gradient_cost(analytic=False)
    # Same gradient, far fewer force evaluations (advantage grows with #DOFs).
    rel = np.linalg.norm(analytic.gradient - fd.gradient) / np.linalg.norm(fd.gradient)
    assert rel < 1e-2
    assert analytic.force_evals < fd.force_evals
