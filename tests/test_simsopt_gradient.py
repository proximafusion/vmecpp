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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from simsopt_vmec_gradient import (  # type: ignore
    VmecBoundaryProblem,
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
