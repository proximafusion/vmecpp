# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the quasi-helical core example."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import quasi_helical_core as qh


def test_without_a_core_the_field_is_quasi_axisymmetric():
    output = qh.solve(qh.START, seed=0.0)[0]
    assert abs(qh.axis_shift(output)) < 1e-12
    assert qh.mirror(output) < 1e-12
    assert np.sum(qh.residuals(output, 0) ** 2) < 1e-6


@pytest.mark.slow
def test_the_core_is_closer_to_quasi_helical_than_to_quasi_axisymmetric():
    output = qh.solve(qh.START)[0]
    assert qh.axis_shift(output) > qh.CRITICAL
    helical, axisymmetric = (np.sum(qh.residuals(output, n) ** 2) for n in (1, 0))
    assert helical < axisymmetric / 3.0


@pytest.mark.slow
def test_shaping_lowers_the_residuals_of_the_core():
    start = np.sum(qh.residuals(qh.solve(qh.START)[0]) ** 2)
    core = qh.solve(qh.optimize(evaluations=30))[0]
    assert qh.axis_shift(core) > qh.CRITICAL
    assert np.sum(qh.residuals(core) ** 2) < start
