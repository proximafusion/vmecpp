# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ on the Taylor state in a flux conserver about the trefoil knot, against a
uniform mu0 J.B / B^2 and the rotational transform of Mercier's formula."""

import sys
from pathlib import Path

import numpy as np
import pytest

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import knotted_taylor_state as kt


@pytest.fixture(scope="module")
def runs():
    return {
        k: vmecpp.run(kt.taylor_input(k), max_threads=1, verbose=False)
        for k in (0.0, 0.1)
    }


def test_current_is_k_b_over_mu0(runs):
    for k, output in runs.items():
        assert np.abs(kt.taylor_state(output) - k).max() < 1e-5


def test_boundary_iota_against_mercier(runs):
    for k, output in runs.items():
        expected = abs(kt.mercier(k)[0])
        assert abs(abs(output.wout.iotaf[-1]) - expected) < 2e-3
