# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ with lrfp on Lundquist's force-free flux rope through field reversal, against
the Bessel field of the straight rope at aspect ratio 30."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j0, j1

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import lundquist_flux_rope as lq


@pytest.fixture(scope="module")
def rope():
    output = vmecpp.run(
        lq.lundquist_input(30.0, lq.NS[:4]), max_threads=1, verbose=False
    )
    return lq.lundquist(output)


def test_current_is_k_b_over_mu0(rope):
    mu, _, _ = rope
    assert np.abs(mu - lq.KA / lq.MINOR).max() < 6e-3


def test_field_reverses_where_the_bessel_field_does(rope):
    _, s_reversal, f = rope
    assert abs(s_reversal - 1.0 / (1.0 - j0(lq.KA))) < 3e-4
    assert abs(f - lq.KA * j0(lq.KA) / (2.0 * j1(lq.KA))) < 2e-3
