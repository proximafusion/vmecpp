# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ against the Taylor-state spheromak, whose boundary reaches the major axis,
from the input of examples/spheromak.py at mpol = 24 and ns = 17 then 33."""

import sys
from pathlib import Path

import numpy as np
import pytest

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import spheromak

# VMEC++ gives the axis +1.5e-3, psi 3.0e-3, the enclosed current 6.2e-4 for
# 0.1 < s < 0.9 and the total current +1.8e-2 against the Taylor state
TOL = {"axis": 2e-3, "psi": 4e-3, "current": 1e-3, "total_current": 2.5e-2}


@pytest.fixture(scope="module")
def deviations():
    profiles = spheromak.taylor_profiles()
    vmec_input = spheromak.spheromak_input(profiles)
    wout = vmecpp.run(vmec_input, max_threads=1, verbose=False).wout
    return spheromak.deviations(wout, profiles)


def test_plasma_stays_at_positive_major_radius(deviations):
    assert deviations["min_r"] >= 0.0


def test_axis(deviations):
    assert abs(deviations["axis"]) < TOL["axis"]


def test_flux_surfaces(deviations):
    assert deviations["psi"].max() < TOL["psi"]


def test_enclosed_current(deviations):
    interior = (deviations["s_half"] > 0.1) & (deviations["s_half"] < 0.9)
    assert np.abs(deviations["current"][interior]).max() < TOL["current"]
    assert abs(deviations["total_current"]) < TOL["total_current"]
