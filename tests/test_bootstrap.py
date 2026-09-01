# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the Redl bootstrap evaluation and the consistency loop."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp._bootstrap import (
    _achieved_j_dot_b,
    _enclosed_current,
    _redl_geometry,
)

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

MU_0 = 4.0e-7 * np.pi


@pytest.fixture(scope="module")
def cth_output() -> vmecpp.VmecOutput:
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    return vmecpp.run(vmec_input, verbose=False, max_threads=1)


def test_geometry_reproduces_wout_averages(cth_output: vmecpp.VmecOutput):
    geometry = _redl_geometry(cth_output)
    wout = cth_output.wout

    # <B^2> against the wout's own flux-surface average.
    bdotb = np.asarray(wout.bdotb)[1:]
    np.testing.assert_allclose(geometry.fsa_B2, bdotb, rtol=1e-2)

    # buco is the Boozer current flux function: extrapolated to the edge it
    # carries the net toroidal current reported as ctor.
    edge = geometry.I[-1] + 0.5 * (geometry.I[-1] - geometry.I[-2])
    assert wout.signgs * 2.0 * np.pi * edge / MU_0 == pytest.approx(wout.ctor, rel=1e-3)


def test_identity_reproduces_jdotb(cth_output: vmecpp.VmecOutput):
    geometry = _redl_geometry(cth_output)
    achieved = _achieved_j_dot_b(geometry)

    ns = cth_output.wout.ns
    s_full = np.arange(ns) / (ns - 1)
    jdotb = np.interp(geometry.s, s_full, np.asarray(cth_output.wout.jdotb))
    mask = np.abs(jdotb) > 0.05 * np.max(np.abs(jdotb))
    ratio = achieved[mask] / jdotb[mask]
    assert 0.98 < float(np.median(ratio)) < 1.02


def test_enclosed_current_round_trip(cth_output: vmecpp.VmecOutput):
    geometry = _redl_geometry(cth_output)
    # Integrating the <J.B> the equilibrium carries must recover its own
    # current flux function.
    achieved = _achieved_j_dot_b(geometry)
    current, _ = _enclosed_current(geometry, achieved)
    scale = np.max(np.abs(geometry.I))
    np.testing.assert_allclose(
        current[1:-1] / scale, geometry.I[1:-1] / scale, atol=2e-2
    )


def test_redl_profile_is_finite_and_nonzero(cth_output: vmecpp.VmecOutput):
    s, j_dot_b = vmecpp.redl_bootstrap_current(
        cth_output,
        ne=lambda s: 2.0e19 * (1.0 - 0.8 * s),
        Te=lambda s: 1000.0 * (1.0 - 0.8 * s),
        Ti=lambda s: 800.0 * (1.0 - 0.8 * s),
    )
    assert s.shape == j_dot_b.shape
    assert np.all(np.isfinite(j_dot_b))
    assert np.max(np.abs(j_dot_b[1:-1])) > 0.0


def test_bootstrap_consistent_converges():
    # cth_like_fixed_bdy has vacuum rotational transform, so the first pass is
    # well-conditioned whatever current it carries.
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    result = vmecpp.bootstrap_consistent(
        vmec_input,
        ne=lambda s: 1.0e20 * (1.0 - 0.8 * s),
        Te=lambda s: 1500.0 * (1.0 - 0.8 * s),
        Ti=lambda s: 1500.0 * (1.0 - 0.8 * s),
        relaxation=0.5,
        tolerance=0.05,
        max_iterations=10,
        verbose=False,
        max_threads=1,
    )
    assert result.residuals[-1] <= 0.05
    assert result.residuals[-1] < result.residuals[0]
    assert result.iterations <= 10
    # The converged equilibrium carries a net current driven by the profiles.
    assert abs(result.output.wout.ctor) > 0.0
    # The current it carries is the Redl current: re-derive both from the
    # final output alone.
    geometry = _redl_geometry(result.output)
    achieved = _achieved_j_dot_b(geometry)
    scale = np.max(np.abs(result.j_dot_B))
    assert float(np.max(np.abs(achieved[1:-1] - result.j_dot_B[1:-1])) / scale) <= 0.05
