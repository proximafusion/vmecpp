# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Free-boundary balance of the CTH-like case against certified enclosures.

`return_vacuum_field` exports NESTOR's boundary vacuum field. The jump of
p + B^2/2 across the boundary at each point of NESTOR's grid is enclosed by
Stellarocq's point certificates (theories/BoxCell.v, check_bpcert_correct)
for every vacuum pressure on the segment between NESTOR's and the one BIEST's
virtual casing gives: CEILING bounds it over the whole grid and FLOOR bounds
NESTOR's jump from below where it is largest. The certified jump falls under
radial refinement when each resolution's ceiling lies below the previous
floor.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from free_boundary_jump import (
    TEST_DATA_DIR,
    boundary_grid,
    cth_like,
    jump,
    jump_between,
    vacuum_interpolant,
)

# Stellarocq's covering of the whole boundary (BoxCell.bt_surface): 512 x 128
# cells tiling a field period, each bounded by two Taylor steps, give
# |jump| <= COVERED at every point of the boundary for every vacuum pressure on
# the segment, the vacuum side being the trigonometric interpolant of each
# answer's grid. Between grid points the bound is set by that interpolant, whose
# angular resolution radial refinement leaves as it is.
COVERED = {25: 8.46407980031941079e-03, 385: 1.05019452171668456e-02}

# ns: (floor on NESTOR's jump at its largest, ceiling over both vacuum answers)
CERTIFIED = {
    25: (7.55646643843636183e-04, 7.56015749652367598e-04),
    49: (7.20184858729281600e-04, 7.20536656831872015e-04),
    97: (6.43324651921008410e-04, 6.43638935017077632e-04),
    193: (5.63679866925871674e-04, 5.63955298772989412e-04),
    385: (4.96479136101348156e-04, 4.96721841363433015e-04),
}


@pytest.fixture(scope="module")
def runs():
    return {ns: cth_like(ns) for ns in (25, 49, 97)}


def test_vacuum_field_exported(runs):
    """The exported field is NESTOR's: its components give the vacuum pressure VMEC++
    balanced, on the grid of the boundary it returns."""
    out = runs[25]
    fb = out.threed1_free_boundary
    assert fb is not None
    vi_nzeta = fb.bsqvacf.shape[0]
    assert fb.brv.shape == fb.bphiv.shape == fb.bzv.shape == fb.bsqvacf.shape
    assert fb.rb.shape == fb.zb.shape == fb.bsqvacf.shape
    b2 = np.asarray(fb.brv) ** 2 + np.asarray(fb.bphiv) ** 2 + np.asarray(fb.bzv) ** 2
    np.testing.assert_allclose(0.5 * b2, fb.bsqvacf, rtol=1e-12)
    theta, zeta = boundary_grid(fb)
    assert len(zeta) == vi_nzeta
    assert len(theta) == fb.bsqvacf.shape[1]


def test_vacuum_field_off_by_default():
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    assert vi.return_vacuum_field is False


@pytest.mark.parametrize("ns", [25, 49, 97])
def test_jump_within_certificate(runs, ns):
    """The largest jump on NESTOR's grid lies in the certified interval, and VMEC++'s
    own extrapolated edge pressure agrees with the certified one."""
    j = jump(runs[ns])
    floor, ceiling = CERTIFIED[ns]
    big = float(np.abs(j["reconstruction"]).max())
    assert floor <= big <= ceiling
    assert float(np.abs(j["vmecpp"]).max()) <= ceiling + 1e-5
    assert float(np.abs(j["edge_difference"]).max()) < 1e-5


def test_certified_jump_falls():
    """Each resolution's ceiling lies below the previous resolution's floor."""
    ns = sorted(CERTIFIED)
    for a, b in pairwise(ns):
        assert CERTIFIED[b][1] < CERTIFIED[a][0]


def test_covering_holds_the_grid_certificates():
    """The bound over the whole boundary is at least the jump certified at the largest
    grid point of the same run."""
    for ns, bound in COVERED.items():
        assert CERTIFIED[ns][0] <= bound


def test_jump_falls_under_refinement(runs):
    big = [float(np.abs(jump(runs[ns])["reconstruction"]).max()) for ns in (25, 49, 97)]
    assert big[0] > big[1] > big[2]


def test_vacuum_interpolant_reproduces_the_grid(runs):
    fb = runs[25].threed1_free_boundary
    theta, zeta = boundary_grid(fb)
    at = vacuum_interpolant(fb, int(runs[25].wout.nfp))
    np.testing.assert_allclose(at(theta, zeta), fb.bsqvacf, rtol=1e-10, atol=1e-14)


def test_jump_between_grid_points_within_covering(runs):
    """Off NESTOR's grid, on a grid four times finer in each angle, the jump stays
    inside the bound the covering proves for the whole boundary."""
    big = float(np.abs(jump_between(runs[25], refine=4)).max())
    assert big <= COVERED[25], big
