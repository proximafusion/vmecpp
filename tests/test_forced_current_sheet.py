# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The forced current sheet of a rippled circular tokamak, as a regression.

`examples/forced_current_sheet.py` runs the circular tokamak of the test data with a
(1,1) boundary ripple and iota = 1/2 at s = 0.625, and reconstructs the radial force
residual of the converged state on that surface by VMEC's own half-grid rule. Its non-
resonant harmonics fall fourfold per doubling of ns, the order of the discretization;
its (2,1) harmonic does not fall, because a nested-surface field cannot balance the
force the ripple drives at the rational surface, whose ideal response is a current
sheet.

The numbers below are enclosures of the equispaced (2,1) harmonic on a 64 by 32 grid of
angles at the converged states of ns = 65, 129 and 257, established by Stellarocq's
Harmonic.dharm_correct (

https://github.com/CharlesCNorton/stellarocq,
gen/forced_sheet.py). Over every
state whose R, Z and lambda coefficients lie within a relative 1e-14 of the
converged ones the harmonic stays above CERTIFIED_FLOOR at each ns.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import forced_current_sheet as fcs

NS = (65, 129, 257)

# certified enclosures of the (2,1) harmonic at the converged state of each ns
CERTIFIED_21 = {
    65: (8.3487e-05, 8.4502e-05),
    129: (8.6756e-05, 9.0721e-05),
    257: (6.1943e-05, 7.7635e-05),
}
# the least lower end of the enclosures over the boxes of relative width 1e-14
CERTIFIED_FLOOR = 5.32e-05


@pytest.fixture(scope="module")
def harmonics():
    out = {}
    for ns in NS:
        wout = fcs.rippled_tokamak(ns)
        j = round((ns - 1) * fcs.S_RATIONAL)
        rs = fcs.node_residual(wout, j)
        out[ns] = {mn: fcs.harmonic(rs, *mn) for mn in ((2, 1), (2, 0), (1, 0))}
    return out


def test_non_resonant_harmonics_fall_fourfold(harmonics):
    for mn in ((2, 0), (1, 0)):
        for coarse, fine in pairwise(NS):
            ratio = abs(harmonics[coarse][mn] / harmonics[fine][mn])
            assert ratio > 3.5, (mn, coarse, fine, ratio)


def test_resonant_harmonic_holds_the_certified_floor(harmonics):
    for ns in NS:
        lo, hi = CERTIFIED_21[ns]
        value = harmonics[ns][(2, 1)]
        assert value >= CERTIFIED_FLOOR, (ns, value)
        assert 0.95 * lo <= value <= 1.05 * hi, (ns, value, lo, hi)


def test_resonant_harmonic_does_not_converge_away(harmonics):
    for coarse, fine in pairwise(NS):
        ratio = harmonics[coarse][(2, 1)] / harmonics[fine][(2, 1)]
        assert 0.5 < ratio < 2.0, (coarse, fine, ratio)
