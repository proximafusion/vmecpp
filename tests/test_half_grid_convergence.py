# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Second-order convergence of the half-grid rule against certified enclosures.

`examples/half_grid_convergence.py` solves the half-grid rule's collocated system
for three manufactured mappings on the annulus 1/4 < s < 1: a three-dimensional
one, the same with non-stellarator-symmetric content, and that at five times the
pressure. The distance max |x_h - x*| of the discrete solution x_h from the
mapping's own coefficients x* is the discretization error.

Stellarocq (https://github.com/CharlesCNorton/stellarocq, gen/mms_colloc.py)
establishes each discrete solution by the interval Newton test on the collocated
system (Colloc.colloc_correct), with every datum of the problem carried at 120 bits:
exactly one zero lies in a box of radius three units of a 58-bit mantissa in each
unknown, around a centre refined by Newton steps on the system's outputs read at
128 bits (Newton.centre_tab, Wide.v). That box gives the error to the digits below.
The enclosures fall by 3.81, 3.92 and 3.97 per doubling of ns for the
three-dimensional mapping and by 3.73 and 3.87 for the others, the second order
HalfGrid.lax_second_order asks of the discretization, and the pressure, which enters
the radial force only as a flux function the rule and its source share, leaves them
identical. The floating-point solve rounds the problem's data to binary64, which
moves its discrete solution by up to 4e-15, and is held to the enclosures within
1e-14. The system is the half-grid residual at collocation points, not VMEC's own
discrete equations, which come from the variation of the energy with spectral
condensation and a constraint on the m = 1 modes, so its errors measure the
half-grid rule.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import half_grid_convergence as hgc

# case: {ns: (lower, upper)} enclosing max |x_h - x*|
CERTIFIED = {
    "3d": {
        9: (2.7343293682873708e-06, 2.734329368287412e-06),
        17: (7.169121707716152e-07, 7.16912170771656e-07),
        33: (1.8303258574207905e-07, 1.8303258574211974e-07),
        65: (4.607481933443167e-08, 4.607481933447233e-08),
    },
    "asymmetric": {
        9: (2.6907305704322986e-05, 2.6907305704323006e-05),
        17: (7.206670532902957e-06, 7.206670532902977e-06),
        33: (1.8626687770335693e-06, 1.8626687770335896e-06),
    },
    "high-beta": {
        9: (2.6907305704322986e-05, 2.6907305704323006e-05),
        17: (7.206670532902957e-06, 7.206670532902977e-06),
        33: (1.8626687770335693e-06, 1.8626687770335896e-06),
    },
}
ABS = 1e-14

# the floating-point solves this file repeats, the fine grids being slow in Python
PAIRS = [(name, ns) for name, table in CERTIFIED.items() for ns in table if ns <= 17]


@pytest.fixture(scope="module")
def errors():
    cases = hgc.cases()
    out = {}
    for name, ns in PAIRS:
        pb = hgc.Problem(cases[name], ns)
        x, _ = pb.solve()
        out[name, ns] = float(np.abs(x - pb.xstar).max())
    return out


@pytest.mark.parametrize(("name", "ns"), PAIRS)
def test_error_within_the_certificate(errors, name, ns):
    lo, hi = CERTIFIED[name][ns]
    err = errors[name, ns]
    assert lo - ABS <= err <= hi + ABS, (name, ns, err, lo, hi)


@pytest.mark.parametrize("name", list(CERTIFIED))
def test_certified_error_falls_fourfold(name):
    """Between consecutive resolutions the ratio of the errors, over every value the two
    enclosures allow, lies between 3.5 and 4.5."""
    table = CERTIFIED[name]
    ns = sorted(table)
    for a, b in pairwise(ns):
        assert table[a][0] / table[b][1] > 3.5, (name, a, b)
        assert table[a][1] / table[b][0] < 4.5, (name, a, b)


def test_pressure_leaves_the_error_unchanged():
    for ns in CERTIFIED["high-beta"]:
        assert CERTIFIED["asymmetric"][ns] == CERTIFIED["high-beta"][ns], ns
