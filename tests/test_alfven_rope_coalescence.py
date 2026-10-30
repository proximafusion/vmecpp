# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The energy of the coalescence example matches a thin ring of the Lundquist field on a
round torus, and the stored merged strand balances pressure."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j0, j1, jv

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import alfven_rope  # type: ignore
import alfven_rope_coalescence  # type: ignore

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def torus():
    merged = alfven_rope_coalescence.Merged(angle=5.0)
    x = np.zeros(len(merged.modes))
    x[0] = merged.radius
    rb, zb = alfven_rope.boundary(merged, x)
    run = vmecpp.run(
        alfven_rope.strand_input(merged, rb, zb), max_threads=1, verbose=False
    )
    return merged, run.wout


def test_thin_ring(torus):
    merged, wout = torus
    *strand, current = alfven_rope.currents(wout, 1, 24, 144)
    boundary = alfven_rope_coalescence.surface(wout)
    # the torus is axisymmetric, so one plane of targets stands for all 144
    energy = alfven_rope_coalescence.energy(strand, boundary, [np.eye(3)], 144, chunk=8)
    ka = merged.ka
    inductance = (
        alfven_rope.MU0
        * alfven_rope.R0
        * (
            np.log(8.0 * alfven_rope.R0 / merged.radius)
            - 1.0
            - 1.5 * j0(ka) * jv(2, ka) / j1(ka) ** 2
        )
    )
    np.testing.assert_allclose(current, merged.current, rtol=1e-6)
    np.testing.assert_allclose(energy, 0.5 * inductance * current**2, rtol=1e-3)


def test_merged_pressure_balance():
    data = json.loads((DATA / "alfven_rope_merged.json").read_text())
    merged = alfven_rope_coalescence.Merged(
        angle=data["angle"], spacing=data["spacing"]
    )
    _, free = alfven_rope.solve(merged, np.array(data["x"]), data["vertical"])
    modes = alfven_rope.projection(merged.m_shape, merged.n_shape) @ alfven_rope.jump(
        free
    )
    np.testing.assert_allclose(modes, 0.0, atol=1e-6)
