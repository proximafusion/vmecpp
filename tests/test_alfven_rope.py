# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The matched rope of examples/alfven_rope.py balances pressure in every shape mode, is
a Taylor state and keeps the point-vortex pitch within its finite size."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import alfven_rope  # type: ignore

DATA = Path(__file__).resolve().parents[1] / "examples" / "data" / "alfven_rope.json"


@pytest.fixture(scope="module")
def matched():
    data = json.loads(DATA.read_text())
    rope = alfven_rope.Rope(angle=data["angle"], spacing=data["spacing"])
    fixed, free = alfven_rope.solve(rope, np.array(data["x"]), data["vertical"])
    return rope, fixed, free


def test_pressure_balance(matched):
    rope, fixed, free = matched
    modes = alfven_rope.projection(rope.m_shape, rope.n_shape) @ alfven_rope.jump(
        fixed, free
    )
    np.testing.assert_allclose(modes, 0.0, atol=1e-6)


def test_taylor_state(matched):
    rope, fixed, _ = matched
    jxb = fixed.jxbout
    mu = alfven_rope.MU0 * np.asarray(jxb.jdotb)[1:-1] / np.asarray(jxb.bdotb)[1:-1]
    np.testing.assert_allclose(mu, rope.ka / rope.radius, rtol=1e-3)


def test_point_vortex_pitch(matched):
    rope, fixed, _ = matched
    _, _, ratio = alfven_rope.pitch(rope, fixed.wout)
    assert abs(ratio - 1.0) < 0.1
