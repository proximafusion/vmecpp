# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The ring of examples/free_boundary_root.py balances pressure in the vertical field of
the thin ring; its vertical shift is stable at decay index 0.2, unstable at -0.2."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import free_boundary_root  # type: ignore

CURRENT = 250.0e3


class Serial:
    """The map of a process pool, run in this process."""

    @staticmethod
    def map(function, values):
        return [function(value) for value in values]


@pytest.fixture(scope="module", params=[-0.2, 0.2])
def matched(request):
    ring = free_boundary_root.ring_matching(CURRENT, request.param)
    x, parameters, _, jump = free_boundary_root.match(
        ring,
        np.zeros(len(ring.modes)),
        [free_boundary_root.balancing_field(CURRENT)],
        Serial(),
        lambda *_: None,
    )
    return request.param, ring, x, parameters[0], jump


def test_pressure_balance(matched):
    *_, jump = matched
    np.testing.assert_allclose(jump, 0.0, atol=1e-9)


def test_thin_ring_vertical_field(matched):
    _, _, _, vertical, _ = matched
    np.testing.assert_allclose(
        vertical, free_boundary_root.balancing_field(CURRENT), rtol=5e-3
    )


def test_vertical_stability(matched):
    decay, ring, x, vertical, _ = matched
    ring = dataclasses.replace(
        ring, field=free_boundary_root.DecayField(decay, vertical), held=()
    )
    values, modes = free_boundary_root.stiffness(ring, x, Serial())
    assert np.sign(values[modes.index((1, 0, 1))]) == -np.sign(decay)
