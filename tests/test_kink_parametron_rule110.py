# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the kink parametron example: its certificate, its gate, adders of its gates
and Rule 110."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import kink_parametron_rule110 as kp


@pytest.fixture(scope="module")
def certified():
    return kp.certify()


def test_writes_over_either_core_take_the_sign_of_the_sum(certified):
    _, writes = certified
    for total, shifts in writes.items():
        np.testing.assert_array_equal(np.sign(shifts), np.sign(total))


def test_parametron_takes_the_majority_and_restores_its_level(certified):
    # One parametron through all inputs, each written over the core the last one held.
    level, _ = certified
    gate = kp.Parametron()
    for a, b, c in itertools.product((-1, 1), repeat=3):
        held = kp.majority(gate, level, a * level, b * level, c * level)
        np.testing.assert_allclose(held, np.sign(a + b + c) * level, rtol=1e-3)


@pytest.mark.slow
def test_adders(certified):
    level, _ = certified
    for a, b, carry in itertools.product((-1, 1), repeat=3):
        gates = [kp.Parametron() for _ in range(3)]
        s, out = kp.full_adder(gates, level, a * level, b * level, carry * level)
        total = sum(x > 0 for x in (a, b, carry))
        assert (int(s > 0), int(out > 0)) == (total % 2, total // 2)
    assert kp.add(level, 7, 9, 4) == 16


@pytest.mark.slow
def test_rule_110_on_parametrons(certified):
    level, _ = certified
    bits = (kp.evolve(level, kp.DE_BRUIJN, 1) > 0).astype(int)
    np.testing.assert_array_equal(bits[1:], kp.by_number(bits[:-1]))
