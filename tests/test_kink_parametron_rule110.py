# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the kink parametron example: its gate, an adder of its gates, and Rule
110."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import kink_parametron_rule110 as kp


@pytest.fixture(scope="module")
def gate():
    return kp.certify()[0]


def full_adder(gate, a, b, carry):
    """The sum and the carry of three bits of +1 and -1, from three majority gates."""
    out = kp.majority(gate, a, b, carry)
    return kp.majority(gate, -out, carry, kp.majority(gate, a, b, -carry)), out


def add(gate, x, y, bits):
    """The sum of the unsigned integers x and y and the carry out of ``bits`` bits."""
    carry, total = -1, 0
    for k in range(bits):
        s, carry = full_adder(gate, 2 * (x >> k & 1) - 1, 2 * (y >> k & 1) - 1, carry)
        total |= int(s > 0) << k
    return total, int(carry > 0)


def test_gate_takes_the_majority(gate):
    assert gate == {-3: -1, -1: -1, 1: 1, 3: 1}
    for a, b, c in itertools.product((-1, 1), repeat=3):
        assert kp.majority(gate, a, b, c) == np.sign(a + b + c)
        assert kp.majority(gate, a, b, 1) == max(a, b)
        assert kp.majority(gate, a, b, -1) == min(a, b)
        assert kp.majority(gate, -a, -a, -a) == -a


def test_gate_holds_its_bit():
    # A hot restart without the input from the equilibrium the input wrote.
    for bit in (1, -1):
        written = kp.equilibrium(bit * kp.UNIT)
        held = kp.equilibrium(0.0, restart_from=written)
        assert np.sign(kp.axis_shift(held)) == bit
        assert abs(kp.axis_shift(held)) > 0.5 * abs(kp.axis_shift(written))


def test_adders(gate):
    for a, b, carry in itertools.product((-1, 1), repeat=3):
        s, out = full_adder(gate, a, b, carry)
        total = sum(x > 0 for x in (a, b, carry))
        assert (int(s > 0), int(out > 0)) == (total % 2, total // 2)
    assert add(gate, 1, 1, 1) == (0, 1)
    assert add(gate, 7, 9, 4) == (0, 1)


def test_rule_110_from_the_gate(gate):
    for left, centre, right in itertools.product((0, 1), repeat=3):
        new = kp.rule_110(gate, 2 * left - 1, 2 * centre - 1, 2 * right - 1)
        assert int(new > 0) == (110 >> (4 * left + 2 * centre + right)) & 1
    bits = (kp.evolve(gate, cells=48, generations=40) > 0).astype(int)
    np.testing.assert_array_equal(bits[1:], kp.by_number(bits[:-1]))
