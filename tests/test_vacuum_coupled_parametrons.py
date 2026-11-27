# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the vacuum-coupled parametron example."""

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import vacuum_coupled_parametrons as vc


@pytest.fixture(scope="module")
def context():
    return vc.context_of()


def test_the_current_elements_give_the_field_of_the_current_loops():
    wout = vmecpp.run(vc.fixed_input(2.0e4), verbose=False).wout
    positions, elements = vc.current_elements(wout)
    ring, ring_elements = vc.current_elements(wout, nv=1)
    radii = np.hypot(ring[:, 0], ring[:, 1])
    currents = ring_elements[:, 1] / (2.0 * np.pi * radii)
    probe = np.array([[1.5, 0.0, -3.6], [2.5, 0.0, -4.6], [0.6, 0.0, -5.6]])
    field = vc.biot_savart(probe, positions, elements)
    b_r, b_z = vc.loop_field(radii, ring[:, 2], currents, probe[:, 0], probe[:, 2])
    np.testing.assert_allclose(field[:, [0, 2]], np.stack([b_r, b_z], -1), rtol=1e-4)


def test_the_coils_hold_the_boundary_of_the_case():
    wout = vc.solve(vc.free_input(2.0e4), vc.field_of(vc.coils())).wout
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    rmnc, zmns = np.asarray(wout.rmnc)[:, -1], np.asarray(wout.zmns)[:, -1]
    shape = [rmnc[(xm == 0) & (xn == 0)][0], rmnc[(xm == 1) & (xn == 0)][0]]
    shape.append(zmns[(xm == 1) & (xn == 0)][0])
    np.testing.assert_allclose(shape, [1.5, 1.0, 1.6], rtol=0.03)


@pytest.mark.slow
def test_a_pair_of_neighbours_inverts_a_target_that_keeps_its_state(context):
    for x in (1, -1):
        assert vc.inverter_and_latch(context, x) == (-x, -x)


@pytest.mark.slow
def test_the_neighbours_above_and_below_write_their_nand(context):
    rows = list(itertools.product((1, -1), repeat=2))
    with ProcessPoolExecutor() as pool:
        states = list(
            pool.map(vc.nand, [context] * len(rows), *zip(*rows, strict=True))
        )
    assert states == [-1 if x > 0 and y > 0 else 1 for x, y in rows]
