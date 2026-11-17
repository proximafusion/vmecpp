# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ on knotted stellarators, the knotatron of Hudson, Startsev and Feibush (2014)
against the paper and the interiors of it and of two other trefoils against tracing."""

import sys
from pathlib import Path

import numpy as np
import pytest

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import knotted_stellarator as ks
from island_interior import axis_curve


@pytest.fixture(scope="module")
def knotatron():
    k = ks.knotatron()
    k["runs"] = {
        p0: vmecpp.run(
            k["input"].model_copy(update={"pres_scale": p0}),
            max_threads=1,
            verbose=False,
        ).wout
        for p0 in (0.0, 500.0, 1000.0)
    }
    return k


def test_coils_and_axis_of_the_paper(knotatron):
    # Hudson et al. report a smallest coil-coil distance of 32 cm and iota 0.261
    assert abs(knotatron["spacing"] - 0.32) < 0.005
    assert abs(abs(knotatron["iota_axis"]) - 0.261) < 2e-3


def test_interior_against_tracing(knotatron):
    wout = knotatron["runs"][0.0]
    iota = np.asarray(wout.iotaf)
    distance = np.hypot(*(axis_curve(wout, knotatron["planes"]) - knotatron["axis"]).T)
    assert distance.max() < 5e-4
    assert abs(iota[0] - knotatron["iota_axis"]) < 1e-3
    assert abs(iota[-1] - knotatron["iota_surface"]) < 2e-4


@pytest.mark.parametrize(
    "name", ["shaped_trefoil", pytest.param("trefoil_3_2", marks=pytest.mark.slow)]
)
def test_other_knots_against_tracing(name):
    k = ks.knotatron(name)
    wout = vmecpp.run(k["input"], max_threads=1, verbose=False).wout
    iota = np.asarray(wout.iotaf)
    distance = np.hypot(*(axis_curve(wout, k["planes"]) - k["axis"]).T)
    assert distance.max() < 5e-4
    assert abs(iota[0] - k["iota_axis"]) < 1e-3
    assert abs(iota[-1] - k["iota_surface"]) < 2e-4


def test_axis_moves_in_proportion_to_beta(knotatron):
    planes = knotatron["planes"]
    vacuum = axis_curve(knotatron["runs"][0.0], planes)
    shift = {
        p0: np.hypot(*(axis_curve(knotatron["runs"][p0], planes) - vacuum).T).max()
        for p0 in (500.0, 1000.0)
    }
    assert shift[500.0] > 1e-3
    assert abs(shift[1000.0] / shift[500.0] - 2.0) < 0.2
