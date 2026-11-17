# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Knots wound from identical circular coils: the coil field against Biot-Savart, the
bounce integrals against quadrature, the residuals of the wells on an analytic surface,
the trefoil found against the knotatron's curve, the wout of its interior and W7-X,
and the knot of plane periods against the wout of its interior."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import circular_coil_knot as ck
from island_interior import axis_curve

# of the trefoil on the surface of its search, 15 cm from its axis
RIPPLE, CLOSURE = 0.04, 0.3


@pytest.fixture(scope="module")
def trefoil():
    design = ck.DESIGNS["trefoil"]
    k = ck.interior(design, 0.15)
    k["traced"] = ck.measures(design, 0.15)
    k["wout"] = vmecpp.run(k["input"], max_threads=1, verbose=False).wout
    return k


@pytest.fixture(scope="module")
def star():
    design = ck.DESIGNS["star"]
    reach, mpol, ntor, planes, periods = ck.SURFACES["star"]
    k = ck.interior(design, reach, mpol, ntor, planes)
    k["traced"] = ck.measures(design, reach, periods)
    k["wout"] = vmecpp.run(k["input"], max_threads=1, verbose=False).wout
    return k


def test_coil_field_against_biot_savart():
    axes = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, -1.0], [-3.0, 1.0, 0.5]])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    centres = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.3], [-0.5, 0.6, 0.1]])
    radius, current = 0.7, 2.0e5
    loops = ck.Loops(centres, axes, radius, current)
    first = np.cross(axes, [0.3, 0.5, 0.8])
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(axes, first)
    # the trapezoidal rule around each coil, the current turning about its axis
    angle = 2.0 * np.pi * np.arange(4096) / 4096
    cos, sin = np.cos(angle)[None, :, None], np.sin(angle)[None, :, None]
    wire = centres[:, None] + radius * (cos * first[:, None] + sin * second[:, None])
    step = radius * (cos * second[:, None] - sin * first[:, None]) * (angle[1])
    # points on the axis of a coil, in its plane and elsewhere
    points = np.array(
        [[0.0, 0.0, 0.25], [0.2, 0.1, 0.0], [1.3, -0.4, 0.2], [-0.1, 0.2, -0.9]]
    )
    offset = points[:, None, None] - wire[None]
    expected = (
        ck.MU0
        * current
        / (4.0 * np.pi)
        * (
            np.cross(step[None], offset)
            / np.linalg.norm(offset, axis=-1, keepdims=True) ** 3
        ).sum(axis=(1, 2))
    )
    np.testing.assert_allclose(loops.b(points), expected, rtol=1e-12, atol=1e-15)


def test_axis_closes_after_the_potential_of_a_period():
    # Ampere's law around the coils of a period: mu0 times their current
    design = ck.DESIGNS["knotatron"]
    field = design.loops()

    def miss(steps):
        r_axis, points, _ = ck.magnetic_axis(
            field, design.period_potential, steps, 3.0, 0.4
        )
        end = points[-1]
        np.testing.assert_allclose(
            end[0] / np.hypot(end[0], end[1]), np.cos(design.period), atol=1e-6
        )
        return np.hypot(np.hypot(end[0], end[1]) - r_axis, end[2])

    # the RK4 steps are of fourth order
    coarse, fine = miss(96), miss(192)
    assert fine < 2e-7
    assert 12.0 < coarse / fine < 20.0


def test_bounce_integrals_against_quadrature():
    # wells of B = 1 - 0.1 cos(zeta), whole in three of four periods
    periods, steps = 4, 400
    zeta = 2.0 * np.pi * np.arange(periods * steps + 1) / steps
    strength = 1.0 - 0.1 * np.cos(zeta)
    weight = np.full(zeta.size, 2.0 * np.pi / steps)
    mean = np.sum(weight * strength) / np.sum(weight)

    def eta(z):
        return (1.0 - 0.1 * np.cos(z)) / mean

    grid, wells = ck.bounce_integrals(strength, 0.3 + np.sin(zeta), weight, 5)
    for level, (_, _, h, i) in zip(grid, wells, strict=True):
        edge = np.arccos((1.0 - level * mean) / 0.1)
        expected_h = quad(
            lambda z, level=level: (
                np.sqrt(level - eta(z))
                * (4.0 / eta(z) - 1.0 / level)
                * (0.3 + np.sin(z))
            ),
            -edge,
            edge,
        )[0]
        expected_i = quad(
            lambda z, level=level: np.sqrt(1.0 - eta(z) / level), -edge, edge
        )[0]
        np.testing.assert_allclose(h, np.full(3, expected_h), rtol=2e-4)
        np.testing.assert_allclose(i, np.full(3, expected_i), rtol=2e-4)
    # a drift odd about the bottom of every well averages out
    _, wells = ck.bounce_integrals(strength, np.sin(zeta), weight, 5)
    for _, _, h, _ in wells:
        np.testing.assert_allclose(h, 0.0, atol=1e-13)
    ripple = ck.effective_ripple(
        strength, np.ones_like(zeta), np.sin(zeta), weight, 1.0
    )
    assert ripple < 1e-15


def test_wells_of_an_analytic_surface():
    # B = 1 + 0.1 cos(zeta) + 0.03 cos(theta) along the field line theta = iota zeta
    periods, steps, iota = 1000, 64, 0.2 * (np.sqrt(5.0) - 1.0) / 2.0
    zeta = 2.0 * np.pi * np.arange(periods * steps + 1) / steps
    strength = 1.0 + 0.1 * np.cos(zeta) + 0.03 * np.cos(iota * zeta)
    # its contours close poloidally between the highest valley and the lowest ridge
    np.testing.assert_allclose(
        ck.poloidal_closure(strength, steps, iota), (1.07 - 0.93) / 0.26, atol=1e-3
    )
    # each line has its ends at 1.1 + 0.03 cos(theta), a turn of the line apart, and
    # its minimum at 0.9 + 0.03 cos(theta)
    found = {k: np.mean(v**2) for k, v in ck.well_residuals(strength, steps).items()}
    turn, scale = 2.0 * np.pi * iota, (0.03 / 0.26) ** 2
    np.testing.assert_allclose(
        found["ends"], scale * (1.0 + (1.0 + np.cos(turn)) / 4.0), rtol=1e-2
    )
    np.testing.assert_allclose(found["lowest"], 1.5 * scale, rtol=1e-2)
    np.testing.assert_allclose(
        found["wells"], scale * (1.0 - np.cos(turn)) / 4.0, rtol=1e-2
    )
    # a field strength of zeta alone is quasi-isodynamic
    plain = 1.0 + 0.1 * np.cos(zeta)
    np.testing.assert_allclose(ck.poloidal_closure(plain, steps, iota), 1.0, atol=1e-12)
    for value in ck.well_residuals(plain, steps).values():
        np.testing.assert_allclose(value, 0.0, atol=1e-12)


def test_interior_against_tracing(trefoil):
    wout = trefoil["wout"]
    iota = np.asarray(wout.iotaf)
    distance = np.hypot(*(axis_curve(wout, trefoil["planes"]) - trefoil["axis"]).T)
    assert distance.max() < 5e-4
    assert abs(abs(iota[0]) - abs(trefoil["traced"]["iota"])) < 2e-3
    assert abs(iota[-1] - trefoil["iota_surface"]) < 2e-4


def test_effective_ripple_of_the_wout_against_the_coils(trefoil):
    wout = trefoil["wout"]
    ripple = ck.wout_ripple(wout, int(wout.ns) - 1)
    np.testing.assert_allclose(ripple, trefoil["traced"]["ripple"], rtol=0.03)


def test_trefoil_against_the_curve_of_the_knotatron(trefoil):
    traced = trefoil["traced"]
    plain = ck.measures(ck.DESIGNS["knotatron"], 0.15)
    assert plain["ripple"] > 1.0
    assert traced["ripple"] < RIPPLE
    assert traced["closure"] > CLOSURE
    # within the bounds of its search, from coils of radius 2/3 m
    scale = ck.STARTS["trefoil"].radius
    assert traced["clearance"] > 0.99 * ck.CLEARANCE * scale
    assert traced["room"] > 0.99 * ck.ROOM * scale
    assert traced["mirror"] < 1.01 * ck.MIRROR


@pytest.mark.slow
def test_star_against_tracing(star):
    wout = star["wout"]
    iota = np.asarray(wout.iotaf)
    distance = np.hypot(*(axis_curve(wout, star["planes"]) - star["axis"]).T)
    assert distance.max() < 5e-4
    assert abs(abs(iota[0]) - abs(star["traced"]["iota"])) < 2e-3
    assert abs(iota[-1] - star["iota_surface"]) < 2e-4


@pytest.mark.slow
def test_star_is_quasi_isodynamic_toward_its_axis(star):
    # its field periods are plane curves
    design = ck.DESIGNS["star"]
    outer = star["traced"]
    reach, *_, periods = ck.SURFACES["star"]
    inner = ck.measures(design, reach / 3.0, periods)
    assert inner["ripple"] < 0.4 * outer["ripple"]
    assert inner["closure"] > 0.8
    assert outer["closure"] > 0.6
    wout = star["wout"]
    ns = int(wout.ns)
    edge = ck.wout_ripple(wout, ns - 1, periods)
    np.testing.assert_allclose(edge, outer["ripple"], rtol=0.05)
    assert ck.wout_ripple(wout, ns // 4, periods) < 0.5 * edge


@pytest.mark.slow
def test_search_lowers_the_effective_ripple():
    start = ck.STARTS["trefoil"]
    found = ck.search(start, evaluations=3, periods=12)
    before = ck.measures(start, ck.OFFSET * start.radius)["ripple"]
    after = ck.measures(found, ck.OFFSET * start.radius)["ripple"]
    assert after < 0.7 * before


@pytest.mark.slow
def test_trefoil_against_w7x():
    # the standard configuration, whose effective ripple is below 1% in the core
    offsets = (0.05, 0.1, 0.15)
    yardstick = ck.w7x(offsets)
    assert all(0.004 < k["ripple"] < 0.008 for k in yardstick)
    design = ck.DESIGNS["trefoil"]
    for offset, k in zip(offsets, yardstick, strict=True):
        found = ck.measures(design, offset)
        assert found["closure"] > k["closure"]
        assert found["wells"] < k["wells"]
