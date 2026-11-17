# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A stellarator wound from identical circular coils: the coil field against Biot-
Savart, the tracing, the distances between the coils of a ring, the effective ripple
along the tree of the wells against quadrature and against levels taken one by one, the
residuals of the wells on an analytic surface, the Pfirsch-Schlueter current of a torus,
the magnetic well of an analytic line, the design found against its start, the limits
of its search and W7-X, and the wout of its interior against the tracing."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import circular_coil_stellarator as ck
from island_interior import axis_curve


@pytest.fixture(scope="module")
def found():
    reach, mpol, ntor, planes, periods = ck.SURFACE
    k = ck.interior(ck.FOUND, reach, mpol, ntor, planes)
    k["traced"] = ck.measures(ck.FOUND, reach, periods)
    k["iota_axis"] = ck.measures(ck.FOUND, 0.01, periods)["iota"]
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
    design = ck.START
    field = design.loops()
    middle = float(np.hypot(*design.points(0.0)[:2]))

    def miss(steps):
        r_axis, points, _ = ck.magnetic_axis(
            field, design.period_potential, steps, middle, 0.6 * design.radius
        )
        end = points[-1]
        np.testing.assert_allclose(
            end[0] / np.hypot(end[0], end[1]), np.cos(design.period), atol=1e-6
        )
        return np.hypot(np.hypot(end[0], end[1]) - r_axis, end[2])

    # the RK4 steps are of fourth order
    coarse, fine = miss(96), miss(192)
    assert fine < 2e-6
    assert 12.0 < coarse / fine < 20.0


def test_gaps_of_a_ring_of_coils():
    # twelve coils of radius a across a circle of radius R: neighbours are nearest on
    # the inside, 2 (R - a) sin(k pi / 12) apart for the k-th neighbour
    major, minor = 2.0, 0.3
    # the direction across the curve is that of its radial excursion, here 1 nm
    ring = ck.Design(
        p=1, q=4, per_period=3, radius=minor, r=(1.0e-9,), zs=(0.0,), r0=major
    )
    gaps = ring.gaps(ring.loops())
    expected = 2.0 * (major - minor) * np.sin(np.pi * np.array([1, 1, 2, 2, 3]) / 12.0)
    np.testing.assert_allclose(gaps, expected, rtol=2e-3)
    np.testing.assert_allclose(ring.loops().clearance(), expected[0], rtol=2e-3)


def test_effective_ripple_against_quadrature():
    # wells of B = 1 - 0.1 cos(zeta), whole in three of four periods
    periods, steps = 4, 400
    zeta = 2.0 * np.pi * np.arange(periods * steps + 1) / steps
    strength = 1.0 - 0.1 * np.cos(zeta)
    weight = np.full(zeta.size, 2.0 * np.pi / steps)
    size = np.ones_like(zeta)
    mean = np.sum(weight * strength) / np.sum(weight)

    def eta(z):
        return (1.0 - 0.1 * np.cos(z)) / mean

    def well(level):
        edge = np.arccos(np.clip((1.0 - level * mean) / 0.1, -1.0, 1.0))
        h = quad(
            lambda z: (
                np.sqrt(max(level - eta(z), 0.0))
                * (4.0 / eta(z) - 1.0 / level)
                * (0.3 + np.sin(z))
            ),
            -edge,
            edge,
            limit=200,
        )[0]
        i = quad(
            lambda z: np.sqrt(max(1.0 - eta(z) / level, 0.0)), -edge, edge, limit=200
        )[0]
        return h * h / (level * level * i)

    wells = 3.0 * quad(well, 0.9 / mean, 1.1 / mean, limit=200)[0]
    # with |grad psi| = 1 and R0 = 1, pi / (8 sqrt 2) over the length of the line
    expected = (np.pi / (8.0 * np.sqrt(2.0)) / (2.0 * np.pi * periods) * wells) ** (
        2.0 / 3.0
    )
    found = ck.effective_ripple(strength, size, 0.3 + np.sin(zeta), weight, 1.0)
    np.testing.assert_allclose(found, expected, rtol=1e-4)
    # a drift odd about the bottom of every well averages out
    assert ck.effective_ripple(strength, size, np.sin(zeta), weight, 1.0) < 1e-15


def test_wells_merge_along_their_tree():
    # ridges of two heights between ends at the higher: the two wells about a lower
    # ridge merge into one that a higher ridge ends
    periods, steps = 4, 400
    zeta = 2.0 * np.pi * np.arange(periods * steps + 1) / steps
    strength = 1.0 + 0.1 * np.cos(zeta) + 0.03 * np.cos(0.5 * zeta)
    drift = 0.3 + np.sin(zeta) + 0.2 * np.cos(0.5 * zeta)
    weight = np.full(zeta.size, 2.0 * np.pi / steps)
    eta = strength * np.sum(weight) / np.sum(weight * strength)
    inner = eta[1:-1]
    valleys = inner[(inner < eta[:-2]) & (inner < eta[2:])]
    ridges = inner[(inner > eta[:-2]) & (inner > eta[2:])]
    wells = sorted(ck.wells_of(eta))
    assert len(wells) == 6
    np.testing.assert_allclose([well[0] for well in wells[:4]], sorted(valleys))
    np.testing.assert_allclose([well[1] for well in wells[:4]], ridges.min())
    np.testing.assert_allclose([well[0] for well in wells[4:]], ridges.min())
    np.testing.assert_allclose([well[1] for well in wells[4:]], eta.max())
    # the levels one by one, each with the wells that the ends of the line do not cut
    total = 0.0
    grid = np.linspace(eta.min(), eta.max(), 4002)[1:-1]
    factor = weight[:, None] * (4.0 / eta[:, None] - 1.0 / grid) * drift[:, None]
    below = np.maximum(grid - eta[:, None], 0.0)
    root = 2.0 / 3.0 * np.diff(below**1.5, axis=0) / -np.diff(eta)[:, None]
    for k, level in enumerate(grid):
        wet = root[:, k] > 0.0
        starts = np.flatnonzero(wet[1:] & ~wet[:-1]) + 1
        stops = np.flatnonzero(~wet[1:] & wet[:-1]) + 1
        if wet[0]:
            stops = stops[1:]
        for a, b in zip(starts, stops, strict=False):
            h = np.sum(root[a:b, k] * 0.5 * (factor[a:b, k] + factor[a + 1 : b + 1, k]))
            i = np.sum(root[a:b, k] * weight[0]) / np.sqrt(level)
            total += h * h / (level * level * i)
    expected = (
        np.pi
        / (8.0 * np.sqrt(2.0))
        / (2.0 * np.pi * periods)
        * total
        * (grid[1] - grid[0])
    ) ** (2.0 / 3.0)
    found = ck.effective_ripple(strength, np.ones_like(zeta), drift, weight, 1.0)
    np.testing.assert_allclose(found, expected, rtol=1e-3)


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


def test_current_ratio_of_a_torus():
    # B = B0 (1 - eps cos(theta)) on the circle of radius eps R about the axis of a
    # torus of major radius R, psi = B0 r^2 / 2, along the line theta = iota phi
    iota, eps, major, b0 = 0.37, 0.01, 3.0, 2.0
    phi = 2.0 * np.pi * np.arange(200 * 64 + 1) / 64
    strength = b0 * (1.0 - eps * np.cos(iota * phi))
    size = np.full(phi.size, b0 * eps * major)
    drift = b0 * b0 * eps * np.sin(iota * phi) / strength
    found = ck.current_ratio(strength, size, drift, b0 * major * 2.0 * np.pi / 64)
    np.testing.assert_allclose(found, 2.0 / iota**2, rtol=1e-3)


def test_magnetic_well_of_an_analytic_line():
    # B = (1 + a cos(zeta)) (1 + d + e cos(theta)) along theta = iota zeta against the
    # axis 1 + a cos(zeta): the mean of 1 / B^2 over theta is (1 + d) / ((1 + d)^2 -
    # e^2)^(3/2) times that of the axis
    a, d, e, iota, steps = 0.3, -0.02, 0.05, 0.2 * (np.sqrt(5.0) - 1.0) / 2.0, 64
    zeta = 2.0 * np.pi * np.arange(120 * steps + 1) / steps
    axis = 1.0 + a * np.cos(zeta[: steps + 1])
    line = (1.0 + a * np.cos(zeta)) * (1.0 + d + e * np.cos(iota * zeta))
    expected = 1.0 - (1.0 + d) / ((1.0 + d) ** 2 - e**2) ** 1.5
    np.testing.assert_allclose(ck.magnetic_well(line, axis), expected, rtol=1e-3)


def test_interior_against_tracing(found):
    wout = found["wout"]
    iota = np.asarray(wout.iotaf)
    distance = np.hypot(*(axis_curve(wout, found["planes"]) - found["axis"]).T)
    assert distance.max() < 3e-3
    assert abs(abs(iota[0]) - abs(found["iota_axis"])) < 2e-3
    assert abs(abs(iota[-1]) - abs(found["iota_surface"])) < 2e-4


def test_effective_ripple_of_the_wout_against_the_coils(found):
    wout = found["wout"]
    periods = ck.SURFACE[4]
    ripple = ck.wout_ripple(wout, int(wout.ns) - 1, periods)
    np.testing.assert_allclose(ripple, found["traced"]["ripple"], rtol=0.1)


def test_magnetic_well_of_the_wout_against_the_coils(found):
    volume = np.asarray(found["wout"].vp)[1:]
    well = (volume[0] - volume[-1]) / volume[0]
    np.testing.assert_allclose(well, found["traced"]["well"], rtol=0.1)


def test_design_found_against_its_start(found):
    traced = found["traced"]
    reach, *_, periods = ck.SURFACE
    start = ck.measures(ck.START, reach, periods)
    assert traced["ripple"] < 0.01
    assert start["ripple"] > 5.0 * traced["ripple"]
    assert start["currents"] > 1.5 * traced["currents"]
    assert traced["closure"] > 0.3
    # within the limits of its search
    assert traced["gaps"].min() > 0.99 * ck.CLEARANCE
    assert traced["room"] > ck.ROOM
    assert traced["mirror_axis"] < 1.01 * ck.MIRROR
    assert ck.FOUND.crossing > 0.99 * ck.CROSSING


@pytest.mark.slow
def test_search_lowers_the_merit():
    def merit(design):
        _, values, currents, limits = ck.search_parts(design)
        return (
            sum(np.sum(v**2) for v in values)
            + currents @ currents
            + 100.0 * np.sum(np.maximum(limits, 0.0) ** 2)
        )

    moved = ck.search(ck.START, workers=4, steps=1)
    assert merit(moved) < 0.5 * merit(ck.START)


@pytest.mark.slow
def test_design_found_against_w7x():
    # the standard configuration, whose effective ripple is below 1% in the core and
    # whose Pfirsch-Schlueter currents are a fraction of those of a tokamak
    yardstick = ck.w7x(ck.OFFSETS)
    assert all(0.004 < k["ripple"] < 0.02 for k in yardstick)
    assert all(0.2 < k["currents"] < 0.5 for k in yardstick)
    periods = ck.SURFACE[4]
    for offset, k in zip(ck.OFFSETS, yardstick, strict=True):
        traced = ck.measures(ck.FOUND, offset, periods)
        assert traced["ripple"] < 0.01
        assert traced["closure"] > k["closure"]
