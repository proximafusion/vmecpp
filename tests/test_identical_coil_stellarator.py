# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the identical-coil stellarator example."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from simsopt.field import BiotSavart

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import identical_coil_stellarator as ck  # type: ignore


@pytest.fixture(scope="module")
def solved():
    k = ck.interior(BiotSavart(ck.coils()), ck.DESIGN)
    k["wout"] = vmecpp.run(k["input"], max_threads=1, verbose=False).wout
    return k


def test_field_of_the_coils_is_stellarator_symmetric():
    # B_R is odd and B_\phi and B_Z are even under (\phi, Z) \to (-\phi, -Z).
    # The field repeats after a field period.
    field = BiotSavart(ck.coils())
    rng = np.random.default_rng(1)
    points = ck.CENTRES + rng.uniform(-0.2, 0.2, ck.CENTRES.shape)
    r = np.hypot(points[:, 0], points[:, 1])
    phi, z = np.arctan2(points[:, 1], points[:, 0]), points[:, 2]

    def cylindrical(r, phi, z):
        cos, sin = np.cos(phi), np.sin(phi)
        field.set_points(np.ascontiguousarray(np.stack([r * cos, r * sin, z], -1)))
        b = np.asarray(field.B())
        return np.stack(
            [b[:, 0] * cos + b[:, 1] * sin, b[:, 1] * cos - b[:, 0] * sin, b[:, 2]], -1
        )

    here = cylindrical(r, phi, z)
    np.testing.assert_allclose(
        cylindrical(r, -phi, -z), here * [-1.0, 1.0, 1.0], rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        cylindrical(r, phi + 2.0 * np.pi / ck.NFP, z), here, rtol=0.0, atol=1e-10
    )


def test_effective_ripple_against_quadrature():
    # B = 1 - 0.1 \cos\zeta has whole wells in three of four periods.
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
    # |\nabla \psi| = 1 and R_0 = 1.
    expected = (np.pi / (8.0 * np.sqrt(2.0)) / (2.0 * np.pi * periods) * wells) ** (
        2.0 / 3.0
    )
    # from the mean B as the reference field to the strongest
    expected *= (strength.max() / mean) ** (4.0 / 3.0)
    found = ck.effective_ripple(strength, size, 0.3 + np.sin(zeta), weight, 1.0)
    np.testing.assert_allclose(found, expected, rtol=1e-4)
    # A drift odd about the bottom of every well averages out.
    assert ck.effective_ripple(strength, size, np.sin(zeta), weight, 1.0) < 1e-15


def test_closure_of_an_analytic_field():
    # The contours of B = 1 + 0.1 \cos\zeta + 0.03 \cos\theta close poloidally between
    # 0.93 and 1.07.
    theta, zeta = np.meshgrid(
        2.0 * np.pi * np.arange(256) / 256,
        2.0 * np.pi * np.arange(256) / 256,
        indexing="ij",
    )
    strength = 1.0 + 0.1 * np.cos(zeta) + 0.03 * np.cos(theta)
    np.testing.assert_allclose(ck.closure(strength), (1.07 - 0.93) / 0.26, rtol=1e-12)
    # B of \zeta alone is quasi-isodynamic.
    np.testing.assert_allclose(ck.closure(1.0 + 0.1 * np.cos(zeta)), 1.0, rtol=1e-12)
    # The contours of the quasi-helical B = 1 + 0.1 \cos(\theta - \zeta) are helical.
    np.testing.assert_allclose(ck.closure(1.0 + 0.1 * np.cos(theta - zeta)), 0.0)


def test_interior_against_tracing(solved):
    wout = solved["wout"]
    iota = np.asarray(wout.iotaf)
    np.testing.assert_allclose(
        ck.axis_curve(wout, solved["planes"]), solved["axis"], rtol=0.0, atol=3e-3
    )
    # At this resolution VMEC++ puts the rotational transform 0.0013 below the traced
    # one on the axis and 0.0001 above it on the boundary.
    np.testing.assert_allclose(
        abs(iota[0]), abs(solved["iota_axis"]), rtol=0.0, atol=7e-3
    )
    np.testing.assert_allclose(
        abs(iota[-1]), abs(solved["iota_surface"]), rtol=0.0, atol=5e-4
    )


def test_effective_ripple_and_closure_of_the_interior(solved):
    _, ripple, closed = ck.measures(solved["wout"])
    assert max(ripple) < 0.025
    assert min(closed) > 0.5


@pytest.mark.slow
def test_effective_ripple_and_closure_against_w7x(solved):
    w7x = ck.interior(BiotSavart(ck.w7x_coils()), ck.W7X)
    _, ripple, closed = ck.measures(
        vmecpp.run(w7x["input"], max_threads=1, verbose=False).wout
    )
    *_, design_closed = ck.measures(solved["wout"])
    assert all(0.003 < value < 0.03 for value in ripple)
    assert all(a > b for a, b in zip(design_closed, closed, strict=True))
