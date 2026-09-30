# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ against an exact three-dimensional MHD equilibrium.

examples/exact_equilibria.py writes VMEC++ inputs for the exact equilibria of
Landreman (arXiv:2609.26742) and measures, at VMEC++'s own points, quantities that do
not depend on its poloidal angle. These tests hold its member "sheared" (eps = 0.6,
S = 2.2, lambda = 2.97, k_b = 0.2: two field periods, stellarator symmetry, iota from
4.34 to 4.37, volume-averaged beta 1.9 per cent) to tolerances at ns = 25 and require
the enclosed current to converge at second order; the script's quick and full suites
scan every member in ns, mpol and ntor.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import exact_equilibria as ee

NS_COARSE, NS, MPOL, NTOR = 13, 25, 12, 10
# Largest deviations from the exact solution at NS: psi relative to its boundary
# value, the axis in metres, the field relative to |B| at the point, and the enclosed
# current and its radial derivative jcurv relative to their largest values. VMEC++
# reaches 5.2e-5, 4.1e-6, 4.9e-5, 4.2e-6 and 3.3e-6, each within 15 per cent of that
# for delt from 0.7 to 1.
TOL = {"psi": 1.2e-4, "axis": 9e-6, "B": 1e-4, "current": 8e-6, "jcurv": 7e-6}


@pytest.fixture(scope="module")
def runs():
    return {ns: ee.run(ee.SHEARED, ns, MPOL, NTOR) for ns in (NS_COARSE, NS)}


def test_converges(runs):
    for ns, wout in runs.items():
        assert max(wout.fsqr, wout.fsqz, wout.fsql) <= ee.ftol_for(ns)


def test_flux_surfaces(runs):
    """On every surface the analytic psi equals the surface's own flux label."""
    error = ee.surface_error(ee.SHEARED, runs[NS])
    assert error <= TOL["psi"], error


def test_magnetic_axis(runs):
    """The axis is the ellipse of semiaxes a(S), b(S) in the plane Z = 0."""
    error = ee.axis_error(ee.SHEARED, runs[NS])
    assert error <= TOL["axis"], error


def test_magnetic_field(runs):
    """At the half-grid points the field is the analytic field there."""
    error = ee.field_error(ee.SHEARED, runs[NS])
    assert error <= TOL["B"], error


def test_enclosed_current(runs):
    """With iota prescribed, the toroidal current inside each half-grid surface is the
    exact solution's, and so is its radial derivative on the full grid."""
    error = ee.current_error(ee.SHEARED, runs[NS])
    assert error <= TOL["current"], error
    error = ee.current_derivative_error(ee.SHEARED, runs[NS])
    assert error <= TOL["jcurv"], error


def test_current_converges_at_second_order(runs):
    """Halving the radial grid spacing divides the error in the enclosed current by
    about four."""
    ratio = ee.current_error(ee.SHEARED, runs[NS_COARSE]) / ee.current_error(
        ee.SHEARED, runs[NS]
    )
    assert ratio >= 3.5, ratio
