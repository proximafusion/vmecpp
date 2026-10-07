# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ against exact three-dimensional MHD equilibria.

examples/exact_equilibria.py writes VMEC++ inputs for the exact equilibria of
Landreman (arXiv:2609.26742) and of Issan et al. (arXiv:2610.07304), whose parameter
tau1 breaks stellarator symmetry, and measures, at VMEC++'s own points, quantities that
do not depend on its poloidal angle. These tests hold its member "sheared" (eps = 0.6,
S = 2.2, lambda = 2.97, k_b = 0.2: two field periods, stellarator symmetry, iota from
4.34 to 4.37, volume-averaged beta 1.9 per cent) and "sheared-asym", the same member
with tau1 = 0.15 run with lasym, to tolerances at ns = 25 and require the enclosed
current to converge at second order; the script's suites scan the members in ns, and
"sheared" also in mpol and ntor.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import exact_equilibria as ee

NS_COARSE, NS, MPOL, NTOR = 13, 25, 12, 10
# Largest deviations from the exact solution at NS: psi relative to its boundary
# value, the axis in metres, the field relative to |B| at the point, and the enclosed
# current and its radial derivative jcurv relative to their largest values. VMEC++
# reaches 7.5e-5, 4.0e-6, 6.7e-5, 4.2e-6 and 3.4e-6 on "sheared" and 8.4e-5, 4.6e-6,
# 6.9e-5, 4.3e-6 and 4.2e-6 on "sheared-asym", each within 16 per cent of that for
# delt from 0.7 to 1.
TOL = {
    "sheared": {"psi": 1.2e-4, "axis": 9e-6, "B": 1e-4, "current": 8e-6, "jcurv": 7e-6},
    "sheared-asym": {
        "psi": 1.3e-4,
        "axis": 1e-5,
        "B": 1.1e-4,
        "current": 8e-6,
        "jcurv": 8e-6,
    },
}


@pytest.fixture(scope="module", params=list(TOL))
def case(request):
    member = ee.MEMBERS[request.param]
    return member, {ns: ee.run(member, ns, MPOL, NTOR) for ns in (NS_COARSE, NS)}


def test_converges(case):
    _, runs = case
    for ns, wout in runs.items():
        assert max(wout.fsqr, wout.fsqz, wout.fsql) <= ee.ftol_for(ns)


def test_flux_surfaces(case):
    """On every surface the analytic psi equals the surface's own flux label."""
    member, runs = case
    error = ee.surface_error(member, runs[NS])
    assert error <= TOL[member.name]["psi"], error


def test_magnetic_axis(case):
    """The axis is the ellipse of semiaxes a(S), b(S) in the plane Z = 0, whatever
    tau1."""
    member, runs = case
    error = ee.axis_error(member, runs[NS])
    assert error <= TOL[member.name]["axis"], error


def test_magnetic_field(case):
    """At the half-grid points the field is the analytic field there."""
    member, runs = case
    error = ee.field_error(member, runs[NS])
    assert error <= TOL[member.name]["B"], error


def test_enclosed_current(case):
    """With iota prescribed, the toroidal current inside each half-grid surface is the
    exact solution's, and so is its radial derivative on the full grid."""
    member, runs = case
    error = ee.current_error(member, runs[NS])
    assert error <= TOL[member.name]["current"], error
    error = ee.current_derivative_error(member, runs[NS])
    assert error <= TOL[member.name]["jcurv"], error


def test_current_converges_at_second_order(case):
    """Halving the radial grid spacing divides the error in the enclosed current by
    about four."""
    member, runs = case
    ratio = ee.current_error(member, runs[NS_COARSE]) / ee.current_error(
        member, runs[NS]
    )
    assert ratio >= 3.5, ratio


# p'(psi) of Issan et al.'s equations 17 and 7
@pytest.mark.parametrize(
    ("member", "dp_dpsi"),
    [
        pytest.param(ee.SHEARED_ASYM, -1.0 / ee.SHEARED_ASYM.lam**2, id="sheared-asym"),
        pytest.param(
            ee.IOTA2_ASYM, -2.0 / (1.0 - ee.IOTA2_ASYM.tau0**2), id="iota2-asym"
        ),
    ],
)
def test_members_without_stellarator_symmetry_are_equilibria(member, dp_dpsi):
    """On the boundary of each member with tau1 or tau0, psi is constant, and J x B =
    p'(psi) grad psi and div B = 0 hold to the accuracy of central differences."""
    theta, phi = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False),
        np.linspace(0.0, np.pi, 12, endpoint=False),
    )
    r, z = member.boundary(theta, phi)
    x = np.array([r * np.cos(phi), r * np.sin(phi), z])
    h = 1e-5
    f = np.array(member.field(*x))
    # d[j, i] is the derivative of B_x, B_y, B_z, psi (i) along x, y, z (j)
    d = np.array(
        [
            (
                np.array(member.field(*(x + h * e)))
                - np.array(member.field(*(x - h * e)))
            )
            / (2.0 * h)
            for e in np.eye(3)[:, :, None, None]
        ]
    )
    b = f[:3]
    j = np.array([d[1, 2] - d[2, 1], d[2, 0] - d[0, 2], d[0, 1] - d[1, 0]])
    grad_psi = d[:, 3]
    force = np.cross(j, b, axis=0) - dp_dpsi * grad_psi
    scale = np.linalg.norm(j, axis=0) * np.linalg.norm(b, axis=0)
    assert np.ptp(f[3]) <= 1e-13 * np.max(f[3])
    assert np.max(np.linalg.norm(force, axis=0) / scale) <= 1e-8
    assert np.max(np.abs(d[0, 0] + d[1, 1] + d[2, 2])) <= 1e-8 * np.max(
        np.abs(d[:, :3])
    )
