# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++ against exact three-dimensional MHD equilibria.

examples/exact_equilibria.py writes VMEC++ inputs for the exact equilibria of
Landreman (arXiv:2609.26742), of Issan et al. (arXiv:2610.07304), whose parameter tau1
breaks stellarator symmetry, and the exact vacuum fields of Sengupta et al.
(arXiv:2610.28466), and measures, at VMEC++'s own points, quantities that do not depend
on its poloidal angle. These tests hold its member "sheared" (eps = 0.6, S = 2.2,
lambda = 2.97, k_b = 0.2: two field periods, stellarator symmetry, iota from 4.34 to
4.37, volume-averaged beta 1.9 per cent) and "sheared-asym", the same member with tau1
= 0.15 run with lasym, to tolerances at ns = 25 and require the enclosed current to
converge at second order. They hold "vacuum", a member of Sengupta et al.'s ellipsoidal
family with zero pressure and zero current, to tolerances at ns = 25 inside its exact
boundary, check that its field is curl-free, divergence-free and tangent to the surfaces
of its label, and that the formulas give the axis transforms the paper states for two
of its members. They check that the closed forms of Issan, Jose, Landreman, Citrin and
Tracey (arXiv:2611.07672) give equilibria with the field periods, stellarator symmetry,
transform, toroidal flux and current the paper states, for a member of each of their
three families with three field periods, and that VMEC++'s force residuals at the exact
states of these members fall at first order in the radial grid spacing, the lambda
residual at order 1.5. The script's suites scan the members in ns, and "sheared" also
in mpol and ntor.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ellipkinc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import exact_equilibria as ee

NS_COARSE, NS, MPOL, NTOR = 13, 25, 12, 10
# Largest deviations from the exact solution at NS: psi relative to its boundary
# value, the axis in metres, the field relative to |B| at the point, and the enclosed
# current and its radial derivative jcurv relative to their largest values. VMEC++
# reaches 9.7e-5, 4.7e-6, 6.7e-5, 4.2e-6 and 3.1e-6 on "sheared" and 1.1e-4, 6.6e-6,
# 6.8e-5, 4.2e-6 and 3.9e-6 on "sheared-asym", each within 0.5 per cent of that for
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


# Largest deviations of "vacuum" from the exact solution at NS: psi relative to its
# boundary value, the axis in metres, the field relative to |B| at the point, and iota
# relative to its largest value, over the whole profile and for s >= 1/4. VMEC++ reaches
# 1.07e-2, 7.3e-3, 1.04e-3, 1.31e-2 and 3.0e-3, each within 3 per cent of that for delt
# from 0.7 to 1.
VACUUM_TOL = {
    "psi": 1.25e-2,
    "axis": 8.5e-3,
    "B": 1.2e-3,
    "iota": 1.5e-2,
    "iota, s >= 1/4": 3.5e-3,
}


@pytest.fixture(scope="module")
def vacuum_run():
    return ee.run(ee.VACUUM, NS, MPOL, NTOR)


def test_vacuum_converges(vacuum_run):
    assert max(vacuum_run.fsqr, vacuum_run.fsqz, vacuum_run.fsql) <= ee.ftol_for(NS)


@pytest.mark.parametrize("quantity", list(VACUUM_TOL))
def test_vacuum_field(vacuum_run, quantity):
    """With zero pressure and zero enclosed current, the surfaces, the axis, the field
    and iota are the exact vacuum field's."""
    error = ee.VACUUM_MEASURES[quantity](ee.VACUUM, vacuum_run)
    assert error <= VACUUM_TOL[quantity], error


def test_vacuum_field_is_curl_and_divergence_free():
    """On the boundary of "vacuum", psi is constant, and curl B, div B and the component
    of B along grad psi vanish to the accuracy of central differences."""
    member = ee.VACUUM
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
    b, grad_psi = f[:3], d[:, 3]
    curl = np.array([d[1, 2] - d[2, 1], d[2, 0] - d[0, 2], d[0, 1] - d[1, 0]])
    scale = np.max(np.abs(d[:, :3]))
    tangency = np.sum(b * grad_psi, axis=0) / (
        np.linalg.norm(b, axis=0) * np.linalg.norm(grad_psi, axis=0)
    )
    np.testing.assert_allclose(f[3], member.dpsi, rtol=1e-12)
    assert np.max(np.abs(curl)) <= 1e-8 * scale
    assert np.max(np.abs(d[0, 0] + d[1, 1] + d[2, 2])) <= 1e-8 * scale
    assert np.max(np.abs(tangency)) <= 1e-8


# Members of Sengupta et al.'s appendix A: the one with the axis transform sqrt(2)/7 in
# closed form (k = 3 - 2 sqrt(2), m_g = k^2, lambda_0 = 21 sqrt(2) - 28), and the
# reference member, whose axis transform their figure 2 gives as 0.04235.
K_72 = 3.0 - 2.0 * math.sqrt(2.0)
LAMBDA_72 = 21.0 * math.sqrt(2.0) - 28.0


@pytest.mark.parametrize(
    ("member", "iota0", "tolerance"),
    [
        pytest.param(
            ee.Vacuum(
                "sqrt(2)/7",
                k=K_72,
                m_g=K_72**2,
                u0=2.0 * ellipkinc(math.asin(LAMBDA_72**-0.5), K_72**2),
                dpsi=0.01,
            ),
            math.sqrt(2.0) / 7.0,
            1e-12,
            id="sqrt(2)/7",
        ),
        pytest.param(
            ee.Vacuum(
                "reference",
                k=0.5,
                m_g=0.5,
                u0=2.0 * ellipkinc(math.pi / 4.0, 0.5),
                dpsi=0.01,
            ),
            0.04235,
            5e-6,
            id="reference",
        ),
    ],
)
def test_vacuum_axis_transform(member, iota0, tolerance):
    """The flux integrals give the axis transforms of the paper's members."""
    assert abs(abs(member.iota_of_s(0.0)) - iota0) <= tolerance


def _cartesian_derivatives(member, x, h=1e-5):
    """The field and psi at the points x, and d[j, i], the derivative of B_x, B_y, B_z,
    psi (i) along x, y, z (j), by central differences."""
    f = np.array(member.field(*x))
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
    return f, d


# Members of the three families of Issan, Jose, Landreman, Citrin and Tracey with three
# field periods: the first family's member with reflection symmetry (their figure 3(a)),
# the second family's member of their figure 3(b), and a member of the third family,
# whose field lines turn three times about the axis.
CLOSED_LINES = [ee.LORTZ_N3, ee.IOTA0_N3, ee.IOTA3]


@pytest.mark.parametrize("member", CLOSED_LINES, ids=lambda m: m.name)
def test_closed_line_members_are_equilibria(member):
    """On the boundary psi = delta, J x B = grad p with p = 2 (delta - psi) and div B = 0
    hold to the accuracy of central differences, and the field and psi repeat with the
    field period and under stellarator symmetry."""
    theta, phi = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False),
        np.linspace(0.0, 2.0 * np.pi / member.nfp, 6, endpoint=False),
    )
    r, z = member.boundary(theta, phi)
    x = np.array([r * np.cos(phi), r * np.sin(phi), z])
    f, d = _cartesian_derivatives(member, x, h=1e-6)
    b = f[:3]
    j = np.array([d[1, 2] - d[2, 1], d[2, 0] - d[0, 2], d[0, 1] - d[1, 0]])
    force = np.cross(j, b, axis=0) + 2.0 * d[:, 3]
    scale = np.linalg.norm(j, axis=0) * np.linalg.norm(b, axis=0)
    np.testing.assert_allclose(f[3], member.delta, rtol=1e-10)
    assert np.max(np.linalg.norm(force, axis=0) / scale) <= 1e-6
    assert np.max(np.abs(d[0, 0] + d[1, 1] + d[2, 2])) <= 1e-6 * np.max(
        np.abs(d[:, :3])
    )
    angle = 2.0 * np.pi / member.nfp
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = np.array(member.field(*np.einsum("ij,j...->i...", rotation, x)))
    np.testing.assert_allclose(
        rotated[:3], np.einsum("ij,j...->i...", rotation, b), atol=1e-11
    )
    np.testing.assert_allclose(rotated[3], f[3], rtol=1e-11)
    # the rotation by pi about the x axis takes (Bx, By, Bz) at x to (-Bx, By, Bz)
    flipped = np.array(member.field(x[0], -x[1], -x[2]))
    np.testing.assert_allclose(
        flipped[:3], b * np.array([-1.0, 1.0, 1.0])[:, None, None], atol=1e-11
    )
    np.testing.assert_allclose(flipped[3], f[3], rtol=1e-11)


@pytest.mark.parametrize(
    ("member", "iota"),
    [
        pytest.param(m, i, id=m.name)
        for m, i in zip(CLOSED_LINES, (0.0, 0.0, 3.0), strict=True)
    ],
)
def test_closed_line_members_have_the_papers_transform(member, iota):
    """VMEC++ is given the turns of a field line about the axis as iota: -s N on the
    third family's branch and none on the others'."""
    np.testing.assert_allclose(member.vmec_input(13, 4, 3).ai, [iota])


@pytest.mark.parametrize("member", CLOSED_LINES, ids=lambda m: m.name)
def test_closed_line_flux_and_current(member):
    """The volume inside the boundary is 2 pi times its toroidal flux (their sections
    3.2 and 4.3), and the circulation of B around the boundary in the plane phi = 0 is 2
    pi buco."""
    nth, nph = 128, 64
    theta = 2.0 * np.pi * np.arange(nth) / nth
    phi = 2.0 * np.pi * np.arange(nph) / (nph * member.nfp)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    r, z = member.boundary(th, ph)
    k = np.fft.fftfreq(nth, 1.0 / nth)[:, None]
    z_theta = np.real(np.fft.ifft(1j * k * np.fft.fft(z, axis=0), axis=0))
    r_theta = np.real(np.fft.ifft(1j * k * np.fft.fft(r, axis=0), axis=0))
    # the area integral of R over each section, integrated over phi
    volume = 2.0 * np.pi * np.mean(0.5 * r * r * z_theta) * 2.0 * np.pi
    np.testing.assert_allclose(
        volume, 2.0 * np.pi * member.profiles["phi_edge"], rtol=1e-9
    )
    x = np.array([r[:, 0], 0.0 * r[:, 0], z[:, 0]])
    bx, _, bz, _ = member.field(*x)
    circulation = np.mean(bx * r_theta[:, 0] + bz * z_theta[:, 0])
    np.testing.assert_allclose(
        circulation, member.buco_of_s(1.0), rtol=1e-8, atol=1e-12
    )


@pytest.mark.parametrize("member", CLOSED_LINES, ids=lambda m: m.name)
def test_closed_line_residuals_fall_at_the_order_of_the_discretization(member):
    """Halving the radial grid spacing halves VMEC++'s R and Z force residuals at the
    exact state and divides the lambda residual by about 2^1.5."""
    mpol, ntor = ee.CLOSED_LINE_MODES[member.name]
    coarse, fine = (ee.exact_residual(member, ns, mpol, ntor) for ns in (NS_COARSE, NS))
    least = {
        "R residual": 2.0**0.95,
        "Z residual": 2.0**0.95,
        "lambda residual": 2.0**1.4,
    }
    ratios = {key: coarse[key] / fine[key] for key in least}
    assert all(ratios[key] >= value for key, value in least.items()), ratios
