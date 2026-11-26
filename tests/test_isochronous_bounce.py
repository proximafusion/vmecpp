# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the isochronous bounce example."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.interpolate
import scipy.special

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import isochronous_bounce as ib


@pytest.fixture(scope="module")
def wout():
    vmec_input = vmecpp.VmecInput.from_file(ib.HUYGENS).model_copy(
        update={
            "ns_array": np.array([13, 25]),
            "ftol_array": np.array([1e-8, 1e-10]),
            "niter_array": np.array([5000, 10000]),
        }
    )
    return vmecpp.run(vmec_input, verbose=False).wout


def geometry_rates(wout, state, moment, inertia):
    """The guiding-center rates from the flux surfaces in cylindrical coordinates."""
    s, theta, zeta, v = state
    ns = int(wout.ns)
    rho = np.sqrt(s)
    m, n = np.asarray(wout.xm, dtype=float), np.asarray(wout.xn, dtype=float)
    full = np.sqrt(np.arange(ns) / (ns - 1))
    half = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))
    r_mn = scipy.interpolate.CubicSpline(full, np.asarray(wout.rmnc).T)
    z_mn = scipy.interpolate.CubicSpline(full, np.asarray(wout.zmns).T)
    c, sn = np.cos(m * theta - n * zeta), np.sin(m * theta - n * zeta)
    r = c @ r_mn(rho)
    e_s = np.array(
        [
            c @ r_mn(rho, 1) * np.cos(zeta),
            c @ r_mn(rho, 1) * np.sin(zeta),
            sn @ z_mn(rho, 1),
        ]
    ) / (2.0 * rho)
    r_theta, z_theta = -sn @ (m * r_mn(rho)), c @ (m * z_mn(rho))
    r_zeta, z_zeta = sn @ (n * r_mn(rho)), -c @ (n * z_mn(rho))
    e_theta = np.array([r_theta * np.cos(zeta), r_theta * np.sin(zeta), z_theta])
    e_zeta = np.array(
        [
            r_zeta * np.cos(zeta) - r * np.sin(zeta),
            r_zeta * np.sin(zeta) + r * np.cos(zeta),
            z_zeta,
        ]
    )
    jacobian = e_s @ np.cross(e_theta, e_zeta)
    grad_s = np.cross(e_theta, e_zeta) / jacobian
    grad_theta = np.cross(e_zeta, e_s) / jacobian
    grad_zeta = np.cross(e_s, e_theta) / jacobian
    m_nyq, n_nyq = np.asarray(wout.xm_nyq, dtype=float), np.asarray(wout.xn_nyq, float)
    c, sn = np.cos(m_nyq * theta - n_nyq * zeta), np.sin(m_nyq * theta - n_nyq * zeta)
    b_mn = scipy.interpolate.CubicSpline(half, np.asarray(wout.bmnc)[:, 1:].T)
    up_mn = scipy.interpolate.CubicSpline(half, np.asarray(wout.bsupumnc)[:, 1:].T)
    along_mn = scipy.interpolate.CubicSpline(half, np.asarray(wout.bsupvmnc)[:, 1:].T)
    strength = c @ b_mn(rho)
    up, along = c @ up_mn(rho), c @ along_mn(rho)
    field = up * e_theta + along * e_zeta
    gradient = (
        (c @ b_mn(rho, 1)) / (2.0 * rho) * grad_s
        - (sn @ (m_nyq * b_mn(rho))) * grad_theta
        + (sn @ (n_nyq * b_mn(rho))) * grad_zeta
    )
    drift = inertia * (v**2 / strength + moment) * np.cross(field, gradient)
    drift /= strength**2
    return np.array(
        [
            drift @ grad_s,
            v * up / strength + drift @ grad_theta,
            v * along / strength + drift @ grad_zeta,
            -moment * field @ gradient / strength,
        ]
    )


def test_bounce_integral_of_a_pendulum():
    # In B = 1 - depth \cos l the bounce integral is 4 K(k) / \sqrt{2 depth}.
    # Here k^2 = (level - 1 + depth) / (2 depth).
    depth = 0.2
    length = np.linspace(-np.pi, np.pi, 4000)
    strength = 1.0 - depth * np.cos(length)
    for level in 1.0 - depth + 2.0 * depth * np.linspace(0.25, 0.75, 11):
        k2 = (level - 1.0 + depth) / (2.0 * depth)
        exact = 4.0 * scipy.special.ellipk(k2) / np.sqrt(2.0 * depth)
        np.testing.assert_allclose(
            ib.bounce_integral(strength, length, level), exact, rtol=1e-5
        )


def test_bounce_periods_of_a_pendulum():
    # B = 1 - depth \cos \phi with dl = d\phi has its maxima at \pm \pi.
    depth = 0.2
    strength = ib.cosine_series([1.0, -depth])
    periods = ib.bounce_periods(
        strength, np.ones_like, -np.pi, 0.0, np.linspace(0.05, 0.95, 19)
    )
    exact = 4.0 * scipy.special.ellipk(np.linspace(0.05, 0.95, 19))
    np.testing.assert_allclose(periods, exact / np.sqrt(2.0 * depth), rtol=1e-12)


def test_huygens_series_keeps_time_and_a_cosine_well_does_not():
    # The truncated Boozer series of the Huygens well against the cosine well of the
    # same depth.
    huygens = ib.boozer_series(0.85 + 0.3 * ib.huygens_shape(ib.SHARE))
    assert np.ptp(ib.boozer_periods(huygens, ib.BAND)[0]) < 1e-3
    assert np.ptp(ib.boozer_periods([1.0, 0.15], ib.BAND)[0]) > 0.1


@pytest.mark.slow
def test_designed_well_keeps_time_from_its_bottom():
    coefficients = ib.design_well(0.85, 1.15)
    periods, depth = ib.boozer_periods(coefficients, ib.DESIGN_BAND)
    assert np.ptp(periods) < 3e-8
    np.testing.assert_allclose(depth, 0.3, rtol=1e-10)


def test_rates_against_the_flux_surfaces(wout):
    # The rates from the covariant components and the Jacobian of the wout against
    # those from the flux surfaces in cylindrical coordinates, on a half-grid surface
    # and on either side of it.
    ns = int(wout.ns)
    s = (np.round(0.2 * (ns - 1) + 0.5) - 0.5) / (ns - 1)
    rates = ib.guiding_centers(wout, tolerance=0.0)
    moment, inertia = 0.6, 0.01 * float(wout.Aminor_p)
    for offset, theta, zeta, v in ((0.0, 0.3, 0.7, 0.2), (0.06, 2.0, 2.5, -0.3)):
        for side in (1.0, -1.0):
            state = np.array(
                [[(np.sqrt(s) + side * offset) ** 2], [theta], [zeta], [v]]
            )
            parallel = rates(state, moment, 0.0)[0][:, 0]
            np.testing.assert_allclose(
                parallel, geometry_rates(wout, state[:, 0], moment, 0.0), rtol=1e-10
            )
            drift = rates(state, moment, inertia)[0][:, 0] - parallel
            expected = geometry_rates(wout, state[:, 0], moment, inertia)
            expected -= geometry_rates(wout, state[:, 0], moment, 0.0)
            np.testing.assert_allclose(drift[0], expected[0], rtol=1e-3)
            # The angular drift takes B_s from the full grid of the wout.
            np.testing.assert_allclose(
                drift[1:], expected[1:], atol=1e-2 * np.abs(expected).max()
            )


def test_guiding_centers_keep_their_energy(wout):
    rates = ib.guiding_centers(wout)
    state = np.array([[0.2, 0.22], [0.5, 2.0], [1.0, 2.5], [0.2, -0.3]])
    moment, inertia = 0.6, 0.01 * float(wout.Aminor_p)
    _, strength = rates(state, moment, inertia)
    energy = 0.5 * state[3] ** 2 + moment * strength
    dt = 0.05
    for _ in range(400):
        k1, _ = rates(state, moment, inertia)
        k2, _ = rates(state + 0.5 * dt * k1, moment, inertia)
        k3, _ = rates(state + 0.5 * dt * k2, moment, inertia)
        k4, _ = rates(state + dt * k3, moment, inertia)
        state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    _, strength = rates(state, moment, inertia)
    np.testing.assert_allclose(
        0.5 * state[3] ** 2 + moment * strength, energy, rtol=1e-8
    )


def test_ticks_without_drift_follow_the_bounce_integral(wout):
    # The periods over their mean against the bounce integrals of a finer trace.
    times, _ = ib.clock(
        wout, [0.2], lines=1, ions=8, ticks=3, steps=200, gyroradius=0.0
    )
    periods = np.diff(times[0, 0], axis=1).mean(1)
    (b, arc, _, _), *_ = ib.first_wells(wout, 0.2, [0.0], steps=2000)[0]
    bottom, top = b.min(), min(b[0], b[-1])
    fields = bottom + (top - bottom) * np.linspace(0.25, 0.75, 8)
    integrals = np.array([ib.bounce_integral(b, arc, f) for f in fields])
    np.testing.assert_allclose(
        periods / periods.mean(), integrals / integrals.mean(), rtol=2e-5
    )


@pytest.mark.slow
def test_axis_of_the_equilibria():
    def spread(path):
        wout = vmecpp.run(vmecpp.VmecInput.from_file(path), verbose=False).wout
        return np.ptp(ib.axis_periods(wout))

    assert spread(ib.HUYGENS) < 1e-3
    assert spread(ib.COSINE) > 5e-2
