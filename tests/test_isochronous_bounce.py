# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests of the isochronous bounce example."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.special

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import isochronous_bounce as ib


@pytest.fixture(scope="module")
def wout():
    vmec_input = vmecpp.VmecInput.from_file(ib.HUYGENS).model_copy(
        update={
            "ns_array": np.array([13]),
            "ftol_array": np.array([1e-9]),
            "niter_array": np.array([5000]),
        }
    )
    return vmecpp.run(vmec_input, verbose=False).wout


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


def test_huygens_well_keeps_time_and_a_cosine_well_does_not():
    huygens = 0.85 + 0.3 * ib.huygens_shape(ib.SHARE)
    cosine = 1.0 - 0.15 * np.cos(2.0 * np.pi * ib.SHARE)
    assert ib.profile_spread(huygens) < 1e-5
    assert ib.profile_spread(cosine) > 0.1


def test_huygens_well_through_the_boozer_angle():
    # The series maps back to arc length through dl = d\varphi / B_0.
    coefficients = ib.boozer_series(0.85 + 0.3 * ib.huygens_shape(ib.SHARE))
    angle = 2.0 * np.pi * np.arange(20001) / 20000
    strength = np.cos(np.outer(angle, np.arange(len(coefficients)))) @ coefficients
    steps = 0.5 * (1.0 / strength[1:] + 1.0 / strength[:-1]) * angle[1]
    length = np.concatenate([[0.0], np.cumsum(steps)])
    bottom, top = strength.min(), strength.max()
    periods = [
        ib.bounce_integral(strength, length, bottom + (top - bottom) * f)
        for f in np.linspace(0.25, 0.75, 21)
    ]
    assert (max(periods) - min(periods)) / np.mean(periods) < 1e-3


def test_guiding_centers_keep_their_energy(wout):
    rates = ib.guiding_centers(wout, 0.2)
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
    times, _ = ib.clock(wout, 0.2, ions=8, ticks=3, gyroradius=0.0)
    b, l, _, _, _ = ib.first_well(wout, 0.2)
    bottom, top = b.min(), min(b[0], b[-1])
    fields = bottom + (top - bottom) * np.linspace(0.25, 0.75, 8)
    integrals = np.array([ib.bounce_integral(b, l, f) for f in fields])
    middle = ib.bounce_integral(b, l, 0.5 * (bottom + top))
    np.testing.assert_allclose(
        np.diff(times, axis=1).mean(1), integrals / middle, rtol=1e-4
    )


@pytest.mark.slow
def test_axis_of_the_equilibria():
    def spread(path):
        wout = vmecpp.run(vmecpp.VmecInput.from_file(path), verbose=False).wout
        return ib.profile_spread(ib.axis_field(wout))

    assert spread(ib.HUYGENS) < 2e-3
    assert spread(ib.COSINE) > 5e-2
