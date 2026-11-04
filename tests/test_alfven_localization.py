# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigvalsh_tridiagonal

import vmecpp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import alfven_localization as al  # noqa: E402

PERIODS = 600


@pytest.fixture(scope="module")
def wout():
    vmec_input = vmecpp.VmecInput.from_file(ROOT / "examples" / "data" / "w7x.json")
    vmec_input.ns_array = np.array([25])
    vmec_input.ftol_array = np.array([1e-11])
    vmec_input.niter_array = np.array([5000])
    return vmecpp.run(vmec_input, verbose=False).wout


@pytest.fixture(scope="module")
def surface(wout):
    js = round(0.75 * (int(wout.ns) - 1))
    h = 2.0 * np.pi / int(wout.nfp) / al.POINTS_PER_PERIOD
    zeta = h * (0.5 + np.arange(al.POINTS_PER_PERIOD * PERIODS))
    return js, h, zeta, *al.line_coefficients(wout, js, zeta)


def test_a_constant_is_the_zero_mode(surface):
    """(P u_tau)_tau + W^2 P u = 0 keeps u = const at W = 0."""
    _, h, _, p, b_zeta = surface
    lowest = eigvalsh_tridiagonal(
        *al.operator(p, b_zeta, h), select="i", select_range=(0, 0)
    )
    assert abs(lowest[0]) < 1e-9


def test_counting_follows_the_alfven_time(surface):
    """The states below W approach W tau / pi."""
    _, h, _, p, b_zeta = surface
    tau = np.sum(h / np.abs(b_zeta))
    w = 4.0 * np.pi * PERIODS / tau
    count = len(
        eigvalsh_tridiagonal(
            *al.operator(p, b_zeta, h), select="v", select_range=(0.0, w**2)
        )
    )
    np.testing.assert_allclose(count, w * tau / np.pi, rtol=0.03)


def test_gaps_sit_at_their_labels(wout, surface):
    """The widest gaps hold a rotation |m iota - n N| / 2."""
    js, h, zeta, p, b_zeta = surface
    iota, nfp = abs(float(wout.iotas[js])), int(wout.nfp)
    unit = abs(float(wout.b0)) / float(wout.Rmajor_p)
    values = np.sqrt(
        np.maximum(
            eigvalsh_tridiagonal(
                *al.operator(p, b_zeta, h),
                select="v",
                select_range=(0.0, (3.0 * unit) ** 2),
            ),
            0.0,
        )
    )
    wide = np.argsort(np.diff(values))[-5:]
    labels = np.array(
        [abs(m * iota - n * nfp) / 2.0 for m in range(13) for n in range(4)]
    )
    rotation = np.pi * (wide + 1) / zeta[-1]
    distance = np.min(np.abs(rotation[:, None] - labels[None, :]), axis=1)
    assert np.all(distance < 5e-3)


def test_end_states_cross_each_gap_as_its_label_says(wout, surface):
    """As alpha advances by 2 pi the states at one end of the line cross a gap of label
    |m iota - n N| / 2 m times, and those at the other end the opposite way."""
    js = surface[0]
    iota, nfp = abs(float(wout.iotas[js])), int(wout.nfp)
    periods, points = 200, 24
    length = periods * 2.0 * np.pi / nfp
    gaps = al.widest_gaps(wout, js, periods, 1.5, 3, points)
    net, occupancy = al.end_state_crossings(wout, js, periods, gaps, 96, points)
    for gap, (left, right), states_in_gap in zip(gaps, net, occupancy, strict=True):
        assert states_in_gap.max() < 1.0
        assert left == -right
        assert left != 0
        fits = [abs(abs(left * iota - n * nfp) / 2.0 - gap[2]) for n in range(-8, 9)]
        assert min(fits) < 2.0 * np.pi / length


def test_quasi_periodic_states_localize_and_periodic_ones_do_not(wout, surface):
    """In a window of the spectrum the states of the true line keep their extent as the
    line doubles, and at a rational slope they grow with it."""
    js = surface[0]
    true_line = {row[0]: row for row in al.table(wout, js, PERIODS, 1.0)}
    localized = [row for row in true_line.values() if row[2] < 20.0]
    assert localized
    for _, _, extent, extent_half in localized:
        np.testing.assert_allclose(extent, extent_half, rtol=0.25)
    for _, _, extent, extent_half in al.table(wout, js, PERIODS, 1.0, slope=1.0):
        assert extent > 1.5 * extent_half
