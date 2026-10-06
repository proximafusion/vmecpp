# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Reversed-field pinches with ``lrfp``: the poloidal flux is the radial coordinate and
``ai`` describes q."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

TEST_DATA = Path(__file__).parent.parent / "src" / "vmecpp" / "cpp" / "vmecpp"
TEST_DATA = TEST_DATA / "test_data"


def test_lrfp_holds_chi_prime_constant_and_phi_prime_at_q_times_it():
    """chi' is constant, phi' = q chi' with q = 0.2 - 0.27 s through the reversal, and
    iota is 1/q on the full grid, where the average of a linear q is exact."""
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA / "rfp_gamma.json")
    wout = vmecpp.run(vmec_input, max_threads=1, verbose=False).wout

    assert wout.lrfp
    s = np.linspace(0.0, 1.0, wout.ns)
    q = 0.2 - 0.27 * s
    chipf = np.asarray(wout.chipf)
    np.testing.assert_allclose(chipf, chipf[0], rtol=1e-13)
    np.testing.assert_allclose(wout.phipf, q * chipf, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(1.0 / np.asarray(wout.iotaf), q, rtol=0.0, atol=1e-13)
    # phiedge is the toroidal flux enclosed by the boundary, though phi reverses
    assert wout.phi[-1] == pytest.approx(vmec_input.phiedge, rel=1e-14)
    assert np.max(wout.phi) > wout.phi[-1]
