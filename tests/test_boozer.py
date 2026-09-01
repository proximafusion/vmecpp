# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The in-core Boozer transform against booz_xform, when that package is available, and
its invariants otherwise."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

TEST_DATA_DIR = (
    Path(__file__).parent.parent / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
)


@pytest.fixture(scope="module")
def cth_like_output() -> vmecpp.VmecOutput:
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    return vmecpp.run(vmec_input, verbose=False)


def test_boozer_transform_invariants(cth_like_output):
    wout = cth_like_output.wout
    boozer = vmecpp.boozer_transform(wout, mboz=8, nboz=8)

    assert boozer.nfp == wout.nfp
    assert len(boozer.xm_b) == 9 + 7 * 17
    assert list(boozer.surfaces) == list(range(1, wout.ns))
    # sqrt(g_B) |B|^2 is a flux function in Boozer coordinates; the spread is
    # the residual of the transformation, set by the force tolerance (1e-6)
    assert np.max(boozer.jacobian_spread) < 1.0e-3
    # the Boozer currents are the (0, 0) covariant field components
    np.testing.assert_allclose(boozer.g_b, wout.bvco[1:], rtol=0, atol=0)
    np.testing.assert_allclose(boozer.i_b, wout.buco[1:], rtol=0, atol=0)
    np.testing.assert_allclose(boozer.iota_b, wout.iotas[1:], rtol=0, atol=0)


def test_boozer_transform_matches_booz_xform(cth_like_output, tmp_path):
    bx = pytest.importorskip("booz_xform")
    wout = cth_like_output.wout
    wout_path = tmp_path / "wout_cth_like_fixed_bdy.nc"
    wout.save(wout_path)

    reference = bx.Booz_xform()
    reference.read_wout(str(wout_path))
    reference.mboz = 8
    reference.nboz = 8
    # booz_xform indexes the half grid from zero; wout columns start at one
    reference.compute_surfs = list(range(wout.ns - 1))
    reference.run()

    boozer = vmecpp.boozer_transform(wout, mboz=8, nboz=8)
    np.testing.assert_array_equal(boozer.xm_b, reference.xm_b)
    np.testing.assert_array_equal(boozer.xn_b, reference.xn_b)
    for name, ours, theirs in [
        ("bmnc_b", boozer.bmnc_b, reference.bmnc_b),
        ("rmnc_b", boozer.rmnc_b, reference.rmnc_b),
        ("zmns_b", boozer.zmns_b, reference.zmns_b),
        ("numns_b", boozer.numns_b, reference.numns_b),
        ("gmnc_b", boozer.gmnc_b, reference.gmnc_b),
    ]:
        scale = np.max(np.abs(np.asarray(theirs)))
        np.testing.assert_allclose(
            np.asarray(ours),
            np.asarray(theirs),
            rtol=0,
            atol=1.0e-9 * scale,
            err_msg=name,
        )
