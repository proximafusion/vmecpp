# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The coupled block of ``axis_block_preconditioner``: R, Z and lambda at the lowest
poloidal modes and n = 0 take their step from the block of the force Jacobian that
couples them."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

TEST_DATA = Path(__file__).resolve().parents[1] / "src" / "vmecpp" / "cpp" / "vmecpp"
TEST_DATA = TEST_DATA / "test_data"


def _input(ftol: float, **update) -> vmecpp.VmecInput:
    return vmecpp.VmecInput.from_file(TEST_DATA / "cth_like_fixed_bdy.json").model_copy(
        update={
            "ns_array": np.asarray([51]),
            "ftol_array": np.asarray([ftol]),
            "niter_array": np.asarray([20000]),
            **update,
        }
    )


def _axis(wout, nzeta=64):
    """The magnetic axis at nzeta toroidal angles of one field period."""
    zeta = np.linspace(0.0, 2.0 * np.pi / wout.nfp, nzeta, endpoint=False)
    m0 = np.asarray(wout.xm) == 0
    angle = -np.outer(zeta, np.asarray(wout.xn)[m0])
    r = np.cos(angle) @ np.asarray(wout.rmnc)[m0, 0]
    z = np.sin(angle) @ np.asarray(wout.zmns)[m0, 0]
    return r, z


def test_the_block_moves_the_axis_the_residuals_leave_behind():
    """At ftol 1e-8 the axis of a run with the block is at least ten times closer to the
    axis of the converged equilibrium than the axis of a run without it."""
    r_ref, z_ref = _axis(vmecpp.run(_input(1e-16), verbose=False).wout)
    distance = {}
    for block in (False, True):
        wout = vmecpp.run(
            _input(1e-8, axis_block_preconditioner=block), verbose=False
        ).wout
        r, z = _axis(wout)
        distance[block] = np.max(np.hypot(r - r_ref, z - z_ref))
    assert distance[True] < distance[False] / 10


def test_the_block_converges_to_the_same_equilibrium():
    """The two runs reach the same equilibrium in slightly different poloidal angles:
    the coefficients agree to the size of that relabelling, the magnetic energy and the
    pressure to rounding."""
    default = vmecpp.run(_input(1e-14), verbose=False).wout
    block = vmecpp.run(
        _input(1e-14, axis_block_preconditioner=True), verbose=False
    ).wout
    for name, atol in [
        ("rmnc", 2e-5),
        ("zmns", 2e-5),
        ("lmns", 5e-4),
        ("iotaf", 5e-5),
        ("presf", 1e-12),
    ]:
        np.testing.assert_allclose(
            np.asarray(getattr(block, name)),
            np.asarray(getattr(default, name)),
            rtol=0.0,
            atol=atol,
            err_msg=name,
        )
    np.testing.assert_allclose(block.wb, default.wb, rtol=1e-9)


def test_the_block_is_refused_for_a_free_boundary_run():
    vmec_input = vmecpp.VmecInput.from_file(
        TEST_DATA / "cth_like_free_bdy.json"
    ).model_copy(update={"axis_block_preconditioner": True})
    with pytest.raises(ValueError, match="fixed-boundary runs only"):
        vmecpp.run(vmec_input, verbose=False)
