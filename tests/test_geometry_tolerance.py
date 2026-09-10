# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Convergence on the state rather than on the force alone.

The force residual is small in directions the equilibrium is still moving along,
so a run that meets ``ftol_array`` can leave the magnetic axis and the rotational
transform short of where they settle. ``geometry_tolerance`` adds the condition
that the flux surfaces have stopped moving, measured over the last ``nstep``
iterations as the Euclidean norm of the change in the R and Z coefficients.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
CASE = TEST_DATA / "cth_like_fixed_bdy.json"


def _single_grid(ns: int = 51) -> vmecpp.VmecInput:
    vmec_input = vmecpp.VmecInput.from_file(CASE)
    vmec_input.ns_array = np.array([ns])
    vmec_input.niter_array = np.array([100000])
    return vmec_input


def _axis_and_iota(wout) -> tuple[float, float]:
    return float(wout.rmnc[0, 0]), float(wout.iotaf[wout.iotaf.size // 2])


def test_unset_tolerance_leaves_the_run_untouched():
    """The default is off, so the run is the one the residual test alone gives."""
    vmec_input = _single_grid()
    vmec_input.ftol_array = np.array([1.0e-8])
    assert vmec_input.geometry_tolerance == 0.0
    loose = vmecpp.run(vmec_input, verbose=False, max_threads=1)

    explicit = vmec_input.model_copy(update={"geometry_tolerance": 0.0})
    again = vmecpp.run(explicit, verbose=False, max_threads=1)

    assert again.wout.niter == loose.wout.niter
    np.testing.assert_array_equal(again.wout.rmnc, loose.wout.rmnc)


def test_a_settled_geometry_is_required_before_convergence():
    """At a tolerance most inputs ship with, the axis and iota are still moving;
    requiring the surfaces to settle recovers them."""
    reference = vmecpp.run(
        _single_grid().model_copy(update={"ftol_array": np.array([1.0e-16])}),
        verbose=False,
        max_threads=1,
    )
    axis_reference, iota_reference = _axis_and_iota(reference.wout)

    loose_input = _single_grid().model_copy(update={"ftol_array": np.array([1.0e-8])})
    loose = vmecpp.run(loose_input, verbose=False, max_threads=1)
    axis_loose, iota_loose = _axis_and_iota(loose.wout)

    settled = vmecpp.run(
        loose_input.model_copy(update={"geometry_tolerance": 1.0e-5}),
        verbose=False,
        max_threads=1,
    )
    axis_settled, iota_settled = _axis_and_iota(settled.wout)

    # the residual alone stops with the axis a fraction of a millimetre out
    assert abs(axis_loose - axis_reference) > 1.0e-4
    assert abs(iota_loose - iota_reference) > 1.0e-4

    # and the same run, held until the surfaces stop moving, lands on it
    assert settled.wout.niter > loose.wout.niter
    assert abs(axis_settled - axis_reference) < 1.0e-6
    assert abs(iota_settled - iota_reference) < 1.0e-5


def test_a_tighter_tolerance_gets_closer():
    reference = vmecpp.run(
        _single_grid().model_copy(update={"ftol_array": np.array([1.0e-16])}),
        verbose=False,
        max_threads=1,
    )
    _, iota_reference = _axis_and_iota(reference.wout)

    errors = []
    for tolerance in (1.0e-5, 1.0e-6):
        output = vmecpp.run(
            _single_grid().model_copy(
                update={
                    "ftol_array": np.array([1.0e-8]),
                    "geometry_tolerance": tolerance,
                }
            ),
            verbose=False,
            max_threads=1,
        )
        errors.append(abs(_axis_and_iota(output.wout)[1] - iota_reference))
    assert errors[1] < errors[0]


def test_a_negative_tolerance_is_rejected():
    vmec_input = _single_grid().model_copy(update={"geometry_tolerance": -1.0})
    with pytest.raises(AttributeError, match="geometry_tolerance"):
        vmecpp.run(vmec_input, verbose=False, max_threads=1)


def test_the_tolerance_survives_a_round_trip(tmp_path):
    vmec_input = _single_grid().model_copy(update={"geometry_tolerance": 2.5e-6})
    path = tmp_path / "input.json"
    path.write_text(vmec_input.to_json())
    assert vmecpp.VmecInput.from_file(path).geometry_tolerance == 2.5e-6
