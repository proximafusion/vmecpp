# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the unpreconditioned internal-basis gradient.

VmecModel.evaluate(precondition=False) returns at the INVARIANT_RESIDUALS checkpoint, so
get_forces() yields the raw, unpreconditioned force: the gradient of VMEC's augmented
functional (MHD energy plus the spectral-condensation and lambda constraints) with
respect to the decomposed (internal-basis) state. This is the gradient an external
optimizer working in the internal basis needs.

The preconditioned force (precondition=True) is the native solver's search direction and
points in a different direction; both vanish at the converged equilibrium.
"""

from pathlib import Path

import numpy as np
import pytest

from vmecpp.cpp import _vmecpp  # type: ignore

SOLOVEV = Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"
CTH_LIKE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "data"
    / "cth_like_fixed_bdy.json"
)


def _model(ns: int = 11):
    indata = _vmecpp.VmecINDATA.from_file(str(SOLOVEV))
    return _vmecpp.VmecModel.create(indata, ns)


def test_raw_force_differs_from_preconditioned():
    m = _model()
    m.evaluate(2, 2, True)
    f_prec = np.asarray(m.get_forces(), float)
    m.evaluate(2, 2, False)
    f_raw = np.asarray(m.get_forces(), float)

    assert np.all(np.isfinite(f_raw))
    assert np.linalg.norm(f_raw) > 0.0
    # The preconditioner is a non-trivial metric: the two vectors are different
    # in direction, not just scale.
    cos = np.dot(f_prec, f_raw) / (np.linalg.norm(f_prec) * np.linalg.norm(f_raw))
    assert abs(cos) < 0.99


def test_raw_force_vanishes_at_equilibrium():
    m = _model()
    m.evaluate(2, 2, False)
    f_initial = np.linalg.norm(np.asarray(m.get_forces(), float))

    m.solve()
    m.evaluate(2, 2, False)
    f_converged = np.linalg.norm(np.asarray(m.get_forces(), float))

    # The augmented-functional gradient is the equilibrium residual: it drops by
    # many orders of magnitude once the native solver has converged.
    assert f_converged < 1e-6 * f_initial


def test_cold_start_is_excluded():
    # evaluate(1, 2) is the cold-start special case (forces initialised to 1.0);
    # the raw-gradient path uses iter1 >= 2, where the force is well defined.
    m = _model()
    m.evaluate(2, 2, False)
    assert np.all(np.isfinite(np.asarray(m.get_forces(), float)))


@pytest.fixture(scope="module")
def force_histories():
    indata = _vmecpp.VmecINDATA.from_file(str(CTH_LIKE))
    reference = _vmecpp.VmecModel.create(indata, 11)
    reference.solve()
    low_residual_state = np.asarray(reference.get_state(), float).copy()

    model = _vmecpp.VmecModel.create(indata, 11)
    high_residual_state = np.asarray(model.get_state(), float).copy()
    model.evaluate(2, 2, False)
    high_after_high = np.asarray(model.get_forces(), float).copy()

    model.set_state(np.ascontiguousarray(low_residual_state))
    model.evaluate(2, 2, False)
    low_after_high = np.asarray(model.get_forces(), float).copy()
    model.evaluate(2, 2, False)
    low_after_low = np.asarray(model.get_forces(), float).copy()

    model.set_state(np.ascontiguousarray(high_residual_state))
    model.evaluate(2, 2, False)
    high_after_low = np.asarray(model.get_forces(), float).copy()

    return high_after_high, high_after_low, low_after_high, low_after_low


def test_raw_force_is_independent_of_high_residual_history(force_histories):
    _, _, low_after_high, low_after_low = force_histories
    np.testing.assert_array_equal(low_after_high, low_after_low)


def test_raw_force_is_independent_of_low_residual_history(force_histories):
    high_after_high, high_after_low, _, _ = force_histories
    np.testing.assert_array_equal(high_after_high, high_after_low)
