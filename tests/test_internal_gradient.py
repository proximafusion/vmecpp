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

from vmecpp.cpp import _vmecpp  # type: ignore

SOLOVEV = Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"


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
