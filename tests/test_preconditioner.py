# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VmecModel.apply_preconditioner exposes VMEC's preconditioner as an operator.

The preconditioner M^-1 is VMEC's hand-built approximate inverse Hessian. The native
solver applies it to the raw force to get its search direction, so
apply_preconditioner(raw force) must equal the preconditioned force exactly. The
operator is linear and, once assembled (via evaluate(precondition=True)), does not
depend on the current state, so it can be reused as a frozen preconditioner for
Krylov/quasi-Newton solvers.
"""

from pathlib import Path

import numpy as np

try:
    from vmecpp.cpp import _vmecpp
except ImportError:
    import _vmecpp

SOLOVEV = Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"


def _model(ns: int = 11):
    return _vmecpp.VmecModel.create(_vmecpp.VmecINDATA.from_file(str(SOLOVEV)), ns)


def test_preconditioner_matches_native_search_direction():
    m = _model()
    m.evaluate(2, 2, True)  # assemble preconditioner + preconditioned force
    f_prec = np.asarray(m.get_forces(), float)
    m.evaluate(2, 2, False)  # raw force (does not reassemble)
    f_raw = np.asarray(m.get_forces(), float)
    minv_fraw = np.asarray(m.apply_preconditioner(f_raw), float)
    assert np.linalg.norm(minv_fraw - f_prec) <= 1e-12 * np.linalg.norm(f_prec)


def test_preconditioner_is_linear_and_finite():
    m = _model()
    m.evaluate(2, 2, True)
    rng = np.random.default_rng(0)
    v = np.ascontiguousarray(rng.standard_normal(np.asarray(m.get_state()).size))
    mv = np.asarray(m.apply_preconditioner(v), float)
    m2v = np.asarray(m.apply_preconditioner(np.ascontiguousarray(2.0 * v)), float)
    assert np.all(np.isfinite(mv))
    assert np.linalg.norm(m2v - 2.0 * mv) <= 1e-12 * np.linalg.norm(mv)


def test_preconditioner_state_invariant_after_assembly():
    m = _model()
    m.evaluate(2, 2, True)
    rng = np.random.default_rng(1)
    x = np.asarray(m.get_state(), float)
    v = np.ascontiguousarray(rng.standard_normal(x.size))
    mv0 = np.asarray(m.apply_preconditioner(v), float)
    # Move to a different state and raw-evaluate (no reassembly).
    m.set_state(np.ascontiguousarray(x + 0.01 * rng.standard_normal(x.size)))
    m.evaluate(2, 2, False)
    mv1 = np.asarray(m.apply_preconditioner(v), float)
    assert np.linalg.norm(mv1 - mv0) <= 1e-12 * np.linalg.norm(mv0)


CTH_LIKE = SOLOVEV.parent / "cth_like_fixed_bdy.json"


def test_checkerboard_lambda_preconditioner_inverts_m0_lambda_block():
    # The m = 0 lambda preconditioner is P = c (M + floor D)^-1 with M the radial m = 0
    # lambda block of the force Jacobian H (c M^-1 = -scale H^-1) and P_diag = c D^-1
    # the diagonal preconditioner, so P (-H / scale + floor P_diag^-1) = I on every
    # row, the axis and LCFS rows included.
    indata = _vmecpp.VmecINDATA.from_file(str(CTH_LIKE))
    floor = 1e-3
    indata.lambda_precondition_checkerboard_terms = floor
    ns = 15
    m = _vmecpp.VmecModel.create(indata, ns)
    indata.lambda_precondition_checkerboard_terms = -1.0
    diag = _vmecpp.VmecModel.create(indata, ns)
    m.evaluate(2, 2, True)
    diag.evaluate(2, 2, True)
    scale = indata.lambda_preconditioner_scale
    size = np.asarray(m.get_state()).size
    lcs = np.arange(size).reshape(6, ns, indata.mpol, indata.ntor + 1)[5]
    for n in range(1, indata.ntor + 1):
        cols = lcs[1:, 0, n]
        k = np.zeros((cols.size, cols.size))
        for i, c in enumerate(cols):
            v = np.zeros(size)
            v[c] = 1.0
            d = np.asarray(diag.apply_preconditioner(v), float)[c]
            hv = np.asarray(m.hessian_vector_product(v, 1e-7), float)
            k[:, i] = np.asarray(m.apply_preconditioner(-hv / scale + floor * v / d))[
                cols
            ]
        np.testing.assert_allclose(k, np.eye(cols.size), atol=3e-2)
