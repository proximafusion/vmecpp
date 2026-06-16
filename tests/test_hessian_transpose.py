# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The transposed exact Hessian-vector product is the exact transpose of H.

VMEC's internal force is a scaled gradient, so H = dF/dx is non-symmetric. The
implicit-function adjoint of a general objective therefore needs H^T, exposed by
``exact_hessian_vector_product_transpose``. This checks H^T against the assembled
Hessian and confirms the O(1) reverse adjoint agrees with the forward
sensitivities on a non-energy objective (where the asymmetry matters).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from vmecpp_adjoint import (
    adjoint_boundary_gradient,
    forward_boundary_gradient,
    make_model,
    partition,
    structural_nullfree_interior,
)

try:
    from vmecpp.cpp import _vmecpp
except ImportError:  # pragma: no cover
    import _vmecpp

_HAS_TRANSPOSE = hasattr(_vmecpp.VmecModel, "exact_hessian_vector_product_transpose")
needs_transpose = pytest.mark.skipif(
    not _HAS_TRANSPOSE, reason="requires an Enzyme build with the transposed HVP"
)


@needs_transpose
def test_transpose_matches_assembled_hessian():
    ns = 11
    model = make_model(ns=ns)
    model.solve()
    x = np.asarray(model.get_state(), float).copy()
    model.set_state(np.ascontiguousarray(x))
    model.evaluate(2, 2, True)
    n = x.size

    hessian = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        hessian[:, i] = np.asarray(
            model.exact_hessian_vector_product(np.ascontiguousarray(e)), float
        )

    rng = np.random.default_rng(0)
    for _ in range(3):
        w = rng.standard_normal(n)
        ht_w = np.asarray(
            model.exact_hessian_vector_product_transpose(np.ascontiguousarray(w)),
            float,
        )
        rel = np.linalg.norm(ht_w - hessian.T @ w) / np.linalg.norm(hessian.T @ w)
        assert rel < 1e-10


@needs_transpose
def test_reverse_adjoint_matches_forward_sensitivities():
    # On a non-energy objective the interior cotangent is nonzero, so a wrong
    # (symmetric) adjoint would disagree. The O(1) reverse adjoint must match the
    # O(n_boundary) forward sensitivities.
    ns = 11
    model = make_model(ns=ns)
    model.solve()
    x = np.asarray(model.get_state(), float).copy()
    interior, boundary = partition(model, ns)
    model.set_state(np.ascontiguousarray(x))
    model.evaluate(2, 2, True)
    keep = structural_nullfree_interior(model, interior)

    rng = np.random.default_rng(1)
    dj = rng.standard_normal(x.size)
    g_reverse, info = adjoint_boundary_gradient(
        model, x, interior, boundary, dj, keep=keep
    )
    g_forward, _ = forward_boundary_gradient(
        model, x, interior, boundary, dj, exact=True
    )
    assert info == 0
    rel = np.linalg.norm(g_reverse - g_forward) / np.linalg.norm(g_forward)
    assert rel < 1e-5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
