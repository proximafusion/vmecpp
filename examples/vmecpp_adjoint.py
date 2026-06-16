# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Sensitivity of a converged VMEC++ equilibrium to its boundary.

A fixed-boundary equilibrium satisfies the interior force balance F_I(x) = 0,
where x is the decomposed internal-basis state and F is VMEC's internal force.
The outermost flux surface (the boundary) is the last radial block of the state
and is held fixed during the solve. For a scalar objective J(x), the implicit
function theorem gives the boundary sensitivity from the equilibrium response
dx_I/dx_B = -H_II^{-1} H_IB, H = dF/dx:

    dJ/dx_Bj = dJ/dx_Bj|_x + dJ/dx_I . dx_I,   H_II dx_I = -H_IB e_j.

Two finite-difference-free forms are provided, both driven by the exact autodiff
Hessian-vector product and preconditioned by VMEC's approximate inverse Hessian
(``apply_preconditioner``):

* ``forward_boundary_gradient``: one Hessian solve per boundary DOF
  (``H_II dx_I = -H_IB e_j``). Works on any Enzyme build.
* ``adjoint_boundary_gradient``: one Hessian solve total, independent of the
  boundary DOF count. VMEC's force is a *scaled* gradient, so H = dF/dx is
  non-symmetric and the adjoint needs the transpose H^T, exposed by
  ``exact_hessian_vector_product_transpose``; the structural null space is
  deflated once (state-independent) and the symmetric preconditioner serves H^T
  as well as H.

Both are far cheaper than the finite-difference reference, which re-solves the
equilibrium once per boundary DOF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

try:
    from vmecpp.cpp import _vmecpp
except ImportError:
    import _vmecpp

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"
)


def make_model(input_path: Path = DEFAULT_INPUT, ns: int = 11):
    return _vmecpp.VmecModel.create(_vmecpp.VmecINDATA.from_file(str(input_path)), ns)


def partition(model, ns: int):
    """Indices of the interior (free) and boundary (LCFS) state components."""
    k = model.mpol * (model.ntor + 1)
    n = np.asarray(model.get_state()).size
    per_span = ns * k
    n_span = n // per_span
    boundary = []
    for s in range(n_span):
        boundary.extend(range(s * per_span + (ns - 1) * k, s * per_span + ns * k))
    boundary = np.array(sorted(boundary))
    interior = np.setdiff1d(np.arange(n), boundary)
    return interior, boundary


def _raw_force(model, x):
    model.set_state(np.ascontiguousarray(x))
    model.evaluate(2, 2, False)
    return np.asarray(model.get_forces(), float)


def _interior_operators(model, x, interior, exact=False):
    # The caller must set the base state to x and assemble the preconditioner
    # (evaluate(2, 2, True)) before using these. With exact=True the analytic
    # autodiff HVP (no force evaluation per matvec) is used; otherwise the
    # finite-difference HVP (two force evaluations per matvec).
    n = x.size
    ni = interior.size
    hvp = model.exact_hessian_vector_product if exact else model.hessian_vector_product

    def hii(vi):
        v = np.zeros(n)
        v[interior] = vi
        return np.asarray(hvp(np.ascontiguousarray(v)), float)[interior]

    def mii(bi):
        v = np.zeros(n)
        v[interior] = bi
        return np.asarray(model.apply_preconditioner(np.ascontiguousarray(v)), float)[
            interior
        ]

    return (
        LinearOperator((ni, ni), matvec=hii),
        LinearOperator((ni, ni), matvec=mii),
    )


def solve_interior(model, x0, interior, boundary, x_boundary, tol=1e-10, max_newton=80):
    """Converge the interior to force balance with the boundary held fixed.

    Preconditioned Newton-Krylov on the interior residual with a backtracking line
    search; the line search is required for stiff 3D equilibria, where the full Newton
    step overshoots.
    """
    x = np.asarray(x0, float).copy()
    x[boundary] = x_boundary
    for _ in range(max_newton):
        f = _raw_force(model, x)
        norm0 = np.linalg.norm(f[interior])
        if norm0 < tol:
            break
        model.set_state(np.ascontiguousarray(x))
        model.evaluate(2, 2, True)  # assemble preconditioner + set base state
        h_op, m_op = _interior_operators(model, x, interior)
        dxi, _ = gmres(h_op, -f[interior], M=m_op, rtol=1e-4, maxiter=300)
        alpha = 1.0
        for _ in range(30):
            xt = x.copy()
            xt[interior] += alpha * dxi
            if np.linalg.norm(_raw_force(model, xt)[interior]) < norm0:
                break
            alpha *= 0.5
        else:
            break  # no decrease found; stop
        x[interior] += alpha * dxi
    return x


def objective_state_gradient(model, x, objective, h=1e-6):
    """Partial derivative dJ/dx at fixed state, by central finite differences."""
    n = x.size
    g = np.zeros(n)
    for i in range(n):
        xp = x.copy()
        xp[i] += h
        model.set_state(np.ascontiguousarray(xp))
        model.evaluate(2, 2, False)
        jp = objective(model)
        xm = x.copy()
        xm[i] -= h
        model.set_state(np.ascontiguousarray(xm))
        model.evaluate(2, 2, False)
        jm = objective(model)
        g[i] = (jp - jm) / (2 * h)
    return g


def boundary_gradient(
    model, x_star, interior, boundary, objective, h=1e-6, exact=False
):
    """dJ/dx_B at the converged equilibrium for a scalar objective.

    The equilibrium response is captured by forward sensitivities driven by the
    exact autodiff Hessian-vector product (no nonlinear re-solves). The state
    cotangent dJ/dx is taken by finite differences over the state here; pass an
    analytic one to ``forward_boundary_gradient`` to remove that step too (the QS
    objective does, via ``qs_boundary_gradient``).
    """
    dj = objective_state_gradient(model, x_star, objective, h)
    grad, _ = forward_boundary_gradient(model, x_star, interior, boundary, dj, exact)
    return grad


def finite_difference_boundary_gradient(
    model, x_star, interior, boundary, objective, dofs, h=1e-5
):
    """Reference gradient: re-solve the interior for each perturbed boundary DOF."""
    g = {}
    for j in dofs:
        xbp = x_star[boundary].copy()
        xbp[j] += h
        xp = solve_interior(model, x_star, interior, boundary, xbp)
        model.set_state(np.ascontiguousarray(xp))
        model.evaluate(2, 2, False)
        jp = objective(model)
        xbm = x_star[boundary].copy()
        xbm[j] -= h
        xm = solve_interior(model, x_star, interior, boundary, xbm)
        model.set_state(np.ascontiguousarray(xm))
        model.evaluate(2, 2, False)
        jm = objective(model)
        g[j] = (jp - jm) / (2 * h)
    return g


def mhd_energy(model):
    return model.mhd_energy


def forward_boundary_gradient(
    model, x_star, interior, boundary, dj, exact=True, rtol=1e-8, maxiter=60
):
    """dJ/dx_B from forward equilibrium sensitivities, finite-difference-free.

    For each boundary DOF j the interior responds along the equilibrium manifold by dx_I
    = -H_II^{-1} H_IB e_j (implicit function theorem, F_I = 0), and dJ/dx_Bj =
    dJ/dx_Bj|_x + dJ/dx_I . dx_I. Every operator is the exact autodiff Hessian-vector
    product.

    The reverse (adjoint) form would need one solve total instead of one per boundary
    DOF, but it requires the transpose H^T: VMEC's force is a *scaled* gradient, so H =
    dF/dx is genuinely non-symmetric (H_BI != H_IB^T) and the naive reverse solve is
    wrong. The scaling cancels in the forward sensitivity (H_II dx = -H_IB e_j has the
    same solution as the symmetric system), so this form is exact. An exact O(1) adjoint
    needs a reverse-mode force VJP.
    """
    hvp = model.exact_hessian_vector_product if exact else model.hessian_vector_product
    n = x_star.size
    model.set_state(np.ascontiguousarray(x_star))
    model.evaluate(2, 2, True)  # assemble preconditioner at x_star
    h_op, m_op = _interior_operators(model, x_star, interior, exact)
    dj_i = dj[interior]
    grad = np.empty(boundary.size)
    failures = 0
    for col, j in enumerate(boundary):
        ej = np.zeros(n)
        ej[j] = 1.0
        hib = np.asarray(hvp(np.ascontiguousarray(ej)), float)[interior]
        dxi, info = gmres(h_op, -hib, M=m_op, rtol=rtol, restart=100, maxiter=maxiter)
        failures += int(info != 0)
        grad[col] = dj[j] + dj_i @ dxi
    return grad, failures


def structural_nullfree_interior(model, interior, n_probe=6, tol=1e-9, seed=0):
    """Interior DOFs that actually enter the force, i.e. not in the augmented Hessian's
    structural null space (state-independent gauge/parity modes). A DOF is kept when
    both its Hessian column and row are nonzero.

    Detected with a few random probes rather than one per column: a column i is
    zero iff (H^T v)[i] = H[:,i].v = 0 for random v, and a row i is zero iff
    (H v)[i] = 0, so O(n_probe) Hessian-vector products find every structural
    zero (a null column/row gives exactly zero for every probe). The set is
    state-independent; detect it once and reuse it across adjoint solves.
    """
    n = int(np.asarray(model.get_state()).size)
    rng = np.random.default_rng(seed)
    col = np.zeros(n)
    row = np.zeros(n)
    for _ in range(n_probe):
        v = np.ascontiguousarray(rng.standard_normal(n))
        col = np.maximum(
            col,
            np.abs(np.asarray(model.exact_hessian_vector_product_transpose(v), float)),
        )
        row = np.maximum(
            row, np.abs(np.asarray(model.exact_hessian_vector_product(v), float))
        )
    thr = tol * max(col.max(), row.max(), 1.0)
    return np.array([i for i in interior if col[i] > thr and row[i] > thr])


def adjoint_boundary_gradient(
    model, x_star, interior, boundary, dj, keep=None, rtol=1e-9, maxiter=400
):
    """dJ/dx_B from the reverse adjoint: one Hessian solve, O(1) in boundary DOF.

    Solves the transposed interior system H_II^T lambda = dJ/dx_I and forms
    dJ/dx_B = dJ/dx_B|_x - H_IB^T lambda. VMEC's force is a scaled gradient, so
    H = dF/dx is non-symmetric and the transpose H^T is required; it is provided
    exactly by ``exact_hessian_vector_product_transpose``. The augmented Hessian
    has a structural null space, deflated via ``keep`` (pass a cached set to keep
    the solve O(1); if None it is detected here at O(n_interior) cost). The
    transposed system is preconditioned by VMEC's approximate inverse Hessian,
    which is symmetric, so it preconditions H^T as well as H.
    """
    n = x_star.size
    model.set_state(np.ascontiguousarray(x_star))
    model.evaluate(2, 2, True)
    if keep is None:
        keep = structural_nullfree_interior(model, interior)
    nk = keep.size

    def ht(e):
        return np.asarray(
            model.exact_hessian_vector_product_transpose(np.ascontiguousarray(e)), float
        )

    def hii_t(v):
        e = np.zeros(n)
        e[keep] = v
        return ht(e)[keep]

    def mii(b):
        e = np.zeros(n)
        e[keep] = b
        return np.asarray(model.apply_preconditioner(np.ascontiguousarray(e)), float)[
            keep
        ]

    h_op = LinearOperator((nk, nk), matvec=hii_t)
    m_op = LinearOperator((nk, nk), matvec=mii)
    lam, info = gmres(h_op, dj[keep], M=m_op, rtol=rtol, restart=200, maxiter=maxiter)
    embedded = np.zeros(n)
    embedded[keep] = lam
    coupling = ht(embedded)[boundary]
    return dj[boundary] - coupling, info
