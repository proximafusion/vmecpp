# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Adjoint sensitivity of a converged VMEC++ equilibrium to its boundary.

A fixed-boundary equilibrium satisfies the interior force balance F_I(x) = 0,
where x is the decomposed internal-basis state and F is the gradient of VMEC's
augmented functional. The outermost flux surface (the boundary) is the last
radial block of the state and is held fixed during the solve. For a scalar
objective J(x), the sensitivity to the boundary degrees of freedom follows from
the implicit function theorem:

    dJ/dx_B = dJ/dx_B|_x - (dF_I/dx_B)^T lambda,   H_II lambda = dJ/dx_I,

with H = dF/dx the (symmetric) Hessian of the augmented functional. Every
operator is matrix-free and already exposed by VmecModel: the Hessian-vector
product (``hessian_vector_product``) and the preconditioner
(``apply_preconditioner``), used to solve the adjoint system. Only one Hessian
solve is needed for the full boundary gradient, versus one equilibrium re-solve
per boundary degree of freedom for finite differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator, gmres

from vmecpp.cpp import _vmecpp  # type: ignore

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


def _interior_operators(model, x, interior):
    # The caller must set the base state to x and assemble the preconditioner
    # (evaluate(2, 2, True)) before using these. hessian_vector_product uses the
    # current state as its base point and restores it, so no per-matvec state
    # update is needed: that keeps each Hessian matvec at two force evaluations.
    n = x.size
    ni = interior.size

    def hii(vi):
        v = np.zeros(n)
        v[interior] = vi
        return np.asarray(model.hessian_vector_product(np.ascontiguousarray(v)), float)[
            interior
        ]

    def mii(bi):
        v = np.zeros(n)
        v[interior] = bi
        return np.asarray(model.apply_preconditioner(np.ascontiguousarray(v)), float)[
            interior
        ]

    return (
        LinearOperator((ni, ni), matvec=hii),  # type: ignore[call-overload]
        LinearOperator((ni, ni), matvec=mii),  # type: ignore[call-overload]
    )


class _VmecPreconditioner(LinearOperator):
    """Adaptive VMEC preconditioner for SciPy's Newton-Krylov solver."""

    def __init__(self, model, x_template, interior):
        self._model = model
        self._x_template = x_template
        self._interior = interior
        self._x = x_template.copy()
        super().__init__(dtype=float, shape=(interior.size, interior.size))

    def update(self, x, _f):
        self._x = self._x_template.copy()
        self._x[self._interior] = x
        self._model.set_state(np.ascontiguousarray(self._x))
        self._model.evaluate(2, 2, True)

    def _matvec(self, x):
        v = np.zeros_like(self._x)
        v[self._interior] = x
        self._model.set_state(np.ascontiguousarray(self._x))
        return np.asarray(self._model.apply_preconditioner(v), float)[self._interior]


def solve_interior(model, x0, interior, boundary, x_boundary, tol=1e-10, max_newton=80):
    """Converge the interior to force balance with the boundary held fixed.

    Use SciPy's Newton-Krylov implementation with VMEC's adaptive preconditioner.

    SciPy owns the nonlinear iteration, inner Krylov solve, and line search. VMEC supplies
    the state-dependent preconditioner through SciPy's public ``inner_M`` interface.
    """
    x_template = np.asarray(x0, float).copy()
    x_template[boundary] = x_boundary

    def residual(xi):
        x = x_template.copy()
        x[interior] = xi
        return _raw_force(model, x)[interior]

    preconditioner = _VmecPreconditioner(model, x_template, interior)
    preconditioner.update(x_template[interior], np.zeros(interior.size))
    solution = root(
        residual,
        x_template[interior],
        method="krylov",
        options={
            "fatol": tol,
            "line_search": "armijo",
            "maxiter": max_newton,
            "jac_options": {
                "method": "lgmres",
                "inner_M": preconditioner,
                "inner_rtol": 1e-4,
                "inner_maxiter": 300,
            },
        },
    )

    x = x_template.copy()
    x[interior] = solution.x
    if not solution.success:
        error_message = (
            f"Interior solve failed: {solution.message}; "
            f"max|F|={np.max(np.abs(solution.fun)):.3e}"
        )
        raise RuntimeError(error_message)
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


def boundary_gradient(model, x_star, interior, boundary, objective, h=1e-6):
    """Adjoint gradient dJ/dx_B at the converged equilibrium x_star."""
    n = x_star.size
    dj = objective_state_gradient(model, x_star, objective, h)
    model.set_state(np.ascontiguousarray(x_star))
    model.evaluate(2, 2, True)  # assemble preconditioner + set base state to x_star
    h_op, m_op = _interior_operators(model, x_star, interior)
    lam, _ = gmres(h_op, dj[interior], M=m_op, rtol=1e-6, restart=100, maxiter=30)
    embedded = np.zeros(n)
    embedded[interior] = lam
    model.set_state(np.ascontiguousarray(x_star))
    model.evaluate(2, 2, False)
    coupling = np.asarray(
        model.hessian_vector_product(np.ascontiguousarray(embedded)), float
    )[boundary]
    return dj[boundary] - coupling


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
