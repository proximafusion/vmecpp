# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Drive a VMEC++ equilibrium from outside with general-purpose optimizers.

VMEC's equilibrium is the stationary point of its augmented functional (MHD
energy plus the spectral-condensation and lambda constraints). The gradient of
that functional in the decomposed internal basis is the raw, unpreconditioned
force exposed by ``VmecModel.evaluate(precondition=False)``. Finding the
equilibrium is therefore the root problem F(x) = 0, which gradient- and
Hessian-based solvers can attack while reusing VMEC++'s forward model, its
preconditioner (``apply_preconditioner``, VMEC's approximate inverse Hessian),
and its Hessian-vector product (``hessian_vector_product``, a directional
derivative of the analytic force computed inside VMEC++).

This module wires that residual to several solvers and is shared by the
benchmark ``main`` below and by the tests:

* preconditioned descent (VMEC's own update direction),
* Jacobian-free Newton-Krylov, plain and preconditioned, and
* a true Newton-Krylov driven by VMEC++'s own Hessian-vector product.

All converge to the same equilibrium as the native solver. Force evaluations are
counted inside VMEC++ (``force_eval_count``) so the comparison is fair across
methods, including the evaluations hidden in Hessian-vector products.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import LinearOperator, lgmres

from vmecpp.cpp import _vmecpp  # type: ignore[import]

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"
)


_BAD_JACOBIAN = 2  # RestartReason::BAD_JACOBIAN (flow_control.h)


def _ensure_valid_initial_jacobian(model, max_reguess: int = 2) -> None:
    """Reguess the magnetic axis until the initial Jacobian is non-singular.

    Inputs that ship no axis (e.g. cma.json, with raxis/zaxis all zero) start
    from a degenerate geometry, so the bare forward model returns at the
    BAD_JACOBIAN checkpoint with zero force. The native solver reguesses the axis
    on the first iterate (vmec.cc SolveEquilibriumLoop); mirror that here via
    reinitialize() so a cold start from any valid INDATA has a defined gradient.
    """
    for _ in range(max_reguess):
        model.evaluate(2, 2, False)
        if model.restart_reason != _BAD_JACOBIAN:
            return
        model.reinitialize()
    model.evaluate(2, 2, False)


def make_model(input_path: Path = DEFAULT_INPUT, ns: int = 11):
    indata = _vmecpp.VmecINDATA.from_file(str(input_path))
    model = _vmecpp.VmecModel.create(indata, ns)
    _ensure_valid_initial_jacobian(model)
    return model


def residual(model):
    """Return F(x) = raw internal-basis force; F(x) = 0 at equilibrium."""

    def F(x):
        model.set_state(np.ascontiguousarray(x, dtype=float))
        model.evaluate(2, 2, False)
        return np.asarray(model.get_forces(), dtype=float)

    return F


@dataclass
class Result:
    name: str
    force_evals: int
    outer_iters: int
    seconds: float
    residual_norm: float
    energy: float


def reference_equilibrium(input_path: Path = DEFAULT_INPUT, ns: int = 11):
    model = make_model(input_path, ns)
    model.solve()
    model.evaluate(2, 2, False)
    return np.asarray(model.get_state(), float), model.mhd_energy


def _finish(model, name, x, outer_iters, t0):
    model.set_state(np.ascontiguousarray(x))
    model.evaluate(2, 2, False)
    return x, Result(
        name,
        model.force_eval_count,
        outer_iters,
        time.perf_counter() - t0,
        float(np.linalg.norm(np.asarray(model.get_forces(), float))),
        model.mhd_energy,
    )


def solve_preconditioned_descent(
    input_path=DEFAULT_INPUT, ns=11, tol=1e-9, delt=0.9, momentum=0.5, max_iter=20000
):
    model = make_model(input_path, ns)
    F = residual(model)
    x = np.asarray(model.get_state(), float).copy()
    v = np.zeros_like(x)
    model.reset_force_eval_count()
    it = 0
    t0 = time.perf_counter()
    for _ in range(max_iter):
        if np.linalg.norm(F(x)) < tol:
            break
        it += 1
        model.set_state(np.ascontiguousarray(x))
        model.evaluate(2, 2, True)  # preconditioned search direction
        fprec = np.asarray(model.get_forces(), float)
        v = momentum * v + delt * fprec
        x = x + delt * v
    return _finish(model, "preconditioned descent", x, it, t0)


def solve_newton_krylov(
    input_path=DEFAULT_INPUT, ns=11, tol=1e-9, max_iter=300, preconditioned=False
):
    model = make_model(input_path, ns)
    F = residual(model)
    x0 = np.asarray(model.get_state(), float)
    inner_m = None
    model.reset_force_eval_count()
    if preconditioned:
        # Assemble VMEC's preconditioner at x0 and use it, frozen, as the inner
        # Krylov preconditioner. M^-1 approximates the inverse Hessian and is
        # state-invariant once assembled.
        model.evaluate(2, 2, True)
        n_dof = x0.size
        inner_m = LinearOperator(  # type: ignore[call-overload]
            (n_dof, n_dof),
            matvec=lambda b: np.asarray(  # type: ignore[call-overload]
                model.apply_preconditioner(np.ascontiguousarray(b)), float
            ),
        )
    t0 = time.perf_counter()
    x = newton_krylov(
        F, x0, f_tol=tol, maxiter=max_iter, method="lgmres", inner_M=inner_m
    )
    name = (
        "Newton-Krylov (preconditioned)" if preconditioned else "Newton-Krylov (JFNK)"
    )
    return _finish(model, name, x, 0, t0)


def solve_newton_krylov_preconditioned(input_path=DEFAULT_INPUT, ns=11, tol=1e-9):
    return solve_newton_krylov(input_path, ns, tol, preconditioned=True)


def solve_newton_hvp(input_path=DEFAULT_INPUT, ns=11, tol=1e-9, max_newton=80):
    """Globalized Newton-Krylov using VMEC++'s finite-difference Hessian-vector product.

    Each Newton step solves H dx = -F with GMRES preconditioned by M^-1 (VMEC's
    approximate inverse Hessian), with Eisenstat-Walker adaptive inner forcing and a
    backtracking line search. H v is hessian_vector_product (a central difference of the
    analytic force; two force evaluations per matvec). Same solver as
    solve_newton_exact_hvp but with the FD HVP, for a like-for-like comparison of the
    HVP backends.
    """
    model = make_model(input_path, ns)
    F = residual(model)
    x = np.asarray(model.get_state(), float).copy()
    n_dof = x.size
    model.reset_force_eval_count()
    t0 = time.perf_counter()
    it = 0
    prev_norm = None
    eta = 0.5
    for _ in range(max_newton):
        fk = F(x)
        norm0 = np.linalg.norm(fk)
        if norm0 < tol:
            break
        it += 1
        if prev_norm is not None:
            eta = min(0.5, max(1e-4, 0.9 * (norm0 / prev_norm) ** 2))
        prev_norm = norm0
        model.set_state(np.ascontiguousarray(x))
        model.evaluate(2, 2, True)  # assemble M at the current iterate
        h_op = LinearOperator(  # type: ignore[call-overload]
            (n_dof, n_dof),
            matvec=lambda v: np.asarray(  # type: ignore[call-overload]
                model.hessian_vector_product(np.ascontiguousarray(v)), float
            ),
        )
        m_op = LinearOperator(  # type: ignore[call-overload]
            (n_dof, n_dof),
            matvec=lambda b: np.asarray(  # type: ignore[call-overload]
                model.apply_preconditioner(np.ascontiguousarray(b)), float
            ),
        )
        dx, _ = lgmres(h_op, -fk, M=m_op, rtol=eta, maxiter=200)
        # Backtracking line search: accept the largest step that reduces ||F||.
        alpha = 1.0
        for _ in range(30):
            if np.linalg.norm(F(x + alpha * dx)) < norm0:
                break
            alpha *= 0.5
        else:
            break  # no decrease found; stop
        x = x + alpha * dx
    return _finish(model, "Newton (VMEC++ HVP + M^-1)", x, it, t0)


def solve_newton_exact_hvp(input_path=DEFAULT_INPUT, ns=11, tol=1e-9, max_newton=80):
    """Globalized Newton-Krylov using VMEC++'s exact autodiff Hessian-vector product
    (``exact_hessian_vector_product``, one Enzyme forward pass) instead of the finite-
    difference HVP. Requires an Enzyme-enabled build; raises AttributeError otherwise.

    The inner GMRES tolerance is set adaptively (Eisenstat-Walker forcing): the
    augmented-Lagrangian Hessian is indefinite, so solving it tightly every step wastes
    hundreds of matvecs early on. Loose-early/tight-late forcing cuts the total matvec
    count several-fold while preserving the asymptotic convergence, which is what
    dominates wall-clock (each matvec is cheap but there are many).
    """
    model = make_model(input_path, ns)
    # The exact HVP freezes the constraint multiplier tcon (it depends on the
    # preconditioner, not just the geometry). Freeze it in the raw force too so
    # the residual and its exact Jacobian are one consistent map; otherwise the
    # HVP drifts from the force on stellarators where the constraint force is
    # significant. The unfrozen evaluation here populates tcon before freezing.
    model.evaluate(2, 2, True)
    model.set_freeze_constraint_multiplier(True)
    F = residual(model)
    x = np.asarray(model.get_state(), float).copy()
    n_dof = x.size
    model.reset_force_eval_count()
    t0 = time.perf_counter()
    it = 0
    prev_norm = None
    eta = 0.5
    for _ in range(max_newton):
        fk = F(x)
        norm0 = np.linalg.norm(fk)
        if norm0 < tol:
            break
        it += 1
        # Eisenstat-Walker choice 2 for the inner forcing term.
        if prev_norm is not None:
            eta = min(0.5, max(1e-4, 0.9 * (norm0 / prev_norm) ** 2))
        prev_norm = norm0
        model.set_state(np.ascontiguousarray(x))
        model.evaluate(2, 2, True)
        h_op = LinearOperator(  # type: ignore[call-overload]
            (n_dof, n_dof),
            matvec=lambda v: np.asarray(  # type: ignore[call-overload]
                model.exact_hessian_vector_product(np.ascontiguousarray(v)),
                float,
            ),
        )
        m_op = LinearOperator(  # type: ignore[call-overload]
            (n_dof, n_dof),
            matvec=lambda b: np.asarray(  # type: ignore[call-overload]
                model.apply_preconditioner(np.ascontiguousarray(b)), float
            ),
        )
        dx, _ = lgmres(h_op, -fk, M=m_op, rtol=eta, maxiter=200)
        alpha = 1.0
        for _ in range(30):
            if np.linalg.norm(F(x + alpha * dx)) < norm0:
                break
            alpha *= 0.5
        else:
            break
        x = x + alpha * dx
    return _finish(model, "Newton (exact autodiff HVP + M^-1)", x, it, t0)


ALL_SOLVERS = (
    solve_preconditioned_descent,
    solve_newton_krylov,
    solve_newton_krylov_preconditioned,
    solve_newton_hvp,
)


def main():
    _, w_star = reference_equilibrium()
    print(f"reference equilibrium (native solve): W = {w_star:.8e}\n")
    print(
        f"{'optimizer':32s} {'F-evals':>8s} {'iters':>6s} {'time[s]':>8s} "
        f"{'||F||':>10s} {'dW vs ref':>10s}"
    )
    for solver in ALL_SOLVERS:
        r = solver()[1]
        print(
            f"{r.name:32s} {r.force_evals:8d} {r.outer_iters:6d} {r.seconds:8.2f} "
            f"{r.residual_norm:10.1e} {abs(r.energy - w_star):10.1e}"
        )


if __name__ == "__main__":
    main()
