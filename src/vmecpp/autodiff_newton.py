"""Interim Newton polishing with the exact, unpreconditioned force Jacobian.

Active entries use the existing random structural probes until WP3 classification is
available. This implementation does not establish the N1 acceptance criterion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vmecpp import autodiff
from vmecpp._adjoint_contracts import Factorization, NewtonResult
from vmecpp.autodiff_diagnostics import raw_force


def _validate_model(model: Any) -> None:
    if model.lasym or getattr(model, "lfreeb", False):
        message = "Newton polishing requires symmetric fixed-boundary models"
        raise ValueError(message)
    if not model.always_fix_m1_gauge:
        message = "Newton polishing requires always_fix_m1_gauge=True"
        raise ValueError(message)
    if getattr(model, "lforbal", False) or not model.has_exact_force_jacobian:
        message = "Newton polishing requires Enzyme exact derivatives and lforbal=false"
        raise RuntimeError(message)
    # VmecModel does not expose lfreeb; use its native fixed-boundary guard.
    model.geometry_state_vjp(
        np.zeros(12 * model.ns * model.mpol * (model.ntor + 1)), np.zeros(0)
    )


def _factorize(model: Any, active: np.ndarray) -> Factorization:
    matrix = autodiff._assemble_block_tridiagonal(model, active)
    matrix.factorize()
    return matrix


def _residual(model: Any, state: np.ndarray) -> np.ndarray:
    force = raw_force(model)
    if force.shape != state.shape or not np.all(np.isfinite(force)):
        message = "Newton polishing requires a finite full-state raw force"
        raise RuntimeError(message)
    if not np.array_equal(model.get_state(), state):
        message = "raw force evaluation changed prescribed state entries"
        raise RuntimeError(message)
    return force


def newton_polish(model: Any, max_steps: int, tol: float) -> NewtonResult:
    """Reduce the active raw force norm with exact Newton steps and backtracking.

    The model is mutated to the last accepted state, including on derivative
    failures. Boundary, pinned gauge, and probe-excluded entries are unchanged.
    Exhaustion or failed backtracking returns ``converged=False``. Factors are
    rebuilt at the returned state, even if no step was accepted.
    """
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 0
    ):
        message = "max_steps must be a finite nonnegative integer"
        raise ValueError(message)
    if not np.isfinite(tol) or tol <= 0:
        message = "tol must be finite and positive"
        raise ValueError(message)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        message = "Newton polishing requires a finite one-dimensional state"
        raise ValueError(message)
    _validate_model(model)
    try:
        force = _residual(model, state)
        interior, _ = autodiff._interior_and_boundary(model)
        interior = np.setdiff1d(interior, autodiff._gauge_entries(model))
        active = autodiff._structural_nullfree_interior(model, interior)
        history = [float(np.linalg.norm(force[active]))]
        if not np.isfinite(history[0]):
            message = "Newton polishing requires a finite active raw force norm"
            raise RuntimeError(message)
        for _ in range(max_steps):
            if history[-1] <= tol:
                break
            factors = _factorize(model, active)
            step = -factors.solve_transpose(force[active])
            if not np.all(np.isfinite(step)):
                break
            accepted = False
            alpha = 1.0
            for _ in range(24):
                candidate = state.copy()
                candidate[active] += alpha * step
                if np.all(np.isfinite(candidate)):
                    try:
                        model.set_state(np.ascontiguousarray(candidate))
                        candidate_force = _residual(model, candidate)
                    except RuntimeError:
                        pass
                    else:
                        norm = float(np.linalg.norm(candidate_force[active]))
                        if np.isfinite(norm) and norm < history[-1]:
                            state = candidate
                            force = candidate_force
                            history.append(norm)
                            accepted = True
                            break
                model.set_state(np.ascontiguousarray(state))
                _residual(model, state)
                alpha *= 0.5
            if not accepted:
                break
        factors = _factorize(model, active)
        return NewtonResult(
            state=state.copy(),
            residual_history=np.asarray(history),
            factorization=factors,
            active=active.copy(),
            converged=history[-1] <= tol,
        )
    finally:
        model.set_state(np.ascontiguousarray(state))
        try:
            _residual(model, state)
        except Exception:
            model.set_state(np.ascontiguousarray(state))
            raise
