"""WP7 residual corrections using exact first derivatives and directional differences.

Option B retains the objective curvature through a reevaluated full-state gradient.
Classification is the existing structural-probe approximation, not WP3's analytic
classification. The model state and raw force are restored after derivative work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np
import scipy.linalg

from vmecpp import autodiff, geometry
from vmecpp._adjoint_contracts import ImplicitVjpDiagnostics
from vmecpp.autodiff_newton import _residual, _validate_model


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).copy()
    if vector.shape != shape or not np.all(np.isfinite(vector)):
        message = f"{name} must be finite with full-state shape {shape}"
        raise ValueError(message)
    return vector


def _require_x64() -> None:
    if not jax.config.x64_enabled:
        message = "Geometry objective state gradients require jax_enable_x64=True"
        raise ValueError(message)


def geometry_objective_state_gradient(
    model: Any, objective: Callable[[geometry.Geometry], jax.Array]
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a reevaluated state-gradient callback for a scalar Geometry objective.

    JAX differentiates ``objective(geometry.from_cpp(model.get_geometry()))``.
    The native geometry VJP maps all twelve coefficient blocks and the poloidal
    flux cotangent into the full scaled state. Toroidal flux and objective
    parameters are fixed. For Mercier, close over a VmecInput with lbsubs=True
    and use autodiff_wout.wout_quantities on the supplied Geometry, not a cached
    wout snapshot. Nonzero ncurr=1 flux cotangents require Enzyme.

    JAX x64 must be enabled before constructing or using the callback; the
    helper raises rather than changing global precision configuration.
    The callback refreshes candidate profiles through the raw-force evaluation,
    then restores caller state and raw force on all exits. It requires the
    pinned-gauge force definition, like corrected_vjp.
    """
    _require_x64()
    if not callable(objective):
        message = "objective must be a callable scalar Geometry objective"
        raise ValueError(message)
    value_and_gradient = jax.value_and_grad(objective)

    def state_gradient(state: np.ndarray) -> np.ndarray:
        _require_x64()
        original = np.asarray(model.get_state(), dtype=np.float64).copy()
        candidate = _finite_vector(state, original.shape, "objective state")
        try:
            model.set_state(np.ascontiguousarray(candidate))
            _residual(model, candidate)
            value, cotangent = value_and_gradient(
                geometry.from_cpp(model.get_geometry())
            )
            if not np.isfinite(float(value)):
                message = "Geometry objective value must be finite"
                raise ValueError(message)
            coefficient_bar = np.concatenate(
                [
                    np.asarray(getattr(cotangent, name), dtype=np.float64).ravel()
                    for name in autodiff._GEOMETRY_COEFFICIENTS
                ]
            )
            poloidal_flux_bar = np.asarray(cotangent.poloidal_flux, dtype=np.float64)
            if not np.all(np.isfinite(coefficient_bar)) or not np.all(
                np.isfinite(poloidal_flux_bar)
            ):
                message = "Geometry objective cotangent must be finite"
                raise ValueError(message)
            return _finite_vector(
                model.geometry_state_vjp(
                    np.ascontiguousarray(coefficient_bar),
                    np.ascontiguousarray(poloidal_flux_bar),
                ),
                candidate.shape,
                "Geometry objective state gradient",
            )
        finally:
            model.set_state(np.ascontiguousarray(original))
            try:
                _residual(model, original)
            finally:
                model.set_state(np.ascontiguousarray(original))

    return state_gradient


def _product(model: Any, vector: np.ndarray, *, transpose: bool) -> np.ndarray:
    operator = (
        model.exact_hessian_vector_product_transpose
        if transpose
        else model.exact_hessian_vector_product
    )
    return _finite_vector(
        operator(np.ascontiguousarray(vector)), vector.shape, "exact force product"
    )


def _relative_residual(product: np.ndarray, rhs: np.ndarray) -> float:
    error = float(np.linalg.norm(product - rhs))
    scale = max(float(np.linalg.norm(rhs)), 1.0e-300)
    relative = error / scale
    if not np.isfinite(relative) or relative > 1.0e-6:
        message = f"WP7 linear solve failed: relative residual {relative:.3e}"
        raise RuntimeError(message)
    return relative


def _norm(vector: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm):
        message = "WP7 requires finite vector norms"
        raise ValueError(message)
    return norm


def _block_rcond(matrix: Any) -> float:
    """Estimate reciprocal 1-norm condition numbers of factored Schur blocks."""
    estimates = []
    for j, members in enumerate(matrix.members):
        if not members.size:
            continue
        lu = matrix._block(matrix.diagonal, j, j)
        lower = np.tril(lu, -1) + np.eye(members.size)
        upper = np.triu(lu)
        # Row pivoting leaves the 1-norm unchanged.
        norm = float(np.linalg.norm(lower @ upper, 1))
        gecon: Any = scipy.linalg.lapack.get_lapack_funcs("gecon", (lu,))
        rcond, info = gecon(lu, norm)
        if info != 0 or not np.isfinite(rcond) or rcond <= 0:
            message = "WP7 singular or nonfinite Schur complement"
            raise RuntimeError(message)
        estimates.append(float(rcond))
    if not estimates:
        message = "WP7 requires a nonempty active force operator"
        raise RuntimeError(message)
    return min(estimates)


def _probe_classes(
    model: Any, interior: np.ndarray, active: np.ndarray, gauge: np.ndarray
) -> dict[str, np.ndarray]:
    state_size = model.get_state().size
    column = np.zeros(state_size)
    row = np.zeros(state_size)
    generator = np.random.default_rng(0)
    for _ in range(6):
        probe = generator.standard_normal(state_size)
        column = np.maximum(column, np.abs(_product(model, probe, transpose=True)))
        row = np.maximum(row, np.abs(_product(model, probe, transpose=False)))
    threshold = 1.0e-9 * max(column.max(), row.max(), 1.0)
    excluded = np.setdiff1d(interior, np.concatenate([active, gauge]))
    return {
        "active": active,
        "pinned": np.sort(
            np.concatenate(
                [
                    gauge,
                    excluded[
                        (row[excluded] <= threshold) & (column[excluded] > threshold)
                    ],
                ]
            )
        ),
        "null": excluded[
            (row[excluded] <= threshold) & (column[excluded] <= threshold)
        ],
        "zero_column": excluded[row[excluded] > threshold],
    }


def corrected_vjp(
    model: Any,
    x_bar: np.ndarray,
    method: str,
    *,
    objective_state_gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    directional_step: float = 1.0e-3,
) -> tuple[np.ndarray, float, ImplicitVjpDiagnostics]:
    """Return Option B boundary gradient, additive value correction, and diagnostics.

    ``objective_state_gradient`` is required for B and must reevaluate J_x in the
    flat scaled state layout, including prescribed entries. Its base gradient
    must match ``x_bar``. Profiles and boundary parameters are held fixed; all
    boundary dependence must enter through state entries. The callback receives
    a copy and must not change model parameters.

    With H = F_x on the active set, delta = -H^-1 F, lambda = H^-T J_x,
    L_xx delta = D_x(J_x - F_x^T lambda) delta, and lambda_dot =
    H^-T L_xx delta. Central differences at x +/- h delta hold lambda fixed.
    ``directional_step`` requests h; the displacement is enlarged to at least
    eps^(1/3) max(1, ||x_active||_inf). Differences are then scaled by
    ||delta||_inf / displacement, retaining a correction proportional to delta
    rather than introducing a finite-difference floor near convergence.
    The returned gradient is g + D_x g delta - F_b^T lambda_dot, with the
    pinned-gauge chain included in every boundary pullback. It is a local
    first-order residual correction, not a converged gradient or an N1 claim.
    Option A is unsupported because it requires objective reevaluation.
    """
    if method == "A":
        message = "WP7 Option A requires objective reevaluation and is unsupported"
        raise NotImplementedError(message)
    if method != "B":
        message = "method must be 'A' or 'B'"
        raise ValueError(message)
    if not callable(objective_state_gradient):
        message = "Option B requires objective_state_gradient; x_bar alone lacks J_xx"
        raise ValueError(message)
    if not np.isfinite(directional_step) or directional_step <= 0:
        message = "directional_step must be finite and positive"
        raise ValueError(message)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        message = "WP7 requires a finite one-dimensional state"
        raise ValueError(message)
    x_bar = _finite_vector(x_bar, state.shape, "x_bar")
    _norm(x_bar)
    _validate_model(model)

    def objective_gradient(candidate: np.ndarray) -> np.ndarray:
        return _finite_vector(
            objective_state_gradient(candidate.copy()),
            state.shape,
            "objective_state_gradient",
        )

    try:
        force = _residual(model, state)
        _norm(force)
        base_gradient = objective_gradient(state)
        if not np.allclose(base_gradient, x_bar, rtol=1.0e-8, atol=1.0e-12):
            message = "objective_state_gradient at base state disagrees with x_bar"
            raise ValueError(message)
        model.set_state(np.ascontiguousarray(state))
        interior, boundary = autodiff._interior_and_boundary(model)
        gauge = autodiff._gauge_entries(model)
        active = autodiff._structural_nullfree_interior(
            model, np.setdiff1d(interior, gauge)
        )
        if not active.size:
            message = "WP7 requires a nonempty active force operator"
            raise RuntimeError(message)
        classes = _probe_classes(model, interior, active, gauge)
        unknown_pinned = np.setdiff1d(classes["pinned"], gauge)
        if unknown_pinned.size or classes["zero_column"].size:
            message = "WP7 requires prescribed gauge pins and no zero-column equations"
            raise RuntimeError(message)
        prescribed = np.concatenate([boundary, gauge])
        matrix = autodiff._assemble_block_tridiagonal(model, active)
        matrix.factorize()
        rcond = _block_rcond(matrix)
        delta = np.zeros_like(state)
        delta[active] = matrix.solve_transpose(-force[active])
        delta = _finite_vector(delta, state.shape, "Newton direction")
        newton_residual = _relative_residual(
            _product(model, delta, transpose=False)[active], -force[active]
        )
        adjoint = np.zeros_like(state)
        adjoint[active] = matrix.solve(x_bar[active])
        adjoint = _finite_vector(adjoint, state.shape, "adjoint")
        transposed = _product(model, adjoint, transpose=True)
        solve_residual = _relative_residual(transposed[active], x_bar[active])

        def pullback(cotangent: np.ndarray) -> np.ndarray:
            selected = np.zeros_like(state)
            selected[prescribed] = cotangent[prescribed]
            selected = autodiff._pull_gauge_back_to_boundary(model, selected, gauge)
            result = autodiff._boundary_from_state_vjp(model, selected)
            if not np.all(np.isfinite(result)):
                message = "WP7 boundary pullback is nonfinite"
                raise RuntimeError(message)
            return result

        gradient = pullback(x_bar - transposed)
        weighted_residual = float(adjoint @ force)
        if not np.isfinite(weighted_residual):
            message = "WP7 adjoint-weighted residual is nonfinite"
            raise RuntimeError(message)
        lxx_delta = np.zeros_like(state)
        directional_gradient = np.zeros_like(gradient)
        delta_scale = float(np.max(np.abs(delta[active])))
        if delta_scale > 0.0:
            state_scale = max(1.0, float(np.max(np.abs(state[active]))))
            displacement = max(
                float(directional_step) * delta_scale,
                float(np.cbrt(np.finfo(np.float64).eps)) * state_scale,
            )
            direction = delta / delta_scale
            lagrangian_gradients = []
            boundary_gradients = []
            for sign in (1.0, -1.0):
                candidate = _finite_vector(
                    state + sign * displacement * direction, state.shape, "candidate"
                )
                if np.array_equal(candidate, state):
                    message = "WP7 directional displacement cannot perturb the state"
                    raise ValueError(message)
                model.set_state(np.ascontiguousarray(candidate))
                _residual(model, candidate)
                cotangent = objective_gradient(candidate)
                model.set_state(np.ascontiguousarray(candidate))
                lagrangian_gradient = cotangent - _product(
                    model, adjoint, transpose=True
                )
                lagrangian_gradients.append(lagrangian_gradient)
                boundary_gradients.append(pullback(lagrangian_gradient))
            derivative_scale = 0.5 * (delta_scale / displacement)
            lxx_delta = _finite_vector(
                (lagrangian_gradients[0] - lagrangian_gradients[1]) * derivative_scale,
                state.shape,
                "L_xx delta",
            )
            directional_gradient = (
                boundary_gradients[0] - boundary_gradients[1]
            ) * derivative_scale
        model.set_state(np.ascontiguousarray(state))
        _residual(model, state)
        adjoint_dot = np.zeros_like(state)
        adjoint_dot[active] = matrix.solve(lxx_delta[active])
        adjoint_dot = _finite_vector(adjoint_dot, state.shape, "second adjoint")
        second_transposed = _product(model, adjoint_dot, transpose=True)
        second_residual = _relative_residual(
            second_transposed[active], lxx_delta[active]
        )
        corrected = gradient + directional_gradient - pullback(second_transposed)
        if not np.all(np.isfinite(corrected)):
            message = "WP7 corrected boundary gradient is nonfinite"
            raise RuntimeError(message)
        diagnostics = ImplicitVjpDiagnostics(
            force_residual=_norm(force[active]),
            solve_residual=solve_residual,
            block_rcond=rcond,
            class_sizes={name: entries.size for name, entries in classes.items()},
            cotangent_on_null=_norm(x_bar[classes["null"]]),
            cotangent_on_pinned=_norm(x_bar[classes["pinned"]]),
            adjoint_weighted_residual=weighted_residual,
            newton_solve_residual=newton_residual,
            second_adjoint_solve_residual=second_residual,
        )
        return corrected, -weighted_residual, diagnostics
    finally:
        model.set_state(np.ascontiguousarray(state))
        try:
            _residual(model, state)
        finally:
            model.set_state(np.ascontiguousarray(state))
