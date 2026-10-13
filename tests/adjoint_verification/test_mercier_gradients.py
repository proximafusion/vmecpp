"""Nonlinear Mercier derivatives, isolated from and coupled to the equilibrium."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmecpp import autodiff, autodiff_newton, autodiff_wout, geometry
from vmecpp.autodiff_diagnostics import raw_force
from vmecpp.cpp import _vmecpp  # type: ignore

from .conftest import requires_enzyme
from .harness import boundary_array, load_input, small_input, solved_model

MERCIER_TERMS = ("DMerc", "DCurr", "DGeod", "DWell")


def test_singular_lbsubs_fallback_has_finite_reverse_derivative():
    # Pedestal and sin(theta) on the smallest symmetric angular grid.
    setup: Any = SimpleNamespace(
        ns=4,
        nzeta=1,
        ntheta_eff=2,
        w_int=np.asarray([0.5, 0.5]),
        signgs=1,
        lbsubs=(
            np.arange(2),
            np.zeros((2, 2)),
            np.asarray([[0.0, 1.0], [0.0, -1.0]]),
            np.zeros((2, 2)),
            np.asarray([1.0, 0.0]),
        ),
    )

    def fields(x):
        metric = tuple(jnp.full((4, 1, 2), entry) for entry in (x**2 + 2, 2 * x, 3 * x))
        half = jnp.ones((3, 1, 2))
        # The first interior surface is invertible; the second has a zero
        # pedestal column throughout a neighborhood, not a branch-threshold kink.
        bsupv = half * jnp.asarray([2.0, 0.0, 0.0])[:, None, None]
        bsubu = x**2 * jnp.asarray([[[0.0, 0.0]], [[1.0, -1.0]], [[2.0, -2.0]]])
        return autodiff_wout._bsubs_from_force_balance(
            setup, metric, x * half, bsupv, half, bsubu, half, jnp.ones(3), jnp.ones(3)
        )

    def objective(x):
        return sum(jnp.sum(field[1:-1] ** 2) for field in fields(x))

    actual = fields(1.0)
    for field, fallback in zip(actual, (3.0, 2.0, 3.0), strict=True):
        np.testing.assert_array_equal(field[2:], fallback)
        np.testing.assert_array_equal(field[0], fallback)
    np.testing.assert_allclose(actual[1][1], [[3.0, -3.0]], rtol=0, atol=1e-14)
    value, derivative = jax.jit(jax.value_and_grad(objective))(1.0)
    np.testing.assert_allclose(value, 62.0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(derivative, 148.0, rtol=1e-13, atol=0)
    _, forward = jax.jvp(objective, (1.0,), (1.0,))
    np.testing.assert_allclose(forward, derivative, rtol=1e-13, atol=0)
    step = 1e-5
    finite_difference = (objective(1 + step) - objective(1 - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-9, atol=0)


def interior_mercier(value: geometry.Geometry, vmec_input):
    fields = autodiff_wout.wout_quantities(value, vmec_input)
    terms = []
    for name in MERCIER_TERMS:
        term = fields[name]
        assert term is not None
        terms.append(term)
    return jnp.stack(terms)[:, 1:-1]


@dataclasses.dataclass(frozen=True)
class MercierObjective:
    """Weighted squared interior diagnostic, with a fixed physical scale."""

    vmec_input: Any
    quantity: str
    scale: float

    def __call__(self, value: geometry.Geometry):
        terms = interior_mercier(value, self.vmec_input)
        term = terms[MERCIER_TERMS.index(self.quantity)] / self.scale
        weights = jnp.linspace(1.0, 2.0, term.size)
        return jnp.mean(weights * term**2)


def objective_state_cotangent(
    model,
    objective: Callable[[geometry.Geometry], jax.Array],
    state: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Reevaluate a nonlinear objective and its xbar, restoring the caller state."""
    original = np.asarray(model.get_state()).copy()
    try:
        if state is not None:
            model.set_state(np.ascontiguousarray(state))
        raw_force(model)
        value, cotangent = jax.value_and_grad(objective)(
            geometry.from_cpp(model.get_geometry())
        )
        coefficient_bar = np.concatenate(
            [
                np.asarray(getattr(cotangent, name)).ravel()
                for name in autodiff._GEOMETRY_COEFFICIENTS
            ]
        )
        state_bar = model.geometry_state_vjp(
            np.ascontiguousarray(coefficient_bar),
            np.ascontiguousarray(cotangent.poloidal_flux),
        )
        return float(value), np.asarray(state_bar).copy()
    finally:
        model.set_state(np.ascontiguousarray(original))
        try:
            raw_force(model)
        finally:
            model.set_state(np.ascontiguousarray(original))


@pytest.fixture(scope="module", params=("small", "cth"))
def mercier_case(request):
    source = (
        small_input()
        if request.param == "small"
        else load_input("cth_like_fixed_bdy", ns=9, mpol=4, ntor=2)
    )
    source.lbsubs = True
    source.always_fix_m1_gauge = True
    output = _vmecpp.run(
        source._to_cpp_vmecindata(),
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
    )
    return source, geometry.make(output), output.wout


def test_native_interior_mercier_fields(mercier_case):
    source, value, native = mercier_case
    assert source.ncurr == 1
    assert source.ntor > 0
    actual = np.asarray(interior_mercier(value, source))
    expected = np.stack([getattr(native, name) for name in MERCIER_TERMS])[:, 1:-1]
    assert np.all(np.max(np.abs(expected), axis=1) > 0)
    # Compare each term on its own scale, not against the larger total DMerc.
    scale = np.max(np.abs(expected), axis=1)[:, None]
    errors = np.max(np.abs(actual - expected) / scale, axis=1)
    print(f"native interior errors {MERCIER_TERMS}: {errors}")
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=0, atol=1e-8)


def _geometry_direction(value, kind):
    direction = jax.tree.map(jnp.zeros_like, value)
    s = jnp.linspace(0.0, 1.0, value.toroidal_flux.size)
    radial = s * (1.0 - s)
    if kind == "poloidal_flux":
        return dataclasses.replace(
            direction, poloidal_flux=0.01 * value.poloidal_flux[-1] * radial
        )
    return dataclasses.replace(
        direction,
        r_cc=direction.r_cc.at[:, 2, 1].set(0.001 * radial),
        z_sc=direction.z_sc.at[:, 2, 1].set(-0.0007 * radial),
        lambda_sc=direction.lambda_sc.at[:, 2, 1].set(0.002 * radial),
        lambda_cs=direction.lambda_cs.at[:, 0, 1].set(0.001 * radial),
    )


@pytest.mark.parametrize("kind", ["shape_lambda", "poloidal_flux"])
def test_fixed_state_mercier_directional_ad_matches_fd(mercier_case, kind):
    """No perturbed solves or Enzyme: differentiate the Geometry output stage."""
    source, value, _ = mercier_case
    direction = _geometry_direction(value, kind)

    def objective(geometry_value):
        return interior_mercier(geometry_value, source)

    _, tangent = jax.jvp(objective, (value,), (direction,))
    tangent = np.asarray(tangent)
    assert np.all(np.isfinite(tangent))
    scale = np.max(np.abs(tangent), axis=1)[:, None]
    assert np.all(scale > 0)
    errors = []
    for step in (3e-4, 1e-4, 3e-5):
        plus = jax.tree.map(lambda x, d, h=step: x + h * d, value, direction)
        minus = jax.tree.map(lambda x, d, h=step: x - h * d, value, direction)
        finite_difference = np.asarray(
            (objective(plus) - objective(minus)) / (2 * step)
        )
        errors.append(np.max(np.abs(finite_difference - tangent) / scale, axis=1))
    errors = np.asarray(errors)
    print(f"fixed-state {kind}: errors by step and {MERCIER_TERMS}: {errors}")
    # Two steps must independently resolve every term across all interior surfaces.
    assert np.all(np.sum(errors < 1e-6, axis=0) >= 2), errors


def test_fixed_state_nonlinear_reverse_gradients(mercier_case):
    source, value, _ = mercier_case
    scales = jnp.max(jnp.abs(interior_mercier(value, source)), axis=1)
    direction = _geometry_direction(value, "shape_lambda")

    def objectives(geometry_value):
        terms = interior_mercier(geometry_value, source) / scales[:, None]
        weights = jnp.linspace(1.0, 2.0, terms.shape[1])
        return jnp.mean(weights * terms**2, axis=1)

    _, pullback = jax.vjp(objectives, value)
    derivatives = []
    for index in range(len(MERCIER_TERMS)):
        cotangent = pullback(jnp.eye(len(MERCIER_TERMS))[index])[0]
        assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(cotangent))
        derivatives.append(
            sum(
                float(jnp.sum(bar * tangent))
                for bar, tangent in zip(
                    jax.tree.leaves(cotangent),
                    jax.tree.leaves(direction),
                    strict=True,
                )
            )
        )
    derivatives = np.asarray(derivatives)
    _, forward = jax.jvp(objectives, (value,), (direction,))
    np.testing.assert_allclose(derivatives, forward, rtol=1e-10, atol=0)
    steps = np.asarray([3e-4, 1e-4, 3e-5])
    finite_differences = []
    for step in steps:
        plus = jax.tree.map(lambda x, d, h=step: x + h * d, value, direction)
        minus = jax.tree.map(lambda x, d, h=step: x - h * d, value, direction)
        fd = np.asarray((objectives(plus) - objectives(minus)) / (2 * step))
        finite_differences.append(fd)
    finite_differences = np.asarray(finite_differences)
    # Squaring the lbsubs current terms makes central FD truncation appreciable.
    # Eliminate its O(h^2) term using two independent Richardson pairs.
    ratios = (steps[:-1] / steps[1:]) ** 2
    extrapolated = finite_differences[1:] + (
        finite_differences[1:] - finite_differences[:-1]
    ) / (ratios[:, None] - 1.0)
    errors = np.abs(extrapolated - derivatives) / np.abs(derivatives)
    print(f"fixed-state nonlinear reverse errors {MERCIER_TERMS}: {errors}")
    assert np.all(np.asarray(errors) < 1e-6), errors


@requires_enzyme
def test_nonlinear_objective_state_cotangent_restores_and_reevaluates():
    source = small_input()
    source.lbsubs = True
    source.always_fix_m1_gauge = True
    model = solved_model(source)
    state = np.asarray(model.get_state()).copy()
    force = raw_force(model)
    terms = np.asarray(
        interior_mercier(geometry.from_cpp(model.get_geometry()), source)
    )
    objective = MercierObjective(source, "DMerc", np.max(np.abs(terms[0])))
    _, base_bar = objective_state_cotangent(model, objective)
    interior, _ = autodiff._interior_and_boundary(model)
    direction = np.zeros_like(state)
    direction[interior] = base_bar[interior] / np.linalg.norm(base_bar[interior])
    step = 1e-6
    plus_value, plus_bar = objective_state_cotangent(
        model, objective, state + step * direction
    )
    minus_value, _ = objective_state_cotangent(
        model, objective, state - step * direction
    )
    np.testing.assert_allclose(
        (plus_value - minus_value) / (2 * step),
        base_bar @ direction,
        rtol=1e-6,
        atol=0,
    )
    assert np.linalg.norm(plus_bar - base_bar) > 0
    np.testing.assert_array_equal(model.get_state(), state)
    np.testing.assert_array_equal(model.get_forces(), force)


@pytest.fixture(scope="module")
def polished_mercier_equilibria():
    source = small_input()
    source.lbsubs = True
    source.always_fix_m1_gauge = True
    boundary = boundary_array(source)
    direction = np.zeros_like(boundary)
    direction[0, 1, source.ntor + 1] = 1.0
    direction[0, 2, source.ntor + 1] = 0.3
    direction[1, 2, source.ntor] = -0.2
    direction /= np.linalg.norm(direction)

    def solve(parameters):
        # Fresh independent solves; no continuation or perturbed-state prediction.
        model = autodiff._solve_model(source._to_cpp_vmecindata(), parameters)
        polished = autodiff_newton.newton_polish(model, 5, 1e-12)
        norm = np.linalg.norm(raw_force(model)[polished.active])
        np.testing.assert_allclose(norm, polished.residual_history[-1], rtol=1e-10)
        if not polished.converged:
            pytest.skip(f"Mercier equilibrium FD unresolved: raw residual {norm:.3e}")
        assert norm <= 1e-12
        return model, geometry.from_cpp(model.get_geometry()), float(norm)

    model, base, residual = solve(boundary)
    steps = np.asarray([1e-3, 3e-4, 1e-4])
    pairs = []
    residuals = [residual]
    for step in steps:
        _, plus, plus_residual = solve(boundary + step * direction)
        _, minus, minus_residual = solve(boundary - step * direction)
        pairs.append((plus, minus))
        residuals.extend((plus_residual, minus_residual))
    return source, model, base, direction, steps, pairs, residuals


@requires_enzyme
@pytest.mark.parametrize("quantity", MERCIER_TERMS)
def test_implicit_nonlinear_mercier_gradient(polished_mercier_equilibria, quantity):
    source, model, base, direction, steps, pairs, residuals = (
        polished_mercier_equilibria
    )
    terms = np.asarray(interior_mercier(base, source))
    scale = float(np.max(np.abs(terms[MERCIER_TERMS.index(quantity)])))
    assert scale > 0
    objective = MercierObjective(source, quantity, scale)
    value, state_bar = objective_state_cotangent(model, objective)
    direct_value, geometry_bar = jax.value_and_grad(objective)(
        geometry.from_cpp(model.get_geometry())
    )
    assert value == float(direct_value)
    assert np.all(np.isfinite(state_bar))
    assert np.linalg.norm(state_bar) > 0
    flat_bar = np.concatenate(
        [np.asarray(leaf).ravel() for leaf in jax.tree.leaves(geometry_bar)]
    )
    state = np.asarray(model.get_state()).copy()
    try:
        gradient, _ = autodiff._implicit_vjp(model, flat_bar)
    finally:
        model.set_state(np.ascontiguousarray(state))
    derivative = float(np.sum(gradient * direction))
    finite_differences = np.asarray(
        [
            float((objective(plus) - objective(minus)) / (2 * step))
            for step, (plus, minus) in zip(steps, pairs, strict=True)
        ]
    )
    assert np.isfinite(derivative)
    assert np.all(np.isfinite(finite_differences))
    magnitude = np.max(np.abs(finite_differences))
    assert magnitude > 0
    errors = np.abs(finite_differences - derivative) / magnitude
    print(
        f"implicit {quantity}: AD={derivative:.12g}, FD={finite_differences}, "
        f"relative errors={errors}, raw residuals={residuals}"
    )
    if abs(finite_differences[-1] - finite_differences[-2]) > 1e-6 * magnitude:
        pytest.skip(f"{quantity} independent equilibrium FD has no resolved plateau")
    np.testing.assert_allclose(finite_differences[-2:], derivative, rtol=1e-6, atol=0)
