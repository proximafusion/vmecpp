# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The exact derivatives with the bootstrap current closed inside the solve.

With ``bootstrap_current`` the enclosed current is the fixed point
``currH = I_bs(x, currH)`` of the Redl closure evaluated on the iterating field.
The local force composition evaluates the closure alongside the force densities
with ``currH`` as an active input, and the exact products fold the response of
the fixed point in through ``(1 - dI_bs/dcurrH)^{-1}``. These tests check the
products against central differences of the self-consistent force, the adjoint
identity between the forward and transposed products, and the end-to-end
gradient of an objective through a solve.

The case is examples/data/cth_like_bootstrap.json at a reduced ns for speed.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres

import vmecpp
from vmecpp import autodiff
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)


def _bootstrap_input() -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file("examples/data/cth_like_bootstrap.json")
    assert source.bootstrap_current
    return source.model_copy(
        update={
            "ns_array": np.asarray([15]),
            "ftol_array": np.asarray([1.0e-16]),
            "niter_array": np.asarray([8000]),
            "bootstrap_tolerance": 1.0e-12,
        }
    )


def _requires_exact_derivatives(indata: vmecpp.VmecInput) -> None:
    ns = int(np.asarray(indata.ns_array)[-1])
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    if not model.has_exact_force_jacobian:
        pytest.skip("needs an Enzyme-enabled build for the exact residual transpose")


def _boundary(indata: vmecpp.VmecInput) -> np.ndarray:
    return np.stack([np.asarray(indata.rbc), np.asarray(indata.zbs)], axis=0).astype(
        np.float64
    )


def _solved_model(indata: vmecpp.VmecInput):
    ns = int(np.asarray(indata.ns_array)[-1])
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    model.solve()
    assert model.status == autodiff._VMEC_STATUS_SUCCESSFUL_TERMINATION
    return model


def test_composition_closure_reproduces_the_solved_current() -> None:
    """The closure block of the composition at the solved state is the current the
    solver iterated to, within the convergence tolerance of the closure."""
    indata = _bootstrap_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    solved = np.asarray(model.bootstrap_current_profile(), dtype=np.float64)
    closure = np.asarray(model.bootstrap_closure_current(), dtype=np.float64)
    assert np.abs(closure - solved).max() < 1.0e-2 * np.abs(solved).max()
    assert np.abs(solved).max() > 0.0


def test_self_consistent_evaluation_reaches_the_fixed_point() -> None:
    indata = _bootstrap_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.set_state(np.ascontiguousarray(state))
    model.evaluate_self_consistent(2, 2, True, 1.0e-13, 400)
    current = np.asarray(model.bootstrap_current_profile(), dtype=np.float64)
    closure = np.asarray(model.bootstrap_closure_current(), dtype=np.float64)
    assert np.abs(closure - current).max() < 1.0e-12 * np.abs(current).max()


def test_exact_hvp_matches_finite_difference_of_the_self_consistent_force() -> None:
    """The forward product against a central difference of the force evaluated at the
    fixed point of the closure for every perturbed state."""
    indata = _bootstrap_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()

    def self_consistent_force(x):
        model.set_state(np.ascontiguousarray(x))
        model.evaluate_self_consistent(2, 2, False, 1.0e-14, 400)
        return np.asarray(model.get_forces(), dtype=np.float64).copy()

    model.set_state(np.ascontiguousarray(state))
    model.evaluate_self_consistent(2, 2, True, 1.0e-14, 400)
    reference_profile = np.asarray(model.bootstrap_current_profile()).copy()
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(state.size)
    direction /= np.linalg.norm(direction)
    hv = np.asarray(
        model.exact_hessian_vector_product(np.ascontiguousarray(direction)),
        dtype=np.float64,
    )

    errors = {}
    for h in (1.0e-4, 1.0e-5, 1.0e-6):
        model.set_bootstrap_current_profile(np.ascontiguousarray(reference_profile))
        plus = self_consistent_force(state + h * direction)
        model.set_bootstrap_current_profile(np.ascontiguousarray(reference_profile))
        minus = self_consistent_force(state - h * direction)
        fd = (plus - minus) / (2 * h)
        errors[h] = np.linalg.norm(hv - fd) / np.linalg.norm(fd)
    assert errors[1.0e-6] < 1.0e-5
    assert errors[1.0e-6] < errors[1.0e-5] < errors[1.0e-4]


def test_exact_hvp_differs_from_the_frozen_current_product() -> None:
    """Holding currH fixed is a different functional: the force response through
    the closure is resolvable."""
    indata = _bootstrap_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(state.size)
    direction /= np.linalg.norm(direction)
    hv = np.asarray(
        model.exact_hessian_vector_product(np.ascontiguousarray(direction)),
        dtype=np.float64,
    )

    # the same pressure as a power series, p = p0 (1 - 0.8 s)^2
    p0 = 1.602176634e-19 * 2.0e18 * 1300.0
    frozen = indata.model_copy(
        update={
            "bootstrap_current": False,
            "pmass_type": "power_series",
            "am": np.asarray([p0, -1.6 * p0, 0.64 * p0]),
            "pres_scale": 1.0,
        }
    )
    frozen_model = _vmecpp.VmecModel.create(frozen._to_cpp_vmecindata(), model.ns)
    frozen_model.set_bootstrap_current_profile(
        np.ascontiguousarray(model.bootstrap_current_profile())
    )
    frozen_model.set_state(np.ascontiguousarray(state))
    frozen_model.evaluate(2, 2, True)
    assert frozen_model.fsqr < 1.0e-8
    hv_frozen = np.asarray(
        frozen_model.exact_hessian_vector_product(np.ascontiguousarray(direction)),
        dtype=np.float64,
    )
    assert np.linalg.norm(hv - hv_frozen) / np.linalg.norm(hv) > 1.0e-4


def test_forward_and_transposed_products_are_adjoint() -> None:
    indata = _bootstrap_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    rng = np.random.default_rng(3)
    v = rng.standard_normal(state.size)
    w = rng.standard_normal(state.size)
    hv = np.asarray(model.exact_hessian_vector_product(np.ascontiguousarray(v)))
    htw = np.asarray(
        model.exact_hessian_vector_product_transpose(np.ascontiguousarray(w))
    )
    np.testing.assert_allclose(float(w @ hv), float(htw @ v), rtol=1.0e-9)


def test_chip_state_vjp_matches_finite_difference_of_the_self_consistent_chip() -> None:
    indata = _bootstrap_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.set_state(np.ascontiguousarray(state))
    model.evaluate_self_consistent(2, 2, True, 1.0e-14, 400)
    reference_profile = np.asarray(model.bootstrap_current_profile()).copy()

    rng = np.random.default_rng(1)
    chip_bar = rng.standard_normal(model.ns - 1)
    state_bar = np.asarray(
        model.chip_state_vjp(np.ascontiguousarray(chip_bar)), dtype=np.float64
    )
    direction = rng.standard_normal(state.size)
    direction /= np.linalg.norm(direction)

    def chip_h(x):
        model.set_bootstrap_current_profile(np.ascontiguousarray(reference_profile))
        model.set_state(np.ascontiguousarray(x))
        model.evaluate_self_consistent(2, 2, True, 1.0e-14, 400)
        return np.asarray(model.chip_h, dtype=np.float64).copy()

    errors = {}
    for h in (1.0e-4, 1.0e-5, 1.0e-6):
        d_chip = (chip_h(state + h * direction) - chip_h(state - h * direction)) / (
            2 * h
        )
        lhs = chip_bar @ d_chip
        rhs = state_bar @ direction
        errors[h] = abs(lhs - rhs) / abs(lhs)
    assert errors[1.0e-6] < 1.0e-5
    assert errors[1.0e-6] < errors[1.0e-5] < errors[1.0e-4]


def _tangent_through_the_solve(indata, boundary, seed_state):
    """Forward sensitivity of the self-consistent solve: the interior state
    tangent solves H dx = -H x_b with the exact product, which carries the
    response of the closure, and maps to the geometry."""
    model = autodiff._solve_model(indata._to_cpp_vmecindata(), boundary)
    state = np.asarray(model.get_state(), dtype=np.float64)
    interior, edge = autodiff._interior_and_boundary(model)
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    keep = autodiff._structural_nullfree_interior(model, interior)
    size = state.size

    def forward(value):
        return np.asarray(
            model.exact_hessian_vector_product(np.ascontiguousarray(value)),
            dtype=np.float64,
        )

    def restricted(value):
        embedded = np.zeros(size)
        embedded[keep] = value
        return forward(embedded)[keep]

    def precondition(value):
        embedded = np.zeros(size)
        embedded[keep] = value
        return np.asarray(
            model.apply_preconditioner(np.ascontiguousarray(embedded)),
            dtype=np.float64,
        )[keep]

    factory: Any = LinearOperator
    operator = factory((keep.size, keep.size), matvec=restricted, dtype=np.float64)
    preconditioner = factory(
        (keep.size, keep.size), matvec=precondition, dtype=np.float64
    )
    seeded = np.zeros(size)
    seeded[edge] = seed_state[edge]
    tangent, info = gmres(
        operator,
        -forward(seeded)[keep],
        M=preconditioner,
        rtol=1.0e-12,
        restart=200,
        maxiter=400,
    )
    assert info == 0
    state_tangent = np.zeros(size)
    state_tangent[keep] = tangent
    state_tangent[edge] = seed_state[edge]

    # the geometry is linear in the state and in chi', and chi' follows the
    # state through the closure; the poloidal flux route is covered by the
    # transposed map, so the tangent is taken on the Fourier coefficients
    step = 1.0e-6

    def coefficients(value):
        model.set_state(np.ascontiguousarray(value))
        flat = autodiff._cpp_geometry_flat(
            model.get_geometry(), model.ns, model.mpol, model.ntor
        )
        return flat[2 * model.ns :]

    result = (
        coefficients(state + step * state_tangent)
        - coefficients(state - step * state_tangent)
    ) / (2.0 * step)
    model.set_state(np.ascontiguousarray(state))
    return result


def _parser_state_tangent(indata, boundary, direction):
    """The fixed-boundary parser is linear, so this difference is exact."""
    ns = int(np.asarray(indata.ns_array)[-1])
    step = 1.0e-6

    def state(value):
        perturbed = indata.model_copy(update={"rbc": value[0], "zbs": value[1]})
        model = _vmecpp.VmecModel.create(perturbed._to_cpp_vmecindata(), ns)
        return np.asarray(model.get_state(), dtype=np.float64)

    return (state(boundary + step * direction) - state(boundary - step * direction)) / (
        2.0 * step
    )


def test_solve_vjp_is_the_transpose_of_the_forward_sensitivity() -> None:
    """The adjoint of the self-consistent solve against its forward sensitivity, both
    through the exact products with the closure; the products themselves are checked
    against the self-consistent force above.

    A re-solve carries the path-dependent m=1 gauge of the native iteration, so a finite
    difference of re-solved equilibria does not resolve this derivative.
    """
    indata = _bootstrap_input()
    _requires_exact_derivatives(indata)
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)

    generator = np.random.default_rng(0)
    direction = generator.standard_normal(boundary.shape)
    ns = int(np.asarray(indata.ns_array)[-1])
    cotangent = np.zeros(solver.output_shape)
    cotangent[2 * ns :] = generator.standard_normal(solver.output_shape[0] - 2 * ns)

    seed_state = _parser_state_tangent(indata, boundary, direction)
    tangent = _tangent_through_the_solve(indata, boundary, seed_state)
    adjoint = solver._backward_callback(boundary, cotangent)

    np.testing.assert_allclose(
        float(cotangent[2 * ns :] @ tangent),
        float((adjoint * direction).sum()),
        rtol=1.0e-6,
    )


def test_iota_gradient_through_a_bootstrap_closed_solve_is_finite() -> None:
    """jax.grad of the mid-radius iota through a solve with the closure on."""
    indata = _bootstrap_input()
    _requires_exact_derivatives(indata)
    solver = autodiff.make_solver(indata)
    ns = int(np.asarray(indata.ns_array)[-1])
    mid = ns // 2

    def iota_mid(value):
        geometry = solver(value)
        dchi = geometry.poloidal_flux[mid + 1] - geometry.poloidal_flux[mid]
        dphi = geometry.toroidal_flux[mid + 1] - geometry.toroidal_flux[mid]
        return dchi / dphi

    value, gradient = jax.value_and_grad(iota_mid)(jnp.asarray(_boundary(indata)))
    gradient = np.asarray(gradient)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(gradient))
    assert np.abs(gradient).max() > 0.0
