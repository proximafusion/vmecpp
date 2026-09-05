from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres

import vmecpp
from vmecpp import autodiff, geometry, qs
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)


def _small_input() -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    return source.model_copy(
        update={
            "ns_array": np.asarray([5]),
            "ftol_array": np.asarray([1.0e-8]),
            "niter_array": np.asarray([200]),
        }
    )


def _small_3d_input() -> vmecpp.VmecInput:
    """A genuinely three-dimensional fixed-boundary, ncurr=0 case.

    Solovev with one toroidal boundary mode. The 2D case cannot exercise the
    ``r_ss``, ``z_cs`` and ``lambda_cs`` blocks at all, and it is exactly those
    that bring in the structural null directions the adjoint has to deflate.
    """
    source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    mpol = source.mpol
    rbc = np.zeros((mpol, 3))
    zbs = np.zeros((mpol, 3))
    rbc[:, 1] = np.asarray(source.rbc)[:, 0]
    zbs[:, 1] = np.asarray(source.zbs)[:, 0]
    rbc[1, 2] = 0.01
    zbs[1, 2] = 0.01
    return source.model_copy(
        update={
            "ntor": 1,
            "rbc": rbc,
            "zbs": zbs,
            "raxis_c": np.asarray([4.0, 0.0]),
            "zaxis_s": np.asarray([0.0, 0.0]),
            "ns_array": np.asarray([5]),
            "ftol_array": np.asarray([1.0e-15]),
            "niter_array": np.asarray([4000]),
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


def _coefficients(value) -> np.ndarray:
    shape = (
        value.dimensions.ns,
        value.dimensions.mpol,
        value.dimensions.ntor + 1,
    )
    names = (
        "r_cc",
        "r_ss",
        "r_sc",
        "r_cs",
        "z_sc",
        "z_cs",
        "z_cc",
        "z_ss",
        "lambda_sc",
        "lambda_cs",
        "lambda_cc",
        "lambda_ss",
    )
    arrays = []
    for name in names:
        raw = np.asarray(getattr(value.coefficients, name), dtype=np.float64)
        arrays.append(np.zeros(shape) if raw.size == 0 else raw.reshape(shape))
    return np.concatenate([array.ravel() for array in arrays])


def test_solver_runs_vmecpp_and_matches_native_geometry() -> None:
    indata = _small_input()
    boundary = _boundary(indata)
    solver = autodiff.make_solver(indata)

    actual = solver(boundary)
    native = _vmecpp.run(indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT)
    expected = geometry.make(native)

    np.testing.assert_allclose(np.asarray(actual.r_cc), np.asarray(expected.r_cc))
    np.testing.assert_allclose(np.asarray(actual.z_sc), np.asarray(expected.z_sc))
    np.testing.assert_allclose(
        np.asarray(actual.lambda_sc), np.asarray(expected.lambda_sc)
    )


def test_solver_is_usable_under_jit() -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata)
    boundary = jnp.asarray(_boundary(indata))
    value = jax.jit(lambda x: solver(x).r_cc[-1, 0, 0])(boundary)
    assert np.isfinite(float(value))


def test_solver_does_not_fall_back_to_finite_differences() -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)
    if _vmecpp.VmecModel.create(
        indata._to_cpp_vmecindata(), 5
    ).has_exact_force_jacobian:
        pytest.skip("exact Enzyme derivative support is enabled")
    with pytest.raises(RuntimeError, match="no exact residual transpose"):
        solver._backward_callback(boundary, np.zeros(solver.output_shape))


def test_geometry_state_vjp_is_the_transpose_in_three_dimensions() -> None:
    """The 2D case leaves the ``lthreed`` branch of the map untested."""
    indata = _small_3d_input()._to_cpp_vmecindata()
    model = _vmecpp.VmecModel.create(indata, 5)
    assert model.lthreed
    state = np.asarray(model.get_state(), dtype=np.float64)
    direction = np.linspace(-0.2, 0.3, state.size)
    base = model.get_geometry()
    model.set_state(state + direction)
    shifted = model.get_geometry()
    model.set_state(state)

    coefficient_direction = _coefficients(shifted) - _coefficients(base)
    coefficient_bar = np.linspace(-0.7, 0.4, coefficient_direction.size)
    state_bar = np.asarray(model.geometry_state_vjp(coefficient_bar))

    np.testing.assert_allclose(
        np.dot(coefficient_bar, coefficient_direction),
        np.dot(state_bar, direction),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def _tangent_through_the_solve(indata, boundary, seed_state):
    """Forward sensitivity of the solve, using the same exact operator the adjoint uses
    but in the opposite direction."""
    model = autodiff._solve_model(indata._to_cpp_vmecindata(), boundary)
    state = np.asarray(model.get_state(), dtype=np.float64)
    interior, edge = autodiff._interior_and_boundary(model)
    model.set_state(np.ascontiguousarray(state))
    model.set_freeze_constraint_multiplier(True)
    try:
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

        step = 1.0e-6  # MakeGeometry is linear in the state, so this is exact

        def flat(value):
            model.set_state(np.ascontiguousarray(value))
            return autodiff._cpp_geometry_flat(
                model.get_geometry(), model.ns, model.mpol, model.ntor
            )

        result = (
            flat(state + step * state_tangent) - flat(state - step * state_tangent)
        ) / (2.0 * step)
        model.set_state(np.ascontiguousarray(state))
        return result
    finally:
        model.set_freeze_constraint_multiplier(False)


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
    """The adjoint is validated against the tangent, not against a finite difference of
    the nonlinear solve.

    A re-solve converges to |F| ~ 5e-9, so a finite-difference reference is
    noise-dominated well above the accuracy of interest, badly so in 3D. The
    forward sensitivity is an independent computation through the same
    equilibrium: it solves with H, while the adjoint solves with H^T.
    """
    indata = _small_3d_input()
    _requires_exact_derivatives(indata)
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)

    generator = np.random.default_rng(0)
    direction = generator.standard_normal(boundary.shape)
    cotangent = generator.standard_normal(solver.output_shape)

    seed_state = _parser_state_tangent(indata, boundary, direction)
    tangent = _tangent_through_the_solve(indata, boundary, seed_state)
    adjoint = solver._backward_callback(boundary, cotangent)

    np.testing.assert_allclose(
        float(cotangent @ tangent), float((adjoint * direction).sum()), rtol=1.0e-6
    )


def test_quasisymmetry_gradient_through_a_three_dimensional_solve() -> None:
    """The end-to-end target: a QS objective differentiated through a stellarator
    equilibrium. This composition is what the geometry contract exists for."""
    indata = _small_3d_input()
    _requires_exact_derivatives(indata)
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)

    def objective(value):
        return qs.quasisymmetry_total(solver(value), [0.5], ntheta=16, nphi=16)

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(boundary))
    gradient = np.asarray(gradient)

    assert float(value) > 0.0  # a 3D equilibrium is not quasi-axisymmetric
    assert np.all(np.isfinite(gradient))
    assert np.abs(gradient).max() > 0.0
