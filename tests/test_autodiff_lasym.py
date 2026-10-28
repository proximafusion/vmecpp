"""The implicit adjoint for equilibria without stellarator symmetry (lasym)."""

from typing import Any

import jax
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres

import vmecpp
from vmecpp import autodiff
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="needs an Enzyme-enabled build for the exact residual transpose",
)


def _asymmetric(source: vmecpp.VmecInput, **update) -> vmecpp.VmecInput:
    rbs = np.zeros_like(np.asarray(source.rbc))
    zbc = np.zeros_like(np.asarray(source.zbs))
    ntor = source.ntor
    rbs[1, ntor] = 0.02
    zbc[1, ntor] = 0.03
    rbs[2, ntor] = 0.01
    zbc[2, ntor] = -0.01
    return source.model_copy(
        update={
            "lasym": True,
            "rbs": rbs,
            "zbc": zbc,
            "raxis_s": np.zeros_like(np.asarray(source.raxis_c)),
            "zaxis_c": np.zeros_like(np.asarray(source.zaxis_s)),
            **update,
        }
    )


@pytest.fixture(scope="module")
def input_2d() -> vmecpp.VmecInput:
    """An up-down asymmetric Solov'ev tokamak."""
    source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    return _asymmetric(
        source,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-15]),
        niter_array=np.asarray([4000]),
    )


@pytest.fixture(scope="module")
def input_3d() -> vmecpp.VmecInput:
    """Solov'ev with one toroidal boundary mode and asymmetric m = 1, 2 terms."""
    source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    mpol = source.mpol
    rbc = np.zeros((mpol, 3))
    zbs = np.zeros((mpol, 3))
    rbc[:, 1] = np.asarray(source.rbc)[:, 0]
    zbs[:, 1] = np.asarray(source.zbs)[:, 0]
    rbc[1, 2] = 0.01
    zbs[1, 2] = 0.01
    symmetric = source.model_copy(
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
    asymmetric = _asymmetric(symmetric)
    rbs = np.asarray(asymmetric.rbs).copy()
    rbs[1, 2] = 0.005
    return asymmetric.model_copy(update={"rbs": rbs})


def _boundary(indata: vmecpp.VmecInput) -> np.ndarray:
    rows = [indata.rbc, indata.zbs]
    if indata.lasym:
        rows += [indata.rbs, indata.zbc]
    return np.stack([np.asarray(row) for row in rows]).astype(np.float64)


def _solved_model(indata: vmecpp.VmecInput):
    return autodiff._solve_model(indata._to_cpp_vmecindata(), _boundary(indata))


@pytest.mark.parametrize("case", ["input_2d", "input_3d"])
def test_transposed_exact_hessian_is_the_transpose(case, request) -> None:
    model = _solved_model(request.getfixturevalue(case))
    assert model.lasym
    state = np.asarray(model.get_state(), dtype=np.float64)
    generator = np.random.default_rng(0)
    v = np.ascontiguousarray(generator.standard_normal(state.size))
    w = np.ascontiguousarray(generator.standard_normal(state.size))
    model.evaluate(2, 2, True)
    product = np.asarray(model.exact_hessian_vector_product(v), dtype=np.float64)
    transposed = np.asarray(
        model.exact_hessian_vector_product_transpose(w), dtype=np.float64
    )
    np.testing.assert_allclose(w @ product, transposed @ v, rtol=1.0e-11)


@pytest.mark.parametrize("case", ["input_2d", "input_3d"])
def test_exact_hessian_vector_product_is_the_derivative_of_the_force(
    case, request
) -> None:
    model = _solved_model(request.getfixturevalue(case))
    state = np.asarray(model.get_state(), dtype=np.float64).copy()

    def raw_force(value):
        model.set_state(np.ascontiguousarray(value))
        model.evaluate(2, 2, False)
        return np.asarray(model.get_forces(), dtype=np.float64).copy()

    direction = np.random.default_rng(1).standard_normal(state.size)
    direction /= np.linalg.norm(direction)
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    product = np.asarray(
        model.exact_hessian_vector_product(np.ascontiguousarray(direction)),
        dtype=np.float64,
    )
    step = 1.0e-6
    difference = (
        raw_force(state + step * direction) - raw_force(state - step * direction)
    ) / (2.0 * step)
    assert np.linalg.norm(product - difference) < 1.0e-6 * np.linalg.norm(difference)


@pytest.mark.parametrize("case", ["input_2d", "input_3d"])
def test_geometry_state_vjp_is_the_transpose(case, request) -> None:
    indata = request.getfixturevalue(case)
    ns = int(np.asarray(indata.ns_array)[-1])
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    state = np.asarray(model.get_state(), dtype=np.float64)
    direction = np.linspace(-0.2, 0.3, state.size)

    def coefficients(value):
        model.set_state(np.ascontiguousarray(value))
        flat = autodiff._cpp_geometry_flat(
            model.get_geometry(), ns, model.mpol, model.ntor
        )
        return flat[2 * ns :]

    coefficient_direction = coefficients(state + direction) - coefficients(state)
    model.set_state(np.ascontiguousarray(state))
    coefficient_bar = np.linspace(-0.7, 0.4, coefficient_direction.size)
    state_bar = np.asarray(model.geometry_state_vjp(coefficient_bar))
    np.testing.assert_allclose(
        coefficient_bar @ coefficient_direction,
        state_bar @ direction,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def _parser_state_tangent(indata, boundary, direction):
    """The asymmetric parser rotates the poloidal angle by an angle that depends on the
    boundary, so it is nonlinear; a central difference is accurate to O(h^2)."""
    ns = int(np.asarray(indata.ns_array)[-1])
    step = 1.0e-6

    def state(value):
        perturbed = indata.model_copy(
            update={"rbc": value[0], "zbs": value[1], "rbs": value[2], "zbc": value[3]}
        )
        model = _vmecpp.VmecModel.create(perturbed._to_cpp_vmecindata(), ns)
        return np.asarray(model.get_state(), dtype=np.float64)

    return (state(boundary + step * direction) - state(boundary - step * direction)) / (
        2.0 * step
    )


def _tangent_through_the_solve(indata, boundary, seed_state):
    """The forward sensitivity, solved with H where the adjoint solves with H^T."""
    model = autodiff._solve_model(indata._to_cpp_vmecindata(), boundary)
    state = np.asarray(model.get_state(), dtype=np.float64)
    interior, edge = autodiff._interior_and_boundary(model)
    gauge = autodiff._gauge_entries(model)
    edge = np.concatenate([edge, gauge])
    interior = np.setdiff1d(interior, gauge)
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

    step = 1.0e-6  # MakeGeometry is linear in the state, so this is exact

    def flat(value):
        model.set_state(np.ascontiguousarray(value))
        return autodiff._cpp_geometry_flat(
            model.get_geometry(), model.ns, model.mpol, model.ntor
        )

    return (flat(state + step * state_tangent) - flat(state - step * state_tangent)) / (
        2.0 * step
    )


@pytest.mark.parametrize("case", ["input_2d", "input_3d"])
def test_solve_vjp_is_the_transpose_of_the_forward_sensitivity(case, request) -> None:
    indata = request.getfixturevalue(case)
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)
    assert boundary.shape == solver.parameter_shape

    generator = np.random.default_rng(0)
    direction = generator.standard_normal(boundary.shape)
    cotangent = generator.standard_normal(solver.output_shape)

    seed_state = _parser_state_tangent(indata, boundary, direction)
    tangent = _tangent_through_the_solve(indata, boundary, seed_state)
    adjoint = solver._backward_callback(boundary, cotangent)[0]

    np.testing.assert_allclose(
        float(cotangent @ tangent), float((adjoint * direction).sum()), rtol=1.0e-6
    )


def test_poloidal_origin_shift_enters_the_gradient(input_3d) -> None:
    """A direction along rbs(m = 1, n = 0) alone changes the solve only through the
    rotation of the poloidal angle and the boundary it rotates."""
    solver = autodiff.make_solver(input_3d)
    boundary = _boundary(input_3d)
    direction = np.zeros(boundary.shape)
    direction[2, 1, input_3d.ntor] = 1.0
    cotangent = np.random.default_rng(2).standard_normal(solver.output_shape)

    seed_state = _parser_state_tangent(input_3d, boundary, direction)
    tangent = _tangent_through_the_solve(input_3d, boundary, seed_state)
    adjoint = solver._backward_callback(boundary, cotangent)[0][0]

    np.testing.assert_allclose(
        float(cotangent @ tangent), float(adjoint[2, 1, input_3d.ntor]), rtol=1.0e-6
    )


def test_run_with_a_traced_asymmetric_boundary(input_3d) -> None:
    """vmecpp.run differentiates through rbs and zbc; the plasma volume does not depend
    on the poloidal angle, so a central difference of full runs resolves it."""
    indata = input_3d.model_copy(
        update={"ftol_array": np.asarray([1.0e-20]), "niter_array": np.asarray([20000])}
    )

    def volume(boundary):
        perturbed = indata.model_copy(
            update={
                "rbc": boundary[0],
                "zbs": boundary[1],
                "rbs": boundary[2],
                "zbc": boundary[3],
            }
        )
        return vmecpp.run(perturbed, verbose=False).wout.volume_p

    boundary = _boundary(indata)
    value, gradient = jax.value_and_grad(volume)(boundary)
    np.testing.assert_allclose(
        float(value), vmecpp.run(indata, verbose=False).wout.volume_p, rtol=1.0e-12
    )
    step = 1.0e-6
    # the roundoff of the volume, divided by the step, bounds the difference's accuracy
    scale = float(np.abs(np.asarray(gradient)).max())
    for entry in ((2, 1, indata.ntor), (3, 2, indata.ntor), (2, 1, indata.ntor + 1)):
        plus, minus = boundary.copy(), boundary.copy()
        plus[entry] += step
        minus[entry] -= step
        difference = (float(volume(plus)) - float(volume(minus))) / (2.0 * step)
        np.testing.assert_allclose(
            float(gradient[entry]), difference, rtol=1.0e-5, atol=1.0e-6 * scale
        )


def test_symmetric_input_through_the_asymmetric_adjoint() -> None:
    """A symmetric boundary run with lasym = True reproduces the symmetric gradient."""
    source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    symmetric = source.model_copy(
        update={
            "ns_array": np.asarray([5]),
            "ftol_array": np.asarray([1.0e-15]),
            "niter_array": np.asarray([4000]),
        }
    )
    asymmetric = symmetric.model_copy(
        update={
            "lasym": True,
            "rbs": np.zeros_like(np.asarray(symmetric.rbc)),
            "zbc": np.zeros_like(np.asarray(symmetric.zbs)),
            "raxis_s": np.zeros_like(np.asarray(symmetric.raxis_c)),
            "zaxis_c": np.zeros_like(np.asarray(symmetric.zaxis_s)),
        }
    )
    cotangent = np.random.default_rng(3).standard_normal(
        autodiff.make_solver(symmetric).output_shape
    )
    symmetric_bar = autodiff.make_solver(symmetric)._backward_callback(
        _boundary(symmetric), cotangent[None]
    )[0][0]
    asymmetric_bar = autodiff.make_solver(asymmetric)._backward_callback(
        _boundary(asymmetric), cotangent[None]
    )[0][0]
    np.testing.assert_allclose(
        asymmetric_bar[:2], symmetric_bar, rtol=1.0e-8, atol=1.0e-10
    )
