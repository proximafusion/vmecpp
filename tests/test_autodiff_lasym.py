"""The implicit adjoint for equilibria without stellarator symmetry (lasym)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.fixture(scope="module")
def input_3d_high_beta(input_3d) -> vmecpp.VmecInput:
    """input_3d at a volume-averaged beta of 7.9 per cent."""
    return input_3d.model_copy(update={"pres_scale": 2.0e4})


CASES = ["input_2d", "input_3d", "input_3d_high_beta"]


def _boundary(indata: vmecpp.VmecInput) -> np.ndarray:
    rows = [indata.rbc, indata.zbs]
    if indata.lasym:
        rows += [indata.rbs, indata.zbc]
    return np.stack([np.asarray(row) for row in rows]).astype(np.float64)


def _solved_model(indata: vmecpp.VmecInput):
    return autodiff._solve_model(indata._to_cpp_vmecindata(), _boundary(indata))


@pytest.mark.parametrize("case", CASES)
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


@pytest.mark.parametrize("case", CASES)
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


@pytest.mark.parametrize("case", CASES)
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


@pytest.mark.parametrize("case", CASES)
def test_solve_vjp_is_the_transpose_of_the_forward_sensitivity(
    case, request, forward_sensitivity
) -> None:
    indata = request.getfixturevalue(case)
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)
    assert boundary.shape == solver.parameter_shape

    generator = np.random.default_rng(0)
    direction = generator.standard_normal(boundary.shape)
    cotangent = generator.standard_normal(solver.output_shape)

    tangent = forward_sensitivity(indata, boundary, direction)
    adjoint = solver._backward_callback(boundary, cotangent)[0]

    np.testing.assert_allclose(
        float(cotangent @ tangent), float((adjoint * direction).sum()), rtol=1.0e-6
    )


def test_poloidal_origin_shift_enters_the_gradient(
    input_3d, forward_sensitivity
) -> None:
    """A direction along rbs(m = 1, n = 0), which sets the poloidal rotation angle."""
    solver = autodiff.make_solver(input_3d)
    boundary = _boundary(input_3d)
    direction = np.zeros(boundary.shape)
    direction[2, 1, input_3d.ntor] = 1.0
    cotangent = np.random.default_rng(2).standard_normal(solver.output_shape)

    tangent = forward_sensitivity(input_3d, boundary, direction)
    adjoint = solver._backward_callback(boundary, cotangent)[0][0]

    np.testing.assert_allclose(
        float(cotangent @ tangent), float(adjoint[2, 1, input_3d.ntor]), rtol=1.0e-6
    )


def test_run_solver_vjp_is_the_transpose_of_the_forward_sensitivity(
    input_3d, forward_sensitivity
) -> None:
    """The VJP of vmecpp.run linearizes at a model hot-restarted from the run's wout."""
    solver = autodiff._RunSolver(input_3d)
    boundary = _boundary(input_3d)
    generator = np.random.default_rng(4)
    direction = generator.standard_normal(boundary.shape)
    cotangent = generator.standard_normal(solver.geometry_size)

    tangent = forward_sensitivity(input_3d, boundary, direction)
    adjoint = solver._backward_callback(boundary, cotangent)[0]

    np.testing.assert_allclose(
        float(cotangent @ tangent), float((adjoint * direction).sum()), rtol=1.0e-6
    )


def test_run_with_a_traced_asymmetric_boundary(input_3d) -> None:
    """A traced asymmetric run reproduces the plain run with the m = 1 gauge pinned."""
    pinned = input_3d.model_copy(update={"always_fix_m1_gauge": True})
    reference = vmecpp.run(pinned, verbose=False).wout
    weights = np.random.default_rng(5).standard_normal(np.shape(reference.rmns))

    def objective(boundary):
        perturbed = input_3d.model_copy(
            update={
                "rbc": boundary[0],
                "zbs": boundary[1],
                "rbs": boundary[2],
                "zbc": boundary[3],
            }
        )
        return jnp.sum(weights * vmecpp.run(perturbed, verbose=False).wout.rmns)

    value, gradient = jax.value_and_grad(objective)(_boundary(input_3d))
    np.testing.assert_allclose(
        float(value), float(np.sum(weights * reference.rmns)), rtol=1.0e-9
    )
    gradient = np.asarray(gradient)
    assert np.all(np.isfinite(gradient))
    assert np.abs(gradient[2:]).max() > 0.0


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
