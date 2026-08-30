import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff, geometry
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


def test_geometry_state_vjp_is_the_transpose_of_the_public_map() -> None:
    indata = _small_input()._to_cpp_vmecindata()
    model = _vmecpp.VmecModel.create(indata, 5)
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
