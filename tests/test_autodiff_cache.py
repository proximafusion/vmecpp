"""``make_solver``'s forward/backward model cache and its failure modes."""

import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff
from vmecpp.cpp import _vmecpp  # type: ignore

pytestmark = pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="make_solver needs an Enzyme-enabled build for the exact residual transpose",
)


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


def test_backward_reuses_the_cached_forward_model(monkeypatch) -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)

    call_count = 0
    original = autodiff._solve_model

    def counting_solve_model(template, value):
        nonlocal call_count
        call_count += 1
        return original(template, value)

    monkeypatch.setattr(autodiff, "_solve_model", counting_solve_model)

    solver._forward_callback(boundary)
    assert call_count == 1

    solver._backward_callback(boundary, np.zeros(solver.output_shape))
    assert call_count == 1  # the cached model from the forward call is reused


def test_cache_miss_falls_back_to_a_fresh_solve(monkeypatch) -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)
    other_boundary = boundary.copy()
    other_boundary[0, 0, 0] *= 1.01

    call_count = 0
    original = autodiff._solve_model

    def counting_solve_model(template, value):
        nonlocal call_count
        call_count += 1
        return original(template, value)

    monkeypatch.setattr(autodiff, "_solve_model", counting_solve_model)

    solver._forward_callback(boundary)
    assert call_count == 1

    # A different boundary is a cache miss and triggers its own solve.
    solver._backward_callback(other_boundary, np.zeros(solver.output_shape))
    assert call_count == 2


def test_cache_evicts_the_least_recently_used_entry(monkeypatch) -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata, cache_size=1)
    boundary = _boundary(indata)
    other_boundary = boundary.copy()
    other_boundary[0, 0, 0] *= 1.01

    call_count = 0
    original = autodiff._solve_model

    def counting_solve_model(template, value):
        nonlocal call_count
        call_count += 1
        return original(template, value)

    monkeypatch.setattr(autodiff, "_solve_model", counting_solve_model)

    solver._forward_callback(boundary)
    solver._forward_callback(other_boundary)
    assert len(solver._cache) == 1
    assert call_count == 2

    # The first boundary was evicted, so this is a fresh solve, not a hit.
    solver._backward_callback(boundary, np.zeros(solver.output_shape))
    assert call_count == 3


def test_cache_size_must_be_positive() -> None:
    indata = _small_input()
    with pytest.raises(ValueError, match="cache_size"):
        autodiff.make_solver(indata, cache_size=0)


def test_on_failure_nan_warns_and_returns_nan_on_solver_failure(monkeypatch) -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata, on_failure="nan")
    boundary = _boundary(indata)

    def failing_solve_model(_template, _value):
        error_message = "synthetic solver failure"
        raise RuntimeError(error_message)

    monkeypatch.setattr(autodiff, "_solve_model", failing_solve_model)

    with pytest.warns(UserWarning, match="synthetic solver failure"):
        result = solver._forward_callback(boundary)
    assert result.shape == solver.output_shape
    assert np.all(np.isnan(result))


def test_on_failure_nan_warns_and_returns_nan_on_adjoint_failure(monkeypatch) -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata, on_failure="nan")
    boundary = _boundary(indata)

    def failing_vjp(_model, _geometry_bar):
        error_message = "synthetic adjoint failure"
        raise RuntimeError(error_message)

    monkeypatch.setattr(autodiff, "_implicit_boundary_vjp", failing_vjp)

    with pytest.warns(UserWarning, match="synthetic adjoint failure"):
        result = solver._backward_callback(boundary, np.zeros(solver.output_shape))
    assert result.shape == solver.parameter_shape
    assert np.all(np.isnan(result))


def test_on_failure_raise_is_the_default_and_propagates(monkeypatch) -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata)
    assert solver.on_failure == "raise"
    boundary = _boundary(indata)

    def failing_solve_model(_template, _value):
        error_message = "synthetic solver failure"
        raise RuntimeError(error_message)

    monkeypatch.setattr(autodiff, "_solve_model", failing_solve_model)

    with pytest.raises(RuntimeError, match="synthetic solver failure"):
        solver._forward_callback(boundary)


def test_on_failure_must_be_a_known_literal() -> None:
    indata = _small_input()
    with pytest.raises(ValueError, match="on_failure"):
        autodiff.make_solver(indata, on_failure="ignore")  # type: ignore[arg-type]
