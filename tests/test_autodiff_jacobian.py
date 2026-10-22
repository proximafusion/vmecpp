"""Jacobians through the implicit adjoint: one factorization of the transposed
interior system serves every cotangent at a solved state."""

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


def _input() -> vmecpp.VmecInput:
    """Solovev with one toroidal boundary mode: three-dimensional, ncurr = 0."""
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


def _boundary(indata: vmecpp.VmecInput) -> jax.Array:
    return jnp.asarray(
        np.stack([np.asarray(indata.rbc), np.asarray(indata.zbs)]), dtype=jnp.float64
    )


def _outputs(solver):
    """Scalars of the solved geometry on two interior surfaces and three spans."""

    def outputs(boundary):
        geometry = solver(boundary)
        return jnp.stack(
            [
                geometry.r_cc[2, 0, 0],
                geometry.r_cc[3, 1, 1],
                geometry.r_ss[2, 2, 1],
                geometry.z_sc[2, 1, 0],
                geometry.z_sc[3, 2, 1],
                geometry.lambda_sc[3, 1, 0],
            ]
        )

    return outputs


def test_jacobian_rows_equal_the_single_cotangent_gradients() -> None:
    indata = _input()
    solver = autodiff.make_solver(indata)
    outputs = _outputs(solver)
    boundary = _boundary(indata)

    jacobian = np.asarray(jax.jacrev(outputs)(boundary))
    rows = np.stack(
        [
            np.asarray(jax.grad(lambda b, k=k: outputs(b)[k])(boundary))
            for k in range(jacobian.shape[0])
        ]
    )

    assert np.abs(rows).max() > 0.0
    np.testing.assert_allclose(
        jacobian, rows, rtol=1.0e-12, atol=1.0e-14 * np.abs(rows).max()
    )


def test_one_solve_and_one_factorization_serve_further_vjps(monkeypatch) -> None:
    """A Jacobian, then a gradient of another objective at the same parameters."""
    indata = _input()
    solver = autodiff.make_solver(indata)
    outputs = _outputs(solver)
    boundary = _boundary(indata)
    calls = {"solve": 0, "assembly": 0}
    solve_model = autodiff._solve_model
    assemble = autodiff._assemble_block_tridiagonal

    def counted_solve(*args, **kwargs):
        calls["solve"] += 1
        return solve_model(*args, **kwargs)

    def counted_assembly(*args, **kwargs):
        calls["assembly"] += 1
        return assemble(*args, **kwargs)

    monkeypatch.setattr(autodiff, "_solve_model", counted_solve)
    monkeypatch.setattr(autodiff, "_assemble_block_tridiagonal", counted_assembly)

    jax.jacrev(outputs)(boundary)
    assert calls == {"solve": 1, "assembly": 1}
    jax.value_and_grad(lambda b: jnp.sum(outputs(b) ** 2))(boundary)
    assert calls == {"solve": 1, "assembly": 1}


def test_gradients_at_two_parameter_points_under_vmap() -> None:
    indata = _input()
    solver = autodiff.make_solver(indata)
    outputs = _outputs(solver)
    boundary = _boundary(indata)
    points = jnp.stack([boundary, boundary.at[0, 1, 2].add(1.0e-3)])

    def first(b):
        return outputs(b)[0]

    batched = np.asarray(jax.vmap(jax.grad(first))(points))
    for k in range(points.shape[0]):
        np.testing.assert_allclose(
            batched[k], np.asarray(jax.grad(first)(points[k])), rtol=1.0e-12, atol=0.0
        )
