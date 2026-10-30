"""The block tridiagonal direct adjoint solve: locality, exact assembly, and the gradient
against finite differences of re-solved equilibria."""

from pathlib import Path

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
    reason="the adjoint solve needs the exact residual transpose (Enzyme-enabled build)",
)

SOLOVEV = Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"


def _3d_input(ns: int = 7, ftol: float = 1.0e-15) -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file(str(SOLOVEV))
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
            "ns_array": np.asarray([ns]),
            "ftol_array": np.asarray([ftol]),
            "niter_array": np.asarray([20000]),
        }
    )


def _boundary(indata: vmecpp.VmecInput) -> np.ndarray:
    return np.stack([np.asarray(indata.rbc), np.asarray(indata.zbs)], axis=0).astype(
        np.float64
    )


def _linearized_model():
    indata = _3d_input()
    model = autodiff._solve_model(indata._to_cpp_vmecindata(), _boundary(indata))
    model.evaluate(2, 2, True)
    interior, _ = autodiff._interior_and_boundary(model)
    interior = np.setdiff1d(interior, autodiff._gauge_entries(model))
    return model, autodiff._structural_nullfree_interior(model, interior)


def _surfaces(model) -> np.ndarray:
    state_size = int(np.asarray(model.get_state()).size)
    modes_per_surface = model.mpol * (model.ntor + 1)
    surface = np.zeros(state_size, dtype=np.int64)
    for span in autodiff._span_slices(model).values():
        surface[span] = np.arange(span.stop - span.start) // modes_per_surface
    return surface


def test_force_depends_only_on_neighboring_surfaces() -> None:
    """The locality the assembly relies on: a solved DOF on surface j reaches solved
    force rows on surfaces j - 1, j, j + 1 only, for every span."""
    model, solved = _linearized_model()
    surface = _surfaces(model)
    state_size = surface.size
    for span in autodiff._span_slices(model).values():
        in_span = solved[(solved >= span.start) & (solved < span.stop)]
        for j in np.unique(surface[in_span]):
            probe = np.zeros(state_size)
            probe[in_span[surface[in_span] == j][0]] = 1.0
            column = np.asarray(model.exact_hessian_vector_product(probe))[solved]
            touched = surface[solved[column != 0.0]]
            assert np.all(np.abs(touched - j) <= 1)


def test_assembled_matrix_reproduces_hessian_vector_products() -> None:
    model, solved = _linearized_model()
    matrix = autodiff._assemble_block_tridiagonal(model, solved)
    state_size = int(np.asarray(model.get_state()).size)
    generator = np.random.default_rng(0)
    for _ in range(3):
        direction = generator.standard_normal(solved.size)
        embedded = np.zeros(state_size)
        embedded[solved] = direction
        expected = np.asarray(model.exact_hessian_vector_product_transpose(embedded))
        np.testing.assert_allclose(
            matrix.matvec(direction), expected[solved], rtol=1.0e-10, atol=1.0e-12
        )


def test_block_lu_solves_the_assembled_system() -> None:
    model, solved = _linearized_model()
    matrix = autodiff._assemble_block_tridiagonal(model, solved)
    rhs = np.random.default_rng(1).standard_normal(solved.size)
    unfactorized = autodiff._assemble_block_tridiagonal(model, solved)
    matrix.factorize()
    solution = matrix.solve(rhs)
    np.testing.assert_allclose(unfactorized.matvec(solution), rhs, atol=1.0e-9)


def test_gradient_matches_finite_differences_of_resolved_equilibria() -> None:
    """Central differences of fully re-solved equilibria.

    The tolerance is the noise floor of the reference, not of the adjoint: re-solves
    converged to ftol = 1e-20 still differ by more than the O(h^2) truncation at smaller
    steps, and the adjoint agrees with the forward sensitivity to 1e-6 in
    test_autodiff.py.
    """
    indata = _3d_input(ftol=1.0e-20)
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)
    weights = np.random.default_rng(2).standard_normal(solver.geometry_size)

    def objective(value):
        geometry = solver(value)
        flat = [geometry.toroidal_flux, geometry.poloidal_flux]
        flat += [
            getattr(geometry, name).ravel() for name in autodiff._GEOMETRY_COEFFICIENTS
        ]
        return jnp.concatenate(flat) @ weights

    gradient = np.asarray(jax.grad(objective)(jnp.asarray(boundary)))
    step = 1.0e-3
    for index in [(0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 1, 2)]:
        plus, minus = boundary.copy(), boundary.copy()
        plus[index] += step
        minus[index] -= step
        reference = (float(objective(plus)) - float(objective(minus))) / (2.0 * step)
        np.testing.assert_allclose(gradient[index], reference, rtol=1.0e-3)
