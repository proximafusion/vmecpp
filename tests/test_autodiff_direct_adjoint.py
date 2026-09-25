"""The block tridiagonal direct adjoint solve: locality, assembly, and equivalence with
GMRES."""

from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.sparse.linalg import splu

import vmecpp
from vmecpp import autodiff, qs
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="the adjoint solve needs the exact residual transpose (Enzyme-enabled build)",
)

SOLOVEV = Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"


def _requires_exact_derivatives(model) -> None:
    if not model.has_exact_force_jacobian:
        pytest.skip("needs an Enzyme-enabled build for the exact residual transpose")


def _2d_model(ns: int = 7):
    indata = _vmecpp.VmecINDATA.from_file(str(SOLOVEV))
    model = _vmecpp.VmecModel.create(indata, ns)
    model.solve()
    return model


def _3d_input() -> vmecpp.VmecInput:
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
            "ns_array": np.asarray([5]),
            "ftol_array": np.asarray([1.0e-15]),
            "niter_array": np.asarray([4000]),
        }
    )


def _3d_model(ns: int = 5):
    indata = _3d_input()
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    model.solve()
    return model


def _boundary(indata: vmecpp.VmecInput) -> np.ndarray:
    return np.stack([np.asarray(indata.rbc), np.asarray(indata.zbs)], axis=0).astype(
        np.float64
    )


def test_force_depends_only_on_neighboring_surfaces() -> None:
    """Verifies the locality claim the block tridiagonal assembly relies on: a unit
    perturbation of a solved DOF at surface j only changes the *solved* force rows at
    surfaces j - 1, j, j + 1.

    Restricted to ``solved`` rows: the fixed boundary surface's r/z entries are not
    force rows in this sense (they hold the prescribed boundary, not a force balance
    the adjoint solves for), so a column can and does reach them from more than one
    surface away without contradicting the block tridiagonal structure of H_SS.
    """
    model = _3d_model()
    _requires_exact_derivatives(model)
    model.evaluate(2, 2, True)
    layout = autodiff.state_layout(model)
    state_size = int(np.asarray(model.get_state()).size)
    solved_set = {int(i) for i in layout.solved}

    def touched_surfaces(index: int) -> set[int]:
        probe = np.zeros(state_size)
        probe[index] = 1.0
        result = np.asarray(model.exact_hessian_vector_product_transpose(probe))
        touched = set()
        for name, span in layout.spans.items():
            values = result[span]
            nonzero_local = np.nonzero(np.abs(values) > 1.0e-10)[0]
            for local_index, surface in zip(
                nonzero_local, layout.surface[name][nonzero_local], strict=True
            ):
                global_index = span.start + int(local_index)
                if global_index in solved_set:
                    touched.add(int(surface))
        return touched

    checked_any = False
    for name, span in layout.spans.items():
        in_span = layout.solved[
            (layout.solved >= span.start) & (layout.solved < span.stop)
        ]
        if in_span.size == 0:
            continue
        local = in_span - span.start
        surface = layout.surface[name][local]
        for target_surface in sorted(set(surface.tolist())):
            index = int(in_span[surface == target_surface][0])
            touched = touched_surfaces(index)
            assert touched <= {target_surface - 1, target_surface, target_surface + 1}
            checked_any = True
    assert checked_any


def test_assembled_matrix_reproduces_hessian_vector_products() -> None:
    model = _3d_model()
    _requires_exact_derivatives(model)
    model.evaluate(2, 2, True)
    layout = autodiff.state_layout(model)
    matrix = autodiff.assemble_block_tridiagonal_hessian(model, layout)

    state_size = int(np.asarray(model.get_state()).size)
    generator = np.random.default_rng(0)
    for _ in range(5):
        direction = generator.standard_normal(layout.solved.size)
        embedded = np.zeros(state_size)
        embedded[layout.solved] = direction
        expected = np.asarray(model.exact_hessian_vector_product_transpose(embedded))[
            layout.solved
        ]
        actual = matrix @ direction
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1.0e-10,
            atol=1.0e-10 * max(1.0, np.abs(expected).max()),
        )


def test_direct_solve_residual_is_small() -> None:
    model = _3d_model()
    _requires_exact_derivatives(model)
    model.evaluate(2, 2, True)
    layout = autodiff.state_layout(model)
    matrix = autodiff.assemble_block_tridiagonal_hessian(model, layout)

    generator = np.random.default_rng(1)
    rhs = generator.standard_normal(layout.solved.size)
    factor = splu(matrix.tocsc())
    solution = factor.solve(rhs)
    residual = matrix @ solution - rhs
    assert np.linalg.norm(residual) < 1.0e-8 * np.linalg.norm(rhs)


def test_direct_and_gmres_vjp_agree() -> None:
    indata = _3d_input()
    model_direct = autodiff._solve_model(indata._to_cpp_vmecindata(), _boundary(indata))
    _requires_exact_derivatives(model_direct)
    model_gmres = autodiff._solve_model(indata._to_cpp_vmecindata(), _boundary(indata))

    generator = np.random.default_rng(2)
    output_shape = autodiff.make_solver(indata).output_shape
    cotangent = generator.standard_normal(output_shape)

    direct = autodiff._implicit_boundary_vjp(
        model_direct, cotangent, adjoint_solver="direct"
    )
    gmres_result = autodiff._implicit_boundary_vjp(
        model_gmres, cotangent, adjoint_solver="gmres"
    )
    np.testing.assert_allclose(direct, gmres_result, rtol=1.0e-5, atol=1.0e-8)


def test_direct_is_the_default_adjoint_solver() -> None:
    indata = _3d_input()
    solver = autodiff.make_solver(indata)
    assert solver.adjoint_solver == "direct"


def test_adjoint_solver_must_be_a_known_literal() -> None:
    indata = _3d_input()
    with pytest.raises(ValueError, match="adjoint_solver"):
        autodiff.make_solver(indata, adjoint_solver="lu")  # type: ignore[arg-type]


def test_quasisymmetry_gradient_matches_between_solvers() -> None:
    indata = _3d_input()
    solver_direct = autodiff.make_solver(indata, adjoint_solver="direct")
    solver_gmres = autodiff.make_solver(indata, adjoint_solver="gmres")
    _requires_exact_derivatives(
        autodiff._solve_model(indata._to_cpp_vmecindata(), _boundary(indata))
    )
    boundary = _boundary(indata)

    def objective(solver):
        def inner(value):
            return qs.quasisymmetry_total(solver(value), [0.5], ntheta=16, nphi=16)

        return inner

    _, gradient_direct = jax.value_and_grad(objective(solver_direct))(
        jax.numpy.asarray(boundary)
    )
    _, gradient_gmres = jax.value_and_grad(objective(solver_gmres))(
        jax.numpy.asarray(boundary)
    )
    np.testing.assert_allclose(
        np.asarray(gradient_direct),
        np.asarray(gradient_gmres),
        rtol=1.0e-4,
        atol=1.0e-6,
    )
