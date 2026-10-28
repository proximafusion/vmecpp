from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres

import vmecpp
from vmecpp import autodiff, autodiff_wout, geometry, qs
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


@pytest.fixture
def indata() -> vmecpp.VmecInput:
    return _small_input()


@pytest.mark.parametrize("converges", [False, True])
def test_solver_success_is_a_traced_boolean(indata, monkeypatch, converges):
    monkeypatch.setattr(_vmecpp, "VMECPP_ENABLE_ENZYME", True)
    if not converges:
        indata.niter_array = np.asarray([1])
    solver = autodiff.make_solver(indata, raise_on_failure=False)
    result = jax.jit(solver.__call__)(_boundary(indata))
    assert result.success.shape == ()
    assert result.success.dtype == jnp.bool_
    assert bool(result.success) == converges
    if converges:
        expected = autodiff.DifferentiableVmec(indata)(_boundary(indata))
        for actual, reference in zip(
            jax.tree.leaves(result.geometry), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(actual, reference)
    else:
        for leaf in jax.tree.leaves(result.geometry):
            assert np.all(np.isnan(leaf))


@pytest.mark.parametrize("converges", [False, True])
def test_run_success_and_nan_fields_under_jit(indata, converges):
    if not converges:
        indata.niter_array = np.asarray([1])

    @jax.jit
    def solve(boundary):
        return vmecpp.run(
            indata.model_copy(update={"rbc": boundary[0], "zbs": boundary[1]}),
            raise_on_failure=False,
            verbose=False,
            max_threads=1,
        )

    result = solve(_boundary(indata))
    assert result.success.shape == ()
    assert result.success.dtype == jnp.bool_
    assert bool(result.success) == converges
    assert result.jxbout is None
    if converges:
        expected = jax.jit(
            lambda boundary: vmecpp.run(
                indata.model_copy(update={"rbc": boundary[0], "zbs": boundary[1]}),
                verbose=False,
                max_threads=1,
            )
        )(_boundary(indata))
        for name in autodiff_wout.WOUT_QUANTITIES:
            actual = getattr(result.wout, name)
            if actual is not None:
                np.testing.assert_allclose(
                    actual, getattr(expected.wout, name), rtol=1.0e-7, atol=1.0e-10
                )
    else:
        for leaf in jax.tree.leaves((result.input, result.wout)):
            assert np.all(np.isnan(leaf))


@pytest.mark.parametrize("api", ["run", "solver"])
def test_failure_branch_has_zero_gradient_under_jit(indata, api):
    indata.niter_array = np.asarray([1])
    solver = autodiff.DifferentiableVmec(indata, raise_on_failure=False)

    def objective(boundary):
        if api == "run":
            result = vmecpp.run(
                indata.model_copy(update={"rbc": boundary[0], "zbs": boundary[1]}),
                raise_on_failure=False,
                verbose=False,
                max_threads=1,
            )
            value = result.wout.aspect
        else:
            result = solver(boundary)
            assert isinstance(result, autodiff.SolveResult)
            value = jnp.sum(result.geometry.r_cc)
        return jax.lax.cond(result.success, lambda: value, lambda: 10.0)

    value, gradient = jax.jit(jax.value_and_grad(objective))(_boundary(indata))
    assert value == 10.0
    np.testing.assert_array_equal(gradient, np.zeros(solver.parameter_shape))


def test_failed_solver_derivative_is_nan(indata):
    indata.niter_array = np.asarray([1])
    solver = autodiff.DifferentiableVmec(indata, raise_on_failure=False)

    def objective(boundary):
        result = solver(boundary)
        assert isinstance(result, autodiff.SolveResult)
        return jnp.sum(result.geometry.r_cc)

    gradient = jax.jit(jax.grad(objective))(_boundary(indata))
    assert np.all(np.isnan(gradient))


def test_failed_solver_batched_cotangents(indata):
    indata.niter_array = np.asarray([1])
    solver = autodiff.DifferentiableVmec(indata, raise_on_failure=False)

    def values(boundary):
        result = solver(boundary)
        assert isinstance(result, autodiff.SolveResult)
        return jnp.stack([jnp.sum(result.geometry.r_cc), jnp.asarray(10.0)])

    jacobian = jax.jit(jax.jacrev(values))(_boundary(indata))
    assert np.all(np.isnan(jacobian[0]))
    np.testing.assert_array_equal(jacobian[1], np.zeros(solver.parameter_shape))


def test_vmap_can_mix_success_and_failure(indata, monkeypatch):
    def forward(self, boundary, profiles):
        del profiles
        if boundary[0, 0, 0] < 0:
            message = "solver failure"
            raise RuntimeError(message)
        return np.full(self.output_shape, np.sum(boundary))

    def backward(self, boundary, bars, profiles):
        del boundary, profiles
        return (
            np.broadcast_to(
                np.sum(bars, axis=-1).reshape(-1, 1, 1, 1),
                (len(bars), *self.parameter_shape),
            ).copy(),
            np.zeros((len(bars), 3, self.ns - 1)),
        )

    monkeypatch.setattr(autodiff.DifferentiableVmec, "_forward_callback", forward)
    monkeypatch.setattr(autodiff.DifferentiableVmec, "_backward_callback", backward)
    solver = autodiff.DifferentiableVmec(indata, raise_on_failure=False)

    def objective(boundary):
        result = solver(boundary)
        assert isinstance(result, autodiff.SolveResult)
        value = jax.lax.cond(
            result.success,
            lambda: jnp.sum(result.geometry.r_cc),
            lambda: 10.0,
        )
        return value, result.success

    boundary = jnp.asarray(_boundary(indata))
    points = jnp.stack([boundary, boundary.at[0, 0, 0].set(-1.0)])
    (values, success), gradient = jax.jit(
        jax.vmap(jax.value_and_grad(objective, has_aux=True))
    )(points)
    np.testing.assert_array_equal(success, [True, False])
    assert values[1] == 10.0
    expected = solver.ns * indata.mpol * (indata.ntor + 1)
    np.testing.assert_array_equal(
        gradient[0], np.full(solver.parameter_shape, expected)
    )
    np.testing.assert_array_equal(gradient[1], np.zeros(solver.parameter_shape))


@pytest.mark.parametrize("api", ["run", "solver"])
def test_failed_profile_gradient_can_use_a_constant_fallback(indata, api):
    indata.niter_array = np.asarray([1])
    solver = autodiff.DifferentiableVmec(indata, raise_on_failure=False)
    boundary = _boundary(indata)

    def objective(ai):
        if api == "run":
            result = vmecpp.run(
                indata.model_copy(update={"ai": ai}),
                raise_on_failure=False,
                verbose=False,
                max_threads=1,
            )
            value = result.wout.aspect
        else:
            result = solver(boundary, profiles={"ai": ai})
            assert isinstance(result, autodiff.SolveResult)
            value = jnp.sum(result.geometry.r_cc)
        return jax.lax.cond(result.success, lambda: value, lambda: 10.0)

    gradient = jax.jit(jax.grad(objective))(jnp.asarray(indata.ai))
    np.testing.assert_array_equal(gradient, np.zeros_like(indata.ai))


def test_nonconverged_outputs_are_not_reported_as_success(indata):
    indata.niter_array = np.asarray([1])
    indata.return_outputs_even_if_not_converged = True
    result = vmecpp.run(indata, raise_on_failure=False, verbose=False, max_threads=1)
    assert not result.success
    for leaf in jax.tree.leaves(result.wout):
        assert np.all(np.isnan(leaf))


def test_runtime_solver_exception_is_reported_as_failure(indata, monkeypatch):
    def fail(*args, **kwargs):
        del args, kwargs
        message = "native solver failure"
        raise RuntimeError(message)

    monkeypatch.setattr(_vmecpp, "run", fail)
    output = vmecpp.run(indata, raise_on_failure=False, verbose=False, max_threads=1)
    assert not output.success
    for leaf in jax.tree.leaves(output.wout):
        assert np.all(np.isnan(leaf))


def test_default_solver_still_raises(indata):
    indata.niter_array = np.asarray([1])
    solver = autodiff.DifferentiableVmec(indata)
    with pytest.raises(RuntimeError, match="did not converge"):
        solver._forward_callback(np.asarray(_boundary(indata)))


def test_nonraising_run_rejects_invalid_input(indata):
    with pytest.raises(ValueError, match="threads"):
        vmecpp.run(indata, raise_on_failure=False, max_threads=0)
    solver = autodiff.DifferentiableVmec(indata, raise_on_failure=False)
    with pytest.raises(ValueError, match="boundary has shape"):
        solver(jnp.zeros((1,)))


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


@pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="needs an Enzyme-enabled build for make_solver's exact residual transpose",
)
def test_solver_runs_vmecpp_and_matches_native_geometry() -> None:
    indata = _small_input()
    boundary = _boundary(indata)
    solver = autodiff.make_solver(indata)

    actual = solver(boundary)
    assert isinstance(actual, geometry.Geometry)
    native = _vmecpp.run(indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT)
    expected = geometry.make(native)

    np.testing.assert_allclose(np.asarray(actual.r_cc), np.asarray(expected.r_cc))
    np.testing.assert_allclose(np.asarray(actual.z_sc), np.asarray(expected.z_sc))
    np.testing.assert_allclose(
        np.asarray(actual.lambda_sc), np.asarray(expected.lambda_sc)
    )


@pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="needs an Enzyme-enabled build for make_solver's exact residual transpose",
)
def test_solver_is_usable_under_jit() -> None:
    indata = _small_input()
    solver = autodiff.make_solver(indata)
    boundary = jnp.asarray(_boundary(indata))

    def objective(value):
        solved = solver(value)
        assert isinstance(solved, geometry.Geometry)
        return solved.r_cc[-1, 0, 0]

    value = jax.jit(objective)(boundary)
    assert np.isfinite(float(value))


@pytest.mark.skipif(
    _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="exact Enzyme derivative support is enabled, so make_solver succeeds",
)
def test_solver_does_not_fall_back_to_finite_differences() -> None:
    indata = _small_input()
    with pytest.raises(RuntimeError, match="VMECPP_ENABLE_ENZYME"):
        autodiff.make_solver(indata)


def test_has_exact_force_jacobian_agrees_with_the_model_property() -> None:
    indata = _small_input()
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), 5)
    # The module-level function is a coarser, static build feature (the
    # compile-time VMECPP_ENABLE_ENZYME macro); the model property additionally
    # depends on the model's force-balance formulation (lforbal), so a true
    # build feature does not imply every model has the exact Jacobian.
    if vmecpp.has_exact_force_jacobian():
        assert model.has_exact_force_jacobian
    else:
        assert not model.has_exact_force_jacobian


def test_make_solver_raises_at_construction_without_the_enzyme_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_vmecpp, "VMECPP_ENABLE_ENZYME", False)
    indata = _small_input()
    with pytest.raises(RuntimeError, match="VMECPP_ENABLE_ENZYME"):
        autodiff.make_solver(indata)


def test_solve_model_raises_when_the_final_multigrid_step_does_not_converge() -> None:
    """A schedule truncated to a coarse, non-final step's ftol/niter must fail loudly.

    Reproduces truncating a multi-step ``ns_array`` to its first steps for a cheap solve:
    the truncated schedule's new last entry keeps the tight ftol that was only meant as a
    hand-over to a finer step, and does not converge standalone at that ftol within its
    niter budget.
    """
    indata = _small_input().model_copy(
        update={
            "ns_array": np.asarray([9]),
            "ftol_array": np.asarray([1.0e-16]),
            "niter_array": np.asarray([50]),
        }
    )
    boundary = _boundary(indata)
    with pytest.raises(RuntimeError, match="did not converge"):
        autodiff._solve_model(indata._to_cpp_vmecindata(), boundary)


def test_solve_model_accepts_a_non_final_step_that_does_not_converge() -> None:
    """A non-final multigrid step is allowed to exhaust its iteration budget.

    Its only job is handing a good initial guess to the next, finer step; only the
    schedule's final step must actually converge.
    """
    indata = _small_input().model_copy(
        update={
            "ns_array": np.asarray([9, 15]),
            "ftol_array": np.asarray([1.0e-16, 1.0e-8]),
            "niter_array": np.asarray([50, 200]),
        }
    )
    boundary = _boundary(indata)
    model = autodiff._solve_model(indata._to_cpp_vmecindata(), boundary)
    assert model.ns == 15
    assert model.fsqr < model.ftolv


@pytest.mark.parametrize("signgs", [-1, 1])
def test_geometry_state_vjp_is_the_transpose_in_three_dimensions(signgs: int) -> None:
    """The 2D case leaves the ``lthreed`` branch of the map untested."""
    vmec_input = _small_3d_input()
    vmec_input.signgs = signgs
    indata = vmec_input._to_cpp_vmecindata()
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
    # The pinned m=1 gauge entries are prescribed by the boundary, like the
    # edge entries; the parser tangent below carries their sqrt(s) profile.
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

    result = (
        flat(state + step * state_tangent) - flat(state - step * state_tangent)
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
    adjoint = solver._backward_callback(boundary, cotangent)[0]

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
        solved = solver(value)
        assert isinstance(solved, geometry.Geometry)
        return qs.quasisymmetry_total(solved, [0.5], ntheta=16, nphi=16)

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(boundary))
    gradient = np.asarray(gradient)

    assert float(value) > 0.0  # a 3D equilibrium is not quasi-axisymmetric
    assert np.all(np.isfinite(gradient))
    assert np.abs(gradient).max() > 0.0


def test_exact_hessian_vector_product_is_the_derivative_of_the_force() -> None:
    """The exact product against a central difference of the raw force, which recomputes
    the spectral-condensation multiplier from the state on every evaluation.

    A boundary-only direction moves the multiplier the most.
    """
    indata = _small_3d_input()
    _requires_exact_derivatives(indata)
    model = autodiff._solve_model(indata._to_cpp_vmecindata(), _boundary(indata))
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    interior, boundary = autodiff._interior_and_boundary(model)

    def raw_force(value):
        model.set_state(np.ascontiguousarray(value))
        model.evaluate(2, 2, False)
        return np.asarray(model.get_forces(), dtype=np.float64).copy()

    generator = np.random.default_rng(0)
    directions = [generator.standard_normal(state.size)]
    for rows in (boundary, interior):
        direction = np.zeros(state.size)
        direction[rows] = generator.standard_normal(rows.size)
        directions.append(direction)
    step = 1.0e-6
    for raw_direction in directions:
        direction = raw_direction / np.linalg.norm(raw_direction)
        model.set_state(np.ascontiguousarray(state))
        model.evaluate(2, 2, True)
        product = np.asarray(
            model.exact_hessian_vector_product(np.ascontiguousarray(direction)),
            dtype=np.float64,
        )
        difference = (
            raw_force(state + step * direction) - raw_force(state - step * direction)
        ) / (2.0 * step)
        assert np.linalg.norm(product - difference) < 1.0e-6 * np.linalg.norm(
            difference
        )
