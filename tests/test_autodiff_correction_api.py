from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from adjoint_verification.harness import small_input

import vmecpp
from vmecpp import autodiff, autodiff_corrections, autodiff_diagnostics, geometry
from vmecpp._adjoint_contracts import ImplicitVjpDiagnostics


@pytest.fixture
def correction_api(monkeypatch):
    original_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
        monkeypatch.setattr(autodiff._vmecpp, "VMECPP_ENABLE_ENZYME", True)
        solver = autodiff.make_solver(source)
        boundary = np.stack([source.rbc, source.zbs])
        state = np.asarray([2.0, 3.0])
        model = SimpleNamespace(
            has_exact_force_jacobian=True,
            get_state=state.copy,
            get_geometry=state.copy,
        )
        solve = Mock(return_value=model)
        monkeypatch.setattr(autodiff, "_solve_model", solve)
        refresh = Mock(return_value=np.zeros_like(state))
        monkeypatch.setattr(autodiff_diagnostics, "raw_force", refresh)
        monkeypatch.setattr(geometry, "from_cpp", jnp.asarray)
        diagnostics = ImplicitVjpDiagnostics(
            force_residual=1.0,
            solve_residual=0.0,
            block_rcond=1.0,
            class_sizes={"active": 2},
            cotangent_on_null=0.0,
            cotangent_on_pinned=0.0,
            adjoint_weighted_residual=0.25,
        )
        gradient = np.ones_like(boundary)
        correction = Mock(return_value=(gradient, -0.25, diagnostics))
        monkeypatch.setattr(autodiff_corrections, "corrected_vjp", correction)
        yield SimpleNamespace(
            solver=solver,
            boundary=boundary,
            state=state,
            model=model,
            solve=solve,
            refresh=refresh,
            correction=correction,
            gradient=gradient,
            diagnostics=diagnostics,
        )
    finally:
        jax.config.update("jax_enable_x64", original_x64)


def test_corrected_api_reevaluates_objective_curvature(correction_api, monkeypatch):
    api = correction_api

    def objective(value):
        return jnp.sum(value**3)

    state_gradient = jax.grad(objective)
    helper = Mock(return_value=state_gradient)
    monkeypatch.setattr(
        autodiff_corrections, "geometry_objective_state_gradient", helper
    )

    value, gradient, diagnostics = api.solver.corrected_value_and_grad(
        objective, api.boundary, directional_step=2.0e-3
    )

    assert value == pytest.approx(35.0 - 0.25)
    assert gradient is api.gradient
    assert diagnostics is api.diagnostics
    helper.assert_called_once_with(api.model, objective)
    args, kwargs = api.correction.call_args
    assert args[0] is api.model
    np.testing.assert_array_equal(args[1], 3.0 * api.state**2)
    assert args[2] == "B"
    assert kwargs["directional_step"] == 2.0e-3
    callback = kwargs["objective_state_gradient"]
    np.testing.assert_array_equal(
        callback(api.state + 1.0), 3.0 * (api.state + 1.0) ** 2
    )
    assert not np.array_equal(callback(api.state + 1.0), args[1])


def test_corrected_api_refreshes_profiles_before_value_and_gradient(
    correction_api, monkeypatch
):
    api = correction_api
    events = []
    api.refresh.side_effect = lambda _model: events.append("raw_force")

    def objective(value):
        events.append("objective_value")
        return jnp.sum(value)

    def state_gradient(state):
        events.append("state_gradient")
        return np.ones_like(state)

    monkeypatch.setattr(
        autodiff_corrections,
        "geometry_objective_state_gradient",
        lambda _model, _objective: state_gradient,
    )
    api.solver.corrected_value_and_grad(objective, api.boundary)

    api.refresh.assert_called_once_with(api.model)
    assert events == ["raw_force", "objective_value", "state_gradient"]


def test_corrected_api_default_step_and_cache_isolation(correction_api, monkeypatch):
    api = correction_api
    monkeypatch.setattr(
        autodiff_corrections,
        "geometry_objective_state_gradient",
        lambda _model, _objective: np.ones_like,
    )
    cache_key = autodiff._key(api.solver._solver_id, api.boundary, {})
    cached = {"state": np.asarray([-7.0])}
    monkeypatch.setattr(autodiff, "_FORWARD_SOLVES", {cache_key: cached})
    original_rbc = api.solver.vmec_input.rbc.copy()
    convert = vmecpp.VmecInput._to_cpp_vmecindata

    def mutate_copy(fixed_input):
        assert fixed_input is not api.solver.vmec_input
        fixed_input.rbc[:] = 0.0
        return convert(fixed_input)

    monkeypatch.setattr(vmecpp.VmecInput, "_to_cpp_vmecindata", mutate_copy)
    api.solver.corrected_value_and_grad(jnp.sum, api.boundary)

    api.solve.assert_called_once()
    assert api.correction.call_args.kwargs["directional_step"] == 1.0e-3
    assert {cache_key: cached} == autodiff._FORWARD_SOLVES
    assert autodiff._FORWARD_SOLVES[cache_key] is cached
    np.testing.assert_array_equal(api.solver.vmec_input.rbc, original_rbc)
    np.testing.assert_array_equal(api.solve.call_args.args[1], api.boundary)


@pytest.mark.parametrize(
    "boundary",
    [
        np.zeros(3),
        np.full((2, 6, 1), np.nan),
        np.full((2, 6, 1), np.inf),
        np.full((2, 6, 1), 1.0j),
        np.full((2, 6, 1), "invalid"),
    ],
)
def test_corrected_api_rejects_invalid_boundary(correction_api, boundary):
    api = correction_api
    if boundary.ndim == 3:
        boundary = np.full(api.boundary.shape, boundary.flat[0])
    with pytest.raises(ValueError, match=r"boundary must be finite.*shape"):
        api.solver.corrected_value_and_grad(jnp.sum, boundary)
    api.solve.assert_not_called()


@pytest.mark.parametrize(
    "objective",
    [
        None,
        lambda _: jnp.ones(2),
        lambda _: jnp.ones(1),
        lambda _: jnp.nan,
        lambda _: jnp.inf,
        lambda _: 1.0j,
        lambda _: "invalid",
    ],
)
def test_corrected_api_rejects_invalid_objective(correction_api, objective):
    api = correction_api
    with pytest.raises(ValueError, match=r"objective.*(callable|finite real scalar)"):
        api.solver.corrected_value_and_grad(objective, api.boundary)
    api.correction.assert_not_called()


def test_corrected_api_requires_x64(correction_api):
    api = correction_api
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64=True"):
            api.solver.corrected_value_and_grad(jnp.sum, api.boundary)
    finally:
        jax.config.update("jax_enable_x64", True)
    api.solve.assert_not_called()


@pytest.mark.parametrize("capability", ["build", "model", "lforbal"])
def test_corrected_api_requires_exact_derivatives(
    correction_api, monkeypatch, capability
):
    api = correction_api
    if capability == "build":
        monkeypatch.setattr(autodiff._vmecpp, "VMECPP_ENABLE_ENZYME", False)
    elif capability == "model":
        api.model.has_exact_force_jacobian = False
    else:
        api.solver.vmec_input.lforbal = True
    with pytest.raises(RuntimeError, match="exact force derivatives"):
        api.solver.corrected_value_and_grad(jnp.sum, api.boundary)
    api.correction.assert_not_called()


@pytest.mark.parametrize("step", [0.0, -1.0, np.nan, np.inf, np.ones(2), 1.0j])
def test_corrected_api_rejects_invalid_step(correction_api, step):
    api = correction_api
    with pytest.raises(
        ValueError, match="directional_step must be finite and positive"
    ):
        api.solver.corrected_value_and_grad(
            jnp.sum, api.boundary, directional_step=step
        )
    api.solve.assert_not_called()


def test_corrected_api_rejects_profile_overrides(correction_api):
    api = correction_api
    with pytest.raises(NotImplementedError, match="profile overrides"):
        api.solver.corrected_value_and_grad(
            jnp.sum, api.boundary, profiles={"phiedge": 1.0}
        )
    api.solve.assert_not_called()


@pytest.mark.parametrize("transform", [jax.jit, jax.grad])
def test_corrected_api_rejects_tracers(correction_api, transform):
    api = correction_api
    with pytest.raises(ValueError, match="native operation; JAX tracers"):
        transform(
            lambda boundary: api.solver.corrected_value_and_grad(jnp.sum, boundary)[0]
        )(api.boundary)
    api.solve.assert_not_called()


def test_corrected_api_rejects_traced_objective_parameters(correction_api):
    api = correction_api
    with pytest.raises(ValueError, match="objective returned a JAX tracer"):
        jax.grad(
            lambda scale: api.solver.corrected_value_and_grad(
                lambda value: scale * jnp.sum(value), api.boundary
            )[0]
        )(2.0)
    api.correction.assert_not_called()


def test_corrected_api_preserves_convergence_failure(correction_api):
    api = correction_api
    api.solve.side_effect = RuntimeError("VMEC++ did not converge")
    with pytest.raises(RuntimeError, match="did not converge"):
        api.solver.corrected_value_and_grad(jnp.sum, api.boundary)
    api.correction.assert_not_called()


def test_geometry_call_keeps_original_profile_route(correction_api, monkeypatch):
    api = correction_api
    sentinel = object()
    profiles = {"phiedge": 1.0}
    solve = Mock(return_value=(sentinel, None))
    monkeypatch.setattr(autodiff.DifferentiableVmec, "_solve", solve)

    assert api.solver(api.boundary, profiles=profiles) is sentinel
    solve.assert_called_once_with(api.boundary, profiles)
    api.solve.assert_not_called()
    api.correction.assert_not_called()


def test_geometry_jax_vjp_remains_uncorrected(correction_api, monkeypatch):
    api = correction_api
    solver = api.solver
    coefficient_offset = 2 * solver.ns

    def forward(_solver, boundary, _profiles):
        flat = np.zeros(solver.output_shape)
        flat[coefficient_offset] = boundary[0, 0, 0]
        return flat

    def backward(_solver, boundary, geometry_bar, _profiles):
        gradient = np.zeros_like(boundary)
        gradient[0, 0, 0] = geometry_bar[coefficient_offset]
        return gradient, np.zeros((3, solver.ns - 1))

    monkeypatch.setattr(autodiff.DifferentiableVmec, "_forward_callback", forward)
    monkeypatch.setattr(autodiff.DifferentiableVmec, "_backward_callback", backward)
    value, gradient = jax.jit(
        jax.value_and_grad(lambda boundary: solver(boundary).r_cc[0, 0, 0] ** 3)
    )(api.boundary)

    assert float(value) == pytest.approx(api.boundary[0, 0, 0] ** 3)
    expected = np.zeros_like(api.boundary)
    expected[0, 0, 0] = 3.0 * api.boundary[0, 0, 0] ** 2
    np.testing.assert_allclose(gradient, expected)
    api.solve.assert_not_called()
    api.correction.assert_not_called()


@pytest.mark.skipif(
    not vmecpp.has_exact_force_jacobian(),
    reason="requires an Enzyme-enabled build for native Option B corrections",
)
@pytest.mark.parametrize("quantity", ["interior_r", "DWell"])
def test_native_corrected_api_matches_direct_option_b(monkeypatch, quantity):
    from adjoint_verification.test_mercier_gradients import (  # noqa: PLC0415
        MERCIER_TERMS,
        MercierObjective,
        interior_mercier,
    )

    original_x64 = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        source = small_input(ftol=1.0e-8)
        source.lbsubs = True
        solver = autodiff.make_solver(source)
        boundary = np.stack([source.rbc, source.zbs])
        original_boundary = boundary.copy()
        original_input = source.model_dump_json()
        cache_key = autodiff._key(solver._solver_id, boundary, {})
        cached = {"state": np.asarray([-7.0])}
        monkeypatch.setattr(autodiff, "_FORWARD_SOLVES", {cache_key: cached})

        def objective(snapshot):
            return jnp.sum(snapshot.r_cc[1:-1, 2, :] ** 2)

        model = autodiff._solve_model(source._to_cpp_vmecindata(), boundary)
        autodiff_diagnostics.raw_force(model)
        if quantity != "interior_r":
            initial_geometry = geometry.from_cpp(model.get_geometry())
            terms = interior_mercier(initial_geometry, source)
            scale = float(np.max(np.abs(terms[MERCIER_TERMS.index(quantity)])))
            assert np.isfinite(scale)
            assert scale > 0
            mercier_objective = MercierObjective(source, quantity, scale)

            def scaled_mercier(snapshot):
                return mercier_objective(snapshot)

            objective = jax.jit(scaled_mercier)

        actual_value, actual_gradient, actual_diagnostics = (
            solver.corrected_value_and_grad(objective, boundary)
        )
        assert {cache_key: cached} == autodiff._FORWARD_SOLVES
        assert autodiff._FORWARD_SOLVES[cache_key] is cached
        np.testing.assert_array_equal(boundary, original_boundary)
        assert source.model_dump_json() == original_input

        autodiff_diagnostics.raw_force(model)
        base_value = float(objective(geometry.from_cpp(model.get_geometry())))
        state_gradient = autodiff_corrections.geometry_objective_state_gradient(
            model, objective
        )
        x_bar = state_gradient(np.asarray(model.get_state()).copy())
        expected_gradient, correction, expected_diagnostics = (
            autodiff_corrections.corrected_vjp(
                model, x_bar, "B", objective_state_gradient=state_gradient
            )
        )

        assert actual_value == pytest.approx(
            base_value + correction, rel=1.0e-10, abs=1.0e-13
        )
        assert actual_value - base_value == pytest.approx(
            -expected_diagnostics.adjoint_weighted_residual,
            rel=1.0e-8,
            abs=1.0e-13,
        )
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=1.0e-9, atol=1.0e-12
        )
        for name, expected in asdict(expected_diagnostics).items():
            actual = getattr(actual_diagnostics, name)
            if expected is None or isinstance(expected, dict):
                assert actual == expected
            else:
                assert actual == pytest.approx(expected, rel=1.0e-9, abs=1.0e-12)
    finally:
        jax.config.update("jax_enable_x64", original_x64)
