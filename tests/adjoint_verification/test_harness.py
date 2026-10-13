"""Shared WP0 configuration, real-physics objectives, and exact-HVP cache."""

import dataclasses
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmecpp import autodiff, autodiff_wout, geometry

from .conftest import requires_enzyme
from .harness import (
    INPUT_NAMES,
    MPOL_VALUES,
    NS_VALUES,
    BoundaryObjective,
    EvaluationSettings,
    InteriorLambdaObjective,
    InteriorM2RObjective,
    InvariantObjective,
    QuasisymmetryObjective,
    RandomLinearObjective,
    boundary_array,
    dense_jacobian,
    load_input,
    objective_suite,
    solved_model,
)


class DummyModel:
    """A nonsymmetric full-state linear force with an exact HVP."""

    has_exact_force_jacobian = True

    def __init__(self):
        self.state = np.arange(4, dtype=np.float64)
        self.matrix = np.arange(16, dtype=np.float64).reshape(4, 4) + np.eye(4)
        self.hvp_calls = []
        self.evaluation = None

    def get_state(self):
        return self.state.copy()

    def get_forces(self):
        return self.matrix @ self.state

    def evaluate(self, *args):
        self.evaluation = args

    def exact_hessian_vector_product(self, direction):
        self.hvp_calls.append(direction.copy())
        return self.matrix @ direction


@pytest.fixture
def cache_directory(tmp_path):
    return tmp_path


@pytest.fixture(scope="module")
def physics_case():
    indata = load_input("solovev", ns=7, mpol=4, ftol=1.0e-11)
    model = solved_model(indata)
    assert max(model.fsqr, model.fsqz, model.fsql) <= indata.ftol_array[-1]
    return indata, geometry.from_cpp(model.get_geometry())


def test_parameterized_inputs_are_pinned_symmetric_and_fixed(vmec_input):
    assert not vmec_input.lasym
    assert not vmec_input.lfreeb
    assert vmec_input.always_fix_m1_gauge
    assert len(vmec_input.ns_array) == len(vmec_input.ftol_array)
    assert len(vmec_input.ns_array) == len(vmec_input.niter_array)


@pytest.mark.parametrize("name", INPUT_NAMES)
def test_inputs_are_independent(name, input_factory):
    first, second = input_factory(name), input_factory(name)
    first.rbc[:] = 0
    first.ai[:] = 0
    first.ns_array[:] = 3
    assert np.any(second.rbc != 0)
    assert np.all(second.ns_array > 3)
    assert not np.shares_memory(first.ai, second.ai)


@pytest.mark.parametrize("mpol", MPOL_VALUES)
@pytest.mark.parametrize("ns", NS_VALUES)
def test_resolution_factory(mpol, ns, resolution_factory):
    value = resolution_factory(mpol, ns)
    assert value.mpol_max == mpol
    assert value.ns_array.tolist() == [ns]
    assert value.rbc.shape == (mpol, 2 * value.ntor_max + 1)
    assert value.zbs.shape == value.rbc.shape
    other = resolution_factory(mpol, ns)
    assert not np.shares_memory(value.rbc, other.rbc)


def test_named_fixtures(request):
    for name in INPUT_NAMES:
        value = request.getfixturevalue(f"{name}_input")
        assert value.always_fix_m1_gauge
    current = request.getfixturevalue("cth_ncurr1_input")
    assert current.ncurr == 1
    small = request.getfixturevalue("small_cth_input")
    assert (small.mpol_max, small.ntor_max) == (4, 2)
    assert small.ns_array.tolist() == [7]
    assert small.ncurr == 1
    assert small.always_fix_m1_gauge


def test_named_fixture_aliases_are_independent(request):
    for name in INPUT_NAMES:
        value = request.getfixturevalue(name)
        original = request.getfixturevalue(f"{name}_input")
        np.testing.assert_array_equal(value.rbc, original.rbc)
        assert not np.shares_memory(value.rbc, original.rbc)
        assert value.always_fix_m1_gauge
    small = request.getfixturevalue("small")
    original = request.getfixturevalue("small_cth_input")
    np.testing.assert_array_equal(small.rbc, original.rbc)
    assert not np.shares_memory(small.rbc, original.rbc)


def test_objective_fixtures(
    invariant_objectives, boundary_objective, relabeling_objectives
):
    assert set(invariant_objectives) == {"volume", "aspect", "iota", "well", "qs"}
    assert all(
        isinstance(invariant_objectives[name], InvariantObjective)
        for name in ("volume", "aspect", "iota", "well")
    )
    assert isinstance(invariant_objectives["qs"], QuasisymmetryObjective)
    assert isinstance(boundary_objective, BoundaryObjective)
    assert isinstance(relabeling_objectives["interior_lambda"], InteriorLambdaObjective)
    assert isinstance(relabeling_objectives["interior_m2_r"], InteriorM2RObjective)


def test_mercier_objective_requires_force_balance_bsubs():
    value = load_input("cth_like_fixed_bdy")
    value.lbsubs = False
    with pytest.raises(ValueError, match="lbsubs=True"):
        InvariantObjective(value, "mercier")
    value.lbsubs = True
    objective = InvariantObjective(value, "mercier")
    value.lbsubs = False
    assert objective.vmec_input.lbsubs


def test_mercier_objective_uses_full_discriminant(monkeypatch, physics_case):
    vmec_input, value = physics_case
    vmec_input = vmec_input.model_copy(update={"lbsubs": True})
    fields = {"DMerc": jnp.asarray([0.0, 3.0, 0.0]), "DWell": jnp.zeros(3)}
    monkeypatch.setattr(autodiff_wout, "wout_quantities", lambda *_: fields)
    assert float(InvariantObjective(vmec_input, "mercier")(value)) == 3.0
    assert "mercier" in objective_suite(vmec_input, value)


def test_factory_preserves_boundary_modes_when_resizing():
    original = load_input("cth_like_fixed_bdy_iota")
    resized = load_input("cth_like_fixed_bdy_iota", ns=11, mpol=8, ntor=6)
    np.testing.assert_array_equal(resized.rbc[: original.mpol_max, 2:-2], original.rbc)
    np.testing.assert_array_equal(resized.zbs[: original.mpol_max, 2:-2], original.zbs)
    boundary = boundary_array(resized)
    boundary[:] = 0
    assert np.any(resized.rbc != 0)


@pytest.mark.parametrize(
    "kwargs", [{"name": "missing"}, {"ns": 2}, {"ftol": np.nan}, {"niter": 0}]
)
def test_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError, match=r"unknown|ns must|ftol must"):
        load_input(**kwargs)


def test_resolution_factory_rejects_outside_grid(resolution_factory):
    with pytest.raises(ValueError, match="resolution"):
        resolution_factory(5, 11)


def test_dense_jacobian_uses_full_exact_unit_hvps():
    model = DummyModel()
    before = model.get_state()
    actual = dense_jacobian(model)
    np.testing.assert_array_equal(actual, model.matrix)
    np.testing.assert_array_equal(model.hvp_calls, np.eye(4))
    np.testing.assert_array_equal(model.get_state(), before)
    assert model.evaluation == (2, 2, True, True)
    assert not np.array_equal(actual, actual.T)


@pytest.mark.parametrize("limits", [{"max_state_size": 3}, {"max_bytes": 127}])
def test_dense_size_limit_precedes_evaluation_and_allocation(limits):
    model = DummyModel()
    with pytest.raises(ValueError, match="size limits"):
        dense_jacobian(model, **limits)
    assert model.evaluation is None
    assert not model.hvp_calls


def test_dense_requires_exact_hvp():
    model = DummyModel()
    model.has_exact_force_jacobian = False
    with pytest.raises(RuntimeError, match="exact HVP"):
        dense_jacobian(model)
    assert not model.hvp_calls


def test_dense_rejects_unpinned_gauge_before_evaluation():
    model = DummyModel()
    with pytest.raises(ValueError, match="always_fix_m1_gauge=True"):
        dense_jacobian(model, evaluation=EvaluationSettings(always_fix_m1_gauge=False))
    assert model.evaluation is None
    assert not model.hvp_calls


def test_cache_requires_full_input(cache_directory):
    with pytest.raises(ValueError, match="full vmec_input"):
        dense_jacobian(DummyModel(), cache_dir=cache_directory)


def test_cache_roundtrip_validates_with_fresh_hvps(cache_directory, solovev_input):
    model = DummyModel()
    args = {"vmec_input": solovev_input, "cache_dir": cache_directory}
    first = dense_jacobian(model, **args)
    assert len(model.hvp_calls) == 4
    model.hvp_calls.clear()
    second = dense_jacobian(model, **args)
    assert len(model.hvp_calls) == 2
    np.testing.assert_array_equal(second, first)
    assert len(list(cache_directory.glob("*.npz"))) == 1


@pytest.mark.parametrize("change", ["state", "input", "build", "evaluation"])
def test_cache_key_covers_numerical_context(change, cache_directory, solovev_input):
    model = DummyModel()
    args = {"vmec_input": solovev_input, "cache_dir": cache_directory}
    dense_jacobian(model, **args)
    model.hvp_calls.clear()
    if change == "state":
        model.state[0] += 0.25
    elif change == "input":
        solovev_input.pres_scale *= 1.5
    elif change == "build":
        args["build_id"] = "different-build"
    else:
        args["evaluation"] = EvaluationSettings(iter2=3)
    dense_jacobian(model, **args)
    assert len(model.hvp_calls) == 4
    assert len(list(cache_directory.glob("*.npz"))) == 2


@pytest.mark.parametrize(
    "corruption", ["bytes", "shape", "dtype", "nan", "checksum", "stale", "metadata"]
)
def test_invalid_cache_is_rebuilt(corruption, cache_directory, solovev_input):
    model = DummyModel()
    args = {"vmec_input": solovev_input, "cache_dir": cache_directory}
    dense_jacobian(model, **args)
    path = next(cache_directory.glob("*.npz"))
    if corruption == "bytes":
        path.write_bytes(b"not an npz")
    else:
        with np.load(path, allow_pickle=False) as data:
            metadata = data["metadata"].copy()
            matrix = data["matrix"].copy()
        if corruption == "shape":
            matrix = matrix[:2]
        elif corruption == "dtype":
            matrix = matrix.astype(np.float32)
        elif corruption == "nan":
            matrix[0, 0] = np.nan
        elif corruption == "metadata":
            metadata = np.asarray("stale metadata")
        else:
            matrix[0, 0] += 100
        checksum = hashlib.sha256(matrix.tobytes()).hexdigest()
        if corruption == "checksum":
            checksum = "wrong"
        np.savez(path, matrix=matrix, metadata=metadata, checksum=checksum)
    model.hvp_calls.clear()
    np.testing.assert_array_equal(dense_jacobian(model, **args), model.matrix)
    assert len(model.hvp_calls) >= 4


def test_cache_probes_detect_hidden_context(cache_directory, solovev_input):
    model = DummyModel()
    args = {"vmec_input": solovev_input, "cache_dir": cache_directory}
    dense_jacobian(model, **args)
    # Changing column zero does not change the force at state[0] == 0.
    model.matrix[:, 0] += 1
    model.hvp_calls.clear()
    np.testing.assert_array_equal(dense_jacobian(model, **args), model.matrix)
    assert len(model.hvp_calls) > 4


@pytest.mark.parametrize("quantity", ["volume", "aspect", "iota", "well"])
def test_invariant_objectives_use_physics_apis(quantity, physics_case):
    indata, value = physics_case
    objective = InvariantObjective(indata, quantity, s=0.5)
    actual = objective(value)
    if quantity == "iota":
        jet = geometry.evaluate(value, jnp.asarray([0.5, 0.0, 0.0]))
        expected = jet[4, 1] / jet[3, 1]
    else:
        fields = autodiff_wout.wout_quantities(value, indata)
        well = fields["DWell"]
        assert well is not None
        expected = (
            jnp.interp(0.5, jnp.linspace(0, 1, len(well)), well)
            if quantity == "well"
            else fields[quantity]
        )
    assert expected is not None
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12)
    # Differentiate the real objective through the full Geometry pytree.
    gradient = jax.grad(objective)(value)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(gradient))
    assert np.isfinite(actual)


def test_objective_suite_and_seeded_random_full_geometry(
    physics_case, random_linear_objective_factory
):
    indata, value = physics_case
    objectives = objective_suite(indata, value, seed=17)
    assert set(objectives) == {
        "volume",
        "aspect",
        "iota",
        "well",
        "qs",
        "boundary",
        "interior_lambda",
        "interior_m2_r",
        "random_linear",
    }
    first = objectives["random_linear"]
    second = random_linear_objective_factory(value, seed=17)
    different = RandomLinearObjective.from_geometry(value, seed=18)
    for a, b, c in zip(
        jax.tree_util.tree_leaves(first.weights),
        jax.tree_util.tree_leaves(second.weights),
        jax.tree_util.tree_leaves(different.weights),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
    gradient = jax.grad(first)(value)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(gradient),
        jax.tree_util.tree_leaves(first.weights),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert len(jax.tree_util.tree_leaves(gradient)) == 14


def test_boundary_and_relabeling_dependent_objective_support(physics_case):
    _, value = physics_case
    boundary = BoundaryObjective()
    lam = InteriorLambdaObjective()
    r_m2 = InteriorM2RObjective()
    changed = dataclasses.replace(
        value,
        lambda_sc=value.lambda_sc.at[2, 2, 0].add(0.2),
        r_cc=value.r_cc.at[2, 2, 0].add(0.3),
    )
    np.testing.assert_array_equal(boundary(value), boundary(changed))
    assert float(lam(value)) != float(lam(changed))
    assert float(r_m2(value)) != float(r_m2(changed))
    gradient = jax.grad(boundary)(value)
    np.testing.assert_array_equal(gradient.r_cc[:-1], 0)
    np.testing.assert_array_equal(gradient.lambda_sc, 0)
    np.testing.assert_array_equal(gradient.toroidal_flux, 0)


def test_qs_objective_uses_real_3d_geometry(small_cth_input):
    model = solved_model(small_cth_input)
    value = geometry.from_cpp(model.get_geometry())
    objective = QuasisymmetryObjective(ntheta=4, nphi=4)
    result = objective(value)
    np.testing.assert_allclose(result, jnp.sum(objective.residuals(value) ** 2))
    assert float(result) > 0
    gradient = jax.grad(objective)(value)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(gradient))


@requires_enzyme
@pytest.mark.parametrize(
    "fixture_name",
    [
        *INPUT_NAMES,
        "cth_ncurr1_input",
        "small",
    ],
)
def test_enzyme_all_fixture_solve_smoke(fixture_name, request):
    indata = request.getfixturevalue(fixture_name)
    model = solved_model(indata)
    assert model.ns == int(indata.ns_array[-1])
    assert max(model.fsqr, model.fsqz, model.fsql) <= model.ftolv
    assert model.has_exact_force_jacobian
    state = np.asarray(model.get_state()).copy()
    EvaluationSettings().evaluate(model)
    direction = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal(len(state))
    )
    product = np.asarray(model.exact_hessian_vector_product(direction))
    assert product.shape == state.shape
    assert np.isfinite(product).all()
    np.testing.assert_array_equal(model.get_state(), state)


@requires_enzyme
@pytest.mark.parametrize("ncurr", [0, 1])
def test_enzyme_solve_and_objective_vjp_smoke(ncurr, input_factory):
    indata = input_factory(
        "cth_like_fixed_bdy_iota" if ncurr == 0 else "cth_like_fixed_bdy",
        ns=7,
        mpol=4,
        ntor=2,
        ftol=1.0e-11,
    )
    solver = autodiff.make_solver(indata)
    objective = InvariantObjective(indata, "iota")
    boundary = jnp.asarray(boundary_array(indata))
    result, gradient = jax.value_and_grad(lambda b: objective(solver(b)))(boundary)
    assert np.isfinite(result)
    assert gradient.shape == boundary.shape
    assert np.isfinite(gradient).all()


@requires_enzyme
def test_enzyme_full_dense_jacobian_smoke(small_cth_input):
    model = solved_model(small_cth_input)
    before = np.asarray(model.get_state()).copy()
    matrix = dense_jacobian(model)
    direction = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal(len(before))
    )
    np.testing.assert_allclose(
        matrix @ direction,
        model.exact_hessian_vector_product(direction),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(model.get_state(), before)
