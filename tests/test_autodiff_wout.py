"""The JAX output stage against the C++ one, the VmecWOut pytree, and gradients."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff, autodiff_wout, geometry
from vmecpp._pydantic_numpy import own_model_fields
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

requires_enzyme = pytest.mark.skipif(
    not vmecpp.has_exact_force_jacobian(),
    reason="autodiff.run needs an Enzyme-enabled build (VMECPP_ENABLE_ENZYME=ON)",
)

# The error of a field is measured against the largest entry of the field, or of
# a related field of the same physical scale where the field itself vanishes
# (no net current, no shear, stellarator symmetry up to roundoff).
_SCALE_OF = {
    "buco": "bvco",
    "jcurv": "jcuru",
    "DShear": "DMerc",
    "DCurr": "DMerc",
    "DWell": "DMerc",
    "DGeod": "DMerc",
    "gmns": "gmnc",
    "bmns": "bmnc",
    "bsubumns": "bsubumnc",
    "bsubvmns": "bsubvmnc",
    "bsubsmnc": "bsubsmns",
    "bsupumns": "bsupumnc",
    "bsupvmns": "bsupvmnc",
    "currumns": "currumnc",
    "currvmns": "currvmnc",
    "rmns": "rmnc",
    "zmnc": "zmns",
    "lmnc": "lmns",
    "lmnc_full": "lmns_full",
    "raxis_cs": "raxis_cc",
    "zaxis_cc": "raxis_cc",
    "zaxis_cs": "raxis_cc",
}

# Relative tolerances above roundoff. The currents, J.B and the radial force
# balance are finite differences in s of B-field Fourier coefficients, which
# amplifies the roundoff of the latter by 1 / (mu_0 ds); the Mercier terms are
# products of such differences.
_TOLERANCE = {
    "jdotb": 1.0e-6,
    "jcuru": 1.0e-6,
    "jcurv": 1.0e-6,
    "ctor": 1.0e-6,
    "currumnc": 1.0e-6,
    "currvmnc": 1.0e-6,
    "currumns": 1.0e-6,
    "currvmns": 1.0e-6,
    "equif": 1.0e-6,
    "DMerc": 1.0e-8,
    "DShear": 1.0e-8,
    "DCurr": 1.0e-8,
    "DWell": 1.0e-8,
    "DGeod": 1.0e-8,
    # evaluated on the state before the final time step by the C++ solver
    "specw": 1.0e-6,
}
_DEFAULT_TOLERANCE = 1.0e-10

# Input file and final ftol: stellarator-symmetric and asymmetric, 2D and 3D,
# fixed and free boundary, prescribed iota and prescribed current. The C++
# solver evaluates the spectral width on the state before the final time step,
# so the parity of specw needs a tight force tolerance.
_CASES = {
    "solovev": 1.0e-14,
    "cma": 1.0e-14,
    "cth_like_fixed_bdy": 1.0e-14,
    "up_down_asym": 1.0e-14,
    "cth_like_fixed_bdy_asym": 1.0e-14,
    "cth_like_free_bdy": 1.0e-14,
}


def _load_input(name: str, ftol: float) -> vmecpp.VmecInput:
    indata = vmecpp.VmecInput.from_file(TEST_DATA_DIR / f"{name}.json")
    ftol_array = np.array(indata.ftol_array, dtype=np.float64)
    ftol_array[-1] = min(ftol_array[-1], ftol)
    niter_array = np.array(indata.niter_array)
    niter_array[-1] = max(niter_array[-1], 50000)
    update = {"ftol_array": ftol_array, "niter_array": niter_array}
    if indata.lfreeb:
        update["mgrid_file"] = str(
            REPO_ROOT / "src" / "vmecpp" / "cpp" / indata.mgrid_file
        )
    return indata.model_copy(update=update)


def _assert_field_close(name, actual, expected, reference_wout) -> None:
    if expected is None or actual is None:
        assert actual is None, name
        assert expected is None, name
        return
    if isinstance(expected, str | bool | int) and not isinstance(expected, float):
        assert actual == expected, name
        return
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, name
    if expected.dtype.kind in "iub":
        np.testing.assert_array_equal(actual, expected, err_msg=name)
        return
    finite = np.isfinite(expected) & (np.abs(expected) < np.finfo(float).max)
    np.testing.assert_array_equal(actual[~finite], expected[~finite], err_msg=name)
    scale = np.abs(expected[finite]).max(initial=0.0)
    if name in _SCALE_OF:
        related = np.asarray(getattr(reference_wout, _SCALE_OF[name]))
        scale = max(scale, np.abs(related).max(initial=0.0))
    if name == "ctor":
        scale = max(scale, abs(reference_wout.rbtor) / autodiff_wout.MU_0)
    tolerance = _TOLERANCE.get(name, _DEFAULT_TOLERANCE)
    np.testing.assert_allclose(
        actual[finite],
        expected[finite],
        rtol=0.0,
        atol=tolerance * scale,
        err_msg=name,
    )


@pytest.fixture(scope="module", params=list(_CASES))
def solved_case(request):
    indata = _load_input(request.param, _CASES[request.param])
    output = _vmecpp.run(
        indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT, max_threads=1
    )
    return indata, output


def test_wout_matches_the_cpp_output_stage(solved_case) -> None:
    """Every VmecWOut field of the JAX output stage against the C++ one.

    Both evaluate the same converged state, so the comparison isolates the output stage
    from the solve.
    """
    indata, output = solved_case
    expected = vmecpp.VmecWOut._from_cpp_wout(output.wout)
    actual = vmecpp._wout_from_output_stage(indata, output)
    fields = own_model_fields(vmecpp.VmecWOut)
    assert set(autodiff_wout.WOUT_QUANTITIES) <= set(fields)
    for name in fields:
        _assert_field_close(
            name, getattr(actual, name), getattr(expected, name), expected
        )


def test_static_fields_match_the_cpp_output_stage(solved_case) -> None:
    indata, output = solved_case
    expected = vmecpp.VmecWOut._from_cpp_wout(output.wout)
    for name, value in autodiff_wout.static_fields(indata).items():
        if expected.lfreeb and name in {"mgrid_mode", "potvac", "xmpot", "xnpot"}:
            continue  # set by the vacuum solve
        _assert_field_close(name, value, getattr(expected, name), expected)


@pytest.mark.parametrize("name", ["solovev", "cth_like_fixed_bdy"])
def test_solver_diagnostics_match_the_cpp_wout(name) -> None:
    """autodiff.run's diagnostics come from VmecModel; vmecpp.run's from the C++ run."""
    indata = vmecpp.VmecInput.from_file(TEST_DATA_DIR / f"{name}.json")
    expected = vmecpp.run(indata, max_threads=1, verbose=False).wout
    boundary = np.stack([np.asarray(indata.rbc), np.asarray(indata.zbs)])
    model = autodiff._solve_model(indata._to_cpp_vmecindata(), boundary)
    diagnostics = autodiff._solver_diagnostics(model)
    assert set(diagnostics) == set(autodiff_wout.UNKNOWN_DIAGNOSTICS)
    for field, value in diagnostics.items():
        np.testing.assert_allclose(
            value, getattr(expected, field), rtol=1.0e-6, atol=0.0, err_msg=field
        )


def test_run_returns_the_output_stage_wout() -> None:
    indata = _load_input("solovev", 1.0e-12)
    wout = vmecpp.run(indata, max_threads=1, verbose=False).wout
    output = _vmecpp.run(
        indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT, max_threads=1
    )
    reference = vmecpp._wout_from_output_stage(indata, output)
    for name in autodiff_wout.WOUT_QUANTITIES:
        value = getattr(wout, name)
        assert value is None or isinstance(value, float | np.ndarray), name
        np.testing.assert_array_equal(value, getattr(reference, name), err_msg=name)


def test_wout_quantities_accept_a_prescribed_iota(solved_case) -> None:
    indata, output = solved_case
    wout = vmecpp.VmecWOut._from_cpp_wout(output.wout)
    quantities = autodiff_wout.wout_quantities(
        geometry.make(output),
        indata,
        iota_half=np.asarray(wout.iotas)[1:],
        mass_half=np.asarray(wout.mass)[1:] * autodiff_wout.MU_0,
    )
    np.testing.assert_allclose(
        np.asarray(quantities["iotas"]), wout.iotas, atol=1.0e-15
    )
    np.testing.assert_allclose(
        np.asarray(quantities["bsupumnc"]),
        wout.bsupumnc,
        atol=1.0e-12 * np.abs(wout.bsupumnc).max(),
    )


@pytest.fixture(scope="module")
def solovev_wout() -> vmecpp.VmecWOut:
    indata = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    return vmecpp.run(indata, max_threads=1, verbose=False).wout


def test_wout_pytree_round_trip(solovev_wout) -> None:
    leaves, treedef = jax.tree_util.tree_flatten(solovev_wout)
    assert len(leaves) == sum(
        getattr(solovev_wout, name) is not None
        for name in autodiff_wout.WOUT_QUANTITIES
    )
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    for name in own_model_fields(vmecpp.VmecWOut):
        np.testing.assert_array_equal(
            getattr(rebuilt, name), getattr(solovev_wout, name), err_msg=name
        )
    scaled = jax.tree_util.tree_map(lambda leaf: 2.0 * leaf, solovev_wout)
    np.testing.assert_allclose(scaled.bmnc, 2.0 * solovev_wout.bmnc)
    assert scaled.niter == solovev_wout.niter
    assert scaled.ns == solovev_wout.ns


def test_wout_diagnostics_do_not_key_the_jit_cache(solovev_wout) -> None:
    @jax.jit
    def scaled_aspect(wout):
        return 2.0 * wout.aspect

    other = solovev_wout.model_copy(update={"niter": solovev_wout.niter + 1})
    scaled_aspect(solovev_wout)
    scaled_aspect(other)
    assert scaled_aspect._cache_size() == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_wout_returned_from_jit(solovev_wout) -> None:
    rebuilt = jax.jit(lambda wout: wout)(solovev_wout)
    assert isinstance(rebuilt, vmecpp.VmecWOut)
    assert isinstance(rebuilt.bmnc, jax.Array)
    assert rebuilt.niter == solovev_wout.niter
    np.testing.assert_array_equal(np.asarray(rebuilt.bmnc), solovev_wout.bmnc)


def test_wout_validates_jax_scalars(solovev_wout) -> None:
    values = solovev_wout.model_dump()
    values["aspect"] = jnp.asarray(values["aspect"])
    values["bmnc"] = jnp.asarray(values["bmnc"])
    wout = vmecpp.VmecWOut.model_validate(values)
    assert isinstance(wout.aspect, jax.Array)
    reloaded = vmecpp.VmecWOut.model_validate_json(wout.model_dump_json())
    assert reloaded.aspect == pytest.approx(solovev_wout.aspect, rel=1.0e-15)


def _cth_like_input(
    ns: int = 15, ftol: float = 1.0e-16, niter: int = 5000
) -> vmecpp.VmecInput:
    """The CTH-like fixed-boundary case with a prescribed iota profile."""
    source = vmecpp.VmecInput.from_file(
        REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json"
    )
    return source.model_copy(
        update={
            "ncurr": 0,
            "piota_type": "power_series",
            "ai": np.asarray([0.35, 0.15]),
            "ns_array": np.asarray([ns]),
            "ftol_array": np.asarray([ftol]),
            "niter_array": np.asarray([niter]),
        }
    )


def _boundary(indata: vmecpp.VmecInput) -> jax.Array:
    return jnp.stack([jnp.asarray(indata.rbc), jnp.asarray(indata.zbs)])


def _requires_exact_derivatives(indata: vmecpp.VmecInput) -> None:
    ns = int(np.asarray(indata.ns_array)[-1])
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    if not model.has_exact_force_jacobian:
        pytest.skip("needs an Enzyme-enabled build for the exact residual transpose")


@requires_enzyme
def test_run_matches_vmecpp_run() -> None:
    """The differentiable entry point re-solves through VmecModel; the two solver paths
    agree to the force tolerance."""
    indata = _cth_like_input()
    reference = vmecpp.run(indata, max_threads=1, verbose=False).wout
    wout = autodiff.run(indata).wout
    assert isinstance(wout, vmecpp.VmecWOut)
    for name in autodiff_wout.WOUT_QUANTITIES:
        value = getattr(wout, name)
        expected = getattr(reference, name)
        if expected is None:
            assert value is None, name
            continue
        scale = max(np.abs(np.asarray(expected)).max(initial=0.0), 1.0e-300)
        np.testing.assert_allclose(
            np.asarray(value), expected, rtol=0.0, atol=1.0e-7 * scale, err_msg=name
        )
    for name, value in autodiff_wout.static_fields(indata).items():
        _assert_field_close(name, getattr(wout, name), value, reference)
    assert wout.ier_flag == 0
    assert wout.niter > 0
    assert wout.fsqt.size == wout.itfsq > 0
    assert max(float(wout.fsqr), float(wout.fsqz), float(wout.fsql)) <= float(
        indata.ftol_array[-1]
    )


@requires_enzyme
def test_run_is_usable_under_jit() -> None:
    indata = _cth_like_input()

    @jax.jit
    def solve(boundary):
        return autodiff.run(indata, boundary=boundary).wout

    wout = solve(_boundary(indata))
    assert np.isfinite(float(wout.aspect))
    assert wout.ier_flag == autodiff_wout.UNKNOWN_DIAGNOSTICS["ier_flag"]
    reference = autodiff.run(indata).wout.aspect
    np.testing.assert_allclose(float(wout.aspect), float(reference), rtol=1.0e-12)


@requires_enzyme
def test_run_result_is_a_pytree() -> None:
    indata = _cth_like_input()
    result = autodiff.run(indata)
    leaves = jax.tree_util.tree_leaves(result)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    rebuilt = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(result), leaves)
    np.testing.assert_array_equal(
        np.asarray(rebuilt.wout.bmnc), np.asarray(result.wout.bmnc)
    )
    assert rebuilt.wout.niter == result.wout.niter


def _quasisymmetry_proxy(wout: vmecpp.VmecWOut) -> jax.Array:
    mid = wout.bmnc.shape[1] // 2
    return jnp.sum(wout.bmnc[1:, mid] ** 2) / wout.bmnc[0, mid] ** 2


def _low_mode_direction(indata: vmecpp.VmecInput, seed: int) -> np.ndarray:
    """A random direction on the m <= 2, |n| <= 1 boundary modes, relative size 1e-3."""
    boundary = np.asarray(_boundary(indata))
    generator = np.random.default_rng(seed)
    direction = np.zeros_like(boundary)
    ntor = indata.ntor
    low = generator.standard_normal((2, 3, 3))
    low[1, 0, 1] = 0.0  # zbs(m=0, n=0) is not a degree of freedom
    direction[:, :3, ntor - 1 : ntor + 2] = low
    return direction / np.linalg.norm(direction) * 1.0e-3 * np.linalg.norm(boundary)


@requires_enzyme
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("objective_name", ["aspect", "quasisymmetry_proxy"])
def test_gradient_matches_central_differences(objective_name: str, seed: int) -> None:
    """jax.grad through the solve and the output stage against a central difference of
    the fully re-solved pipeline.

    With ftol = 1e-14 the re-solve noise in the objective is far below the difference
    quotient (floor of order 1e-8 relative at a 1e-3 relative step); the remaining
    discrepancy is the O(h^2) truncation of the central difference and the 1e-8 adjoint
    solve tolerance, measured at 1e-5 to 6e-5 relative.
    """
    indata = _cth_like_input(ns=25, ftol=1.0e-14, niter=20000)
    _requires_exact_derivatives(indata)

    def objective(boundary):
        wout = autodiff.run(indata, boundary=boundary).wout
        if objective_name == "aspect":
            return wout.aspect
        return _quasisymmetry_proxy(wout)

    boundary = _boundary(indata)
    value, gradient = jax.value_and_grad(objective)(boundary)
    gradient = np.asarray(gradient)
    assert np.all(np.isfinite(gradient))

    direction = _low_mode_direction(indata, seed)
    plus = float(objective(boundary + direction))
    minus = float(objective(boundary - direction))
    finite_difference = 0.5 * (plus - minus)
    directional = float(np.sum(gradient * direction))
    print(
        f"{objective_name} seed={seed}: value={float(value):.6e} "
        f"grad.d={directional:.6e} central={finite_difference:.6e} "
        f"rel diff={abs(directional - finite_difference) / abs(finite_difference):.2e}"
    )
    np.testing.assert_allclose(directional, finite_difference, rtol=2.0e-2)
