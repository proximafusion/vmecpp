"""The JAX output stage against the C++ one, and gradients through it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff, autodiff_wout, geometry
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)

# Every array of WoutArrays with a counterpart in VmecWOut. The full-grid
# lambda is compared as lmns_full and, in its classic half-grid form, as lmns.
_COMPARED_ARRAYS = (
    "rmnc",
    "zmns",
    "lmns",
    "lmns_full",
    "gmnc",
    "bmnc",
    "bsubumnc",
    "bsubvmnc",
    "bsupumnc",
    "bsupvmnc",
    "bsubsmns",
    "iotas",
    "iotaf",
    "phi",
    "chi",
    "phipf",
    "chipf",
    "presf",
    "pres",
    "xm",
    "xn",
    "xm_nyq",
    "xn_nyq",
    "aspect",
    "volume_p",
    "volavgB",
    "betatotal",
    "b0",
)


def _cth_like_input(
    ns: int = 15, ftol: float = 1.0e-16, niter: int = 5000
) -> vmecpp.VmecInput:
    """The CTH-like fixed-boundary case with a prescribed iota profile."""
    source = vmecpp.VmecInput.from_file("examples/data/cth_like_fixed_bdy.json")
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


def _small_w7x_input() -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file("examples/data/w7x.json")
    mpol, ntor = 5, 4
    return source.model_copy(
        update={
            "mpol": mpol,
            "ntor": ntor,
            "rbc": vmecpp.VmecInput.resize_2d_coeff(source.rbc, mpol, ntor),
            "zbs": vmecpp.VmecInput.resize_2d_coeff(source.zbs, mpol, ntor),
            "raxis_c": np.asarray(source.raxis_c)[: ntor + 1],
            "zaxis_s": np.asarray(source.zaxis_s)[: ntor + 1],
            "ncurr": 0,
            "piota_type": "power_series",
            "ai": np.asarray([0.85, 0.15]),
            "ns_array": np.asarray([11]),
            "ftol_array": np.asarray([1.0e-14]),
            "niter_array": np.asarray([5000]),
        }
    )


def _boundary(indata: vmecpp.VmecInput) -> jax.Array:
    return jnp.stack([jnp.asarray(indata.rbc), jnp.asarray(indata.zbs)])


def _requires_exact_derivatives(indata: vmecpp.VmecInput) -> None:
    ns = int(np.asarray(indata.ns_array)[-1])
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    if not model.has_exact_force_jacobian:
        pytest.skip("needs an Enzyme-enabled build for the exact residual transpose")


def _reference(name: str, wout: vmecpp.VmecWOut) -> np.ndarray:
    return np.asarray(getattr(wout, name))


@pytest.fixture(scope="module", params=["cth_like", "w7x_small"])
def solved_case(request):
    indata = _cth_like_input() if request.param == "cth_like" else _small_w7x_input()
    output = _vmecpp.run(
        indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT, max_threads=1
    )
    wout = vmecpp.VmecWOut._from_cpp_wout(output.wout)
    return indata, geometry.make(output), wout


def test_wout_arrays_match_the_cpp_output_stage(solved_case) -> None:
    """Every array of the port against the C++ output stage of the same state.

    The geometry is taken from the completed C++ run, so the comparison isolates the
    output stage from the solve. Tolerance is roundoff relative to the largest entry of
    each array.
    """
    indata, solved, wout = solved_case
    actual = autodiff_wout.wout_arrays(solved, indata)
    for name in _COMPARED_ARRAYS:
        expected = _reference(name, wout)
        value = np.asarray(getattr(actual, name))
        assert value.shape == expected.shape, name
        scale = max(float(np.abs(expected).max()), 1.0e-300)
        difference = float(np.abs(value - expected).max())
        print(f"{name}: max abs diff {difference:.3e} (max |ref| {scale:.3e})")
        np.testing.assert_allclose(value, expected, rtol=0.0, atol=1.0e-12 * scale)


def test_wout_arrays_accept_a_prescribed_iota(solved_case) -> None:
    indata, solved, wout = solved_case
    iota_half = np.asarray(wout.iotas)[1:]
    actual = autodiff_wout.wout_arrays(solved, indata, iota_half=iota_half)
    np.testing.assert_allclose(np.asarray(actual.iotas), wout.iotas, atol=1.0e-15)
    np.testing.assert_allclose(np.asarray(actual.bsupumnc), wout.bsupumnc, atol=1.0e-13)


def test_run_matches_vmecpp_run() -> None:
    """The differentiable entry point re-solves through VmecModel; the two solver paths
    agree to the force tolerance."""
    indata = _cth_like_input()
    reference = vmecpp.run(indata, max_threads=1, verbose=False).wout
    result = autodiff.run(indata)
    for name in ("rmnc", "zmns", "bmnc", "gmnc", "iotaf"):
        np.testing.assert_allclose(
            np.asarray(getattr(result.wout, name)),
            _reference(name, reference),
            rtol=0.0,
            atol=1.0e-9 * float(np.abs(_reference(name, reference)).max()),
        )
    np.testing.assert_allclose(float(result.wout.aspect), reference.aspect, rtol=1.0e-9)
    np.testing.assert_allclose(
        float(result.wout.volume_p), reference.volume_p, rtol=1.0e-9
    )


def test_vmecpp_run_differentiable_flag_dispatches() -> None:
    indata = _cth_like_input()
    result = vmecpp.run(indata, differentiable=True)
    assert isinstance(result, autodiff.DifferentiableRun)
    assert isinstance(result.wout, autodiff_wout.WoutArrays)
    assert result.wout.bmnc.shape == (len(result.wout.xm_nyq), 15)
    with pytest.raises(ValueError, match="restart_from"):
        vmecpp.run(
            indata, differentiable=True, restart_from=vmecpp.run(indata, verbose=False)
        )


def test_run_is_usable_under_jit() -> None:
    indata = _cth_like_input()

    @jax.jit
    def aspect(boundary):
        return autodiff.run(indata, boundary=boundary).wout.aspect

    value = aspect(_boundary(indata))
    assert np.isfinite(float(value))
    reference = autodiff.run(indata).wout.aspect
    np.testing.assert_allclose(float(value), float(reference), rtol=1.0e-12)


def test_wout_arrays_are_a_pytree() -> None:
    indata = _cth_like_input()
    result = autodiff.run(indata)
    leaves = jax.tree_util.tree_leaves(result)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    rebuilt = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(result), leaves)
    np.testing.assert_array_equal(
        np.asarray(rebuilt.wout.bmnc), np.asarray(result.wout.bmnc)
    )


def _quasisymmetry_proxy(wout: autodiff_wout.WoutArrays) -> jax.Array:
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
