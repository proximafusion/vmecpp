import jax
import jax.numpy as jnp
import numpy as np

import vmecpp
from vmecpp import geometry
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)


def test_jax_geometry_matches_cpp_evaluator() -> None:
    indata = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    output = _vmecpp.run(indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT)
    cpp_geometry = _vmecpp.make_geometry(output)
    jax_geometry = geometry.from_cpp(cpp_geometry)
    coordinates = jnp.asarray([0.37, 0.42, -0.18])

    actual = np.asarray(geometry.evaluate(jax_geometry, coordinates))
    expected_point = cpp_geometry.evaluate(*map(float, coordinates))
    expected = np.asarray(
        [
            expected_point.r,
            expected_point.z,
            expected_point.lambda_,
            expected_point.toroidal_flux,
            expected_point.poloidal_flux,
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_geometry_of_a_domain_traversed_twice_matches_one_transit() -> None:
    """cth_like written as ten field periods over two transits has the geometry of the
    five-period run at every point, in the C++ evaluator and in the JAX one."""
    base = vmecpp.VmecInput.from_file(
        "src/vmecpp/cpp/vmecpp/test_data/cth_like_fixed_bdy.json"
    )
    twice = base.model_copy(update={"nfp": 2 * base.nfp, "toroidal_transits": 2})
    coordinates = [jnp.asarray([0.37, 0.42, zeta]) for zeta in (-0.18, 0.9, 2.5)]
    values = []
    for indata in (base, twice):
        output = _vmecpp.run(
            indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT
        )
        cpp_geometry = _vmecpp.make_geometry(output)
        jax_geometry = geometry.from_cpp(cpp_geometry)
        points = [cpp_geometry.evaluate(*map(float, c)) for c in coordinates]
        values.append(
            (
                np.asarray([[p.r, p.z, p.lambda_] for p in points]),
                np.asarray([geometry.evaluate(jax_geometry, c) for c in coordinates]),
            )
        )
    for once, twice_traversed in zip(values[0], values[1], strict=True):
        np.testing.assert_allclose(twice_traversed, once, rtol=1e-13, atol=1e-13)


def test_jax_geometry_has_state_and_coordinate_vjps() -> None:
    indata = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    output = _vmecpp.run(indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT)
    jax_geometry = geometry.make(output)
    coordinates = jnp.asarray([0.43, 0.31, 0.0])
    seed = jnp.arange(50.0).reshape(5, 10) / 50.0

    _, pullback = jax.vjp(geometry.evaluate, jax_geometry, coordinates)
    geometry_bar, coordinate_bar = pullback(seed)

    assert geometry_bar.r_cc.shape == jax_geometry.r_cc.shape
    assert np.isfinite(np.asarray(geometry_bar.r_cc)).all()
    assert np.isfinite(np.asarray(coordinate_bar)).all()
