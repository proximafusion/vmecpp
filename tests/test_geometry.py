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
