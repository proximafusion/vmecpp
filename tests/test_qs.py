import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import vmecpp
from vmecpp import geometry, qs
from vmecpp.cpp import _vmecpp  # type: ignore


def _solovev_geometry() -> geometry.Geometry:
    indata = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    output = _vmecpp.run(indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT)
    return geometry.make(output)


def test_axisymmetric_geometry_has_zero_qa_residual() -> None:
    residual = qs.quasisymmetry_residual(_solovev_geometry(), 0.6, ntheta=12, nzeta=12)
    assert float(residual) < 1e-12


def test_qs_objective_is_differentiable_through_geometry() -> None:
    base = _solovev_geometry()
    shape = (*base.r_cc.shape[:2], 2)
    r_cc = jnp.zeros(shape)
    r_cc = r_cc.at[..., 0].set(base.r_cc[..., 0])
    r_cc = r_cc.at[:, 1, 1].set(0.02)
    perturbed = dataclasses.replace(
        base,
        r_cc=r_cc,
        r_ss=jnp.zeros(shape),
        r_sc=jnp.zeros(shape),
        r_cs=jnp.zeros(shape),
        z_sc=jnp.pad(base.z_sc, ((0, 0), (0, 0), (0, 1))),
        z_cs=jnp.zeros(shape),
        z_cc=jnp.zeros(shape),
        z_ss=jnp.zeros(shape),
        lambda_sc=jnp.pad(base.lambda_sc, ((0, 0), (0, 0), (0, 1))),
        lambda_cs=jnp.zeros(shape),
        lambda_cc=jnp.zeros(shape),
        lambda_ss=jnp.zeros(shape),
        nfp=1,
    )

    objective = lambda amplitude: qs.quasisymmetry_residual(  # noqa: E731
        dataclasses.replace(perturbed, r_cc=perturbed.r_cc.at[:, 1, 1].set(amplitude)),
        0.6,
        ntheta=12,
        nzeta=12,
    )
    value, derivative = jax.value_and_grad(objective)(jnp.asarray(0.02))
    assert float(value) > 0.0
    assert np.isfinite(float(derivative))
    assert abs(float(derivative)) > 0.0
