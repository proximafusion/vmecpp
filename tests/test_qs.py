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


def test_reconstructed_field_strength_matches_the_vmec_spectrum() -> None:
    """|B| rebuilt from the geometry jets must be VMEC's own |B|.

    The axisymmetry test above cannot establish this: a zeta-independent
    equilibrium gives zero non-quasi-axisymmetric power for any zeta-independent
    function of the geometry, correct or not. VMEC's ``bmnc`` spectrum is an
    independent oracle for the reconstruction itself.

    ``bmnc`` is a half-grid quantity while the geometry evaluator interpolates on
    the full grid, so the comparison carries a first-order radial offset; a wrong
    reconstruction would be off by tens of percent, not by a fraction of one.
    """
    source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    indata = source.model_copy(
        update={
            "ns_array": np.asarray([51]),
            "ftol_array": np.asarray([1.0e-14]),
            "niter_array": np.asarray([20000]),
        }
    )
    output = _vmecpp.run(indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT)
    wout = output.wout
    jax_geometry = geometry.from_cpp(_vmecpp.make_geometry(output))

    bmnc = np.asarray(wout.bmnc)
    if bmnc.shape[1] == wout.ns:
        bmnc = bmnc.T
    xm = np.asarray(wout.xm_nyq)
    xn = np.asarray(wout.xn_nyq)

    surface = int(0.6 * (wout.ns - 1))
    s = (surface + 0.5) / (wout.ns - 1)  # the half-grid location of bmnc[surface]
    for theta, zeta in ((0.3, 0.2), (1.1, -0.4), (2.4, 0.9)):
        expected = float(np.sum(bmnc[surface] * np.cos(xm * theta - xn * zeta)))
        actual = float(
            qs.magnetic_field_strength(jax_geometry, jnp.asarray([s, theta, zeta]))
        )
        assert abs(actual - expected) / abs(expected) < 4.0e-3
