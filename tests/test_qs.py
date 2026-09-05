import jax
import jax.numpy as jnp
import numpy as np

import vmecpp
from vmecpp import geometry, qs
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)


def _run(indata):
    return _vmecpp.run(indata._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT)


def _solovev(ns: int = 51) -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    return source.model_copy(
        update={
            "ns_array": np.asarray([ns]),
            "ftol_array": np.asarray([1.0e-14]),
            "niter_array": np.asarray([20000]),
        }
    )


def _rippled_solovev(ns: int = 51, ripple: float = 0.01) -> vmecpp.VmecInput:
    """Solovev with one toroidal boundary mode, so it is genuinely 3D."""
    source = vmecpp.VmecInput.from_file("examples/data/solovev.json")
    mpol = source.mpol
    rbc = np.zeros((mpol, 3))
    zbs = np.zeros((mpol, 3))
    rbc[:, 1] = np.asarray(source.rbc)[:, 0]
    zbs[:, 1] = np.asarray(source.zbs)[:, 0]
    rbc[1, 2] = ripple
    zbs[1, 2] = ripple
    return source.model_copy(
        update={
            "ntor": 1,
            "rbc": rbc,
            "zbs": zbs,
            "raxis_c": np.asarray([4.0, 0.0]),
            "zaxis_s": np.asarray([0.0, 0.0]),
            "ns_array": np.asarray([ns]),
            "ftol_array": np.asarray([1.0e-14]),
            "niter_array": np.asarray([20000]),
        }
    )


def test_reconstructed_field_strength_matches_the_vmec_spectrum() -> None:
    """|B| rebuilt from the geometry jets must be VMEC's own |B|.

    A quasisymmetry residual near zero cannot establish this on its own: an
    axisymmetric equilibrium gives zero non-quasi-axisymmetric power for any
    zeta-independent function of the geometry, correct or not. VMEC's ``bmnc``
    spectrum is an independent oracle for the reconstruction itself.

    ``bmnc`` is a half-grid quantity while the evaluator interpolates on the
    full grid, so the agreement is first order in the radial spacing: the
    relative difference at one point falls 5.19e-3, 2.59e-3, 1.33e-3, 6.62e-4
    for ns = 31, 61, 121, 241. A wrong reconstruction would be off by tens of
    percent.
    """
    output = _run(_solovev(ns=51))
    wout = output.wout
    jax_geometry = geometry.from_cpp(_vmecpp.make_geometry(output))

    bmnc = np.asarray(wout.bmnc)
    if bmnc.shape[1] == wout.ns:
        bmnc = bmnc.T
    xm = np.asarray(wout.xm_nyq)
    xn = np.asarray(wout.xn_nyq)

    surface = int(0.6 * (wout.ns - 1))
    s = (surface + 0.5) / (wout.ns - 1)
    for theta, zeta in ((0.3, 0.2), (1.1, -0.4), (2.4, 0.9)):
        expected = float(np.sum(bmnc[surface] * np.cos(xm * theta - xn * zeta)))
        actual = float(
            qs.magnetic_field_strength(jax_geometry, jnp.asarray([s, theta, zeta]))
        )
        assert abs(actual - expected) / abs(expected) < 4.0e-3


def test_matches_simsopt_quasisymmetry_ratio_residual(tmp_path) -> None:
    """The objective is SIMSOPT's, so SIMSOPT is the oracle.

    This is the whole point of the rewrite: a pure Python and JAX objective
    that computes the same number as ``QuasisymmetryRatioResidual``, including
    the ``sqrt(g)`` flux-surface measure and the multi-surface sum.
    """
    simsopt_vmec = __import__("simsopt.mhd", fromlist=["Vmec"])
    diagnostics = __import__(
        "simsopt.mhd.vmec_diagnostics", fromlist=["QuasisymmetryRatioResidual"]
    )

    indata = _rippled_solovev(ns=51)
    output = _run(indata)
    jax_geometry = geometry.from_cpp(_vmecpp.make_geometry(output))

    wout_path = tmp_path / "wout_qs_reference.nc"
    vmecpp.run(indata, verbose=False).wout.save(wout_path)

    surfaces = [0.3, 0.6, 0.9]
    reference = diagnostics.QuasisymmetryRatioResidual(
        simsopt_vmec.Vmec(str(wout_path)),
        surfaces,
        helicity_m=1,
        helicity_n=0,
        ntheta=63,
        nphi=64,
    ).compute()

    actual = np.asarray(
        qs.quasisymmetry_residuals(
            jax_geometry, surfaces, helicity_m=1, helicity_n=0, ntheta=63, nphi=64
        )
    )
    expected = reference.residuals1d
    cosine = float(
        expected @ actual / np.linalg.norm(expected) / np.linalg.norm(actual)
    )
    assert cosine > 0.9999
    np.testing.assert_allclose(float(np.sum(actual**2)), reference.total, rtol=1.0e-3)


def test_qs_objective_is_differentiable_through_geometry() -> None:
    output = _run(_rippled_solovev(ns=31))
    base = geometry.from_cpp(_vmecpp.make_geometry(output))

    def objective(scale):
        scaled = jax.tree_util.tree_map(lambda leaf: leaf * scale, base)
        return qs.quasisymmetry_total(
            scaled, [0.6], helicity_m=1, helicity_n=0, ntheta=16, nphi=16
        )

    value, derivative = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert float(value) > 0.0
    assert np.isfinite(float(derivative))
