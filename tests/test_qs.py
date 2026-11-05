import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import _qs, geometry
from vmecpp import simsopt_compat as qs
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


@pytest.fixture(scope="module")
def asymmetric():
    """A tilted three-dimensional boundary whose equilibrium fills every asymmetric
    array of the geometry."""
    source = _rippled_solovev(ns=51)
    rbs = np.zeros_like(source.rbc)
    zbc = np.zeros_like(source.rbc)
    zbc[1, 1] = 0.2
    rbs[1, 2] = zbc[1, 2] = rbs[2, 1] = 0.05
    output = _run(
        source.model_copy(
            update={
                "lasym": True,
                "rbs": rbs,
                "zbc": zbc,
                "raxis_s": np.zeros(2),
                "zaxis_c": np.zeros(2),
            }
        )
    )
    return geometry.from_cpp(_vmecpp.make_geometry(output)), output.wout


def test_asymmetric_field_strength_matches_the_vmec_spectrum(asymmetric) -> None:
    """|B| and its angular derivatives against ``bmnc`` and ``bmns``, and the part of
    |B| odd under (theta, zeta) -> (-theta, -zeta) against ``bmns`` alone.

    The half-grid spectrum agrees to first order in the radial spacing.
    """
    solved, wout = asymmetric
    surface = int(0.6 * (wout.ns - 1))
    theta = 2.0 * np.pi * np.arange(16) / 16
    zeta = 2.0 * np.pi * np.arange(16) / (16 * wout.nfp)
    fields = _qs._surface_fields(
        solved, (surface + 0.5) / (wout.ns - 1), jnp.asarray(theta), jnp.asarray(zeta)
    )
    m = np.asarray(wout.xm_nyq)[:, None, None]
    n = np.asarray(wout.xn_nyq)[:, None, None]
    angle = m * theta[:, None] - n * zeta
    bmnc = np.asarray(wout.bmnc)[:, surface, None, None]
    bmns = np.asarray(wout.bmns)[:, surface, None, None]
    cosine_part = np.sum(bmnc * np.cos(angle), axis=0)
    sine_part = np.sum(bmns * np.sin(angle), axis=0)
    d_theta = np.sum(m * (bmns * np.cos(angle) - bmnc * np.sin(angle)), axis=0)
    d_zeta = np.sum(n * (bmnc * np.sin(angle) - bmns * np.cos(angle)), axis=0)

    mod_b = np.asarray(fields["mod_b"])
    np.testing.assert_allclose(mod_b, cosine_part + sine_part, rtol=4.0e-3)
    np.testing.assert_allclose(
        0.5 * (mod_b - np.roll(mod_b[::-1, ::-1], 1, axis=(0, 1))),
        sine_part,
        rtol=0.0,
        atol=2.0e-2 * np.abs(sine_part).max(),
    )
    np.testing.assert_allclose(
        fields["d_mod_b_d_theta"],
        d_theta,
        rtol=0.0,
        atol=3.0e-2 * np.abs(d_theta).max(),
    )
    np.testing.assert_allclose(
        fields["d_mod_b_d_zeta"],
        d_zeta,
        rtol=0.0,
        atol=3.0e-2 * np.abs(d_zeta).max(),
    )


def test_asymmetric_surface_grid_matches_the_point_evaluation(asymmetric) -> None:
    """The tensor contractions against the explicit jets of ``geometry.evaluate``."""
    solved, _ = asymmetric
    s = 0.6
    theta = jnp.asarray([0.0, 1.1, 2.7, 4.3])
    zeta = jnp.asarray([0.0, 1.9, 0.8, 5.1])
    fields = _qs._surface_fields(solved, s, theta, zeta)
    for i in range(theta.size):
        point = jnp.asarray([s, theta[i], zeta[i]])
        frame = _qs._frame(solved, point)
        for name in (
            "sqrt_g",
            "b_sup_theta",
            "b_sup_zeta",
            "b_sub_theta",
            "b_sub_zeta",
            "mod_b",
        ):
            np.testing.assert_allclose(
                fields[name][i, i], frame[name], rtol=1.0e-13, err_msg=name
            )
        gradient = jax.grad(qs.magnetic_field_strength, argnums=1)(solved, point)
        np.testing.assert_allclose(
            fields["d_mod_b_d_theta"][i, i], gradient[1], rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            fields["d_mod_b_d_zeta"][i, i], gradient[2], rtol=0.0, atol=1.0e-14
        )


def _in_shifted_angles(solved, shift_theta: float, shift_zeta: float):
    """The geometry in the angles theta + shift_theta, zeta + shift_zeta."""
    m = np.arange(solved.r_cc.shape[1])[:, None]
    n = np.arange(solved.r_cc.shape[2])[None, :] * solved.nfp
    cos_m, sin_m = np.cos(m * shift_theta), np.sin(m * shift_theta)
    cos_n, sin_n = np.cos(n * shift_zeta), np.sin(n * shift_zeta)

    def shifted(cc, ss, sc, cs):
        cc, sc, cs, ss = (
            cc * cos_m - sc * sin_m,
            cc * sin_m + sc * cos_m,
            cs * cos_m - ss * sin_m,
            cs * sin_m + ss * cos_m,
        )
        return (
            cc * cos_n - cs * sin_n,
            sc * sin_n + ss * cos_n,
            sc * cos_n - ss * sin_n,
            cc * sin_n + cs * cos_n,
        )

    r_cc, r_ss, r_sc, r_cs = shifted(solved.r_cc, solved.r_ss, solved.r_sc, solved.r_cs)
    z_cc, z_ss, z_sc, z_cs = shifted(solved.z_cc, solved.z_ss, solved.z_sc, solved.z_cs)
    lambda_cc, lambda_ss, lambda_sc, lambda_cs = shifted(
        solved.lambda_cc, solved.lambda_ss, solved.lambda_sc, solved.lambda_cs
    )
    return geometry.Geometry(
        solved.toroidal_flux,
        solved.poloidal_flux,
        r_cc,
        r_ss,
        r_sc,
        r_cs,
        z_sc,
        z_cs,
        z_cc,
        z_ss,
        lambda_sc,
        lambda_cs,
        lambda_cc,
        lambda_ss,
        solved.nfp,
    )


@pytest.fixture(scope="module")
def rippled():
    return geometry.from_cpp(_vmecpp.make_geometry(_run(_rippled_solovev(ns=31))))


@pytest.mark.parametrize(
    ("cells_theta", "cells_zeta", "helicity_n"), [(1, 0, 0), (0, 1, 1), (3, 5, 1)]
)
def test_angles_shifted_by_grid_cells_shift_the_residuals(
    rippled, cells_theta: int, cells_zeta: int, helicity_n: int
) -> None:
    """A symmetric equilibrium in shifted angles fills the asymmetric arrays, and its
    residuals are the same numbers moved along the grid."""
    ntheta, nphi = 16, 12
    surfaces = [0.3, 0.7]

    def residuals(solved):
        return np.asarray(
            qs.quasisymmetry_residuals(
                solved,
                surfaces,
                helicity_m=1,
                helicity_n=helicity_n,
                ntheta=ntheta,
                nphi=nphi,
            )
        ).reshape(len(surfaces), ntheta, nphi)

    reference = residuals(rippled)
    actual = residuals(
        _in_shifted_angles(
            rippled,
            2.0 * np.pi * cells_theta / ntheta,
            2.0 * np.pi * cells_zeta / (rippled.nfp * nphi),
        )
    )
    np.testing.assert_allclose(
        actual,
        np.roll(reference, (cells_theta, cells_zeta), axis=(1, 2)),
        rtol=0.0,
        atol=1.0e-12 * np.abs(reference).max(),
        equal_nan=False,
    )
