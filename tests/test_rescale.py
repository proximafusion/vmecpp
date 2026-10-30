from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.fixture(scope="module")
def cth_like_free_boundary():
    """A converged three-dimensional free-boundary equilibrium, its input and the coil
    response table it was computed from (built from the coils file, so the test does not
    depend on the git-lfs mgrid file)."""
    makegrid_params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    makegrid_params.number_of_r_grid_points = 31
    makegrid_params.number_of_phi_grid_points = 36
    makegrid_params.number_of_z_grid_points = 20
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", makegrid_params
    )
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    return vmec_input, response, vmecpp.run(vmec_input, response, verbose=False)


def test_equilibrium_rescale():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    oq_initial = vmecpp.run(vmec_input)

    b_scale = 1.5
    r_scale = 2.0

    oq_rescaled = vmecpp.rescale(
        oq_initial, b_scale=b_scale, r_scale=r_scale, scale_pressure=True
    )

    input_scaled = oq_initial.input.model_copy(deep=True)
    input_scaled.phiedge *= b_scale * (r_scale**2)
    input_scaled.pres_scale *= b_scale**2
    input_scaled.curtor *= b_scale * r_scale

    input_scaled.rbc *= r_scale
    input_scaled.zbs *= r_scale
    if input_scaled.rbs is not None:
        input_scaled.rbs *= r_scale
    if input_scaled.zbc is not None:
        input_scaled.zbc *= r_scale

    input_scaled.raxis_c *= r_scale
    input_scaled.zaxis_s *= r_scale

    oq_full_run = vmecpp.run(input_scaled)

    # Volume should scale as r_scale^3
    np.testing.assert_allclose(
        oq_rescaled.wout.volume_p,
        oq_initial.wout.volume_p * (r_scale**3),
        rtol=1e-9,
        atol=1e-10 * (r_scale**3),
    )

    # Pressure should scale as b_scale^2
    np.testing.assert_allclose(
        oq_rescaled.wout.pres,
        oq_initial.wout.pres * (b_scale**2),
        rtol=1e-9,
        atol=1e-10 * (b_scale**2),
    )

    # Magnetic field should scale as b_scale
    np.testing.assert_allclose(
        oq_rescaled.wout.bmnc,
        oq_initial.wout.bmnc * b_scale,
        rtol=1e-9,
        atol=1e-10 * b_scale,
    )

    # Betas should be invariant under this scaling (pressure ~ B^2)
    np.testing.assert_allclose(
        oq_rescaled.wout.betatotal, oq_initial.wout.betatotal, rtol=1e-9, atol=1e-10
    )

    np.testing.assert_allclose(
        oq_rescaled.wout.bmnc, oq_full_run.wout.bmnc, rtol=1e-9, atol=1e-10 * b_scale
    )
    np.testing.assert_allclose(
        oq_rescaled.wout.rmnc, oq_full_run.wout.rmnc, rtol=1e-9, atol=1e-10 * r_scale
    )
    np.testing.assert_allclose(
        oq_rescaled.wout.zmns, oq_full_run.wout.zmns, rtol=1e-9, atol=1e-10 * r_scale
    )


def test_rescale_free_boundary_scales_the_coil_currents(cth_like_free_boundary):
    """The vacuum field is produced by the coil currents, so B -> b_scale * B holds only
    if extcur scales with it; otherwise the rescaled state is not in force balance
    against the unchanged external field."""
    vmec_input, response, oq_initial = cth_like_free_boundary
    b_scale = 1.5

    oq_rescaled = vmecpp.rescale(
        oq_initial, b_scale=b_scale, r_scale=1.0, magnetic_field=response
    )

    np.testing.assert_allclose(
        oq_rescaled.input.extcur,
        np.asarray(vmec_input.extcur, dtype=float) * b_scale,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        oq_rescaled.wout.bmnc,
        np.asarray(oq_initial.wout.bmnc) * b_scale,
        rtol=1e-9,
        atol=1e-10 * b_scale,
    )
    # still an equilibrium: leaving extcur unscaled leaves fsqr at order 1 here
    assert oq_rescaled.wout.fsqr < 1.0e-6
    assert oq_rescaled.wout.fsqz < 1.0e-6


def test_rescale_rejects_radial_scaling_of_a_free_boundary_equilibrium(
    cth_like_free_boundary,
):
    """The mgrid fixes the grid extent and the coil geometry, so r_scale cannot be
    applied to a free-boundary equilibrium."""
    _, _, oq_initial = cth_like_free_boundary

    with pytest.raises(ValueError, match="mgrid"):
        vmecpp.rescale(oq_initial, b_scale=1.0, r_scale=2.0)
