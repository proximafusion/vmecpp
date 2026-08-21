from pathlib import Path

import numpy as np

import vmecpp

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


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
