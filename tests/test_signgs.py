from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def _mirror_image(vmec_input: vmecpp.VmecInput) -> vmecpp.VmecInput:
    """The equilibrium reflected through the midplane, described in the right-handed
    (signgs = +1) coordinates: Z -> -Z negates the sine coefficients of the boundary and
    of the axis, and the toroidal current, which is given in the laboratory sense,
    reverses with the reflection while the toroidal flux keeps its sign."""
    mirrored = vmec_input.model_copy(deep=True)
    mirrored.zbs = -np.asarray(mirrored.zbs)
    mirrored.zaxis_s = -np.asarray(mirrored.zaxis_s)
    mirrored.curtor = -mirrored.curtor
    mirrored.signgs = 1
    return mirrored


def test_signgs_must_be_a_sign():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    vmec_input.signgs = 2
    with pytest.raises((RuntimeError, AttributeError), match="signgs"):
        vmecpp.run(vmec_input, verbose=False)


def test_right_handed_coordinates_give_the_mirror_image():
    """Running the reflected boundary with signgs = +1 must reproduce the same physics
    as the original with signgs = -1: every scalar and profile is identical, the Fourier
    coefficients of R, lambda and |B| are identical, and the ones of Z, the Jacobian and
    the current are negated."""
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    wout = vmecpp.run(vmec_input, verbose=False).wout
    mirrored = vmecpp.run(_mirror_image(vmec_input), verbose=False).wout

    assert wout.signgs == -1
    assert mirrored.signgs == 1
    assert mirrored.niter == wout.niter

    identical = [
        "aspect",
        "volume_p",
        "wb",
        "wp",
        "betatotal",
        "betapol",
        "betator",
        "b0",
        "rbtor",
        "volavgB",
        "Aminor_p",
        "Rmajor_p",
        "iotaf",
        "iotas",
        "presf",
        "pres",
        "phipf",
        "bvco",
        "buco",
        "vp",
        "DMerc",
        "DShear",
        "DWell",
        "DCurr",
        "DGeod",
        "raxis_cc",
        "rmnc",
        "lmns",
        "lmns_full",
        "bmnc",
        "bsupumnc",
        "bsupvmnc",
        "bsubumnc",
        "bsubvmnc",
        "bsubsmns",
        "currumnc",
        "currvmnc",
        "bdotb",
    ]
    for name in identical:
        np.testing.assert_allclose(
            np.asarray(getattr(mirrored, name), dtype=float),
            np.asarray(getattr(wout, name), dtype=float),
            rtol=1e-10,
            atol=1e-13,
            err_msg=name,
        )
    negated = [
        "zmns",
        "zaxis_cs",
        "gmnc",
        "ctor",
        "jcurv",
        "jcuru",
        "jdotb",
        "chi",
        "phips",
    ]
    for name in negated:
        np.testing.assert_allclose(
            np.asarray(getattr(mirrored, name), dtype=float),
            -np.asarray(getattr(wout, name), dtype=float),
            rtol=1e-10,
            atol=1e-13,
            err_msg=name,
        )
