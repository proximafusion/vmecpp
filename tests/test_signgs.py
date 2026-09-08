"""Running an equilibrium in right-handed coordinates (signgs = +1) must give the same
physics as running its mirror image in the left-handed ones (signgs = -1)."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

# Reflecting the equilibrium through the midplane (Z -> -Z) and describing it in the
# right-handed coordinates leaves every scalar, every profile and the Fourier
# coefficients of R, lambda, |B| and the field components unchanged, and negates the
# ones of Z and of the Jacobian. The current reverses with the reflection while the
# toroidal flux keeps its sign, so the current diagnostics change sign as well.
IDENTICAL = [
    "aspect",
    "volume_p",
    "wb",
    "wp",
    "betatotal",
    "betapol",
    "betator",
    "betaxis",
    "b0",
    "rbtor",
    "rbtor0",
    "volavgB",
    "Aminor_p",
    "Rmajor_p",
    "rmax_surf",
    "rmin_surf",
    "zmax_surf",
    "iotaf",
    "iotas",
    "presf",
    "pres",
    "phipf",
    "chipf",
    "phi",
    "bvco",
    "buco",
    "vp",
    "specw",
    "DMerc",
    "DShear",
    "DWell",
    "DCurr",
    "DGeod",
    "raxis_cc",
    "raxis_cs",
    "rmnc",
    "rmns",
    "lmns",
    "lmnc",
    "lmns_full",
    "lmnc_full",
    "bmnc",
    "bmns",
    "bsupumnc",
    "bsupumns",
    "bsupvmnc",
    "bsupvmns",
    "bsubumnc",
    "bsubumns",
    "bsubvmnc",
    "bsubvmns",
    "bsubsmns",
    "bsubsmnc",
    "currumnc",
    "currumns",
    "currvmnc",
    "currvmns",
    "bdotb",
    "bdotgradv",
    "potvac",
]
NEGATED = [
    "signgs",
    "zmns",
    "zmnc",
    "zaxis_cs",
    "zaxis_cc",
    "gmnc",
    "gmns",
    "ctor",
    "jcurv",
    "jcuru",
    "jdotb",
]


def _mirror_image(vmec_input: vmecpp.VmecInput) -> vmecpp.VmecInput:
    """The equilibrium reflected through the midplane, in right-handed coordinates."""
    mirrored = vmec_input.model_copy(deep=True)
    mirrored.zbs = -np.asarray(mirrored.zbs)
    mirrored.zaxis_s = -np.asarray(mirrored.zaxis_s)
    if mirrored.lasym:
        mirrored.zbc = -np.asarray(mirrored.zbc)
        mirrored.zaxis_c = -np.asarray(mirrored.zaxis_c)
    mirrored.curtor = -mirrored.curtor
    mirrored.signgs = 1
    return mirrored


def _mirror_table(
    table: vmecpp.MagneticFieldResponseTable,
) -> vmecpp.MagneticFieldResponseTable:
    """The response table of the reflected coils carrying the reversed current: the
    cylindrical field components at (R, phi, -Z) with B_Z negated."""
    p = table.parameters
    shape = (
        -1,
        p.number_of_phi_grid_points,
        p.number_of_z_grid_points,
        p.number_of_r_grid_points,
    )

    def reflect(component, sign):
        flipped = sign * np.asarray(component).reshape(shape)[:, :, ::-1, :]
        return np.ascontiguousarray(flipped).reshape(flipped.shape[0], -1)

    parameters = p.model_copy(deep=True)
    parameters.z_grid_minimum = -p.z_grid_maximum
    parameters.z_grid_maximum = -p.z_grid_minimum
    return vmecpp.MagneticFieldResponseTable(
        parameters=parameters,
        b_r=reflect(table.b_r, 1.0),
        b_p=reflect(table.b_p, 1.0),
        b_z=reflect(table.b_z, -1.0),
    )


def _cth_like_response_table(
    raise_by: float = 0.0,
) -> vmecpp.MagneticFieldResponseTable:
    """The cth_like coils on the shipped makegrid parameters, optionally raised by
    `raise_by` in Z, which makes the vacuum field up-down asymmetric."""
    parameters = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    evaluated_on = parameters.model_copy(deep=True)
    # The field of the raised coils at Z is the field of the coils as given at
    # Z - raise_by, so it is evaluated on the grid shifted the other way and then
    # labelled with the unshifted grid. The shifted grid is not symmetric about the
    # midplane, so the half-period shortcut does not apply to it.
    evaluated_on.assume_stellarator_symmetry = False
    evaluated_on.z_grid_minimum -= raise_by
    evaluated_on.z_grid_maximum -= raise_by
    table = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", evaluated_on
    )
    return vmecpp.MagneticFieldResponseTable(
        parameters=parameters,
        b_r=np.array(table.b_r),
        b_p=np.array(table.b_p),
        b_z=np.array(table.b_z),
    )


def _assert_mirror_image(wout: vmecpp.VmecWOut, mirrored: vmecpp.VmecWOut) -> None:
    assert wout.signgs == -1
    assert mirrored.signgs == 1
    assert mirrored.niter == wout.niter
    for sign, names in ((1.0, IDENTICAL), (-1.0, NEGATED)):
        for name in names:
            expected = getattr(wout, name)
            actual = getattr(mirrored, name)
            if expected is None or actual is None:
                assert expected is None, name
                assert actual is None, name
                continue
            expected = np.asarray(expected, dtype=float)
            actual = np.asarray(actual, dtype=float)
            scale = np.abs(expected).max() if expected.size else 0.0
            np.testing.assert_allclose(
                actual, sign * expected, rtol=0.0, atol=1e-7 * scale, err_msg=name
            )


def test_signgs_must_be_a_sign():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    vmec_input.signgs = 2
    with pytest.raises((RuntimeError, AttributeError), match="signgs"):
        vmecpp.run(vmec_input, verbose=False)


# solovev is axisymmetric, li383_low_res starts without an axis guess and recomputes it
# after a bad initial Jacobian, and the two asymmetric cases pass through the poloidal
# shift of the boundary and the asymmetric force paths.
@pytest.mark.parametrize(
    "case",
    [
        "cth_like_fixed_bdy",
        "solovev",
        "li383_low_res",
        "cth_like_fixed_bdy_asym",
        "up_down_asym",
    ],
)
def test_right_handed_coordinates_give_the_mirror_image(case: str):
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / f"{case}.json")
    wout = vmecpp.run(vmec_input, verbose=False).wout
    mirrored = vmecpp.run(_mirror_image(vmec_input), verbose=False).wout
    _assert_mirror_image(wout, mirrored)


@pytest.mark.parametrize(
    ("case", "raise_coils_by"),
    [("cth_like_free_bdy", 0.0), ("cth_like_free_bdy_asym", 0.005)],
)
def test_right_handed_coordinates_give_the_mirror_image_free_boundary(
    case: str, raise_coils_by: float
):
    """The vacuum field of the reflected coils is the reflected response table."""
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / f"{case}.json")
    table = _cth_like_response_table(raise_coils_by)
    wout = vmecpp.run(vmec_input, magnetic_field=table, verbose=False).wout
    mirrored = vmecpp.run(
        _mirror_image(vmec_input), magnetic_field=_mirror_table(table), verbose=False
    ).wout
    _assert_mirror_image(wout, mirrored)
    if raise_coils_by != 0.0:
        # the raised coils move the plasma off the midplane
        assert abs(np.asarray(wout.zaxis_cc)[0]) > 1e-3


def test_hot_restart_from_a_right_handed_state():
    vmec_input = _mirror_image(
        vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy_asym.json")
    )
    output = vmecpp.run(vmec_input, verbose=False)
    restarted = vmecpp.run(vmec_input, restart_from=output, verbose=False)
    assert restarted.wout.niter <= 3
    np.testing.assert_allclose(
        restarted.wout.rmnc, output.wout.rmnc, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        restarted.wout.zmns, output.wout.zmns, rtol=0, atol=1e-12
    )


def test_flipping_signgs_alone_relabels_theta():
    """With only signgs changed, the boundary is flipped in theta to match the.

    requested handedness, so the wout describes the same equilibrium in the poloidal
    angle pi - theta: coefficient (m, n) becomes (-1)^m times coefficient (m, -n), the
    rotational transform changes sign and the current does not.
    """
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    wout = vmecpp.run(vmec_input, verbose=False).wout
    flipped_input = vmec_input.model_copy(deep=True)
    flipped_input.signgs = 1
    flipped = vmecpp.run(flipped_input, verbose=False).wout
    assert flipped.niter == wout.niter

    xm = np.asarray(wout.xm, dtype=int)
    xn = np.asarray(wout.xn, dtype=int)
    index = {(m, n): i for i, (m, n) in enumerate(zip(xm, xn, strict=True))}
    relabelled = np.array(
        [
            index[(m, -n)] if m > 0 else index[(0, n)]
            for m, n in zip(xm, xn, strict=True)
        ]
    )

    def relabel(coefficients, sine: bool):
        # the m = 0 rows fold n < 0 into n > 0, where a sine coefficient changes sign
        parity = np.where(xm > 0, (-1.0) ** xm, -1.0 if sine else 1.0)
        return parity[:, None] * np.asarray(coefficients)[relabelled, :]

    # R and lambda are relabelled as they stand; Z reverses with the poloidal
    # direction on top of that
    for name, sine, sign in (
        ("rmnc", False, 1.0),
        ("zmns", True, -1.0),
        ("lmns", True, 1.0),
    ):
        expected = np.asarray(getattr(wout, name))
        np.testing.assert_allclose(
            sign * relabel(getattr(flipped, name), sine),
            expected,
            rtol=0,
            atol=1e-7 * np.abs(expected).max(),
            err_msg=name,
        )
    np.testing.assert_allclose(flipped.iotaf, -np.asarray(wout.iotaf), rtol=1e-7)
    for name in ("volume_p", "wb", "betatotal", "ctor"):
        assert getattr(flipped, name) == pytest.approx(getattr(wout, name), rel=1e-7)
