# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Exact-equivalence tests for non-stellarator-symmetric (lasym) equilibria.

Each test applies a transformation to a stellarator-symmetric configuration that needs
the asymmetric representation (rbs/zbc) but reproduces the symmetric equilibrium's
physics, exercising the asymmetric DFTs end to end.
"""

from pathlib import Path

import netCDF4
import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

# The 12 asymmetric arrays a LASYM wout file must contain in addition to the
# axis coefficient pair raxis_cs/zaxis_cc.
ASYMMETRIC_WOUT_ARRAYS = [
    "rmns",
    "zmnc",
    "lmnc",
    "gmns",
    "bmns",
    "bsubumns",
    "bsubvmns",
    "bsubsmnc",
    "bsupumns",
    "bsupvmns",
    "currumns",
    "currvmns",
]


def _run(vmec_input):
    return vmecpp.run(vmec_input, max_threads=1, verbose=False).wout


def _enable_lasym(vmec_input, **overrides):
    """Return a lasym copy of a symmetric input with zero (then overridden) asym
    fields."""
    zeros2d = np.zeros_like(np.asarray(vmec_input.rbc))
    zeros1d = np.zeros(vmec_input.ntor + 1)
    update = {
        "lasym": True,
        "rbs": zeros2d.copy(),
        "zbc": zeros2d.copy(),
        "raxis_s": zeros1d.copy(),
        "zaxis_c": zeros1d.copy(),
    }
    update.update(overrides)
    return vmec_input.model_copy(update=update)


def _assert_same_physics(ref, test, vol_rtol, beta_atol, iota_atol):
    assert abs(test.volume_p - ref.volume_p) <= vol_rtol * abs(ref.volume_p)
    assert abs(test.betatotal - ref.betatotal) <= beta_atol
    assert np.max(np.abs(np.asarray(test.iotaf) - np.asarray(ref.iotaf))) <= iota_atol


def _assert_same_geometry(ref, test, atol):
    """Assert rmnc and zmns match coefficient-for-coefficient, not just scalars."""
    np.testing.assert_allclose(np.asarray(test.rmnc), np.asarray(ref.rmnc), atol=atol)
    np.testing.assert_allclose(np.asarray(test.zmns), np.asarray(ref.zmns), atol=atol)


def test_lasym_reduces_to_symmetric_2d():
    """2D: lasym=True with zero asymmetric content reproduces the symmetric run."""
    base = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    sym = _run(base)
    asym = _run(_enable_lasym(base))
    assert np.max(np.abs(np.asarray(asym.rmns))) < 1e-12
    _assert_same_geometry(sym, asym, atol=1e-12)
    _assert_same_physics(sym, asym, vol_rtol=1e-9, beta_atol=1e-9, iota_atol=1e-8)


def test_lasym_reduces_to_symmetric_3d():
    """3D: lasym=True with zero asymmetric content reproduces the symmetric run."""
    base = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    sym = _run(base)
    asym = _run(_enable_lasym(base))
    assert np.max(np.abs(np.asarray(asym.rmns))) < 1e-12
    _assert_same_geometry(sym, asym, atol=1e-12)
    _assert_same_physics(sym, asym, vol_rtol=1e-9, beta_atol=1e-9, iota_atol=1e-8)


def test_z_shift_preserves_physics():
    """A rigid z-shift needs the asymmetric zbc[0,0] but leaves the physics
    unchanged."""
    base = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    sym = _run(base)
    dz = 0.05
    zbc = np.zeros_like(np.asarray(base.zbs))
    zbc[0, base.ntor] = dz
    zaxis_c = np.zeros(base.ntor + 1)
    zaxis_c[0] = dz
    shifted = _run(_enable_lasym(base, zbc=zbc, zaxis_c=zaxis_c))
    _assert_same_physics(sym, shifted, vol_rtol=1e-9, beta_atol=1e-9, iota_atol=1e-8)
    # the wout z-shift lands in the m=0 asymmetric coefficients (cos parity)
    assert abs(np.asarray(shifted.zmnc)[0, -1] - dz) < 1e-10


def _toroidally_rotated_lasym_input(base, zeta0):
    """The boundary rotated by zeta0, in the equivalent lasym representation."""
    nfp, ntor, mpol = base.nfp, base.ntor, base.mpol
    rbc = np.asarray(base.rbc, float)
    zbs = np.asarray(base.zbs, float)
    rbc2 = np.zeros_like(rbc)
    rbs = np.zeros_like(rbc)
    zbs2 = np.zeros_like(zbs)
    zbc = np.zeros_like(zbs)
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            j = n + ntor
            c = np.cos(n * nfp * zeta0)
            s = np.sin(n * nfp * zeta0)
            rbc2[m, j] = rbc[m, j] * c
            rbs[m, j] = rbc[m, j] * s
            zbs2[m, j] = zbs[m, j] * c
            zbc[m, j] = -zbs[m, j] * s
    return _enable_lasym(base, rbc=rbc2, zbs=zbs2, rbs=rbs, zbc=zbc)


class CthRotation:
    """Shared toroidally-rotated CTH-like lasym run and its analytic inputs."""

    def __init__(self):
        base = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
        # converge tightly so iota is resolved well below the rotation's numerical
        # drift (the shipped input only asks for ftol 1e-6)
        base = base.model_copy(
            update={
                "ftol_array": np.full(np.asarray(base.ftol_array).shape, 1e-12),
                "niter_array": np.full(
                    np.asarray(base.niter_array).shape, 5000, dtype=np.int64
                ),
            }
        )
        self.base = base
        self.zeta0 = 0.05
        self.rbc = np.asarray(base.rbc, float)
        self.zbs = np.asarray(base.zbs, float)
        self.sym = _run(base)
        self.rotated = _run(_toroidally_rotated_lasym_input(base, self.zeta0))

    def boundary_reference(self, theta, zeta):
        """Analytic rotated boundary: the input boundary evaluated at zeta+zeta0."""
        nfp, ntor, mpol = self.base.nfp, self.base.ntor, self.base.mpol
        r_ref = np.zeros_like(theta)
        z_ref = np.zeros_like(theta)
        for m in range(mpol):
            for n in range(-ntor, ntor + 1):
                j = n + ntor
                angle = m * theta - n * nfp * (zeta + self.zeta0)
                r_ref += self.rbc[m, j] * np.cos(angle)
                z_ref += self.zbs[m, j] * np.sin(angle)
        return r_ref, z_ref


@pytest.fixture(scope="module")
def cth_rotation():
    return CthRotation()


def test_toroidal_rotation_preserves_physics(cth_rotation):
    """A toroidal rotation mixes symmetric modes into asymmetric ones; physics is
    unchanged."""
    rotated = cth_rotation.rotated
    # genuine asymmetric geometry was produced (not gauged away)
    assert np.max(np.abs(np.asarray(rotated.rmns))) > 1e-4
    _assert_same_physics(
        cth_rotation.sym, rotated, vol_rtol=1e-8, beta_atol=1e-8, iota_atol=1e-5
    )


def test_toroidal_rotation_wout_reconstructs_boundary(cth_rotation):
    """The returned rmnc/rmns/zmnc/zmns reconstruct the prescribed fixed boundary.

    Off-grid evaluation against the analytic input boundary, with nonzero m=0/n!=0
    asymmetric content and both signs of n exercised; no phase, sign, gain, or alignment
    freedom is allowed (regression for issue #675).
    """
    base = cth_rotation.base
    wout = cth_rotation.rotated
    nfp = base.nfp

    xm = np.asarray(wout.xm)
    xn = np.asarray(wout.xn)
    rmnc = np.asarray(wout.rmnc)[:, -1]
    zmns = np.asarray(wout.zmns)[:, -1]
    rmns = np.asarray(wout.rmns)[:, -1]
    zmnc = np.asarray(wout.zmnc)[:, -1]

    # m=0, n>0 discriminator: the rotation puts RBC[0,n]*sin(n*nfp*zeta0) into
    # rmns and -ZBS[0,n]*sin(n*nfp*zeta0) into zmnc; a symmetrized or
    # sign-flipped conversion zeroes or negates these.
    ntor = base.ntor
    for n in range(1, ntor + 1):
        (i,) = np.nonzero((xm == 0) & (xn == n * nfp))[0]
        j = n + ntor
        expected_rmns = cth_rotation.rbc[0, j] * np.sin(n * nfp * cth_rotation.zeta0)
        expected_zmnc = -cth_rotation.zbs[0, j] * np.sin(n * nfp * cth_rotation.zeta0)
        assert abs(expected_rmns) > 1e-6  # the discriminator has signal
        np.testing.assert_allclose(rmns[i], expected_rmns, rtol=0, atol=1e-10)
        np.testing.assert_allclose(zmnc[i], expected_zmnc, rtol=0, atol=1e-10)

    # off-grid boundary reconstruction against the analytic input boundary
    rng = np.random.default_rng(675)
    theta = rng.uniform(0.0, 2.0 * np.pi, 187)
    zeta = rng.uniform(0.0, 2.0 * np.pi / nfp, 187)
    angle = np.outer(xm, theta) - np.outer(xn, zeta)
    r_rec = rmnc @ np.cos(angle) + rmns @ np.sin(angle)
    z_rec = zmns @ np.sin(angle) + zmnc @ np.cos(angle)
    r_ref, z_ref = cth_rotation.boundary_reference(theta, zeta)

    num = np.sqrt(np.sum((r_rec - r_ref) ** 2 + (z_rec - z_ref) ** 2))
    den = np.sqrt(np.sum(r_ref**2 + z_ref**2))
    assert num / den < 1e-12
    assert np.max(np.abs(r_rec - r_ref)) < 1e-12
    assert np.max(np.abs(z_rec - z_ref)) < 1e-12


def test_lasym_wout_save_roundtrip(cth_rotation, tmp_path):
    """Save() must write a complete LASYM wout file that loads back identically."""
    wout = cth_rotation.rotated
    out_path = tmp_path / "wout_lasym_roundtrip.nc"
    wout.save(out_path)

    with netCDF4.Dataset(out_path, "r") as fnc:
        assert int(fnc["lasym__logical__"][()]) == 1
        missing = [v for v in ASYMMETRIC_WOUT_ARRAYS if v not in fnc.variables]
        assert missing == []
        assert "raxis_cs" in fnc.variables
        assert "zaxis_cc" in fnc.variables

    reloaded = vmecpp.VmecWOut.from_wout_file(out_path)
    for name in [
        *ASYMMETRIC_WOUT_ARRAYS,
        "rmnc",
        "zmns",
        "lmns",
        "raxis_cs",
        "zaxis_cc",
    ]:
        saved = np.asarray(getattr(wout, name))
        loaded = np.asarray(getattr(reloaded, name))
        assert saved.shape == loaded.shape
        np.testing.assert_array_equal(saved, loaded, err_msg=name)


# Every wout Fourier pair (cos-array, sin-array) obeys, mode by mode with
# phi = xn*zeta0, the rigid-rotation law C' = C cos(phi) - S sin(phi),
# S' = C sin(phi) + S cos(phi). Tolerances are ~3x the discretization-inherent
# law deviations, which PARVMEC 9.0 reproduces to the printed digit on this
# case; the lasym output defects fixed for issue #675 violated them by factors
# of 8 to 230.
ROTATION_LAW_PAIRS = [
    ("rmnc", "rmns", "xn", 2e-3),
    ("zmnc", "zmns", "xn", 2e-3),
    ("lmnc", "lmns", "xn", 5e-2),
    ("lmnc_full", "lmns_full", "xn", 8e-2),
    ("gmnc", "gmns", "xn_nyq", 1e-4),
    ("bmnc", "bmns", "xn_nyq", 2e-3),
    ("bsubumnc", "bsubumns", "xn_nyq", 1e-3),
    ("bsubvmnc", "bsubvmns", "xn_nyq", 5e-3),
    ("bsubsmnc", "bsubsmns", "xn_nyq", 5e-3),
    ("bsupumnc", "bsupumns", "xn_nyq", 2e-1),
    ("bsupvmnc", "bsupvmns", "xn_nyq", 5e-3),
    ("currumnc", "currumns", "xn_nyq", 1e4),
    ("currvmnc", "currvmns", "xn_nyq", 5e2),
]


def test_toroidal_rotation_law_all_wout_arrays(cth_rotation):
    """Every wout Fourier pair obeys the rigid-rotation law on every surface.

    The rotation angle is an integer number of zeta grid steps, so the rotated discrete
    problem is an exact grid permutation of the symmetric one and the law is exact up to
    the (code-independent) discretization asymmetry between the reduced-theta symmetric
    and full-theta lasym representations.
    """
    base = cth_rotation.base
    sym = cth_rotation.sym
    zeta0 = 3 * (2.0 * np.pi) / (base.nfp * base.nzeta)
    rotated = _run(_toroidally_rotated_lasym_input(base, zeta0))

    for cos_name, sin_name, xn_name, atol in ROTATION_LAW_PAIRS:
        cos_sym_attr = getattr(sym, cos_name)
        sin_sym_attr = getattr(sym, sin_name)
        ref = np.asarray(cos_sym_attr if cos_sym_attr is not None else sin_sym_attr)
        cos_sym = (
            np.asarray(cos_sym_attr) if cos_sym_attr is not None else np.zeros_like(ref)
        )
        sin_sym = (
            np.asarray(sin_sym_attr) if sin_sym_attr is not None else np.zeros_like(ref)
        )
        cos_rot = np.asarray(getattr(rotated, cos_name))
        sin_rot = np.asarray(getattr(rotated, sin_name))
        phi = (np.asarray(getattr(sym, xn_name), dtype=float) * zeta0)[:, None]
        cos_pred = cos_sym * np.cos(phi) - sin_sym * np.sin(phi)
        sin_pred = cos_sym * np.sin(phi) + sin_sym * np.cos(phi)
        np.testing.assert_allclose(
            cos_rot, cos_pred, rtol=0, atol=atol, err_msg=cos_name
        )
        np.testing.assert_allclose(
            sin_rot, sin_pred, rtol=0, atol=atol, err_msg=sin_name
        )

    # axis coefficient pairs follow the m=0 rows of the R and Z laws
    phi_axis = base.nfp * np.arange(base.ntor + 1, dtype=float) * zeta0
    np.testing.assert_allclose(
        np.asarray(rotated.raxis_cc),
        np.asarray(sym.raxis_cc) * np.cos(phi_axis),
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(rotated.raxis_cs),
        np.asarray(sym.raxis_cc) * np.sin(phi_axis),
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(rotated.zaxis_cs),
        np.asarray(sym.zaxis_cs) * np.cos(phi_axis),
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(rotated.zaxis_cc),
        -np.asarray(sym.zaxis_cs) * np.sin(phi_axis),
        rtol=0,
        atol=1e-4,
    )


def test_lasym_wout_save_failure_leaves_no_partial_file(cth_rotation, tmp_path):
    """A failing save() must not leave a partial wout file behind."""
    wout = cth_rotation.rotated
    out_path = tmp_path / "wout_partial.nc"
    sentinel = b"pre-existing contents"
    out_path.write_bytes(sentinel)

    # an extra field of unsupported type makes the writer raise after the
    # regular fields have already been written
    broken = wout.model_copy(update={"unwritable_extra": {1, 2, 3}})
    with pytest.raises(ValueError, match="unsupported type"):
        broken.save(out_path)

    # the previous file contents are untouched and no temporaries are left
    assert out_path.read_bytes() == sentinel
    assert list(tmp_path.glob("*.tmp")) == []
