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


def _numeric_fields(model, prefix=""):
    """Every numeric field of an output model, recursing into nested models."""
    fields = {}
    for name in type(model).model_fields:
        if name == "input":
            continue
        value = getattr(model, name)
        if hasattr(type(value), "model_fields"):
            fields.update(_numeric_fields(value, prefix + name + "."))
        elif isinstance(value, (bool, str)):
            continue
        elif isinstance(value, (int, float, np.ndarray)):
            fields[prefix + name] = np.asarray(value, dtype=float)
    return fields


def test_lasym_reduces_to_symmetric_3d_all_outputs():
    """3D: every jxbout, Mercier, threed1 and wout quantity of the lasym run
    reproduces the symmetric run on the shared poloidal range."""
    base = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    sym = vmecpp.run(base, max_threads=1, verbose=False)
    asym = vmecpp.run(_enable_lasym(base), max_threads=1, verbose=False)

    nzeta = base.nzeta
    # theta is the fastest index within a surface; the symmetric run stores the
    # reduced poloidal range, the lasym run the full one
    n_theta_reduced = sym.jxbout.itheta.shape[1] // nzeta
    n_theta_full = asym.jxbout.itheta.shape[1] // nzeta
    assert n_theta_full == 2 * (n_theta_reduced - 1)

    sym_fields = _numeric_fields(sym)
    asym_fields = _numeric_fields(asym)
    for name, expected in sym_fields.items():
        if expected.size == 0:
            continue
        actual = asym_fields[name]
        reference = expected
        if actual.shape != reference.shape:
            assert actual.shape[:-1] == reference.shape[:-1], name
            if reference.shape[-1] == n_theta_reduced:
                actual = actual[..., :n_theta_reduced]
            else:
                assert reference.shape[-1] == nzeta * n_theta_reduced, name
                actual = actual.reshape((*actual.shape[:-1], nzeta, n_theta_full))
                actual = actual[..., :n_theta_reduced]
                reference = reference.reshape(
                    (*reference.shape[:-1], nzeta, n_theta_reduced)
                )
        scale = max(np.max(np.abs(reference)), 1e-300)
        np.testing.assert_allclose(
            actual, reference, rtol=1e-8, atol=1e-8 * scale, err_msg=name
        )


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


def _toroidally_rotated_input(vmec_input, zeta0):
    """A lasym input with boundary and axis rotated by zeta0: X'(zeta) = X(zeta +
    zeta0)."""
    nfp, ntor = vmec_input.nfp, vmec_input.ntor
    phase = nfp * np.arange(-ntor, ntor + 1) * zeta0
    c, s = np.cos(phase), np.sin(phase)
    rbc, rbs, zbs, zbc = (
        np.asarray(getattr(vmec_input, name), float)
        for name in ("rbc", "rbs", "zbs", "zbc")
    )
    phase_axis = nfp * np.arange(ntor + 1) * zeta0
    ca, sa = np.cos(phase_axis), np.sin(phase_axis)
    raxis_c, raxis_s, zaxis_s, zaxis_c = (
        np.asarray(getattr(vmec_input, name), float)
        for name in ("raxis_c", "raxis_s", "zaxis_s", "zaxis_c")
    )
    return vmec_input.model_copy(
        update={
            "rbc": rbc * c - rbs * s,
            "rbs": rbs * c + rbc * s,
            "zbs": zbs * c + zbc * s,
            "zbc": zbc * c - zbs * s,
            "raxis_c": raxis_c * ca - raxis_s * sa,
            "raxis_s": raxis_s * ca + raxis_c * sa,
            "zaxis_s": zaxis_s * ca + zaxis_c * sa,
            "zaxis_c": zaxis_c * ca - zaxis_s * sa,
        }
    )


class CthRotation:
    """Shared toroidally-rotated CTH-like lasym run and its analytic inputs."""

    def __init__(self):
        base = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
        # converge tightly so the comparisons are not limited by the iteration
        # tolerance (the shipped input only asks for ftol 1e-6)
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
        self.rotated = _run(_toroidally_rotated_input(_enable_lasym(base), self.zeta0))

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
        cth_rotation.sym, rotated, vol_rtol=1e-12, beta_atol=1e-12, iota_atol=1e-10
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


# (cos, sin) Fourier pairs of the output and the mode numbers that set their
# toroidal phase xn * zeta0 (axis: n * nfp * zeta0).
OUTPUT_FOURIER_PAIRS = [
    *(
        (f"wout.{c}", f"wout.{s}", xn)
        for c, s, xn in [
            ("rmnc", "rmns", "xn"),
            ("zmnc", "zmns", "xn"),
            ("lmnc", "lmns", "xn"),
            ("lmnc_full", "lmns_full", "xn"),
            ("gmnc", "gmns", "xn_nyq"),
            ("bmnc", "bmns", "xn_nyq"),
            ("bsubumnc", "bsubumns", "xn_nyq"),
            ("bsubvmnc", "bsubvmns", "xn_nyq"),
            ("bsubsmnc", "bsubsmns", "xn_nyq"),
            ("bsupumnc", "bsupumns", "xn_nyq"),
            ("bsupvmnc", "bsupvmns", "xn_nyq"),
            ("currumnc", "currumns", "xn_nyq"),
            ("currvmnc", "currvmns", "xn_nyq"),
        ]
    ),
    ("wout.raxis_cc", "wout.raxis_cs", "axis"),
    ("wout.zaxis_cc", "wout.zaxis_cs", "axis"),
    ("threed1_axis.raxis_symm", "threed1_axis.raxis_asym", "axis"),
    ("threed1_axis.zaxis_asym", "threed1_axis.zaxis_symm", "axis"),
]


def _pair_phases(wout, kind, zeta0):
    if kind == "axis":
        return wout.nfp * np.arange(wout.ntor + 1) * zeta0
    return (np.asarray(getattr(wout, kind), float) * zeta0)[:, np.newaxis]


def _output_rotated_back(output, zeta0, zeta_steps):
    """Every numeric output field, with the toroidal rotation by zeta0 undone.

    Fourier pairs rotate by their mode phase; the real-space jxbout arrays are rolled by
    zeta_steps grid points along zeta, which requires zeta0 to be a whole number of zeta
    grid steps.
    """
    fields = _numeric_fields(output)
    for cos_name, sin_name, kind in OUTPUT_FOURIER_PAIRS:
        phi = _pair_phases(output.wout, kind, zeta0)
        c, s = fields[cos_name], fields[sin_name]
        fields[cos_name] = c * np.cos(phi) + s * np.sin(phi)
        fields[sin_name] = -c * np.sin(phi) + s * np.cos(phi)
    nzeta = output.input.nzeta
    for name, value in fields.items():
        if name.startswith("jxbout.") and value.ndim == 2:
            grid = value.reshape(value.shape[0], nzeta, -1)
            fields[name] = np.roll(grid, zeta_steps, axis=1).reshape(value.shape)
    return fields


def _assert_fields_match(expected, actual, rtol, atol, names):
    """Report every mismatched field, not only the first."""
    mismatches = []
    for name in sorted(names):
        try:
            np.testing.assert_allclose(
                actual[name], expected[name], rtol=rtol, atol=atol, err_msg=name
            )
        except AssertionError as error:
            mismatches.append(str(error))
    assert not mismatches, "\n".join(mismatches)


# Not invariant under a toroidal rotation: quantities evaluated in the zeta = 0 and
# zeta = pi / nfp planes, and the final force residuals.
ROTATION_VARIANT_FIELDS = {
    "wout.b0",
    "wout.fsqr",
    "wout.fsqz",
    "wout.fsql",
    "wout.fsqt",
    *(
        f"threed1_geometric_magnetic.{name}"
        for name in (
            "b0",
            "rcen",
            "aminr1",
            "waist",
            "height",
            "ygeo",
            "yinden",
            "yellip",
            "ytrian",
            "yshift",
        )
    ),
    *(f"threed1_shafranov_integrals.{name}" for name in ("f_geo", "delta2", "s12")),
}

# Force-balance residuals, which vanish at convergence up to the iteration noise.
RESIDUAL_FIELDS = {
    "wout.equif",
    "threed1_first_table.radial_force",
    "jxbout.jsups3",
    "jxbout.amaxfor",
    "jxbout.aminfor",
    "jxbout.avforce",
    "jxbout.jcrossb",
    "jxbout.jxb_gradp",
}

# Current densities and quantities derived from them, which reach 1e12 and take their
# round-off from radial derivatives.
CURRENT_DENSITY_FIELDS = {
    *(
        f"wout.{name}"
        for name in (
            "currumnc",
            "currumns",
            "currvmnc",
            "currvmns",
            "jcuru",
            "jcurv",
            "jdotb",
            "ctor",
        )
    ),
    *(
        f"jxbout.{name}"
        for name in (
            "itheta",
            "izeta",
            "bdotk",
            "jdotb",
            "jdotb_sqrtg",
            "jsupu3",
            "jsupv3",
            "jpar2",
            "jperp2",
        )
    ),
    *(f"threed1_first_table.{name}" for name in ("avg_jsupu", "avg_jsupv", "j_dot_b")),
    *(
        f"threed1_geometric_magnetic.{name}"
        for name in (
            "toroidal_current",
            "jpar_perp",
            "jparPS_perp",
            "loc_jpar_perp",
            "loc_jparPS_perp",
        )
    ),
}


def _lasym_input(name, **overrides):
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / f"{name}.json")
    update = {
        "ftol_array": np.full(len(vmec_input.ftol_array), 1e-12),
        "niter_array": np.full(len(vmec_input.niter_array), 20000, dtype=np.int64),
    }
    update.update(overrides)
    vmec_input = vmec_input.model_copy(update=update)
    return vmec_input if vmec_input.lasym else _enable_lasym(vmec_input)


# 3D stellarator-symmetric content, a non-stellarator-symmetric boundary, and a
# beta = 4.3% case.
ROTATION_CASES = {
    "cth_like_fixed_bdy": {},
    "cth_like_fixed_bdy_asym": {},
    "li383_low_res": {"nzeta": 12},
}


@pytest.mark.parametrize("case", ROTATION_CASES)
def test_toroidal_rotation_equivariance(case):
    """Rotating boundary and axis by whole zeta grid steps permutes the discrete
    problem: undoing the rotation on the output reproduces the unrotated lasym run
    field by field, and the force-residual progression is the same."""
    base = _lasym_input(case, **ROTATION_CASES[case])
    zeta_steps = 3
    zeta0 = zeta_steps * 2.0 * np.pi / (base.nfp * base.nzeta)
    reference = vmecpp.run(base, max_threads=1, verbose=False)
    rotated = vmecpp.run(
        _toroidally_rotated_input(base, zeta0), max_threads=1, verbose=False
    )

    assert rotated.wout.itfsq == reference.wout.itfsq
    np.testing.assert_allclose(rotated.wout.fsqt, reference.wout.fsqt, rtol=1e-4)
    expected = _numeric_fields(reference)
    actual = _output_rotated_back(rotated, zeta0, zeta_steps)
    fields = expected.keys() - ROTATION_VARIANT_FIELDS
    _assert_fields_match(
        expected,
        actual,
        rtol=1e-8,
        atol=1e-8,
        names=fields - RESIDUAL_FIELDS - CURRENT_DENSITY_FIELDS,
    )
    _assert_fields_match(
        expected, actual, rtol=1e-8, atol=1e-3, names=CURRENT_DENSITY_FIELDS
    )
    _assert_fields_match(expected, actual, rtol=1e-5, atol=1e-4, names=RESIDUAL_FIELDS)


def test_toroidal_rotation_off_grid_preserves_flux_functions():
    """A rotation by a fraction of a zeta grid step leaves the flux-surface quantities
    unchanged up to the aliasing of the zeta grid."""
    base = _lasym_input("cth_like_fixed_bdy")
    zeta0 = 1.37 * 2.0 * np.pi / (base.nfp * base.nzeta)
    reference = _run(base)
    rotated = _run(_toroidally_rotated_input(base, zeta0))
    for name in [
        "volume_p",
        "betatotal",
        "wb",
        "wp",
        "iotaf",
        "vp",
        "DWell",
        "DMerc",
        "jcurv",
        "bvco",
    ]:
        expected = np.asarray(getattr(reference, name))
        np.testing.assert_allclose(
            getattr(rotated, name),
            expected,
            rtol=0,
            atol=1e-8 * np.max(np.abs(expected)),
            err_msg=name,
        )


def _poloidally_shifted_input(vmec_input, theta0):
    """The same boundary with theta -> theta + theta0."""
    phase = np.arange(vmec_input.mpol)[:, np.newaxis] * theta0
    c, s = np.cos(phase), np.sin(phase)
    rbc, rbs, zbs, zbc = (
        np.asarray(getattr(vmec_input, name), float)
        for name in ("rbc", "rbs", "zbs", "zbc")
    )
    return vmec_input.model_copy(
        update={
            "rbc": rbc * c + rbs * s,
            "rbs": rbs * c - rbc * s,
            "zbs": zbs * c - zbc * s,
            "zbc": zbc * c + zbs * s,
        }
    )


def _poloidally_reversed_input(vmec_input):
    """The same boundary with theta -> -theta: mode (m, n) moves to (m, -n) and the sine
    coefficients change sign; the m = 0 row does not depend on theta."""
    update = {}
    for name, sign in (("rbc", 1.0), ("rbs", -1.0), ("zbs", -1.0), ("zbc", 1.0)):
        coefficients = np.asarray(getattr(vmec_input, name), float)
        reversed_coefficients = sign * coefficients[:, ::-1]
        reversed_coefficients[0] = coefficients[0]
        update[name] = reversed_coefficients
    return vmec_input.model_copy(update=update)


# simsopt tests/test_files/input.basic_non_stellsym: a vacuum field whose boundary
# needs both the poloidal shift to rbs(1,0) = zbc(1,0) and the theta flip.
BASIC_NON_STELLSYM_INDATA = """&INDATA
  LASYM = T
  DELT = 0.9
  NFP = 1
  NCURR = 1
  MPOL = 2
  NTOR = 2
  NZETA = 10
  NTHETA = 10
  NS_ARRAY = 13 25 51
  FTOL_ARRAY = 1e-08 1e-10 1e-12
  NITER_ARRAY = 2000 4000 20000
  NSTEP = 200
  GAMMA = 0.0
  PHIEDGE = -119.416456038
  CURTOR = 0.0
  SPRES_PED = 1.0
  RAXIS_CC = 6.698820504223022
  ZAXIS_CC = 0.5203227532718783
  RBC(0,0) = 5.0,   ZBS(0,0) = 0.0,   RBS(0,0) = 0.0,  ZBC(0,0) = 0.0
  RBC(0,1) = 1.25,  ZBS(0,1) = -1.25, RBS(0,1) = 0.1,  ZBC(0,1) = -0.1
  RBC(1,0) = 1.25,  ZBS(1,0) = -1.5,  RBS(1,0) = 0.1,  ZBC(1,0) = -0.1
  RBC(1,1) = 0.5,   ZBS(1,1) = -0.5,  RBS(1,1) = 0.5,  ZBC(1,1) = -0.5
/
"""

# Current-density outputs of the vacuum case, which are iteration noise.
VACUUM_CURRENT_FIELDS = {
    *(
        f"wout.{name}"
        for name in (
            "ctor",
            "buco",
            "jcuru",
            "jcurv",
            "jdotb",
            "currumnc",
            "currumns",
            "currvmnc",
            "currvmns",
        )
    ),
    "threed1_geometric_magnetic.toroidal_current",
    *(
        f"threed1_first_table.{name}"
        for name in ("buco_full", "avg_jsupu", "avg_jsupv", "j_dot_b")
    ),
    "mercier.toroidal_current",
    "mercier.d_toroidal_current_d_s",
    *CURRENT_DENSITY_FIELDS,
}

FORCE_RESIDUALS = {"wout.fsqr", "wout.fsqz", "wout.fsql", "wout.fsqt"}


@pytest.fixture(scope="module")
def basic_non_stellsym_input(tmp_path_factory):
    indata = tmp_path_factory.mktemp("indata") / "input.basic_non_stellsym"
    indata.write_text(BASIC_NON_STELLSYM_INDATA)
    return vmecpp.VmecInput.from_file(indata)


def _poloidal_case(name, basic_non_stellsym_input):
    if name == "basic_non_stellsym":
        return basic_non_stellsym_input, VACUUM_CURRENT_FIELDS
    return _lasym_input(name), set()


def _assert_same_outputs(expected, actual, noise_fields):
    fields = expected.keys() - FORCE_RESIDUALS - RESIDUAL_FIELDS - noise_fields
    _assert_fields_match(
        expected,
        actual,
        rtol=1e-7,
        atol=1e-7,
        names=fields - CURRENT_DENSITY_FIELDS,
    )
    _assert_fields_match(
        expected,
        actual,
        rtol=1e-7,
        atol=1e-3,
        names=fields & CURRENT_DENSITY_FIELDS,
    )


def _assert_jacobian_matches_signgs(wout):
    """Signgs is the sign of the Jacobian sqrt(g) on every surface."""
    assert np.all(wout.signgs * np.asarray(wout.gmnc)[0, 1:] > 0.0)
    assert np.all(np.asarray(wout.vp)[1:] > 0.0)


@pytest.mark.parametrize("case", ["cth_like_fixed_bdy_asym", "basic_non_stellsym"])
def test_poloidal_shift_is_gauged_away(case, basic_non_stellsym_input):
    """A boundary relabelled by theta -> theta + theta0 is shifted back to the gauge
    rbs(1,0) = zbc(1,0): every output reproduces the unshifted run."""
    base, noise_fields = _poloidal_case(case, basic_non_stellsym_input)
    reference = vmecpp.run(base, max_threads=1, verbose=False)
    shifted = vmecpp.run(
        _poloidally_shifted_input(base, 0.3), max_threads=1, verbose=False
    )
    _assert_jacobian_matches_signgs(shifted.wout)
    _assert_same_outputs(
        _numeric_fields(reference), _numeric_fields(shifted), noise_fields
    )


@pytest.mark.parametrize("case", ["cth_like_fixed_bdy_asym", "basic_non_stellsym"])
def test_poloidal_reversal_preserves_physics(case, basic_non_stellsym_input):
    """A boundary relabelled by theta -> -theta runs in the opposite poloidal direction,
    so exactly one of the two runs flips theta -> pi - theta. The outputs differ by
    theta -> theta + pi, a factor (-1)^m on every Fourier coefficient."""
    base, noise_fields = _poloidal_case(case, basic_non_stellsym_input)
    reference = vmecpp.run(base, max_threads=1, verbose=False).wout
    reversed_ = vmecpp.run(
        _poloidally_reversed_input(base), max_threads=1, verbose=False
    ).wout
    _assert_jacobian_matches_signgs(reference)
    _assert_jacobian_matches_signgs(reversed_)
    parity = {
        len(reference.xm): (-1.0) ** np.asarray(reference.xm),
        len(reference.xm_nyq): (-1.0) ** np.asarray(reference.xm_nyq),
    }
    expected = _numeric_fields(reference, prefix="wout.")
    actual = _numeric_fields(reversed_, prefix="wout.")
    for name, value in actual.items():
        if value.ndim == 2:
            actual[name] = parity[value.shape[0]][:, np.newaxis] * value
    _assert_same_outputs(expected, actual, noise_fields | {"wout.equif"})
