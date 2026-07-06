# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Exact-equivalence tests for non-stellarator-symmetric (lasym) equilibria.

Each test applies a transformation to a stellarator-symmetric configuration that needs
the asymmetric representation (rbs/zbc) but reproduces the symmetric equilibrium's
physics, exercising the asymmetric DFTs end to end.
"""

from pathlib import Path

import numpy as np

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


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


def test_toroidal_rotation_preserves_physics():
    """A toroidal rotation mixes symmetric modes into asymmetric ones; physics is
    unchanged."""
    base = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    # converge tightly so iota is resolved well below the rotation's numerical drift
    # (the shipped input only asks for ftol 1e-6)
    base = base.model_copy(
        update={
            "ftol_array": np.full(np.asarray(base.ftol_array).shape, 1e-12),
            "niter_array": np.full(
                np.asarray(base.niter_array).shape, 5000, dtype=np.int64
            ),
        }
    )
    nfp, ntor, mpol = base.nfp, base.ntor, base.mpol
    sym = _run(base)
    zeta0 = 0.05
    rbc = np.asarray(base.rbc, float)
    zbs = np.asarray(base.zbs, float)
    rbc2 = np.zeros_like(rbc)
    rbs = np.zeros_like(rbc)
    zbs2 = np.zeros_like(zbs)
    zbc = np.zeros_like(zbs)
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            j = n + ntor
            N = n * nfp
            c, s = np.cos(N * zeta0), np.sin(N * zeta0)
            rbc2[m, j] = rbc[m, j] * c
            rbs[m, j] = rbc[m, j] * s
            zbs2[m, j] = zbs[m, j] * c
            zbc[m, j] = -zbs[m, j] * s
    rotated = _run(_enable_lasym(base, rbc=rbc2, zbs=zbs2, rbs=rbs, zbc=zbc))
    # genuine asymmetric geometry was produced (not gauged away)
    assert np.max(np.abs(np.asarray(rotated.rmns))) > 1e-4
    _assert_same_physics(sym, rotated, vol_rtol=1e-8, beta_atol=1e-8, iota_atol=1e-5)
