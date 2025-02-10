# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for VMEC++'s'SIMSOPT compatibility layer."""

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from vmecpp import simsopt_compat

# We don't want to install tests and test data as part of the package,
# but scikit-build-core + hatchling does not support editable installs,
# so the tests live in the sources but the vmecpp module lives in site_packages.
# Therefore, in order to find the test data we use the relative path to this file.
# I'm very open to alternative solutions :)
REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.fixture
def json_input_filepath() -> Path:
    return TEST_DATA_DIR / "solovev.json"


@pytest.fixture
def vmec(json_input_filepath) -> simsopt_compat.Vmec:
    return simsopt_compat.Vmec(json_input_filepath)


@pytest.fixture
def wout() -> netCDF4.Dataset:
    return netCDF4.Dataset(TEST_DATA_DIR / "wout_solovev.nc", "r")


def test_aspect(vmec, wout):
    vmec.run()
    aspect = vmec.aspect()
    expected_aspect = wout.variables["aspect"][()]
    np.testing.assert_allclose(aspect, expected_aspect, rtol=1e-11, atol=0.0)


def test_volume(vmec, wout):
    vmec.run()
    volume = vmec.volume()
    expected_volume = wout.variables["volume_p"][()]
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-11, atol=0.0)


def test_iota_axis(vmec, wout):
    vmec.run()
    iota_axis = vmec.iota_axis()
    expected_iota_axis = wout.variables["iotaf"][()][0]
    np.testing.assert_allclose(iota_axis, expected_iota_axis, rtol=1e-11, atol=0.0)


def test_iota_edge(vmec, wout):
    vmec.run()
    iota_edge = vmec.iota_edge()
    expected_iota_edge = wout.variables["iotaf"][()][-1]
    np.testing.assert_allclose(iota_edge, expected_iota_edge, rtol=1e-11, atol=0.0)


def test_mean_iota(vmec, wout):
    vmec.run()
    mean_iota = vmec.mean_iota()
    expected_mean_iota = np.mean(wout.variables["iotas"][()][1:])
    np.testing.assert_allclose(mean_iota, expected_mean_iota, rtol=1e-11, atol=0.0)


def test_mean_shear(vmec, wout):
    vmec.run()
    mean_shear = vmec.mean_shear()
    # Compute mean shear as in simsopt
    s_full_grid = np.linspace(0, 1, wout.variables["ns"][()])
    ds = s_full_grid[1] - s_full_grid[0]
    s_half_grid = s_full_grid[1:] - 0.5 * ds
    iotas = wout.variables["iotas"][()][1:]
    iota_fit = np.polynomial.Polynomial.fit(s_half_grid, iotas, deg=1)
    expected_mean_shear = iota_fit.deriv()(0)
    np.testing.assert_allclose(mean_shear, expected_mean_shear, rtol=1e-11, atol=0.0)


@pytest.mark.parametrize(
    "attribute_name,mnmax_size_name",  # noqa: PT006 what ruff wants does not work
    [
        ("rmnc", "mnmax"),
        ("zmns", "mnmax"),
        ("lmns", "mnmax"),
        ("bmnc", "mnmax_nyq"),
        ("bsubumnc", "mnmax_nyq"),
        ("bsubvmnc", "mnmax_nyq"),
        ("bsupumnc", "mnmax_nyq"),
        ("bsupvmnc", "mnmax_nyq"),
        ("bsubsmns", "mnmax_nyq"),
        ("gmnc", "mnmax_nyq"),
        # NOTE: VMEC++ does not implement these and they are not used anywhere
        # ("currumnc", "mnmax_nyq"),
        # ("currvmnc", "mnmax_nyq"),
    ],
)
def test_wout_attributes_shape(vmec, attribute_name, mnmax_size_name):
    vmec.run()
    ns = vmec.wout.ns
    attribute_value = getattr(vmec.wout, attribute_name)
    expected_shape = (getattr(vmec.wout, mnmax_size_name), ns)
    assert attribute_value.shape == expected_shape


def test_changing_boundary():
    # this test only makes sense for a circular tokamak setup
    vmec = simsopt_compat.Vmec(TEST_DATA_DIR / "circular_tokamak.json")
    original_rc00 = vmec.boundary.get_rc(0, 0)
    vmec.run()
    assert vmec.wout is not None
    aspect_ratio = vmec.wout.aspect
    dofs = vmec.boundary.get_dofs()
    vmec.boundary.set_dofs(dofs * 2)
    vmec.boundary.set_rc(0, 0, original_rc00)
    expected_aspect_ratio = aspect_ratio / 2
    vmec.recompute_bell()
    vmec.run()
    np.testing.assert_allclose(
        vmec.wout.aspect, expected_aspect_ratio, rtol=5e-2, atol=0.0
    )


def test_changing_mpol_ntor(vmec):
    vmec.run()

    def expected_shape(mpol: int, ntor: int):
        # corresponds to Sizes::mnmax
        mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1)
        return (mnmax, vmec.wout.ns)

    assert vmec.wout.rmnc.shape == expected_shape(vmec.indata.mpol, vmec.indata.ntor)
    assert vmec.wout.zmns.shape == expected_shape(vmec.indata.mpol, vmec.indata.ntor)

    # Set VMEC poloidal and toroidal modes:
    # this mimics the way starfinder uses VMEC, changing
    # indata between one run and the next
    new_mpol = 7
    new_ntor = 1
    vmec.set_mpol_ntor(new_mpol=new_mpol, new_ntor=new_ntor)
    vmec.run()

    assert vmec.indata.mpol == new_mpol
    assert vmec.indata.ntor == new_ntor
    assert vmec.wout.rmnc.shape == expected_shape(new_mpol, new_ntor)
    assert vmec.wout.zmns.shape == expected_shape(new_mpol, new_ntor)
