# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for VMEC++'s'SIMSOPT compatibility layer."""

import math
from pathlib import Path

import netCDF4
import numpy as np
import pytest
from simsopt import mhd as simsopt_mhd

from vmecpp import _util, ensure_vmec2000_input, simsopt_compat

# We don't want to install tests and test data as part of the package,
# but scikit-build-core + hatchling does not support editable installs,
# so the tests live in the sources but the vmecpp module lives in site_packages.
# Therefore, in order to find the test data we use the relative path to this file.
# I'm very open to alternative solutions :)
REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.fixture(scope="module", params=["solovev.json", "input.cma"])
def input_file_path(request) -> Path:
    return TEST_DATA_DIR / request.param


@pytest.fixture(scope="module")
def vmec(input_file_path) -> simsopt_compat.Vmec:
    vmec = simsopt_compat.Vmec(input_file_path)
    vmec.run()
    return vmec


@pytest.fixture
def reference_wout(input_file_path) -> netCDF4.Dataset:
    if "solovev" in input_file_path.name:
        return netCDF4.Dataset(TEST_DATA_DIR / "wout_solovev.nc", "r")

    assert "cma" in input_file_path.name
    return netCDF4.Dataset(TEST_DATA_DIR / "wout_cma.nc", "r")


# regression test for #174
def test_run_with_relative_path(input_file_path):
    with _util.change_working_directory_to(TEST_DATA_DIR):
        vmec = simsopt_compat.Vmec(input_file_path.name)
        vmec.run()


def test_aspect(vmec, reference_wout):
    aspect = vmec.aspect()
    expected_aspect = reference_wout.variables["aspect"][()]
    np.testing.assert_allclose(aspect, expected_aspect, rtol=1e-11, atol=0.0)


def test_volume(vmec, reference_wout):
    volume = vmec.volume()
    expected_volume = reference_wout.variables["volume_p"][()]
    np.testing.assert_allclose(volume, expected_volume, rtol=1e-11, atol=0.0)


def test_iota_axis(vmec, reference_wout):
    iota_axis = vmec.iota_axis()
    expected_iota_axis = reference_wout.variables["iotaf"][()][0]
    np.testing.assert_allclose(iota_axis, expected_iota_axis, rtol=1e-11, atol=1e-11)


def test_iota_edge(vmec, reference_wout):
    iota_edge = vmec.iota_edge()
    expected_iota_edge = reference_wout.variables["iotaf"][()][-1]
    np.testing.assert_allclose(iota_edge, expected_iota_edge, rtol=1e-11, atol=0.0)


def test_mean_iota(vmec, reference_wout):
    mean_iota = vmec.mean_iota()
    expected_mean_iota = np.mean(reference_wout.variables["iotas"][()][1:])
    np.testing.assert_allclose(mean_iota, expected_mean_iota, rtol=1e-11, atol=0.0)


def test_mean_shear(vmec, reference_wout):
    mean_shear = vmec.mean_shear()
    # Compute mean shear as in simsopt
    s_full_grid = np.linspace(0, 1, reference_wout.variables["ns"][()])
    ds = s_full_grid[1] - s_full_grid[0]
    s_half_grid = s_full_grid[1:] - 0.5 * ds
    iotas = reference_wout.variables["iotas"][()][1:]
    iota_fit = np.polynomial.Polynomial.fit(s_half_grid, iotas, deg=1)
    expected_mean_shear = iota_fit.deriv()(0)
    np.testing.assert_allclose(mean_shear, expected_mean_shear, rtol=1e-11, atol=5e-11)


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
    new_ntor = 5
    vmec.set_mpol_ntor(new_mpol=new_mpol, new_ntor=new_ntor)
    vmec.run()

    assert vmec.indata.mpol == new_mpol
    assert vmec.indata.ntor == new_ntor
    assert vmec.wout.rmnc.shape == expected_shape(new_mpol, new_ntor)
    assert vmec.wout.zmns.shape == expected_shape(new_mpol, new_ntor)


def test_ensure_vmec2000_input_from_vmecpp_input():
    # we only install VMEC2000 on Ubu 22.04, not on MacOS and Ubuntu 24.04
    pytest.importorskip("vmec")

    vmecpp_input_file = TEST_DATA_DIR / "cma.json"

    with ensure_vmec2000_input(vmecpp_input_file) as converted_indata_file:
        vmec2000 = simsopt_mhd.Vmec(str(converted_indata_file))

    vmecpp = simsopt_compat.Vmec(str(vmecpp_input_file))

    # vmec2000.indata has way many more variables than vmecpp.indata, so we test
    # the common subset.
    for varname in dir(vmecpp.indata):
        # These are not present in the legacy VMEC2000 INDATA namelist,
        # therefore skip them.
        if varname.startswith("_") or varname in [
            "free_boundary_method",
            "return_outputs_even_if_not_converged",
        ]:
            continue

        vmecpp_var = getattr(vmecpp.indata, varname)
        if callable(vmecpp_var):
            continue  # this is a method, not a variable

        varname_vmec2000 = varname
        if varname[1:-1] == "axis_":
            # these are called differently in VMEC2000, e.g. raxis_c -> raxis_cc
            varname_vmec2000 = f"{varname[:-1]}c{varname[-1]}"
        vmec2000_var = getattr(vmec2000.indata, varname_vmec2000)

        if isinstance(vmecpp_var, str | int | bool):
            if isinstance(vmec2000_var, bytes):
                vmec2000_var = vmec2000_var.decode().strip()
            elif varname in {"ntheta", "nzeta"}:
                assert vmecpp_var == 0  # like in the input file
                assert vmec2000_var == 16  # the default VMEC2000 sets if it's == 0
            else:
                assert vmecpp_var == vmec2000_var
        elif isinstance(vmecpp_var, float):
            assert math.isclose(vmecpp_var, vmec2000_var)
        else:
            assert isinstance(vmecpp_var, np.ndarray)

            # NOTE: these are differences in behavior between VMEC++ and VMEC2000,
            # not an issue with the file format conversion.
            if varname == "ac_aux_f":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 101))
            elif varname == "ac_aux_s":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([-1.0] * 101))
            elif varname == "ai":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 21))
            elif varname == "ai_aux_f":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 101))
            elif varname == "ai_aux_s":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([-1.0] * 101))
            elif varname == "am_aux_f":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 101))
            elif varname == "am_aux_s":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([-1.0] * 101))
            elif varname == "extcur":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 300))
            else:
                # VMEC2000 pads the arrays with zeros, VMEC++ instantiates them
                # with the right length
                if len(vmecpp_var.shape) == 1:
                    vmec2000_var_truncated = vmec2000_var[: len(vmecpp_var)]
                else:
                    assert vmecpp.indata is not None  # for pyright
                    # RBS and ZBC might be just empty
                    if varname in {"rbs", "zbc"} and not vmecpp.indata.lasym:
                        vmec2000_var_truncated = np.zeros(shape=(0, 0))
                    else:
                        # must be 2D RBC, ZBS. here there is a triple mismatch:
                        # 1. VMEC2000 uses layout (n, m) while VMEC++ uses (m, n)
                        # 2. VMEC2000 pre-allocates an array with shape (203, 101)
                        #    while VMEC++ allocates according to mpol, ntor
                        # 3. for the `n` index values are laid out as
                        #    [-ntor, ..., 0, ..., ntor] (with ntor being the one from
                        #    the file for VMEC++, and 101 for VMEC2000), so we need to
                        #    truncate the entries in vmec2000_var symmetrically around
                        #    the center
                        # First we transpose and truncate the rows:
                        vmec2000_var_truncated = vmec2000_var.T[
                            : vmecpp_var.shape[0], :
                        ]
                        # Now we truncate the columns symmetrically around the center:
                        ntor = vmecpp.indata.ntor
                        vmec2000_var_truncated = vmec2000_var_truncated[
                            :, 101 - ntor : 101 + ntor + 1
                        ]
                np.testing.assert_allclose(vmecpp_var, vmec2000_var_truncated)
