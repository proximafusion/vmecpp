# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import vmecpp

# We don't want to install tests and test data as part of the package,
# but scikit-build-core + hatchling does not support editable installs,
# so the tests live in the sources but the vmecpp module lives in site_packages.
# Therefore, in order to find the test data we use the relative path to this file.
# I'm very open to alternative solutions :)
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


# Regression test for #189
@pytest.fixture(params=[str, Path])
def path_type(request) -> str | Path:
    return request.param


@pytest.mark.parametrize(
    ("indata_file", "reference_wout_file"),
    [
        ("cma.json", "wout_cma.nc"),
        (
            "cth_like_free_bdy.json",
            "wout_cth_like_free_bdy.nc",
        ),
    ],
)
def test_save_to_netcdf(indata_file, reference_wout_file, path_type):
    indata = vmecpp.VmecInput.from_file(TEST_DATA_DIR / indata_file)
    if indata.lfreeb:
        indata.mgrid_file = str(
            REPO_ROOT / "src" / "vmecpp" / "cpp" / indata.mgrid_file
        )
    vmec_output = vmecpp.run(indata)
    fortran_wout = vmec_output.wout

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir, "wout_test.nc")
        fortran_wout.save(path_type(out_path))

        test_dataset = netCDF4.Dataset(out_path, "r")

    expected_dataset = netCDF4.Dataset(TEST_DATA_DIR / reference_wout_file, "r")

    for varname, expected_value in expected_dataset.variables.items():
        if varname in vmecpp.VmecWOut._MISSING_FORTRAN_VARIABLES:
            continue
        test_value = test_dataset[varname]
        error_msg = f"mismatch in {varname}"

        # string
        if expected_value.dtype == np.dtype("S1"):
            if varname == "mgrid_file":
                # the `mgrid_file` entry in the reference wout file only contains the
                # base name, while the one produced by VMEC++ contains a path relative
                # to the root of the repo
                assert test_value[:].tobytes().decode().strip() == indata.mgrid_file
            else:
                np.testing.assert_equal(
                    test_value[:], expected_value[:], err_msg=error_msg
                )
            continue

        expected_dims = expected_value.dimensions

        # Check dimensions for an array or tensor (also works for scalars)
        for d in expected_dims:
            assert (
                test_dataset.dimensions[d].size == expected_dataset.dimensions[d].size
            )
        # np.asarray is needed to convert the masked array to a regular array.
        # nan is a valid value for some fields (e.g. extcur) and can't be compared otherwise.
        np.testing.assert_allclose(
            np.asarray(test_value[:]),
            np.asarray(expected_value[:]),
            err_msg=error_msg,
            rtol=1e-6,
            atol=1e-7,
            equal_nan=True,
        )
