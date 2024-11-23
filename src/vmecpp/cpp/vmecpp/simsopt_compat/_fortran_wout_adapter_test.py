import math
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

from vmecpp.cpp import _vmecpp as vmec  # pants: no-infer-dep
from vmecpp.cpp.vmecpp.simsopt_compat import (
    VARIABLES_MISSING_FROM_FORTRAN_WOUT_ADAPTER,
    FortranWOutAdapter,
)


def test_save_to_netcdf():
    indata = vmec.VmecINDATAPyWrapper.from_file("vmecpp/test_data/cma.json")
    cpp_wout = vmec.run(indata).wout

    fortran_wout = FortranWOutAdapter.from_vmecpp_wout(cpp_wout)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir, "wout_test.nc")
        fortran_wout.save(out_path)

        test_dataset = netCDF4.Dataset(out_path, "r")

    expected_dataset = netCDF4.Dataset("vmecpp/test_data/wout_cma.nc", "r")

    for varname, expected_value in expected_dataset.variables.items():
        if varname in VARIABLES_MISSING_FROM_FORTRAN_WOUT_ADAPTER:
            continue

        test_value = test_dataset[varname]
        error_msg = f"mismatch in {varname}"

        # string
        if expected_value.dtype == np.dtype("S1"):
            np.testing.assert_equal(test_value[:], expected_value[:], err_msg=error_msg)
            continue

        expected_dims = expected_value.dimensions
        assert test_value.dimensions == expected_dims, error_msg

        # scalar
        if expected_dims == ():
            assert math.isclose(
                test_value[:], expected_value[:], abs_tol=1e-7
            ), error_msg
            continue

        # array or tensor
        for d in expected_dims:
            assert (
                test_dataset.dimensions[d].size == expected_dataset.dimensions[d].size
            )
        np.testing.assert_allclose(
            test_value[:], expected_value[:], err_msg=error_msg, rtol=1e-6, atol=1e-7
        )
