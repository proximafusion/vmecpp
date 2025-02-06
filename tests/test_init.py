# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for VMEC++'s Python API.

Here we just test that the Python bindings and the general API works as expected.
Physics correctness is checked at the level of the C++ core.
"""

import math
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
REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.mark.parametrize(
    ("max_threads", "input_file", "verbose"),
    [(None, "cma.json", True), (1, "input.cma", False)],
)
def test_run(max_threads, input_file, verbose):
    """Test that the Python API works with different combinations of parameters."""

    input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / input_file)
    out = vmecpp.run(input, max_threads=max_threads, verbose=verbose)

    assert out.wout is not None


def test_get_outputs_if_non_converged_if_wanted():
    """Test that one can get the VMEC++ outputs even if a run did not converge."""

    input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")

    # only allow one iteration - VMEC++ will not converge that fast
    input.niter_array[-1] = 1

    # instruct VMEC++ to return the outputs, even if it did not converge
    input.return_outputs_even_if_not_converged = True

    out = vmecpp.run(input)

    assert out.wout is not None
    assert out.wout.niter == 2


# We trust the C++ tests to cover the hot restart functionality properly,
# here we just want to test that the Python API for it works.
def test_run_with_hot_restart():
    input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")

    # base run
    out = vmecpp.run(input, verbose=False)

    # now with hot restart
    # (only a single multigrid step is supported)
    input.ns_array = input.ns_array[-1:]
    input.ftol_array = input.ftol_array[-1:]
    input.niter_array = input.niter_array[-1:]
    hot_restarted_out = vmecpp.run(input, verbose=False, restart_from=out)

    assert hot_restarted_out.wout.niter == 2


@pytest.fixture(scope="module")
def cma_output() -> vmecpp.VmecOutput:
    input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")
    out = vmecpp.run(input, verbose=False)
    return out


def test_vmecwout_save(cma_output):
    with tempfile.NamedTemporaryFile() as tmp_file:
        cma_output.wout.save(tmp_file.name)

        assert Path(tmp_file.name).exists()

        test_dataset = netCDF4.Dataset(tmp_file.name, "r")

    expected_dataset = netCDF4.Dataset(TEST_DATA_DIR / "wout_cma.nc", "r")

    for varname, expected_value in expected_dataset.variables.items():
        if varname in vmecpp.VmecWOut._MISSING_FORTRAN_VARIABLES:
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


def test_jxbout_bindings(cma_output):
    for varname in [
        "itheta",
        "izeta",
        "bdotk",
        "jsupu3",
        "jsupv3",
        "jsups3",
        "bsupu3",
        "bsupv3",
        "jcrossb",
        "jxb_gradp",
        "jdotb_sqrtg",
        "sqrtg3",
        "bsubu3",
        "bsubv3",
        "bsubs3",
    ]:
        assert len(getattr(cma_output.jxbout, varname).shape) == 2

    for varname in [
        "amaxfor",
        "aminfor",
        "avforce",
        "pprim",
        "jdotb",
        "bdotb",
        "bdotgradv",
        "jpar2",
        "jperp2",
        "phin",
    ]:
        assert len(getattr(cma_output.jxbout, varname).shape) == 1


def test_mercier_bindings(cma_output):
    for varname in [
        "s",
        "toroidal_flux",
        "iota",
        "shear",
        "d_volume_d_s",
        "well",
        "toroidal_current",
        "d_toroidal_current_d_s",
        "pressure",
        "d_pressure_d_s",
        "DMerc",
        "Dshear",
        "Dwell",
        "Dcurr",
        "Dgeod",
    ]:
        assert len(getattr(cma_output.mercier, varname).shape) == 1


def test_threed1volumetrics_bindings(cma_output):
    for varname in [
        "int_p",
        "avg_p",
        "int_bpol",
        "avg_bpol",
        "int_btor",
        "avg_btor",
        "int_modb",
        "avg_modb",
        "int_ekin",
        "avg_ekin",
    ]:
        assert isinstance(getattr(cma_output.threed1_volumetrics, varname), float)
