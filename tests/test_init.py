# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for VMEC++'s Python API.

Here we just test that the Python bindings and the general API works as expected.
Physics correctness is checked at the level of the C++ core.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import vmecpp
from vmecpp.cpp import _vmecpp  # pyright: ignore[reportAttributeAccessIssue]

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

    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / input_file)
    vmec_output = vmecpp.run(vmec_input, max_threads=max_threads, verbose=verbose)

    assert vmec_output.wout is not None


@pytest.mark.parametrize(
    ("mgrid_path", "expected_exception"),
    [
        ("cma.json", RuntimeError),  # Invalid netcdf
        ("does_not_exist", RuntimeError),
        # TODO(jurasic) Enable test after switching netcdf_io to absl::Status
        # ("wout_cma.nc", RuntimeError),  # Valid netcdf, but invalid mgrid
    ],
)
def test_raise_invalid_mgrid(mgrid_path: str, expected_exception):
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")
    vmec_input.lfreeb = True
    vmec_input.mgrid_file = str(TEST_DATA_DIR / mgrid_path)
    with pytest.raises(expected_exception):
        vmecpp.run(vmec_input, max_threads=1)


def test_get_outputs_if_non_converged_if_wanted():
    """Test that one can get the VMEC++ outputs even if a run did not converge."""

    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")

    # only allow one iteration - VMEC++ will not converge that fast
    vmec_input.niter_array[-1] = 1

    # instruct VMEC++ to return the outputs, even if it did not converge
    vmec_input.return_outputs_even_if_not_converged = True

    vmec_output = vmecpp.run(vmec_input)

    assert vmec_output.wout is not None
    assert vmec_output.wout.niter == 2

    # actually check that some arrays,
    # which were previously only filled if VMEC converged,
    # also get populated now
    assert not np.all(vmec_output.jxbout.jxb_gradp == 0.0)


# We trust the C++ tests to cover the hot restart functionality properly,
# here we just want to test that the Python API for it works.
def test_run_with_hot_restart():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")

    # base run
    vmec_output = vmecpp.run(vmec_input, verbose=False)

    # now with hot restart
    # (only a single multigrid step is supported)
    vmec_input.ns_array = vmec_input.ns_array[-1:]
    vmec_input.ftol_array = vmec_input.ftol_array[-1:]
    vmec_input.niter_array = vmec_input.niter_array[-1:]
    vmec_output_hot_restarted = vmecpp.run(
        vmec_input, verbose=False, restart_from=vmec_output
    )

    assert vmec_output_hot_restarted.wout.niter == 2


@pytest.fixture(scope="module")
def cma_output() -> vmecpp.VmecOutput:
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")
    vmec_output = vmecpp.run(vmec_input, verbose=False)
    return vmec_output


def test_vmecwout_load_from_fortran():
    """Test that a wout file produced by PARVMEC can be loaded by VmecWOut."""
    wout_filename = REPO_ROOT / "examples" / "data" / "wout_cth_like_fixed_bdy.nc"
    loaded_wout = vmecpp.VmecWOut.from_wout_file(wout_filename)
    assert loaded_wout is not None


def test_vmecinput_io():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "input_save_test.json"
        vmec_input.save(tmp_file)
        vmec_input_reloaded = vmecpp.VmecInput.from_file(tmp_file)

        for attr in vars(vmec_input):
            error_msg = f"mismatch in {attr}"
            np.testing.assert_equal(
                actual=getattr(vmec_input_reloaded, attr),
                desired=getattr(vmec_input, attr),
                err_msg=error_msg,
            )


def test_asymmetric_tokamak_input_io():
    input_file = TEST_DATA_DIR / "input.up_down_asymmetric_tokamak"

    vmec_input = vmecpp.VmecInput.from_file(input_file)

    # Verify it's an asymmetric run
    assert vmec_input.lasym is True

    # Verify asymmetric arrays are properly initialized (not None)
    assert vmec_input.rbs is not None
    assert vmec_input.zbc is not None
    assert vmec_input.raxis_s is not None
    assert vmec_input.zaxis_c is not None

    # Verify array shapes are correct for ntor=0, mpol=5
    expected_shape = (5, 1)  # (mpol, 2*ntor+1)
    assert vmec_input.rbs.shape == expected_shape
    assert vmec_input.zbc.shape == expected_shape

    expected_axis_shape = (1,)  # (ntor+1,)
    assert vmec_input.raxis_s.shape == expected_axis_shape
    assert vmec_input.zaxis_c.shape == expected_axis_shape

    assert vmec_input.rbs[1, 0] == pytest.approx(0.6)  # RBS(0,1) from input file
    assert vmec_input.rbs[2, 0] == pytest.approx(0.12)  # RBS(0,2) from input file

    # Verify other asymmetric arrays are initialized to zero
    assert vmec_input.zbc[0, 0] == pytest.approx(0.0)
    assert vmec_input.raxis_s[0] == pytest.approx(0.0)
    assert vmec_input.zaxis_c[0] == pytest.approx(0.0)


_MISSING_FORTRAN_VARIABLES = [
    "lrecon__logical__",
    "lmove_axis__logical__",
    "mnyq",
    "nnyq",
    "currumnc",
    "currvmnc",
    "curlabel",
    "potvac",
    "nobser",
    "nobd",
    "nbsets",
    "mnmaxpot",
    "potsin",
    "xmpot",
    "xnpot",
    "bsubumnc_sur",
    "bsubvmnc_sur",
    "bsupumnc_sur",
    "bsupvmnc_sur",
]
"""The complete list of variables that can be found in Fortran VMEC wout files but not
in wout files produced by VMEC++."""


def test_vmecwout_io(cma_output: vmecpp.VmecOutput):
    with tempfile.NamedTemporaryFile() as tmp_file:
        cma_output.wout.save(tmp_file.name)

        assert Path(tmp_file.name).exists()

        # check that from_wout_file can load the file as well
        loaded_wout = vmecpp.VmecWOut.from_wout_file(tmp_file.name)
        assert loaded_wout is not None

        # test contents of loaded wout against original in-memory wout
        for attr in vars(cma_output.wout):
            error_msg = f"mismatch in {attr}"
            np.testing.assert_equal(
                actual=getattr(loaded_wout, attr),
                desired=getattr(cma_output.wout, attr),
                err_msg=error_msg,
            )

        test_dataset = netCDF4.Dataset(tmp_file.name, "r")

    expected_dataset = netCDF4.Dataset(TEST_DATA_DIR / "wout_cma.nc", "r")

    expected_variables = set(expected_dataset.variables.keys()) - set(
        _MISSING_FORTRAN_VARIABLES
    )
    assert expected_variables == set(test_dataset.variables.keys())

    for varname, expected_value in expected_dataset.variables.items():
        if varname in _MISSING_FORTRAN_VARIABLES:
            continue

        test_value = test_dataset[varname]
        error_msg = f"mismatch in {varname}"

        # string
        if expected_value.dtype == np.dtype("S1"):
            np.testing.assert_equal(test_value[:], expected_value[:], err_msg=error_msg)
            continue

        expected_dims = expected_value.dimensions
        assert test_value.dimensions == expected_dims, error_msg

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


@pytest.mark.parametrize(
    ("indata_file", "reference_wout_file", "path_type"),
    [
        ("cma.json", "wout_cma.nc", str),
        ("cma.json", "wout_cma.nc", Path),
        ("cth_like_free_bdy.json", "wout_cth_like_free_bdy.nc", str),
    ],
)
def test_against_reference_wout(indata_file, reference_wout_file, path_type):
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

    # Some variables need enlarged tolerances,
    # because they are not numerically well-defined.
    enlarged_tolerances = {
        "jdotb": {"rtol": 1.0e-5, "atol": 1.0e-4},
    }

    for varname, expected_value in expected_dataset.variables.items():
        if varname in _MISSING_FORTRAN_VARIABLES:
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

        rtol = enlarged_tolerances.get(varname, {"rtol": 1.0e-6})["rtol"]
        atol = enlarged_tolerances.get(varname, {"atol": 1.0e-7})["atol"]

        # np.asarray is needed to convert the masked array to a regular array.
        # nan is a valid value for some fields (e.g. extcur) and can't be compared otherwise.
        np.testing.assert_allclose(
            np.asarray(test_value[:]),
            np.asarray(expected_value[:]),
            err_msg=error_msg,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def test_vmecwout_extra_fields_io(cma_output: vmecpp.VmecOutput):
    """Support for unknown fields in wout files."""
    cma_output_copy = cma_output.model_copy(deep=True)
    del cma_output  # Prevent accidental use of the original object
    cma_output_copy.wout = cma_output_copy.wout.model_copy(
        update={
            "extra_field": np.array([1.0, 2.0, 3.0]),
            "extra_string": "string",
            "extra_float": 3.14,
            "extra_int": 42,
            "extra_2darray": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    with tempfile.NamedTemporaryFile() as tmp_file:
        cma_output_copy.wout.save(tmp_file.name)

        assert Path(tmp_file.name).exists()

        # check that from_wout_file can load the file as well
        loaded_wout = vmecpp.VmecWOut.from_wout_file(tmp_file.name)
        assert loaded_wout is not None

        # test contents of loaded wout against original in-memory wout
        for attr in vars(cma_output_copy.wout):
            error_msg = f"mismatch in {attr}"
            np.testing.assert_equal(
                actual=getattr(loaded_wout, attr),
                desired=getattr(cma_output_copy.wout, attr),
                err_msg=error_msg,
            )


def test_jxbout_bindings(cma_output: vmecpp.VmecOutput):
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


def test_mercier_bindings(cma_output: vmecpp.VmecOutput):
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


def test_threed1volumetrics_bindings(cma_output: vmecpp.VmecOutput):
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


def test_is_vmec2000_input():
    vmec2000_input_file = TEST_DATA_DIR / "input.cma"
    vmecpp_input_file = TEST_DATA_DIR / "cma.json"

    assert vmecpp.is_vmec2000_input(vmec2000_input_file)
    assert not vmecpp.is_vmec2000_input(vmecpp_input_file)


# Regression test #232
def test_is_vmec2000_input_with_comment():
    vmec2000_input_file = TEST_DATA_DIR / "input.solovev_analytical"
    assert vmecpp.is_vmec2000_input(vmec2000_input_file)


def test_ensure_vmec2000_input_noop():
    vmec2000_input_file = TEST_DATA_DIR / "input.cma"

    with vmecpp.ensure_vmec2000_input(vmec2000_input_file) as indata_file:
        assert indata_file == vmec2000_input_file


def test_ensure_vmec2000_input_with_null():
    # Test that the null values are handled gracefully and removed from the VMEC2000 input file
    vmec_input = vmecpp.VmecInput.default()
    assert vmec_input.rbs is None
    with tempfile.TemporaryDirectory() as tmp_dir:
        vmec_input.rbc = np.array([[1.0, 2.0, 3.0]])
        vmecpp_input_file = Path(tmp_dir) / "test_null.json"
        vmec_input.save(vmecpp_input_file)
        with vmecpp.ensure_vmec2000_input(vmecpp_input_file) as converted_indata_file:
            indata_namelist = converted_indata_file.read_text()
            assert "rbs" not in indata_namelist
            assert "rbc" in indata_namelist, indata_namelist


def test_ensure_vmecpp_input_noop():
    vmecpp_input_file = TEST_DATA_DIR / "cma.json"

    with vmecpp.ensure_vmecpp_input(vmecpp_input_file) as new_input_file:
        assert new_input_file == vmecpp_input_file


def test_ensure_vmecpp_input():
    vmec2000_input_file = TEST_DATA_DIR / "input.cma"

    with vmecpp.ensure_vmecpp_input(vmec2000_input_file) as vmecpp_input_file:
        assert vmecpp_input_file == TEST_DATA_DIR / f"cma.{os.getpid()}.json"
        with open(vmecpp_input_file) as f:
            vmecpp_input_dict = json.load(f)
            # check the output is remotely sensible: we don't want to test indata_to_json's
            # correctness here, just that nothing went terribly wrong
            assert vmecpp_input_dict["mpol"] == 5
            assert vmecpp_input_dict["ntor"] == 6


# Regression test for PR #181
def test_raise_invalid_threadcount():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")
    with pytest.raises(RuntimeError):
        vmecpp.run(vmec_input, max_threads=-1)
    with pytest.raises(RuntimeError):
        vmecpp.run(vmec_input, max_threads=0)


def test_vmec_input_validation():
    test_file = TEST_DATA_DIR / "solovev.json"
    vmec_input = vmecpp.VmecInput.from_file(test_file)

    # Why do we not compare `json_dict = json.loads(test_file.read_text())` ?
    # The test_file json may exclude fields that have default values,
    # while the parsed versions should have all fields populated.
    indata_dict_from_json = json.loads(vmec_input._to_cpp_vmecindata().to_json())
    # TODO(jurasic): iteration_style is not yet present in VmecInput, since there's only one option atm.
    del indata_dict_from_json["iteration_style"]
    vmec_input_dict_from_json = json.loads(vmec_input.model_dump_json())

    if not vmec_input.lasym:
        for lasym_field in ["rbs", "zbc", "raxis_s", "zaxis_c"]:
            del vmec_input_dict_from_json[lasym_field]

    assert indata_dict_from_json == vmec_input_dict_from_json


def test_vmec_output_serialization(cma_output: vmecpp.VmecOutput):
    # Since VmecOutput contains the other objects, we don't need to test them individually.
    serialized_output = cma_output.model_dump_json()
    deserialized_output = vmecpp.VmecOutput.model_validate_json(serialized_output)

    # Test nested objects (VmecWOut, JxbOut, Mercier, etc.)
    for field in vmecpp.VmecOutput.model_fields:
        deserialized_field = getattr(deserialized_output, field)
        output_field = getattr(cma_output, field)
        # Check the individual fields of the nested object
        for attr in vars(output_field):
            error_msg = f"mismatch in {attr}"

            np.testing.assert_equal(
                actual=getattr(deserialized_field, attr),
                desired=getattr(output_field, attr),
                err_msg=error_msg,
            )


def test_aux_arrays_from_cpp_wout():
    """Test that auxiliary arrays are correctly padded when empty, and padding doesn't
    accidentally overwrite any values."""

    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")
    vmec_output = vmecpp.run(vmec_input, verbose=False)
    cpp_wout = vmec_output.wout._to_cpp_wout()

    def assert_aux_defaults(wout: vmecpp.VmecWOut):
        # Check padding values and length
        assert len(wout.am_aux_s) == vmecpp.ndfmax
        assert len(wout.am_aux_f) == vmecpp.ndfmax
        assert len(wout.ac_aux_s) == vmecpp.ndfmax
        assert len(wout.ac_aux_f) == vmecpp.ndfmax
        assert len(wout.ai_aux_s) == vmecpp.ndfmax
        assert len(wout.ai_aux_f) == vmecpp.ndfmax

        # Verify _aux_s arrays are padded with -1
        np.testing.assert_allclose(wout.am_aux_s, -1.0)
        np.testing.assert_allclose(wout.ac_aux_s, -1.0)
        np.testing.assert_allclose(wout.ai_aux_s, -1.0)

        # Verify _aux_f arrays are padded with 0
        np.testing.assert_allclose(wout.am_aux_f, 0.0)
        np.testing.assert_allclose(wout.ac_aux_f, 0.0)
        np.testing.assert_allclose(wout.ai_aux_f, 0.0)

    assert_aux_defaults(vmecpp.VmecWOut._from_cpp_wout(cpp_wout))

    # Set all aux arrays to empty
    cpp_wout.am_aux_s = np.array([])
    cpp_wout.am_aux_f = np.array([])
    cpp_wout.ac_aux_s = np.array([])
    cpp_wout.ac_aux_f = np.array([])
    cpp_wout.ai_aux_s = np.array([])
    cpp_wout.ai_aux_f = np.array([])

    # Check that defaults work for empty aux arrays
    assert_aux_defaults(vmecpp.VmecWOut._from_cpp_wout(cpp_wout))

    # The defaults don't overwrite the original values, only pad them
    cpp_wout.am_aux_s = np.array([2.0, 3.0])
    cpp_wout.am_aux_f = np.array([2.0, 3.0])
    wout = vmecpp.VmecWOut._from_cpp_wout(cpp_wout)
    with pytest.raises(AssertionError):
        assert_aux_defaults(wout)
    np.testing.assert_almost_equal(wout.am_aux_s[:2], np.array([2.0, 3.0]))
    np.testing.assert_almost_equal(wout.am_aux_f[:2], np.array([2.0, 3.0]))


def test_populate_raw_profile_knots():
    vmec_input = vmecpp.VmecInput.default()
    vmec_input.ns_array = np.array([5, 9])

    def f(s):
        return s**2

    vmecpp.populate_raw_profile(vmec_input, "pressure", f)

    s_values = set()
    for ns in vmec_input.ns_array:
        delta = 1.0 / float(ns - 1)
        s_values.update(i * delta for i in range(ns))
        s_values.update((i - 0.5) * delta for i in range(ns))
    expected_knots = np.array(sorted(s_values), dtype=float)

    n = len(expected_knots)
    np.testing.assert_allclose(vmec_input.am_aux_s[:n], expected_knots)
    np.testing.assert_allclose(vmec_input.am_aux_f[:n], expected_knots**2)
    assert vmec_input.pmass_type == "line_segment"


def test_default_preset():
    # Default construction doesn't throw an exception
    default_preset = vmecpp.VmecInput.default()
    # Sample a few of the default values that should be set
    assert default_preset.nfp == 1
    assert default_preset.mpol == 6
    assert not default_preset.lasym
    assert default_preset.ns_array == np.array([31])


def test_python_defaults_match_cpp_defaults():
    python_defaults = vmecpp.VmecInput()
    cpp_defaults = vmecpp.VmecInput._from_cpp_vmecindata(_vmecpp.VmecINDATA())

    for field in vmecpp.VmecInput.model_fields:
        py_val = getattr(python_defaults, field)
        cpp_val = getattr(cpp_defaults, field)
        if isinstance(py_val, np.ndarray):
            np.testing.assert_array_equal(py_val, cpp_val)
        else:
            assert py_val == cpp_val


def test_ctrl_c_interrupts_run():
    """Test that a VMEC++ run can be interrupted with SIGINT (Ctrl+C).

    Launches a subprocess that starts a long VMEC++ run, sends SIGINT after the run has
    started, and verifies that the process terminates with KeyboardInterrupt rather than
    running to completion.
    """
    script = f"""\
import sys
import vmecpp
from pathlib import Path

vmec_input = vmecpp.VmecInput.from_file(
    Path({str(TEST_DATA_DIR)!r}) / "cma.json"
)
# Use many iterations to ensure the run doesn't finish before the signal
vmec_input.niter_array[-1] = 100000
vmec_input.ftol_array[-1] = 1.0e-18  # Don't terminate due to tolerance
try:
    vmecpp.run(vmec_input, verbose=True, max_threads=1)
    # Should not reach here
    print("RUN_COMPLETED")
except KeyboardInterrupt:
    print("KEYBOARD_INTERRUPT")
"""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        # Merge stderr into stdout so editable-install build output
        # doesn't block the stdout readline loop
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for the subprocess to signal that it's about to start the run.
    deadline = time.monotonic() + 60
    partial_output = ""
    assert proc.stdout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        partial_output += line
        # The tabular output of the first force iteration
        if "    1 |" in line:
            break
    else:
        raise RuntimeError(
            "The vmecpp subprocess did not start in time. Progress: \n" + partial_output
        )

    proc.send_signal(signal.SIGINT)

    remaining_output = proc.communicate(timeout=30)[0]
    output = partial_output + remaining_output
    assert "KEYBOARD_INTERRUPT" in output, (
        f"Expected KeyboardInterrupt but got:\noutput: {output}"
    )
    assert "RUN_COMPLETED" not in output
