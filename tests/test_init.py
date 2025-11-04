# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for VMEC++'s Python API.

Here we just test that the Python bindings and the general API works as expected.
Physics correctness is checked at the level of the C++ core.
"""

import json
import os
import tempfile
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
    # TODO(jurasic): These quantities are not yet present in VmecInput, since there's only one option atm.
    del indata_dict_from_json["free_boundary_method"]
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


def test_interpolate_to():
    vmec_input_filename = TEST_DATA_DIR / "cma.json"
    vmec_input = vmecpp.VmecInput.from_file(vmec_input_filename)

    mpol = vmec_input.mpol
    ntor = vmec_input.ntor
    ns_old = vmec_input.ns_array[0]
    ns_new = vmec_input.ns_array[1]

    scalxc_old = np.zeros([ns_old])
    sqrtSF1_old = np.sqrt(1.0 / (ns_old - 1.0))
    for jF in range(ns_old):
        sqrtSF = np.sqrt(jF / (ns_old - 1.0))
        scalxc_old[jF] = 1.0 / max(sqrtSF, sqrtSF1_old)

    scalxc_new = np.zeros([ns_new])
    sqrtSF1_new = np.sqrt(1.0 / (ns_new - 1.0))
    for jF in range(ns_new):
        sqrtSF = np.sqrt(jF / (ns_new - 1.0))
        scalxc_new[jF] = 1.0 / max(sqrtSF, sqrtSF1_new)

    # load the interp reference data from src/vmecpp/cpp/vmecpp_large_cpp_tests/test_data/cma/interp/interp_00051_000001_01.cma.json
    LARGE_TEST_DATA_DIR = (
        REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp_large_cpp_tests" / "test_data"
    )
    interp_reference_file = Path(
        LARGE_TEST_DATA_DIR / "cma" / "interp" / "interp_00051_000001_01.cma.json"
    )

    vmec_input_only_first_step = vmec_input.model_copy(deep=True)
    vmec_input_only_first_step.ns_array = np.array([vmec_input.ns_array[0]])
    vmec_input_only_first_step.ftol_array = np.array([vmec_input.ftol_array[0]])
    vmec_input_only_first_step.niter_array = np.array([vmec_input.niter_array[0]])

    vmec_output = vmecpp.run(vmec_input_only_first_step)

    assert vmec_output.wout.ns == vmec_input_only_first_step.ns_array[0]

    interp_reference = {}
    with open(interp_reference_file) as f:
        interp_reference = json.load(f)

    xold_ref = np.array(interp_reference["xold"])
    xnew_ref = np.array(interp_reference["xnew"])

    # 3 (R,Z,L), 2 (SC/CS,SS/CC), 25 (ns), 7(ntor+1), 5(mpol)
    # print(xold_ref.shape)  # (3, 2, 25, 7, 5)
    # print(xnew_ref.shape)  # (3, 2, 51, 7, 5)

    mscale = np.ones(mpol)
    mscale[1:] *= np.sqrt(2.0)

    nscale = np.ones(ntor + 1)
    nscale[1:] *= np.sqrt(2.0)

    ################### other conversion start ###################

    # Allocate reconstructed internal arrays (mixed versions for m=1 already)
    rmncc_rec = np.zeros((ns_old, ntor + 1, mpol))
    rmnss_rec = np.zeros((ns_old, ntor + 1, mpol))
    zmnsc_rec = np.zeros((ns_old, ntor + 1, mpol))
    zmncs_rec = np.zeros((ns_old, ntor + 1, mpol))
    lmnsc_rec = np.zeros((ns_old, ntor + 1, mpol))
    lmncs_rec = np.zeros((ns_old, ntor + 1, mpol))

    # Loop over full radial surfaces
    for jF in range(ns_old):
        # Initialize external index
        mn = 0

        # Handle m = 0 block: n = 0..ntor
        m = 0
        for n in range(ntor + 1):
            # Compute scale factor t1
            t1 = mscale[m] * nscale[n]

            # Invert rmnc_ref for rmncc
            rmncc_rec[jF, n, m] = vmec_output.wout.rmnc[mn, jF] / t1

            # Invert zmns_ref for zmncs_ (note the minus sign in forward)
            zmncs_rec[jF, n, m] = -vmec_output.wout.zmns[mn, jF] / t1

            lmncs_rec[jF, n, m] = -vmec_output.wout.lmns_full[mn, jF] / t1

            # Advance external index
            mn += 1

        # Handle m >= 1 blocks
        for m in range(1, mpol):
            # Precompute scale vector for all |n|
            t1_absn = mscale[m] * nscale

            # Handle n = 0 directly
            # Extract scale for |n|=0
            t10 = t1_absn[0]

            # Assign from external with simple inversion
            rmncc_rec[jF, 0, m] = vmec_output.wout.rmnc[mn + ntor, jF] / t10
            zmnsc_rec[jF, 0, m] = vmec_output.wout.zmns[mn + ntor, jF] / t10
            lmnsc_rec[jF, 0, m] = vmec_output.wout.lmns_full[mn + ntor, jF] / t10

            # Loop positive n only and use pair (+n, -n) to solve the 2x2 systems
            for npos in range(1, ntor + 1):
                # mn index for n=0 sits at mn + ntor

                # Compute mn index for -npos
                mn_neg = mn + ntor - npos

                # Compute mn index for +npos
                mn_pos = mn + ntor + npos

                # Fetch scale
                t1 = t1_absn[npos]

                rmncc_rec[jF, npos, m] = (
                    vmec_output.wout.rmnc[mn_pos, jF]
                    + vmec_output.wout.rmnc[mn_neg, jF]
                ) / t1
                rmnss_rec[jF, npos, m] = (
                    vmec_output.wout.rmnc[mn_pos, jF]
                    - vmec_output.wout.rmnc[mn_neg, jF]
                ) / t1

                zmnsc_rec[jF, npos, m] = (
                    vmec_output.wout.zmns[mn_pos, jF]
                    + vmec_output.wout.zmns[mn_neg, jF]
                ) / t1
                zmncs_rec[jF, npos, m] = (
                    vmec_output.wout.zmns[mn_neg, jF]
                    - vmec_output.wout.zmns[mn_pos, jF]
                ) / t1

                lmnsc_rec[jF, npos, m] = (
                    vmec_output.wout.lmns_full[mn_pos, jF]
                    + vmec_output.wout.lmns_full[mn_neg, jF]
                ) / t1
                lmncs_rec[jF, npos, m] = (
                    vmec_output.wout.lmns_full[mn_neg, jF]
                    - vmec_output.wout.lmns_full[mn_pos, jF]
                ) / t1

            # Advance external index block for this m by (2*ntor + 1)
            mn += 2 * ntor + 1

        # activate m=1 constraint
        old_rss = rmnss_rec[jF, :, 1].copy()
        rmnss_rec[jF, :, 1] = 0.5 * (old_rss + zmncs_rec[jF, :, 1])
        zmncs_rec[jF, :, 1] = 0.5 * (old_rss - zmncs_rec[jF, :, 1])

        # apply scalxc factors
        rmncc_rec[jF, :, 1::2] *= scalxc_old[jF]
        rmnss_rec[jF, :, 1::2] *= scalxc_old[jF]
        zmnsc_rec[jF, :, 1::2] *= scalxc_old[jF]
        zmncs_rec[jF, :, 1::2] *= scalxc_old[jF]
        lmnsc_rec[jF, :, 1::2] *= scalxc_old[jF]
        lmncs_rec[jF, :, 1::2] *= scalxc_old[jF]

    # fix lambda scaling
    lmnsc_rec *= -1.0
    lmncs_rec *= -1.0

    # set lambda to zero on-axis, except for odd-m extrap below
    lmnsc_rec[0, :, :] = 0.0
    lmncs_rec[0, :, :] = 0.0

    # extrapolate odd-m to axis
    rmncc_rec[0, :, 1::2] = 2.0 * rmncc_rec[1, :, 1::2] - rmncc_rec[2, :, 1::2]
    rmnss_rec[0, :, 1::2] = 2.0 * rmnss_rec[1, :, 1::2] - rmnss_rec[2, :, 1::2]
    zmnsc_rec[0, :, 1::2] = 2.0 * zmnsc_rec[1, :, 1::2] - zmnsc_rec[2, :, 1::2]
    zmncs_rec[0, :, 1::2] = 2.0 * zmncs_rec[1, :, 1::2] - zmncs_rec[2, :, 1::2]
    lmnsc_rec[0, :, 1::2] = 2.0 * lmnsc_rec[1, :, 1::2] - lmnsc_rec[2, :, 1::2]
    lmncs_rec[0, :, 1::2] = 2.0 * lmncs_rec[1, :, 1::2] - lmncs_rec[2, :, 1::2]

    ################### other conversion end ###################

    np.testing.assert_array_almost_equal(xold_ref[0, 0, :, :, :], rmncc_rec, decimal=13)
    np.testing.assert_array_almost_equal(xold_ref[0, 1, :, :, :], rmnss_rec, decimal=13)
    np.testing.assert_array_almost_equal(xold_ref[1, 0, :, :, :], zmnsc_rec, decimal=13)
    np.testing.assert_array_almost_equal(xold_ref[1, 1, :, :, :], zmncs_rec, decimal=13)
    np.testing.assert_array_almost_equal(xold_ref[2, 0, :, :, :], lmnsc_rec, decimal=11)
    np.testing.assert_array_almost_equal(xold_ref[2, 1, :, :, :], lmncs_rec, decimal=11)

    # implement interpoation and check interpolated state vector against xnew_ref

    s_old = np.linspace(0.0, 1.0, ns_old, endpoint=True)
    s_new = np.linspace(0.0, 1.0, ns_new, endpoint=True)

    rmncc_new = np.zeros([ns_new, ntor + 1, mpol])
    rmnss_new = np.zeros([ns_new, ntor + 1, mpol])
    zmnsc_new = np.zeros([ns_new, ntor + 1, mpol])
    zmncs_new = np.zeros([ns_new, ntor + 1, mpol])
    lmnsc_new = np.zeros([ns_new, ntor + 1, mpol])
    lmncs_new = np.zeros([ns_new, ntor + 1, mpol])

    for n in range(ntor + 1):
        for m in range(mpol):
            rmncc_new[:, n, m] = np.interp(x=s_new, xp=s_old, fp=rmncc_rec[:, n, m])
            rmnss_new[:, n, m] = np.interp(x=s_new, xp=s_old, fp=rmnss_rec[:, n, m])
            zmnsc_new[:, n, m] = np.interp(x=s_new, xp=s_old, fp=zmnsc_rec[:, n, m])
            zmncs_new[:, n, m] = np.interp(x=s_new, xp=s_old, fp=zmncs_rec[:, n, m])
            lmnsc_new[:, n, m] = np.interp(x=s_new, xp=s_old, fp=lmnsc_rec[:, n, m])
            lmncs_new[:, n, m] = np.interp(x=s_new, xp=s_old, fp=lmncs_rec[:, n, m])

    rmncc_new[:, :, 1::2] /= scalxc_new[:, np.newaxis, np.newaxis]
    rmnss_new[:, :, 1::2] /= scalxc_new[:, np.newaxis, np.newaxis]
    zmnsc_new[:, :, 1::2] /= scalxc_new[:, np.newaxis, np.newaxis]
    zmncs_new[:, :, 1::2] /= scalxc_new[:, np.newaxis, np.newaxis]
    lmnsc_new[:, :, 1::2] /= scalxc_new[:, np.newaxis, np.newaxis]
    lmncs_new[:, :, 1::2] /= scalxc_new[:, np.newaxis, np.newaxis]

    # zero out odd-m on axis extrapolation leftovers
    rmncc_new[0, :, 1::2] = 0.0
    rmnss_new[0, :, 1::2] = 0.0
    zmnsc_new[0, :, 1::2] = 0.0
    zmncs_new[0, :, 1::2] = 0.0
    lmnsc_new[0, :, 1::2] = 0.0
    lmncs_new[0, :, 1::2] = 0.0

    # for title, act, ref in zip(
    #     [
    #         "rmncc",
    #         "rmnss",
    #         "zmnsc",
    #         "zmncs",
    #         "lmnsc",
    #         "lmncs",
    #     ],
    #     [rmncc_new, rmnss_new, zmnsc_new, zmncs_new, lmnsc_new, lmncs_new],
    #     [
    #         xnew_ref[0, 0, :, :, :],
    #         xnew_ref[0, 1, :, :, :],
    #         xnew_ref[1, 0, :, :, :],
    #         xnew_ref[1, 1, :, :, :],
    #         xnew_ref[2, 0, :, :, :],
    #         xnew_ref[2, 1, :, :, :],
    #     ],
    #     strict=False,
    # ):
    #     for jF in range(ns_new):
    #         # first check: compare xold against vmec_output.wout
    #         plt.figure()
    #         plt.subplot(1, 3, 1)
    #         plt.title("ref")
    #         plt.imshow(ref[jF, :, :], origin="lower", aspect="auto")
    #         plt.colorbar()
    #         plt.subplot(1, 3, 2)
    #         plt.title(f"{title} act j={jF}")
    #         plt.imshow(act[jF, :, :], origin="lower", aspect="auto")
    #         plt.colorbar()
    #         plt.subplot(1, 3, 3)
    #         plt.title("err")
    #         plt.imshow(
    #             np.log10(
    #                 1.0e-30
    #                 + np.abs(act[jF, :, :] - ref[jF, :, :])
    #                 / (1.0 + np.abs(ref[jF, :, :]))
    #             ),
    #             origin="lower",
    #             aspect="auto",
    #         )
    #         plt.colorbar()
    #         plt.tight_layout()

    # plt.show()

    np.testing.assert_array_almost_equal(xnew_ref[0, 0, :, :, :], rmncc_new, decimal=13)
    np.testing.assert_array_almost_equal(xnew_ref[0, 1, :, :, :], rmnss_new, decimal=13)
    np.testing.assert_array_almost_equal(xnew_ref[1, 0, :, :, :], zmnsc_new, decimal=13)
    np.testing.assert_array_almost_equal(xnew_ref[1, 1, :, :, :], zmncs_new, decimal=13)
    np.testing.assert_array_almost_equal(xnew_ref[2, 0, :, :, :], lmnsc_new, decimal=11)
    np.testing.assert_array_almost_equal(xnew_ref[2, 1, :, :, :], lmncs_new, decimal=11)

    ############## start of conversion ##############

    # zero out spurious extrapolation at axis from interp leftovers in reference in odd-m modes
    xold_ref[:, :, 0, :, 1::2] = 0.0

    # undo scalxc in reference
    xold_ref[:, :, :, :, 1::2] /= scalxc_old[
        np.newaxis, np.newaxis, :, np.newaxis, np.newaxis
    ]

    rmncc = xold_ref[0, 0, :, :, :]
    rmnss = xold_ref[0, 1, :, :, :]
    zmnsc = xold_ref[1, 0, :, :, :]
    zmncs = xold_ref[1, 1, :, :, :]
    lmnsc = xold_ref[2, 0, :, :, :]
    lmncs = xold_ref[2, 1, :, :, :]

    mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1)

    # Copy internal arrays so the in-place "M=1 conversion" does not mutate inputs
    rmnss_ = rmnss.copy()
    zmncs_ = zmncs.copy()

    # Apply the m=1 internal→physical mixing only for 3D
    # Select the m=1 slice for all jF and all n>=0
    old_rss = rmnss_[:, :, 1].copy()
    # rmnss <- old_rss + zmncs
    rmnss_[:, :, 1] = old_rss + zmncs_[:, :, 1]
    # zmncs <- old_rss - zmncs
    zmncs_[:, :, 1] = old_rss - zmncs_[:, :, 1]

    # Allocate external arrays
    rmnc_ref = np.zeros((mnmax, ns_old))
    zmns_ref = np.zeros((mnmax, ns_old))
    lmns_ref = np.zeros((mnmax, ns_old))

    # Process each full radial surface jF independently
    for jF in range(ns_old):
        mn = 0

        # Handle m = 0 block: n = 0..ntor
        m0 = 0
        for n in range(ntor + 1):
            # Compute the scaling t1 = mscale[m]*nscale[|n|]
            t1 = mscale[m0] * nscale[n]

            # Accumulate rmnc from rmncc(jF, |n|, m=0)
            rmnc_ref[mn, jF] = t1 * rmncc[jF, n, m0]

            # For 3D, set zmns from zmncs with minus sign
            zmns_ref[mn, jF] = -t1 * zmncs_[jF, n, m0]

            lmns_ref[mn, jF] = -t1 * lmncs[jF, n, m0]

            # Increment external mode index
            mn += 1

        # extrapolate lambda to axis for m=0
        if jF == 0:
            for n in range(ntor + 1):
                t1 = mscale[m0] * nscale[n]
                lmns_ref[n, jF] = -t1 * (2.0 * lmncs[1, n, m0] - lmncs[2, n, m0])

        # Handle m >= 1 blocks
        for m in range(1, mpol):
            for n in range(-ntor, ntor + 1):
                # Use |n| for indexing the internal arrays/nscale
                abs_n = abs(n)

                # Build scaling t1 = mscale[m]*nscale[|n|]
                t1 = mscale[m] * nscale[abs_n]

                # Handle the n == 0 case (no 1/2 and no sign mixing)
                if n == 0:
                    # Set external rmnc/zmns/lmns directly from internal cc/sc
                    rmnc_ref[mn, jF] = t1 * rmncc[jF, abs_n, m]
                    zmns_ref[mn, jF] = t1 * zmnsc[jF, abs_n, m]
                    lmns_ref[mn, jF] = t1 * lmnsc[jF, abs_n, m]
                # For jF == 0, leave zeros (already initialized)
                elif jF > 0:  # n != 0 AND jF > 0
                    sign_n = int(np.sign(n))

                    # Start with 1/2 of cc/sc parts
                    # For 3D, add/subtract the mixed cs/ss parts with sign of n
                    rmnc_ref[mn, jF] = (
                        t1 * (rmncc[jF, abs_n, m] + sign_n * rmnss_[jF, abs_n, m]) / 2.0
                    )
                    zmns_ref[mn, jF] = (
                        t1 * (zmnsc[jF, abs_n, m] - sign_n * zmncs_[jF, abs_n, m]) / 2.0
                    )
                    lmns_ref[mn, jF] = (
                        t1 * (lmnsc[jF, abs_n, m] - sign_n * lmncs[jF, abs_n, m]) / 2.0
                    )

                # Increment external mode index
                mn += 1

    lmns_ref *= -1.0

    ############## end of conversion ##############

    # # first check: compare xold against vmec_output.wout
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.title("rmnc ref")
    # plt.imshow(rmnc_ref, origin="lower", aspect="auto")
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.title("rmnc act")
    # plt.imshow(vmec_output.wout.rmnc, origin="lower", aspect="auto")
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.title("rmnc err")
    # plt.imshow(
    #     np.log10(
    #         1.0e-30
    #         + np.abs(vmec_output.wout.rmnc - rmnc_ref) / (1.0 + np.abs(rmnc_ref))
    #     ),
    #     origin="lower",
    #     aspect="auto",
    # )
    # plt.colorbar()
    # plt.tight_layout()

    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.title("zmns ref")
    # plt.imshow(zmns_ref, origin="lower", aspect="auto")
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.title("zmns act")
    # plt.imshow(vmec_output.wout.zmns, origin="lower", aspect="auto")
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.title("zmns err")
    # plt.imshow(
    #     np.log10(
    #         1.0e-30
    #         + np.abs(vmec_output.wout.zmns - zmns_ref) / (1.0 + np.abs(zmns_ref))
    #     ),
    #     origin="lower",
    #     aspect="auto",
    # )
    # plt.colorbar()
    # plt.tight_layout()

    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.title("lmns ref")
    # plt.imshow(lmns_ref, origin="lower", aspect="auto")
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.title("lmns act")
    # plt.imshow(vmec_output.wout.lmns_full, origin="lower", aspect="auto")
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.title("lmns err")
    # plt.imshow(
    #     np.log10(
    #         1.0e-30
    #         + np.abs(vmec_output.wout.lmns_full - lmns_ref)
    #         / (1.0 + np.abs(lmns_ref))
    #     ),
    #     origin="lower",
    #     aspect="auto",
    # )
    # plt.colorbar()
    # plt.tight_layout()

    # plt.show()

    np.testing.assert_array_almost_equal(rmnc_ref, vmec_output.wout.rmnc, decimal=13)
    np.testing.assert_array_almost_equal(zmns_ref, vmec_output.wout.zmns, decimal=13)
    np.testing.assert_array_almost_equal(
        lmns_ref, vmec_output.wout.lmns_full, decimal=11
    )

    ###############

    # TODO(jons):
    # 1. How do I need to provide rmnc, zmns, lmns_full to vmecpp.run() as HotRestartState,
    #    in order for the iterations to proceed as if a regular multi-grid step is continued?
    # 2. Do I need to adjust other flow control variables? Probably not, since they are reset anyway between steps.
