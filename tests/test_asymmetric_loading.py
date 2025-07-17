"""Test asymmetric tokamak input file loading with automatic array initialization.

This test validates that the input.up_down_asymmetric_tokamak file can be loaded
successfully with the asymmetric array initialization fix in __init__.py.
"""

import pytest

import vmecpp


def test_asymmetric_tokamak_input_loading():
    """Test that asymmetric tokamak input file loads without validation errors."""
    # Load the asymmetric tokamak test case
    input_file = "examples/data/input.up_down_asymmetric_tokamak"

    # This should work with the fix in __init__.py that initializes
    # missing asymmetric arrays to zero for asymmetric runs
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

    # Verify the key asymmetric coefficients from the input file
    # Note: Input file has ntor=0, so only n=0 coefficients exist
    # RBS(0,1) in INDATA format = (n=0, m=1) maps to rbs[1, 0] in array format
    # RBS(0,2) in INDATA format = (n=0, m=2) maps to rbs[2, 0] in array format
    assert vmec_input.rbs[1, 0] == pytest.approx(0.6)  # RBS(0,1) from input file
    assert vmec_input.rbs[2, 0] == pytest.approx(0.12)  # RBS(0,2) from input file

    # Verify other asymmetric arrays are initialized to zero
    assert vmec_input.zbc[0, 0] == pytest.approx(0.0)
    assert vmec_input.raxis_s[0] == pytest.approx(0.0)
    assert vmec_input.zaxis_c[0] == pytest.approx(0.0)


def test_asymmetric_arrays_initialization_for_asymmetric_run():
    """Test that missing asymmetric arrays are properly initialized for lasym=True."""
    # This test verifies the fix in _from_cpp_vmecindatapywrapper function
    # Load an asymmetric input that may have missing asymmetric arrays
    input_file = "examples/data/input.up_down_asymmetric_tokamak"

    vmec_input = vmecpp.VmecInput.from_file(input_file)

    # All asymmetric arrays should be initialized (not None) for asymmetric runs
    assert vmec_input.lasym is True
    assert vmec_input.rbs is not None
    assert vmec_input.zbc is not None
    assert vmec_input.raxis_s is not None
    assert vmec_input.zaxis_c is not None

    # Arrays should have proper shapes based on mpol and ntor
    mpol = vmec_input.mpol
    ntor = vmec_input.ntor
    boundary_shape = (mpol, 2 * ntor + 1)
    axis_shape = (ntor + 1,)

    assert vmec_input.rbs.shape == boundary_shape
    assert vmec_input.zbc.shape == boundary_shape
    assert vmec_input.raxis_s.shape == axis_shape
    assert vmec_input.zaxis_c.shape == axis_shape
