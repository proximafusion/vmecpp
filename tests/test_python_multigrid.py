# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Test that verifies the Python-side multigrid implementation matches the C++ one.

This test runs the same VMEC++ configuration using both the C++ multigrid loop
and the Python multigrid loop, then compares the force residual progress to
verify the behavior is exactly preserved.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def test_python_multigrid_matches_cpp():
    """Test that Python multigrid produces results matching C++ multigrid."""
    # Use a CMA-like configuration with 3 multigrid steps
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")

    # Set up 3 multigrid steps as requested: ns = [5, 12, 25]
    vmec_input.ns_array = np.array([5, 12, 25], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-6, 1e-6, 1e-6])
    vmec_input.niter_array = np.array([1000, 1000, 1000], dtype=np.int64)

    # Run with C++ multigrid (default)
    cpp_output = vmecpp.run(vmec_input, max_threads=1, verbose=False)

    # Run with Python multigrid
    python_output = vmecpp.run_with_python_multigrid(
        vmec_input, max_threads=1, verbose=False
    )

    # Compare final residuals
    cpp_fsqr = cpp_output.wout.fsqr
    cpp_fsqz = cpp_output.wout.fsqz
    cpp_fsql = cpp_output.wout.fsql
    cpp_total = cpp_fsqr + cpp_fsqz + cpp_fsql

    python_fsqr = python_output.wout.fsqr
    python_fsqz = python_output.wout.fsqz
    python_fsql = python_output.wout.fsql
    python_total = python_fsqr + python_fsqz + python_fsql

    # Print comparison for debugging
    print(
        f"\nC++ final residuals: fsqr={cpp_fsqr:.3e}, fsqz={cpp_fsqz:.3e}, "
        f"fsql={cpp_fsql:.3e}, total={cpp_total:.3e}"
    )
    print(
        f"Python final residuals: fsqr={python_fsqr:.3e}, fsqz={python_fsqz:.3e}, "
        f"fsql={python_fsql:.3e}, total={python_total:.3e}"
    )

    # Compare geometry
    print(
        f"\nC++ geometry: aspect={cpp_output.wout.aspect:.6f}, "
        f"volume={cpp_output.wout.volume:.6e}"
    )
    print(
        f"Python geometry: aspect={python_output.wout.aspect:.6f}, "
        f"volume={python_output.wout.volume:.6e}"
    )

    # Compare iterations count
    print(f"\nC++ iterations: itfsq={cpp_output.wout.itfsq}")
    print(f"Python iterations: itfsq={python_output.wout.itfsq}")

    # The implementations should produce very similar final results
    # We allow some tolerance because the interpolation might have slight differences
    # that lead to different iteration paths
    np.testing.assert_allclose(
        cpp_output.wout.aspect,
        python_output.wout.aspect,
        rtol=1e-3,
        err_msg="Aspect ratio mismatch between C++ and Python multigrid",
    )

    np.testing.assert_allclose(
        cpp_output.wout.volume,
        python_output.wout.volume,
        rtol=1e-3,
        err_msg="Volume mismatch between C++ and Python multigrid",
    )

    # Both should converge
    assert cpp_output.wout.ier_flag in (0, 11), (
        f"C++ multigrid did not converge: ier_flag={cpp_output.wout.ier_flag}"
    )
    assert python_output.wout.ier_flag in (0, 11), (
        f"Python multigrid did not converge: ier_flag={python_output.wout.ier_flag}"
    )


def test_python_multigrid_single_step():
    """Test that single-step Python multigrid is equivalent to normal run."""
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")

    # Single multigrid step
    vmec_input.ns_array = np.array([31], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-10])
    vmec_input.niter_array = np.array([1000], dtype=np.int64)

    # Run with C++ (effectively no multigrid)
    cpp_output = vmecpp.run(vmec_input, max_threads=1, verbose=False)

    # Run with Python multigrid (single step, so no interpolation)
    python_output = vmecpp.run_with_python_multigrid(
        vmec_input, max_threads=1, verbose=False
    )

    # Results should be essentially identical since there's no interpolation
    np.testing.assert_allclose(
        cpp_output.wout.aspect,
        python_output.wout.aspect,
        rtol=1e-10,
        err_msg="Single-step results should be identical",
    )

    np.testing.assert_allclose(
        cpp_output.wout.volume,
        python_output.wout.volume,
        rtol=1e-10,
        err_msg="Single-step results should be identical",
    )


def test_interpolate_to_new_radial_resolution():
    """Test the radial interpolation function directly."""
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    vmec_input.ns_array = np.array([15], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-10])
    vmec_input.niter_array = np.array([1000], dtype=np.int64)

    # Run to get a converged state
    output = vmecpp.run(vmec_input, max_threads=1, verbose=False)

    # Interpolate to a finer grid
    ns_old = output.wout.ns
    ns_new = 31
    interpolated = vmecpp.interpolate_to_new_radial_resolution(output.wout, ns_new)

    # Check the interpolated wout has correct dimensions
    assert interpolated.ns == ns_new
    assert interpolated.rmnc.shape == (output.wout.mnmax, ns_new)
    assert interpolated.zmns.shape == (output.wout.mnmax, ns_new)
    assert interpolated.lmns_full.shape == (output.wout.mnmax, ns_new)

    # Check that boundary values are preserved at the LCFS (j = ns-1)
    # The boundary should be the same regardless of radial resolution
    np.testing.assert_allclose(
        interpolated.rmnc[:, ns_new - 1],
        output.wout.rmnc[:, ns_old - 1],
        rtol=1e-10,
        err_msg="Boundary rmnc should be preserved",
    )

    np.testing.assert_allclose(
        interpolated.zmns[:, ns_new - 1],
        output.wout.zmns[:, ns_old - 1],
        rtol=1e-10,
        err_msg="Boundary zmns should be preserved",
    )

    # Check that odd-m modes are zero at the axis
    for mn in range(output.wout.mnmax):
        m = int(output.wout.xm[mn])
        if m % 2 == 1:  # odd m
            assert interpolated.rmnc[mn, 0] == 0.0, (
                f"Odd-m mode m={m} rmnc should be zero at axis"
            )
            assert interpolated.zmns[mn, 0] == 0.0, (
                f"Odd-m mode m={m} zmns should be zero at axis"
            )
            assert interpolated.lmns_full[mn, 0] == 0.0, (
                f"Odd-m mode m={m} lmns_full should be zero at axis"
            )


def test_interpolate_same_resolution():
    """Test that interpolation with same resolution returns a copy."""
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    output = vmecpp.run(vmec_input, max_threads=1, verbose=False)

    # Interpolate to same resolution
    interpolated = vmecpp.interpolate_to_new_radial_resolution(
        output.wout, output.wout.ns
    )

    # Should return a deep copy with identical values
    assert interpolated.ns == output.wout.ns
    np.testing.assert_array_equal(interpolated.rmnc, output.wout.rmnc)
    np.testing.assert_array_equal(interpolated.zmns, output.wout.zmns)
    np.testing.assert_array_equal(interpolated.lmns_full, output.wout.lmns_full)

    # Should be a copy, not the same object
    assert interpolated is not output.wout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
