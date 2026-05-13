# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Test that verifies the Python-side multigrid implementation matches the C++ one.

This test runs the same VMEC++ configuration using both the C++ multigrid loop
and the Python multigrid loop, then compares the full force residual time series
to verify the behavior is exactly reproduced at machine precision.

The Python multigrid must reproduce machine-precision-equivalent force_residual
arrays because:
1. The first multigrid step starts from identical initial conditions.
2. The interpolation between steps uses the same algorithm as C++:
   s-space linear interpolation with 1/sqrt(s) mode scaling (scalxc),
   with axis extrapolation for odd-m modes.
3. Identical initial conditions at each step lead to equivalent iteration paths.

Note on floating-point precision: The Python implementation encodes and decodes
the internal state through the wout external representation (which multiplies
and divides by mscale/nscale Fourier basis normalisation factors). This
introduces O(1e-15) floating-point rounding differences compared to the C++
multigrid which interpolates the internal state directly. The tests therefore
allow absolute differences up to 1e-13, well within machine precision.

The test uses the solovev.json configuration (axisymmetric tokamak, ntor=0)
because for 3D configurations the m1Constraint mixing of rss/zcs modes in
InitFromState introduces a systematic difference from the C++ multigrid path,
which does not apply m1Constraint during interpolation.
"""

from pathlib import Path

import numpy as np

import vmecpp
from vmecpp._multigrid import interpolate_to_new_radial_resolution

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def _run_python_multigrid_collect_all_residuals(
    vmec_input: vmecpp.VmecInput,
    *,
    max_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Python multigrid and collect force residuals from all steps.

    Returns:
        Tuple of (force_residual_r, force_residual_z, force_residual_lambda)
        arrays concatenated from all multigrid steps.
    """
    ns_array = vmec_input.ns_array.copy()
    ftol_array = vmec_input.ftol_array.copy()
    niter_array = vmec_input.niter_array.copy()

    all_fr: list[np.ndarray] = []
    all_fz: list[np.ndarray] = []
    all_fl: list[np.ndarray] = []

    vmec_output = None

    for _igrid, (ns, ftol, niter) in enumerate(
        zip(ns_array, ftol_array, niter_array, strict=True)
    ):
        step_input = vmec_input.model_copy(deep=True)
        step_input.ns_array = np.array([ns], dtype=np.int64)
        step_input.ftol_array = np.array([ftol])
        step_input.niter_array = np.array([niter], dtype=np.int64)

        if vmec_output is not None:
            old_ns = vmec_output.wout.ns
            if ns > old_ns:
                interpolated_wout = interpolate_to_new_radial_resolution(
                    vmec_output.wout, ns
                )
                restart_output = vmec_output.model_copy(deep=True)
                restart_output.wout = interpolated_wout
                restart_output.input = restart_output.input.model_copy(deep=True)
                restart_output.input.ns_array = np.array([ns], dtype=np.int64)
                restart_output.input.ftol_array = np.array([ftol])
                restart_output.input.niter_array = np.array([niter], dtype=np.int64)
            else:
                restart_output = vmec_output.model_copy(deep=True)
                restart_output.input = restart_output.input.model_copy(deep=True)
                restart_output.input.ns_array = np.array([ns], dtype=np.int64)
                restart_output.input.ftol_array = np.array([ftol])
                restart_output.input.niter_array = np.array([niter], dtype=np.int64)

            vmec_output = vmecpp.run(
                step_input,
                max_threads=max_threads,
                verbose=False,
                restart_from=restart_output,
            )
        else:
            vmec_output = vmecpp.run(
                step_input,
                max_threads=max_threads,
                verbose=False,
            )

        all_fr.append(vmec_output.wout.force_residual_r)
        all_fz.append(vmec_output.wout.force_residual_z)
        all_fl.append(vmec_output.wout.force_residual_lambda)

    return (
        np.concatenate(all_fr),
        np.concatenate(all_fz),
        np.concatenate(all_fl),
    )


def test_python_multigrid_force_residual_series_matches_cpp():
    """Test that the full force_residual time series matches between Python and C++.

    The Python multigrid must reproduce force_residual arrays to machine precision. Any
    systematic discrepancy larger than floating-point rounding (~1e-15 absolute)
    indicates that the interpolation between multigrid steps differs from the C++
    InterpolateToNextMultigridStep.

    Uses solovev.json (ntor=0, axisymmetric tokamak) where the m1Constraint is a no-op
    and the Python hot-restart round-trip is lossless up to floating-point rounding from
    the mscale/nscale basis normalisation.
    """
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")

    # Two multigrid steps to exercise the interpolation path
    vmec_input.ns_array = np.array([5, 25], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-10, 1e-10])
    vmec_input.niter_array = np.array([1000, 1000], dtype=np.int64)

    # Run with C++ multigrid (default) - force_residual arrays span all steps
    cpp_output = vmecpp.run(vmec_input, max_threads=1, verbose=False)

    # Run with Python multigrid, collecting per-step force_residual arrays
    py_fr, py_fz, py_fl = _run_python_multigrid_collect_all_residuals(
        vmec_input, max_threads=1
    )

    cpp_fr = cpp_output.wout.force_residual_r
    cpp_fz = cpp_output.wout.force_residual_z
    cpp_fl = cpp_output.wout.force_residual_lambda

    # The lengths must match: same number of iterations in each step
    assert len(py_fr) == len(cpp_fr), (
        f"force_residual_r length mismatch: Python={len(py_fr)}, C++={len(cpp_fr)}"
    )
    assert len(py_fz) == len(cpp_fz), (
        f"force_residual_z length mismatch: Python={len(py_fz)}, C++={len(cpp_fz)}"
    )
    assert len(py_fl) == len(cpp_fl), (
        f"force_residual_lambda length mismatch: Python={len(py_fl)}, C++={len(cpp_fl)}"
    )

    # The force_residual time series must match at machine precision.
    # The absolute tolerance of 1e-13 accounts for O(1e-15) floating-point
    # rounding differences from the mscale/nscale round-trip in the wout
    # representation (which introduces extra floating-point multiplications
    # compared to the C++ multigrid that operates on the internal state directly).
    np.testing.assert_allclose(
        py_fr,
        cpp_fr,
        rtol=0,
        atol=1e-13,
        err_msg="force_residual_r time series differs beyond machine precision",
    )
    np.testing.assert_allclose(
        py_fz,
        cpp_fz,
        rtol=0,
        atol=1e-13,
        err_msg="force_residual_z time series differs beyond machine precision",
    )
    np.testing.assert_allclose(
        py_fl,
        cpp_fl,
        rtol=0,
        atol=1e-13,
        err_msg="force_residual_lambda time series differs beyond machine precision",
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
    # The interpolation code explicitly sets these to zero, so they should be
    # exactly 0.0, but we use assert_allclose with tight tolerance for robustness
    for mn in range(output.wout.mnmax):
        m = int(output.wout.xm[mn])
        if m % 2 == 1:  # odd m
            np.testing.assert_allclose(
                interpolated.rmnc[mn, 0],
                0.0,
                atol=1e-15,
                err_msg=f"Odd-m mode m={m} rmnc should be zero at axis",
            )
            np.testing.assert_allclose(
                interpolated.zmns[mn, 0],
                0.0,
                atol=1e-15,
                err_msg=f"Odd-m mode m={m} zmns should be zero at axis",
            )
            np.testing.assert_allclose(
                interpolated.lmns_full[mn, 0],
                0.0,
                atol=1e-15,
                err_msg=f"Odd-m mode m={m} lmns_full should be zero at axis",
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
