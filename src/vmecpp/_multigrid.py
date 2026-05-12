# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Python-side multigrid iteration for VMEC++.

This module provides a Python implementation of the multigrid iteration loop,
which allows for more flexible control over the iteration process.

The C++ implementation remains the default, but this Python implementation
can be used for experimentation and debugging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vmecpp import (
        MagneticFieldResponseTable,
        OutputMode,
        VmecInput,
        VmecOutput,
        VmecWOut,
    )


logger = logging.getLogger(__name__)


def interpolate_to_new_radial_resolution(
    wout: VmecWOut,
    ns_new: int,
) -> VmecWOut:
    """Interpolate the VMEC state to a new radial resolution.

    This function performs radial interpolation of the Fourier coefficients
    (rmnc, zmns, lmns_full) from the old radial grid to a new finer grid.
    This is the Python equivalent of Vmec::InterpolateToNextMultigridStep in C++.

    The interpolation includes:
    1. Extrapolation of odd-m modes at the magnetic axis
    2. Linear interpolation in sqrt(s) space with proper scaling
    3. Zeroing of odd-m modes at the origin

    Args:
        wout: The VmecWOut object containing the current state
        ns_new: The new number of flux surfaces (must be >= wout.ns)

    Returns:
        A new VmecWOut object with interpolated coefficients at the new resolution
    """
    ns_old = wout.ns
    mnmax = wout.mnmax
    xm = wout.xm

    if ns_new < ns_old:
        msg = f"ns_new ({ns_new}) must be >= ns_old ({ns_old})"
        raise ValueError(msg)

    if ns_new == ns_old:
        # No interpolation needed, return a copy
        return wout.model_copy(deep=True)

    # Create output arrays
    rmnc_new = np.zeros((mnmax, ns_new))
    zmns_new = np.zeros((mnmax, ns_new))
    lmns_full_new = np.zeros((mnmax, ns_new))

    # Get source data
    rmnc_old = wout.rmnc
    zmns_old = wout.zmns
    lmns_full_old = wout.lmns_full

    # Compute radial grids in sqrt(s) space.
    # VMEC uses sqrt(s) as the interpolation coordinate because:
    # 1. The Fourier coefficients for odd-m modes (m=1,3,5,...) have an inherent
    #    sqrt(s) dependence near the magnetic axis due to the coordinate singularity.
    # 2. Interpolating in sqrt(s) space gives better accuracy near the axis,
    #    where the flux surfaces are closely spaced in physical space.
    # 3. This choice matches the radial grid spacing used internally by VMEC.
    old_sqrt_s_full = np.sqrt(np.linspace(0.0, 1.0, ns_old, endpoint=True))
    new_sqrt_s_full = np.sqrt(np.linspace(0.0, 1.0, ns_new, endpoint=True))

    # Compute scalxc factors (inverse sqrt(s) for odd-m scaling)
    # scalxc[j] = 1/sqrt(s[j]) for j > 0, scalxc[0] = scalxc[1]
    old_scalxc = np.zeros(ns_old)
    old_scalxc[1:] = 1.0 / old_sqrt_s_full[1:]
    old_scalxc[0] = old_scalxc[1]

    new_scalxc = np.zeros(ns_new)
    new_scalxc[1:] = 1.0 / new_sqrt_s_full[1:]
    new_scalxc[0] = new_scalxc[1]

    # Step 1: Extrapolate odd-m modes at axis (j=0) using linear extrapolation
    # from j=1 and j=2: x[0] = 2*x[1] - x[2]
    # This is done on the scaled coefficients (multiplied by scalxc)
    for mn in range(mnmax):
        m = int(xm[mn])
        if m % 2 == 1:  # odd m
            # Scale the coefficients
            rmnc_scaled = rmnc_old[mn, :].copy() * old_scalxc
            zmns_scaled = zmns_old[mn, :].copy() * old_scalxc
            lmns_full_scaled = lmns_full_old[mn, :].copy() * old_scalxc

            # Extrapolate to axis
            rmnc_scaled[0] = 2.0 * rmnc_scaled[1] - rmnc_scaled[2]
            zmns_scaled[0] = 2.0 * zmns_scaled[1] - zmns_scaled[2]
            lmns_full_scaled[0] = 2.0 * lmns_full_scaled[1] - lmns_full_scaled[2]

            # Interpolate to new grid
            rmnc_interp = np.interp(new_sqrt_s_full, old_sqrt_s_full, rmnc_scaled)
            zmns_interp = np.interp(new_sqrt_s_full, old_sqrt_s_full, zmns_scaled)
            lmns_full_interp = np.interp(
                new_sqrt_s_full, old_sqrt_s_full, lmns_full_scaled
            )

            # Un-scale
            rmnc_new[mn, :] = rmnc_interp / new_scalxc
            zmns_new[mn, :] = zmns_interp / new_scalxc
            lmns_full_new[mn, :] = lmns_full_interp / new_scalxc

            # Zero odd-m modes at axis
            rmnc_new[mn, 0] = 0.0
            zmns_new[mn, 0] = 0.0
            lmns_full_new[mn, 0] = 0.0
        else:  # even m
            # Simple interpolation without scaling
            rmnc_new[mn, :] = np.interp(
                new_sqrt_s_full, old_sqrt_s_full, rmnc_old[mn, :]
            )
            zmns_new[mn, :] = np.interp(
                new_sqrt_s_full, old_sqrt_s_full, zmns_old[mn, :]
            )
            lmns_full_new[mn, :] = np.interp(
                new_sqrt_s_full, old_sqrt_s_full, lmns_full_old[mn, :]
            )

    # Create a new wout with interpolated values
    new_wout = wout.model_copy(deep=True)
    new_wout.ns = ns_new
    new_wout.rmnc = rmnc_new
    new_wout.zmns = zmns_new
    new_wout.lmns_full = lmns_full_new

    return new_wout


def run_with_python_multigrid(
    input: VmecInput,
    magnetic_field: MagneticFieldResponseTable | None = None,
    *,
    max_threads: int | None = None,
    verbose: bool | int | OutputMode = True,
) -> VmecOutput:
    """Run VMEC++ with Python-side multigrid iteration.

    This function orchestrates the multigrid iteration from Python, running
    VMEC++ with a single multigrid step at a time and performing the radial
    interpolation between steps in Python.

    This is equivalent to the C++ multigrid loop, but allows for more flexible
    control over the iteration process from Python.

    Args:
        input: A VmecInput instance, corresponding to the contents of a classic VMEC
            input file
        magnetic_field: If present, VMEC++ will pass the magnetic field object in
            memory instead of reading it from an mgrid file (only relevant in
            free-boundary runs).
        max_threads: Maximum number of threads that VMEC++ should spawn. If None,
            a number of threads equal to the number of logical cores is used.
        verbose: Controls the output format. Accepts bool for backward compatibility.

    Returns:
        A VmecOutput object containing the final equilibrium solution.
    """
    import vmecpp  # noqa: PLC0415  # Import here to avoid circular imports

    # Extract multigrid parameters
    ns_array = input.ns_array.copy()
    ftol_array = input.ftol_array.copy()
    niter_array = input.niter_array.copy()

    num_grids = len(ns_array)

    if num_grids == 0:
        msg = "ns_array must have at least one element"
        raise ValueError(msg)

    # Validate arrays have same length
    if len(ftol_array) != num_grids or len(niter_array) != num_grids:
        msg = "ns_array, ftol_array, and niter_array must have the same length"
        raise ValueError(msg)

    logger.info(
        f"Running VMEC++ with Python-side multigrid: {num_grids} steps, "
        f"ns={list(ns_array)}"
    )

    vmec_output: vmecpp.VmecOutput | None = None

    for igrid, (ns, ftol, niter) in enumerate(
        zip(ns_array, ftol_array, niter_array, strict=True)
    ):
        logger.info(
            f"Multigrid step {igrid + 1}/{num_grids}: ns={ns}, ftol={ftol}, "
            f"niter={niter}"
        )

        # Create input for this step with single-element arrays
        step_input = input.model_copy(deep=True)
        step_input.ns_array = np.array([ns], dtype=np.int64)
        step_input.ftol_array = np.array([ftol])
        step_input.niter_array = np.array([niter], dtype=np.int64)

        if vmec_output is not None:
            # Interpolate from previous step to current resolution
            old_ns = vmec_output.wout.ns

            if ns > old_ns:
                # Need to interpolate to finer grid
                logger.debug(f"Interpolating from ns={old_ns} to ns={ns}")
                interpolated_wout = interpolate_to_new_radial_resolution(
                    vmec_output.wout, ns
                )
                # Create a new VmecOutput with the interpolated wout for hot restart
                restart_output = vmec_output.model_copy(deep=True)
                restart_output.wout = interpolated_wout
                # Update the input's ns_array to match the interpolated state
                # The C++ CheckInitialState checks that the last element of
                # initial_state.indata.ns_array matches the new indata.ns_array[0]
                restart_output.input = restart_output.input.model_copy(deep=True)
                restart_output.input.ns_array = np.array([ns], dtype=np.int64)
                restart_output.input.ftol_array = np.array([ftol])
                restart_output.input.niter_array = np.array([niter], dtype=np.int64)
            else:
                # Same resolution, just restart
                restart_output = vmec_output.model_copy(deep=True)
                # Update ns_array to match new input
                restart_output.input = restart_output.input.model_copy(deep=True)
                restart_output.input.ns_array = np.array([ns], dtype=np.int64)
                restart_output.input.ftol_array = np.array([ftol])
                restart_output.input.niter_array = np.array([niter], dtype=np.int64)

            # Run with hot restart
            vmec_output = vmecpp.run(
                step_input,
                magnetic_field=magnetic_field,
                max_threads=max_threads,
                verbose=verbose,
                restart_from=restart_output,
            )
        else:
            # First step - run without restart
            vmec_output = vmecpp.run(
                step_input,
                magnetic_field=magnetic_field,
                max_threads=max_threads,
                verbose=verbose,
            )

        logger.info(
            f"Multigrid step {igrid + 1}/{num_grids} completed: "
            # fsqt can be empty if return_outputs_even_if_not_converged=True
            # and VMEC fails very early in the iteration
            f"fsqt={vmec_output.wout.fsqt[-1] if len(vmec_output.wout.fsqt) > 0 else 'N/A'}"
        )

    # Return the final output with the original input for consistency
    assert vmec_output is not None
    vmec_output.input = input

    return vmec_output
