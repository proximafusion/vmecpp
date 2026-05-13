# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Python-side multigrid iteration for VMEC++.

This module provides a Python implementation of the multigrid iteration loop, which
allows for more flexible control over the iteration process.

The C++ implementation remains the default, but this Python implementation can be used
for experimentation and debugging.
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

    The interpolation algorithm exactly matches the C++ implementation:
    1. Extrapolation of odd-m modes at the magnetic axis (linear extrapolation
       of scaled coefficients: x[0] = 2*x[1] - x[2])
    2. Linear interpolation in s (normalized toroidal flux) space with odd-m
       mode scaling by 1/sqrt(s) (the scalxc factor from Hirshman et al.)
    3. Zeroing of odd-m modes at the origin

    The s-space linear interpolation formula from the C++ code is:
        js1 = floor(jNew * (ns_old - 1) / (ns_new - 1))
        js2 = min(js1 + 1, ns_old - 1)
        xint = (s[jNew] - s[js1]) / hs_old, clamped to [0, 1]
        result[jNew] = (1 - xint) * scaled[js1] + xint * scaled[js2]

    Note on lambda (lmns_full): In the C++ multigrid, the internal state
    lmnsc is interpolated directly without any lamscale renormalization.
    When the wout encoding is used (which encodes lmnsc * lamscale_old / phipF),
    passing the interpolated values through the hot-restart mechanism (which
    decodes with lamscale_new) introduces an apparent lamscale_old/lamscale_new
    factor. However, this factor is absorbed into the iteration dynamics and
    does NOT need to be corrected: empirically the initial force residuals
    match the C++ multigrid to machine precision without any lamscale correction.

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

    # Compute the C++ interpolation indices and weights for each new grid point.
    # This exactly replicates InterpolateToNextMultigridStep in vmec.cc:
    #   js1[jNew] = (jNew * (ns_old - 1)) / (ns_new - 1)  [integer division]
    #   js2[jNew] = min(js1 + 1, ns_old - 1)
    #   s1[jNew] = js1 * hs_old
    #   xint[jNew] = (s[jNew] - s1[jNew]) / hs_old, clamped to [0, 1]
    # The C++ uses linear interpolation in s = j/(ns-1) space, not sqrt(s)-space.
    hs_old = 1.0 / (ns_old - 1.0)
    jnew_arr = np.arange(ns_new)
    js1_arr = (jnew_arr * (ns_old - 1)) // (ns_new - 1)
    js2_arr = np.minimum(js1_arr + 1, ns_old - 1)
    s_jnew = jnew_arr / (ns_new - 1.0)
    s1_arr = js1_arr * hs_old
    xint_arr = (s_jnew - s1_arr) / hs_old
    xint_arr = np.clip(xint_arr, 0.0, 1.0)

    # Compute scalxc factors (inverse sqrt(s) for odd-m scaling).
    # scalxc[j] = 1/sqrt(s[j]) for j > 0, and scalxc[0] = scalxc[1]
    # (constant extrapolation of the scalxc factor to the axis).
    # This matches the C++ RadialProfiles::evalRadialProfiles() definition:
    #   scalxc[0] = 1 / max(sqrtSF[0], sqrtS1) = 1/sqrtS1
    # where sqrtS1 = sqrt(s[1]) = sqrt(1/(ns-1)).
    old_s_full = np.linspace(0.0, 1.0, ns_old)
    new_s_full = np.linspace(0.0, 1.0, ns_new)

    old_scalxc = np.zeros(ns_old)
    old_scalxc[1:] = 1.0 / np.sqrt(old_s_full[1:])
    old_scalxc[0] = old_scalxc[1]

    new_scalxc = np.zeros(ns_new)
    new_scalxc[1:] = 1.0 / np.sqrt(new_s_full[1:])
    new_scalxc[0] = new_scalxc[1]

    # Step 1: Extrapolate odd-m modes at axis (j=0) using linear extrapolation
    # from j=1 and j=2: x_scaled[0] = 2*x_scaled[1] - x_scaled[2]
    # This is done on the scaled coefficients (multiplied by scalxc = 1/sqrt(s)).
    for mn in range(mnmax):
        m = int(xm[mn])
        if m % 2 == 1:  # odd m
            # Scale the coefficients by scalxc = 1/sqrt(s)
            rmnc_scaled = rmnc_old[mn, :].copy() * old_scalxc
            zmns_scaled = zmns_old[mn, :].copy() * old_scalxc
            lmns_full_scaled = lmns_full_old[mn, :].copy() * old_scalxc

            # Extrapolate scaled coefficients to axis using linear extrapolation
            rmnc_scaled[0] = 2.0 * rmnc_scaled[1] - rmnc_scaled[2]
            zmns_scaled[0] = 2.0 * zmns_scaled[1] - zmns_scaled[2]
            lmns_full_scaled[0] = 2.0 * lmns_full_scaled[1] - lmns_full_scaled[2]

            # Interpolate to new grid using s-space linear interpolation
            rmnc_interp = (1.0 - xint_arr) * rmnc_scaled[
                js1_arr
            ] + xint_arr * rmnc_scaled[js2_arr]
            zmns_interp = (1.0 - xint_arr) * zmns_scaled[
                js1_arr
            ] + xint_arr * zmns_scaled[js2_arr]
            lmns_full_interp = (1.0 - xint_arr) * lmns_full_scaled[
                js1_arr
            ] + xint_arr * lmns_full_scaled[js2_arr]

            # Un-scale by new_scalxc = 1/sqrt(s_new)
            rmnc_new[mn, :] = rmnc_interp / new_scalxc
            zmns_new[mn, :] = zmns_interp / new_scalxc
            lmns_full_new[mn, :] = lmns_full_interp / new_scalxc

            # Zero odd-m modes at axis
            rmnc_new[mn, 0] = 0.0
            zmns_new[mn, 0] = 0.0
            lmns_full_new[mn, 0] = 0.0
        else:  # even m
            # Even-m modes: no scalxc factor (scalxc = 1.0 for even-m),
            # so just interpolate in s-space directly
            rmnc_new[mn, :] = (1.0 - xint_arr) * rmnc_old[
                mn, js1_arr
            ] + xint_arr * rmnc_old[mn, js2_arr]
            zmns_new[mn, :] = (1.0 - xint_arr) * zmns_old[
                mn, js1_arr
            ] + xint_arr * zmns_old[mn, js2_arr]
            lmns_full_new[mn, :] = (1.0 - xint_arr) * lmns_full_old[
                mn, js1_arr
            ] + xint_arr * lmns_full_old[mn, js2_arr]

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

    restart_output: vmecpp.VmecOutput | None = None

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

        if restart_output is not None:
            # Interpolate from previous step to current resolution
            old_ns = restart_output.wout.ns

            if ns > old_ns:
                # Need to interpolate to finer grid
                logger.debug(f"Interpolating from ns={old_ns} to ns={ns}")
                interpolated_wout = interpolate_to_new_radial_resolution(
                    restart_output.wout, ns
                )
                # Create a new VmecOutput with the interpolated wout for hot restart
                restart_output.wout = interpolated_wout
                # Update the input's ns_array to match the interpolated state
                # The C++ CheckInitialState checks that the last element of
                # initial_state.indata.ns_array matches the new indata.ns_array[0]
                restart_output.input = restart_output.input.model_copy(deep=True)
                restart_output.input.ns_array = np.array([ns], dtype=np.int64)
                restart_output.input.ftol_array = np.array([ftol])
                restart_output.input.niter_array = np.array([niter], dtype=np.int64)
            else:
                # Update ns_array to match new input
                restart_output.input = restart_output.input.model_copy(deep=True)
                restart_output.input.ns_array = np.array([ns], dtype=np.int64)
                restart_output.input.ftol_array = np.array([ftol])
                restart_output.input.niter_array = np.array([niter], dtype=np.int64)

            # Run with hot restart
            restart_output = vmecpp.run(
                step_input,
                magnetic_field=magnetic_field,
                max_threads=max_threads,
                verbose=verbose,
                restart_from=restart_output,
            )
        else:
            # First step - run without restart
            restart_output = vmecpp.run(
                step_input,
                magnetic_field=magnetic_field,
                max_threads=max_threads,
                verbose=verbose,
            )

        logger.info(
            f"Multigrid step {igrid + 1}/{num_grids} completed: "
            # fsqt can be empty if return_outputs_even_if_not_converged=True
            # and VMEC fails very early in the iteration
            f"fsqt={restart_output.wout.fsqt[-1] if len(restart_output.wout.fsqt) > 0 else 'N/A'}"
        )

    # Return the final output with the original input for consistency
    assert restart_output is not None
    restart_output.input = input

    return restart_output
