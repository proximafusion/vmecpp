# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""This example demonstrates how to successively increase the Fourier resolution, along
with the number of flux surfaces, in a VMEC++ equilibrium computation.

This is expected to increase the robustness of the equilibrium computation.
"""

from pathlib import Path

import numpy as np

# Import the VMEC++ Python module.
import vmecpp

# Load the VMEC++ JSON indata file.
# Its keys have 1:1 correspondence with those in a classic Fortran INDATA file.
vmec_input_filename = Path("examples") / "data" / "cth_like_fixed_bdy.json"
vmec_input = vmecpp.VmecInput.from_file(vmec_input_filename)

# Note: The `cth_like_fixed_bdy` example case has ns=15, mpol=5, ntor=4.

# Make the initial VMEC++ run for the original input.
# This is expected to have the lowest resolution.
vmec_output = vmecpp.run(vmec_input)

# Define additional steps with increased resolution.
ns_array = [31]
# mpol_array = [7, 9] ; ntor_array = [7, 9]
mpol_array = [5]
ntor_array = [4]

# Go through resolution steps and run VMEC++ for each step.
for i_step, (ns, mpol, ntor) in enumerate(
    zip(ns_array, mpol_array, ntor_array, strict=False)
):
    print(f"Running step {i_step + 1} with ns={ns}, mpol={mpol}, ntor={ntor}")

    old_ns = vmec_output.wout.ns
    old_mpol = vmec_output.wout.mpol
    old_ntor = vmec_output.wout.ntor

    old_mnmax = (old_ntor + 1) + (old_mpol - 1) * (2 * old_ntor + 1)

    print(f"interpolate ns = {old_ns:3d}  mpol = {old_mpol:2d}  ntor = {old_ntor:2d}")
    print(f"         to ns = {ns:3d}  mpol = {mpol:2d}  ntor = {ntor:2d}")

    # make sure that resolution only increases - decreasing resolution not implemented yet
    assert old_ns <= ns
    assert old_mpol <= mpol
    assert old_ntor <= ntor

    # Set resolution parameters for current step.
    vmec_input.ns_array = np.array([ns])
    vmec_input.mpol = mpol
    vmec_input.ntor = ntor

    # set axis of new input to axis from previous output
    vmec_input.raxis_c = np.zeros(ntor + 1)
    vmec_input.zaxis_s = np.zeros(ntor + 1)
    vmec_input.raxis_c[: old_ntor + 1] = vmec_output.wout.raxis_cc
    vmec_input.zaxis_s[: old_ntor + 1] = vmec_output.wout.zaxis_cs

    # Resize boundary coefficients to new resolution.
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
        vmec_input.rbc, mpol_new=mpol, ntor_new=ntor
    )
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
        vmec_input.zbs, mpol_new=mpol, ntor_new=ntor
    )

    # Overwrite input boundary with the boundary from the previous output.
    # This should be particularly useful for free-boundary runs.
    # In a fixed-boundary run, this should effectively be a no-op.
    old_mn = 0
    m = 0
    for n in range(old_ntor + 1):
        vmec_input.rbc[m, ntor + n] = vmec_output.wout.rmnc[old_mn, old_ns - 1]
        vmec_input.zbs[m, ntor + n] = vmec_output.wout.zmns[old_mn, old_ns - 1]
        old_mn += 1
    for m in range(1, old_mpol):
        for n in range(-old_ntor, old_ntor + 1):
            vmec_input.rbc[m, ntor + n] = vmec_output.wout.rmnc[old_mn, old_ns - 1]
            vmec_input.zbs[m, ntor + n] = vmec_output.wout.zmns[old_mn, old_ns - 1]
            old_mn += 1
    assert old_mn == old_mnmax

    # Make a deep copy of the previous output object for being able to change its contents
    # without loosing access to the previous resolution to restart from.
    vmec_output_to_restart_from = vmecpp.interpolate_to(
        vmec_output=vmec_output, new_ns=ns, new_mpol=mpol, new_ntor=ntor
    )

    # Run VMEC++ with the new resolution, based on the previous (interpolated and zero-padded) result.
    vmec_output = vmecpp.run(vmec_input, restart_from=vmec_output_to_restart_from)
