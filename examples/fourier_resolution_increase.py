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
    vmec_output_to_restart_from = vmec_output.model_copy(deep=True)

    # Perform radial interpolation and zero-padding in order to produce the initial guess
    # for the next resolution step based on the previous step.
    mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1)
    vmec_output_to_restart_from.wout.rmnc = np.zeros([mnmax, ns])
    vmec_output_to_restart_from.wout.zmns = np.zeros([mnmax, ns])
    vmec_output_to_restart_from.wout.lmns_full = np.zeros([mnmax, ns])

    if ns == old_ns:
        # no radial interpolation if radial resolution does not change
        for j in range(ns):
            mn = 0
            m = 0
            for n in range(ntor + 1):
                for old_mn in range(old_mnmax):
                    if (
                        m == vmec_output.wout.xm[old_mn]
                        and n * vmec_output.wout.nfp == vmec_output.wout.xn[old_mn]
                    ):
                        vmec_output_to_restart_from.wout.rmnc[mn, j] = (
                            vmec_output.wout.rmnc[old_mn, j]
                        )
                        vmec_output_to_restart_from.wout.zmns[mn, j] = (
                            vmec_output.wout.zmns[old_mn, j]
                        )
                        vmec_output_to_restart_from.wout.lmns_full[mn, j] = (
                            vmec_output.wout.lmns_full[old_mn, j]
                        )
                mn += 1
            for m in range(1, mpol):
                for n in range(-ntor, ntor + 1):
                    for old_mn in range(old_mnmax):
                        if (
                            m == vmec_output.wout.xm[old_mn]
                            and n * vmec_output.wout.nfp == vmec_output.wout.xn[old_mn]
                        ):
                            vmec_output_to_restart_from.wout.rmnc[mn, j] = (
                                vmec_output.wout.rmnc[old_mn, j]
                            )
                            vmec_output_to_restart_from.wout.zmns[mn, j] = (
                                vmec_output.wout.zmns[old_mn, j]
                            )
                            vmec_output_to_restart_from.wout.lmns_full[mn, j] = (
                                vmec_output.wout.lmns_full[old_mn, j]
                            )
                    mn += 1
    else:
        # perform radial interpolation as well as zero-padding in Fourier space

        old_sqrt_s_full = np.sqrt(np.linspace(0.0, 1.0, old_ns, endpoint=True))
        new_sqrt_s_full = np.sqrt(np.linspace(0.0, 1.0, ns, endpoint=True))

        old_scalxc = np.zeros(old_ns)
        old_scalxc[1:] = 1.0 / old_sqrt_s_full[1:]
        old_scalxc[0] = old_scalxc[1]

        new_scalxc = np.zeros(ns)
        new_scalxc[1:] = 1.0 / new_sqrt_s_full[1:]
        new_scalxc[0] = new_scalxc[1]

        # FIXME(jons): Something is still fishy here:
        # When using this interpolation method for just increasing radial resolution, without chaning the Fourier resolution, the resulting force residuals decay is not the same as when using the radial multi-grid method natively implemented in VMEC++ directly.
        # TODO(jons): One could also think about using this comparison as a test for this script.

        def get_interpolated_slice_from_previous_run(
            vmec_output: vmecpp.VmecOutput,
            old_sqrt_s_full: np.ndarray,
            new_sqrt_s_full: np.ndarray,
            old_scalxc: np.ndarray,
            new_scalxc: np.ndarray,
            old_mn: int,
            m: int,
        ):
            # extract radial slice at matching source Fourier mode
            rmnc_slice = vmec_output.wout.rmnc[old_mn, :].copy()
            zmns_slice = vmec_output.wout.zmns[old_mn, :].copy()
            lmns_full_slice = vmec_output.wout.lmns_full[old_mn, :].copy()

            if m % 2 == 1:
                # Apply odd-m interpolation weights.
                rmnc_slice *= old_scalxc
                zmns_slice *= old_scalxc
                lmns_full_slice *= old_scalxc

                # Extrapolate odd-m modes in source output from first two flux surfaces.
                rmnc_slice[0] = 2.0 * rmnc_slice[1] - rmnc_slice[2]
                zmns_slice[0] = 2.0 * zmns_slice[1] - zmns_slice[2]
                lmns_full_slice[0] = 2.0 * lmns_full_slice[1] - lmns_full_slice[2]

            # perform radial interpolation to new resolution
            rmnc_interp = np.interp(new_sqrt_s_full, old_sqrt_s_full, rmnc_slice)
            zmns_interp = np.interp(new_sqrt_s_full, old_sqrt_s_full, zmns_slice)
            lmns_full_interp = np.interp(
                new_sqrt_s_full, old_sqrt_s_full, lmns_full_slice
            )

            if m % 2 == 1:
                # un-do odd-m interpolation weights in interpolated data
                rmnc_interp /= new_scalxc
                zmns_interp /= new_scalxc
                lmns_full_interp /= new_scalxc

                # set odd-m modes in target output to zero at the magnetic axis
                rmnc_interp[0] = 0.0
                zmns_interp[0] = 0.0
                lmns_full_interp[0] = 0.0

            return rmnc_interp, zmns_interp, lmns_full_interp

        mn = 0
        m = 0
        for n in range(ntor + 1):
            for old_mn in range(old_mnmax):
                if (
                    m == vmec_output.wout.xm[old_mn]
                    and n * vmec_output.wout.nfp == vmec_output.wout.xn[old_mn]
                ):
                    rmnc_interp, zmns_interp, lmns_full_interp = (
                        get_interpolated_slice_from_previous_run(
                            vmec_output=vmec_output,
                            old_sqrt_s_full=old_sqrt_s_full,
                            new_sqrt_s_full=new_sqrt_s_full,
                            old_scalxc=old_scalxc,
                            new_scalxc=new_scalxc,
                            old_mn=old_mn,
                            m=m,
                        )
                    )
                    vmec_output_to_restart_from.wout.rmnc[mn, :] = rmnc_interp
                    vmec_output_to_restart_from.wout.zmns[mn, :] = zmns_interp
                    vmec_output_to_restart_from.wout.lmns_full[mn, :] = lmns_full_interp
            mn += 1
        for m in range(1, mpol):
            for n in range(-ntor, ntor + 1):
                for old_mn in range(old_mnmax):
                    if (
                        m == vmec_output.wout.xm[old_mn]
                        and n * vmec_output.wout.nfp == vmec_output.wout.xn[old_mn]
                    ):
                        rmnc_interp, zmns_interp, lmns_full_interp = (
                            get_interpolated_slice_from_previous_run(
                                vmec_output=vmec_output,
                                old_sqrt_s_full=old_sqrt_s_full,
                                new_sqrt_s_full=new_sqrt_s_full,
                                old_scalxc=old_scalxc,
                                new_scalxc=new_scalxc,
                                old_mn=old_mn,
                                m=m,
                            )
                        )
                        vmec_output_to_restart_from.wout.rmnc[mn, :] = rmnc_interp
                        vmec_output_to_restart_from.wout.zmns[mn, :] = zmns_interp
                        vmec_output_to_restart_from.wout.lmns_full[mn, :] = (
                            lmns_full_interp
                        )
                mn += 1
        assert mn == mnmax

    # Run VMEC++ with the new resolution, based on the previous (interpolated and zero-padded) result.
    vmec_output = vmecpp.run(vmec_input, restart_from=vmec_output_to_restart_from)
