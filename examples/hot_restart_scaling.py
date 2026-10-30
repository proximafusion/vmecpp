# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""This example demonstrates how the size of the perturbation on the plasma boundary
geometry in a fixed-boundary equilibrium computation affects the number of iterations
required for a hot-restarted equilibrium to converge again."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import the VMEC++ Python module.
import vmecpp

# Load the VMEC++ JSON indata file.
# Its keys have 1:1 correspondence with those in a classic Fortran INDATA file.
vmec_input_filename = Path("examples") / "data" / "cth_like_fixed_bdy.json"
vmec_input = vmecpp.VmecInput.from_file(vmec_input_filename)

vmec_input.ftol_array[0] = 1.0e-13

# make initial run to restart from
output_quantities = vmecpp.run(vmec_input)

niter_initial = output_quantities.wout.niter

# pertubation amplitudes
rbc_01_scale = [
    1.0 + 1.0e-15,
    1.0 + 1.0e-14,
    1.0 + 1.0e-13,
    1.0 + 1.0e-12,
    1.0 + 1.0e-11,
    1.0 + 1.0e-10,
    1.0 + 1.0e-9,
    1.0 + 1.0e-8,
    1.0 + 1.0e-7,
    1.0 + 1.0e-6,
    1.0 + 1.0e-5,
    1.0 + 1.0e-4,
    1.0 + 1.0e-3,
    1.0 + 1.0e-2,
    # 1.0 + 1.0e-1 does not converge at all anymore
]
niter = []

for s in rbc_01_scale:
    # copy over un-perturbed inputs to reset previous rbc pertubation
    perturbed_indata = vmec_input.model_copy()

    # apply pertubation; indices are [m, ntor + n] (to allow for negative n)
    perturbed_indata.rbc[1, vmec_input.ntor + 0] *= s

    restarted_wout = vmecpp.run(perturbed_indata, restart_from=output_quantities)

    niter.append(restarted_wout.wout.niter)

plt.figure()

plot_relative = False
if plot_relative:
    # plot new number of iterations relative to original ones
    plt.semilogx(np.subtract(rbc_01_scale, 1.0), np.divide(niter, niter_initial), "o-")
    plt.axhline(y=1, linestyle="--")
else:
    # plot absolute number of new iterations
    plt.semilogx(np.subtract(rbc_01_scale, 1.0), niter, "o-")
    plt.axhline(y=niter_initial, linestyle="--")

plt.xlabel("perturbation of RBC(n=0,m=1)")
plt.ylabel("iterations for re-convergence")
plt.grid(True)

plt.show()
