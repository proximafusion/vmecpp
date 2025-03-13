# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""This script assesses the sensitivity of a VMEC equilibrium to the initial state to
restart that computation from.

It therefore performs the following steps:
1. Run a fixed-boundary VMEC++ case to convergence.
2. Randomly perturb all coefficients of boundary slightly (1.0e-8) for 20 cases.
3. Re-run VMEC++ for all of these perturbed cases,
   restarting from the original run -> this is quite fast.
4. Now re-run the single original input 20 times,
   each time restarting from the outputs of the perturbed cases
   -> this is also quite fast
5. Compare the 20 (slightly different) equilibria
   obtained from each of the runs of the original input,
   each one restarted from a slighly different previous state
  (i.e. the outputs of the perturbed equilibria).
As a starting point, we then look at the iota profiles:
* The overall shape of the profile is well preserved.
* Slight variations on the order of 1e-7 are present.
* Recall that each of these outputs is a valid VMEC++ run
  for the identical input file;
  just the state that this case was restarted from was varied.
"""

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

# 1. make initial run to restart from
output_quantities = vmecpp.run(vmec_input)

rng = np.random.default_rng(seed=42)


# 2. randomly perturb the initial input
def random_pertubation(indata, perturbation_amplitude):
    """Return a copy of the input indata with a random perturbation applied to the
    boundary."""
    perturbed_indata = indata.model_copy()
    mpol = indata.mpol
    ntor = indata.ntor
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            # coerce to 5 sigma -> Gaussian noise in principle
            # allows for arbitrarily large samples to be drawn,
            # but we don't want crazy large pertubation
            rnd = np.max([-5, np.min([5, rng.random()])])
            perturbation = perturbation_amplitude * rnd

            perturbed_indata.rbc[m, ntor + n] *= 1.0 + perturbation
            perturbed_indata.zbs[m, ntor + n] *= 1.0 + perturbation
    return perturbed_indata


# This is the number of perturbed equilibria that we sample around
# the original equilibrium. We then compute the initial equilibrium again
# for this number of times, but each restarted from a different one
# of the random samples around the original.
# This should allow us to assess the robustness of the original equilibrium
# against the initial guess for the iterative VMEC++ computation.
number_of_wiggles = 20

# standard deviation of the random relative pertubation
# to be applied to all Fourier coefficients of the plasma boundary
perturbation_amplitude = 1.0e-8

# perturb the initial input and run VMEC++ for each of these
perturbed_outputs = []
for i in range(number_of_wiggles):
    print("perturbation", i)

    # last part of 2.: actally perform the perturbations
    perturbed_indata = random_pertubation(
        vmec_input, perturbation_amplitude=perturbation_amplitude
    )

    # 3. run the perturbed cases
    out = vmecpp.run(perturbed_indata, restart_from=output_quantities)
    perturbed_outputs.append(out)

# 4. now re-run the initial input, but restarting from the perturbed outputs
wiggled_original = []
for i in range(number_of_wiggles):
    out = vmecpp.run(vmec_input, restart_from=perturbed_outputs[i])
    wiggled_original.append(out)

# 5. plot the variation in the iota profile
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(output_quantities.wout.iotaf, "kx-")
for i in range(number_of_wiggles):
    wout = wiggled_original[i].wout
    plt.plot(wout.iotaf, ".-")
plt.grid(True)
plt.title("iotaf")
plt.subplot(2, 1, 2)
for i in range(number_of_wiggles):
    wout = wiggled_original[i].wout
    raErr = (wout.iotaf - output_quantities.wout.iotaf) / (
        1.0 + np.abs(output_quantities.wout.iotaf)
    )
    plt.plot(raErr, ".-")
plt.xlabel("flux surface index")
plt.ylabel("deviations from initial iotaf")
plt.grid(True)
plt.tight_layout()

plt.show()
