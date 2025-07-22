#!/usr/bin/env python3
"""Simple asymmetric test."""

import vmecpp
import numpy as np

# Load Solovev input
vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")

# Enable asymmetric mode
vmec_input.lasym = True

# Set tiny asymmetric perturbation
vmec_input.rbs = np.zeros((2 * vmec_input.ntor + 1, vmec_input.mpol))
vmec_input.rbs[vmec_input.ntor, 0] = 0.001  # Tiny perturbation

# Reduce iterations
vmec_input.niter_array = np.array([5], dtype=np.int64)
vmec_input.ftol_array = np.array([1e-4])

print(f"Running asymmetric with rbs[{vmec_input.ntor},0] = {vmec_input.rbs[vmec_input.ntor, 0]}")

result = vmecpp.run(vmec_input, verbose=False)
print("SUCCESS!")