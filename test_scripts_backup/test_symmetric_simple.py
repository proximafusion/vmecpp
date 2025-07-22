#!/usr/bin/env python3
"""Simple symmetric test."""

import vmecpp
import numpy as np

# Load Solovev input (purely symmetric)
vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")

# Reduce iterations
vmec_input.niter_array = np.array([5], dtype=np.int64)
vmec_input.ftol_array = np.array([1e-6])

print(f"lasym={vmec_input.lasym}, ntor={vmec_input.ntor}, mpol={vmec_input.mpol}")

# Run VMEC
result = vmecpp.run(vmec_input, verbose=False)
print("SUCCESS!")