#!/usr/bin/env python3
"""Test M=1 constraint debug output."""

import os
import sys

import vmecpp

# Redirect stderr to capture debug output
os.environ["VMECPP_VERBOSE"] = "1"

# Load the circular tokamak (symmetric) input
vmec_input = vmecpp.VmecInput.from_file(
    "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.circular_tokamak"
)

print(f"Input LASYM = {vmec_input.lasym}")
print(f"Input RBC(1,0) = {vmec_input.rbc[1,0]}")
print(f"Input ZBS(1,0) = {vmec_input.zbs[1,0]}")

# Run VMEC and let it fail - we just want to see the debug output
try:
    output = vmecpp.run(vmec_input)
except Exception as e:
    print(f"\nVMEC failed as expected: {e}")
    sys.exit(0)
