#!/usr/bin/env python3
"""Test convergence after asymmetric fix"""

from vmecpp import _vmecpp
import os

print("Testing Solovev equilibrium convergence after asymmetric fix...")

# Create indata object
indata = _vmecpp.VmecINDATAPyWrapper()

# Read the input file
input_file = os.path.abspath("src/vmecpp/cpp/vmecpp/test_data/input.solovev")
print(f"Reading input file: {input_file}")
indata.readFromFile(input_file)

# Run VMEC
print(f"Running VMEC with lasym = {indata.lasym}")
result = _vmecpp.run(indata)

# Check result
if result and result.fsqr < 1e-14:
    print(f"SUCCESS: Solovev equilibrium converged! fsqr = {result.fsqr}")
    print(f"         Iterations: {result.iter}")
    print(f"         MHD Energy: {result.wdot}")
else:
    print(f"FAILED: Solovev equilibrium did not converge")
    if result:
        print(f"        fsqr = {result.fsqr}")
        print(f"        iter = {result.iter}")