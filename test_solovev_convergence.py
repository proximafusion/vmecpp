#!/usr/bin/env python3
"""Test Solovev convergence after fix"""

from vmecpp import _vmecpp

print("Testing Solovev equilibrium convergence...")

# Create indata object
indata = _vmecpp.VmecINDATAPyWrapper()

# Read the input file
indata.read_indata("src/vmecpp/cpp/vmecpp/test_data/input.solovev")

print(f"Running VMEC with lasym = {indata.lasym}")

# Run VMEC
result = _vmecpp.run(indata)

# Check result
if result and result.fsqr < 1e-14:
    print(f"SUCCESS: Solovev equilibrium converged!")
    print(f"  fsqr = {result.fsqr}")
    print(f"  iter = {result.iter}")
    print(f"  MHD Energy = {result.wdot}")
else:
    print(f"FAILED: Solovev equilibrium did not converge")
    if result:
        print(f"  fsqr = {result.fsqr}")
        print(f"  iter = {result.iter}")