#!/usr/bin/env python3
"""Test convergence after asymmetric fix"""

import vmecpp

print("Testing Solovev equilibrium convergence after asymmetric fix...")

# Create vmec object
vmec = vmecpp.VMEC()

# Load Solovev test input
vmec.indata.read_indata_file("src/vmecpp/cpp/vmecpp/test_data/input.solovev")

# Run VMEC
result = vmec.run()

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