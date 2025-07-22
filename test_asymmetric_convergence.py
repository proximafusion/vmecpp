#!/usr/bin/env python3
"""Test asymmetric case for convergence"""
from vmecpp.cpp import _vmecpp as vmec

# Test asymmetric case
indata = vmec.VmecINDATAPyWrapper.from_file(
    "src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak_simple.json"
)
assert indata.lasym, "Must be asymmetric"

# Limit iterations to test convergence
indata.nstep = 100  # Limit iterations

print("Testing asymmetric equilibrium convergence...")
try:
    output = vmec.run(indata, verbose=False)
    print(f"✓ Asymmetric case completed!")
    print(f"  Final ier_flag = {output.wout.ier_flag}")
    if output.wout.ier_flag == 0:
        print("  ✓ Converged successfully!")
    else:
        print(f"  ✗ Did not converge (ier_flag = {output.wout.ier_flag})")
    print(f"  Final fsqr = {output.wout.fsqr}")
    print(f"  Final fsqz = {output.wout.fsqz}")
    print(f"  Final fsql = {output.wout.fsql}")
except Exception as e:
    print(f"✗ Error: {e}")
    raise