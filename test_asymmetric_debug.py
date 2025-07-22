#!/usr/bin/env python3
"""Test asymmetric equilibrium with debug output enabled"""

import os
import subprocess

# First, rebuild with debug flags
print("=" * 80)
print("Rebuilding VMEC++ with asymmetric debug flags...")
print("=" * 80)

# Add debug flags to build
build_env = os.environ.copy()
build_env['CXXFLAGS'] = '-DDEBUG_ASYMMETRIC -DDEBUG_JACOBIAN -DDEBUG_CONSTRAINT_FORCE'

# Rebuild
build_cmd = ['pip', 'install', '-e', '.']
subprocess.run(build_cmd, env=build_env, check=True)

print("\n" + "=" * 80)
print("Testing asymmetric equilibrium with debug output...")
print("=" * 80)

# Test the asymmetric case
from vmecpp.cpp import _vmecpp as vmec

test_file = "src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak.json"

try:
    print(f"\nLoading asymmetric input: {test_file}")
    indata = vmec.VmecINDATAPyWrapper.from_file(test_file)
    
    print(f"Configuration: LASYM={indata.lasym}, NFP={indata.nfp}, MPOL={indata.mpol}, NTOR={indata.ntor}")
    print(f"NS array: {list(indata.ns_array)}")
    
    # Check asymmetric arrays
    print("\nAsymmetric boundary coefficients:")
    if hasattr(indata, 'rbs') and len(indata.rbs) > 0:
        print(f"  RBS shape: {indata.rbs.shape}")
        print(f"  First few RBS values: {indata.rbs.flatten()[:5]}")
    if hasattr(indata, 'zbc') and len(indata.zbc) > 0:
        print(f"  ZBC shape: {indata.zbc.shape}")
        print(f"  First few ZBC values: {indata.zbc.flatten()[:5]}")
    
    print("\n" + "-" * 40)
    print("Running VMEC++ (debug output will appear below)...")
    print("-" * 40 + "\n")
    
    # Run with limited iterations to see debug output
    indata.niter_array = [10]  # Limit iterations
    output = vmec.run(indata)
    
    print("\n" + "-" * 40)
    print("Run completed")
    print("-" * 40)
    
except Exception as e:
    print(f"\nERROR: {e}")
    print("\nThis error is expected - we're debugging the issue")
    
print("\n" + "=" * 80)
print("Debug test completed. Check output above for diagnostic information.")
print("=" * 80)