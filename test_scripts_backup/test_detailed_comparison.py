#!/usr/bin/env python3
"""Detailed comparison test for asymmetric equilibria."""

import vmecpp
import numpy as np

def test_detailed_comparison():
    """Run detailed comparison with jVMEC output."""
    print("=== DETAILED ASYMMETRIC EQUILIBRIA COMPARISON ===\n")
    
    # Load the asymmetric tokamak input
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Run only 1 iteration to see initial values
    vmec_input.niter_array = np.array([1], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-3])
    
    print(f"Configuration:")
    print(f"  lasym = {vmec_input.lasym}")
    print(f"  ntor = {vmec_input.ntor}")
    print(f"  mpol = {vmec_input.mpol}")
    print(f"  ns = {vmec_input.ns_array[0]}")
    print(f"  nfp = {vmec_input.nfp}")
    
    # Check boundary coefficients
    print(f"\nBoundary coefficients:")
    print(f"  RBC(0,0) = {vmec_input.rbc[0,0]}")
    print(f"  RBC(0,1) = {vmec_input.rbc[1,0]}")
    print(f"  ZBS(0,1) = {vmec_input.zbs[1,0]}")
    print(f"  RBS(0,1) = {vmec_input.rbs[1,0]}")  # Asymmetric!
    print(f"  RBS(0,2) = {vmec_input.rbs[2,0]}")  # Asymmetric!
    
    print("\nRunning VMEC++ with verbose output...\n")
    
    try:
        result = vmecpp.run(vmec_input, verbose=True)
        print(f"\nCompleted 1 iteration successfully")
        print(f"Force residual after 1 iteration: {result.wout.fsqr:.2e}")
    except Exception as e:
        print(f"\nFailed after 1 iteration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detailed_comparison()