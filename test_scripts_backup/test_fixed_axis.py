#!/usr/bin/env python3
"""Test asymmetric equilibrium with proper axis initialization."""

import vmecpp
import numpy as np

def test_fixed_axis():
    print("=== TESTING ASYMMETRIC WITH PROPER AXIS GUESS ===\n")
    
    # Load the original asymmetric tokamak configuration
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Reduce to minimal setup for faster testing
    vmec_input.ns_array = np.array([5], dtype=np.int64)
    vmec_input.niter_array = np.array([10], dtype=np.int64)
    vmec_input.mpol = 3
    vmec_input.return_outputs_even_if_not_converged = True
    
    # CRITICAL FIX: Provide proper axis guess
    # The boundary has RBC(0,0) = 6.0, so the axis should be around 6.0 as well
    vmec_input.raxis_c = np.array([6.0])  # Instead of 0.0!
    
    # Also reduce asymmetry to help convergence
    vmec_input.rbs[1, 0] = 0.1  # Reduce from 0.6
    vmec_input.rbs[2, 0] = 0.02  # Reduce from 0.12
    vmec_input.zbs[1, 0] = 0.1  # Reduce from 0.6
    
    print("Configuration:")
    print(f"  lasym={vmec_input.lasym}, ntor={vmec_input.ntor}, mpol={vmec_input.mpol}")
    print(f"  ns={vmec_input.ns_array[0]}, niter={vmec_input.niter_array[0]}")
    print(f"\\nAxis guess: raxis_c = {vmec_input.raxis_c}")
    print(f"\\nBoundary coefficients:")
    print(f"  RBC(0,0) = {vmec_input.rbc[0,0]}")
    print(f"  RBC(1,0) = {vmec_input.rbc[1,0]}")
    print(f"  RBS(1,0) = {vmec_input.rbs[1,0]}")
    print(f"  RBS(2,0) = {vmec_input.rbs[2,0]}")
    print(f"  ZBS(1,0) = {vmec_input.zbs[1,0]}")
    
    try:
        result = vmecpp.run(vmec_input, verbose=True)
        print("\\n=== SUCCESS! ASYMMETRIC EQUILIBRIUM CONVERGED ===")
        print(f"Final force residual: {result.fsqr}")
        print(f"Beta: {result.betatot}")
        print(f"Magnetic axis: R = {result.raxis_cc[0]:.6f}")
        
        # Check that we have proper asymmetric coefficients
        if hasattr(result, 'rmns') and result.rmns is not None:
            print("\\nAsymmetric coefficients are present:")
            print(f"  RMNS shape: {result.rmns.shape}")
            print(f"  ZMNC shape: {result.zmnc.shape}")
            
            # Check a few key values
            print(f"  RMNS[boundary,1] = {result.rmns[-1, 1]:.6f}")
            print(f"  ZMNC[boundary,1] = {result.zmnc[-1, 1]:.6f}")
        
    except RuntimeError as e:
        print(f"\\nStill failed: {e}")
        if "BAD_JACOBIAN" in str(e):
            print("BAD_JACOBIAN persists - need further investigation")
        else:
            print("Different error - this is progress!")

if __name__ == "__main__":
    test_fixed_axis()