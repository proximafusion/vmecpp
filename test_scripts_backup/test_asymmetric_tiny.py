#!/usr/bin/env python3
"""Test asymmetric convergence with tiny perturbation."""

import vmecpp
import numpy as np

def test_asymmetric_tiny():
    """Test asymmetric equilibrium with tiny perturbation."""
    print("Testing asymmetric equilibrium with tiny perturbation...")
    
    # Load Solovev input
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")
    
    # Enable asymmetric mode
    vmec_input.lasym = True
    
    # Set very small asymmetric perturbation
    # RBS[0] corresponds to m=0, n=1 asymmetric mode  
    vmec_input.RBS = np.zeros((vmec_input.mPol, 2 * vmec_input.nTor + 1))
    vmec_input.RBS[0, vmec_input.nTor] = 0.001  # Tiny perturbation
    
    # Reduce iterations for testing
    vmec_input.niter_array = np.array([20], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-5])
    
    print(f"Running with RBS[0,{vmec_input.nTor}] = {vmec_input.RBS[0, vmec_input.nTor]}")
    print(f"lasym={vmec_input.lasym}, ntor={vmec_input.ntor}, mpol={vmec_input.mpol}")
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        print(f"\nSUCCESS! Final force residual: {result.wout.fsqr:.2e}")
        return True
    except Exception as e:
        print(f"\nFAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_asymmetric_tiny()
    print(f"\nAsymmetric test {'PASSED' if success else 'FAILED'}")