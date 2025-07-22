#!/usr/bin/env python3
"""Minimal symmetric test with error handling."""

import vmecpp
import numpy as np
import traceback

def test_symmetric():
    """Test symmetric equilibrium."""
    print("Testing pure symmetric equilibrium...")
    
    try:
        # Load Solovev input (purely symmetric)
        print("Loading input file...")
        vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")
        
        # Reduce iterations for testing
        vmec_input.niter_array = np.array([5], dtype=np.int64)
        vmec_input.ftol_array = np.array([1e-6])
        
        print(f"Input loaded: lasym={vmec_input.lasym}, ntor={vmec_input.ntor}, mpol={vmec_input.mpol}")
        print(f"ns={vmec_input.ns}, nzeta={vmec_input.nzeta}")
        
        print("Starting VMEC run...")
        result = vmecpp.run(vmec_input, verbose=True)
        print(f"\nSUCCESS! Final force residual: {result.wout.fsqr:.2e}")
        return True
    except Exception as e:
        print(f"\nFAILED with exception: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_symmetric()
    print(f"\nSymmetric test {'PASSED' if success else 'FAILED'}")