#!/usr/bin/env python3
"""Test purely symmetric equilibrium."""

import vmecpp
import numpy as np

def test_symmetric():
    """Test symmetric equilibrium."""
    print("Testing pure symmetric equilibrium...")
    
    # Load Solovev input (purely symmetric)
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")
    
    # Reduce iterations for testing
    vmec_input.niter_array = np.array([50], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-8])
    
    print(f"Running symmetric: lasym={vmec_input.lasym}, ntor={vmec_input.ntor}, mpol={vmec_input.mpol}")
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        print(f"\nSUCCESS! Final force residual: {result.wout.fsqr:.2e}")
        return True
    except Exception as e:
        print(f"\nFAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_symmetric()
    print(f"\nSymmetric test {'PASSED' if success else 'FAILED'}")