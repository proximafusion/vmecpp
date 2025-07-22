#!/usr/bin/env python3
"""Debug symmetric equilibrium."""

import vmecpp
import numpy as np

def test_symmetric():
    """Test symmetric equilibrium."""
    print("Testing pure symmetric equilibrium with minimal output...")
    
    # Load Solovev input (purely symmetric)
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")
    
    # Reduce iterations for testing
    vmec_input.niter_array = np.array([5], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-6])
    
    print(f"lasym={vmec_input.lasym}, ntor={vmec_input.ntor}, mpol={vmec_input.mpol}")
    
    try:
        # Disable all debug output
        result = vmecpp.run(vmec_input, verbose=False)
        print(f"SUCCESS!")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    test_symmetric()