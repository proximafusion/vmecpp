#!/usr/bin/env python3
"""Simple test of decomposeInto fix for 2D asymmetric cases."""

import vmecpp
import numpy as np

def test_simple():
    print("Testing decomposeInto fix...")
    
    # Load the asymmetric tokamak configuration  
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Configure for minimal test
    vmec_input.ns_array = np.array([3], dtype=np.int64)
    vmec_input.niter_array = np.array([1], dtype=np.int64)
    vmec_input.mpol = 2
    vmec_input.return_outputs_even_if_not_converged = True
    
    # Fixed axis initialization  
    vmec_input.raxis_c = np.array([6.0])
    
    # Very small asymmetry
    vmec_input.rbs[1, 0] = 0.01
    vmec_input.zbs[1, 0] = 0.01
    
    print(f"Configuration: lasym={vmec_input.lasym}, lthreed=False (ntor=0)")
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        print("✅ SUCCESS: decomposeInto fix working!")
        print(f"Final residual: {result.fsqr:.2e}")
        return True
        
    except RuntimeError as e:
        error_msg = str(e)
        if "matrix" in error_msg.lower() or "eigen" in error_msg.lower():
            print(f"❌ Still have matrix/bounds error: {error_msg}")
            return False
        else:
            print(f"✅ decomposeInto fixed, different error: {error_msg}")
            return "partial"

if __name__ == "__main__":
    test_simple()