#!/usr/bin/env python3
"""Find exact crash location with incremental testing."""

import vmecpp
import numpy as np

def test_crash_location():
    print("Finding crash location...")
    
    # Load the asymmetric tokamak configuration  
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Configure for minimal test - very small grid
    vmec_input.ns_array = np.array([3], dtype=np.int64) 
    vmec_input.niter_array = np.array([1], dtype=np.int64)
    vmec_input.mpol = 2  # Small mpol to reduce grid size
    vmec_input.return_outputs_even_if_not_converged = True
    
    # Fixed axis initialization
    vmec_input.raxis_c = np.array([6.0])
    
    # Very tiny asymmetry
    vmec_input.rbs[1, 0] = 0.001
    vmec_input.zbs[1, 0] = 0.001
    
    print(f"Minimal test: ns=3, mpol=2, tiny asymmetry")
    
    try:
        result = vmecpp.run(vmec_input, verbose=True)  # Use verbose to see more output
        print("✅ SUCCESS!")
        return True
        
    except Exception as e:
        print(f"❌ CRASH: {e}")
        return False

if __name__ == "__main__":
    test_crash_location()