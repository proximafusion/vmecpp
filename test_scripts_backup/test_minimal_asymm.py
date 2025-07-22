#!/usr/bin/env python3
"""Test minimal asymmetric case."""

import vmecpp
import numpy as np

def test_minimal():
    print("=== MINIMAL ASYMMETRIC TEST ===\n")
    
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Absolute minimum setup
    vmec_input.ns_array = np.array([3], dtype=np.int64)  # Only 3 surfaces
    vmec_input.niter_array = np.array([2], dtype=np.int64)  # Only 2 iterations
    vmec_input.mpol = 2  # Only m=0,1
    vmec_input.return_outputs_even_if_not_converged = True
    
    # Proper axis guess
    vmec_input.raxis_c = np.array([6.0])
    
    # Minimal asymmetry
    vmec_input.rbs[1, 0] = 0.01
    vmec_input.rbs[2, 0] = 0.0  # Remove m=2 mode entirely 
    vmec_input.zbs[1, 0] = 0.01
    
    print(f"Minimal config: ns={vmec_input.ns_array[0]}, mpol={vmec_input.mpol}, niter={vmec_input.niter_array[0]}")
    print(f"raxis_c = {vmec_input.raxis_c[0]}")
    print(f"RBS(1,0) = {vmec_input.rbs[1,0]}")
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        print("\\n✅ MINIMAL ASYMMETRIC CASE WORKED!")
        print(f"Final residual: {result.fsqr}")
        
    except RuntimeError as e:
        print(f"\\nFailed: {e}")
        if "DenseCoeffsBase.h:366" in str(e):
            print("Eigen bounds error - likely in matrix operations")
        elif "BAD_JACOBIAN" in str(e):
            print("Still Jacobian issue")
        else:
            print("Different error")

if __name__ == "__main__":
    test_minimal()