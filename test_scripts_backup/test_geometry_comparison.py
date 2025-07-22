#!/usr/bin/env python3
"""Compare geometry values between symmetric and asymmetric cases."""

import vmecpp
import numpy as np

def test_geometry_values():
    """Compare geometry at each theta point."""
    print("=== Testing Geometry Values ===\n")
    
    # Load asymmetric input
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Set VERY small asymmetric perturbation
    vmec_input.rbs[1,0] = 0.001  # Very small asymmetric perturbation
    vmec_input.rbs[2,0] = 0.0   # Zero out higher modes
    
    # Reduce iterations - we just want to see initial geometry
    vmec_input.niter_array = np.array([5], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-6])
    
    print(f"Running with tiny asymmetric perturbation: RBS(0,1)={vmec_input.rbs[1,0]}")
    print(f"lasym={vmec_input.lasym}, ntor={vmec_input.ntor}, mpol={vmec_input.mpol}")
    
    try:
        # Run with verbose output
        result = vmecpp.run(vmec_input, verbose=True)
        print("\nSURPRISINGLY, IT RAN!")
        print(f"Final force residual: {result.wout.fsqr:.2e}")
        
    except Exception as e:
        print(f"\nFailed as expected: {e}")
        
    # Now try with zero asymmetric perturbation
    print("\n" + "="*60)
    print("Now testing with ZERO asymmetric perturbation...")
    vmec_input.rbs[1,0] = 0.0
    
    try:
        result = vmecpp.run(vmec_input, verbose=True)
        print("\nWith zero perturbation, it runs!")
        print(f"Final force residual: {result.wout.fsqr:.2e}")
        
    except Exception as e:
        print(f"\nEven with zero perturbation, it failed: {e}")

if __name__ == "__main__":
    test_geometry_values()