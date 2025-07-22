#!/usr/bin/env python3
"""Test asymmetric equilibria with smaller perturbations."""

import vmecpp
import numpy as np

def test_small_asymmetric():
    """Test convergence with small asymmetric perturbation."""
    print("Testing asymmetric convergence with small perturbation...")
    
    # Load asymmetric input
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Reduce asymmetric perturbation
    vmec_input.rbs[1,0] = 0.05  # Much smaller than 0.6
    vmec_input.rbs[2,0] = 0.01  # Much smaller than 0.12
    
    # Also reduce iterations for testing
    vmec_input.niter_array = np.array([200], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-8])
    
    print(f"Running asymmetric equilibrium: lasym={vmec_input.lasym}")
    print(f"Reduced asymmetric perturbations: RBS(0,1)={vmec_input.rbs[1,0]}, RBS(0,2)={vmec_input.rbs[2,0]}")
    
    try:
        # Run with verbose output
        result = vmecpp.run(vmec_input, verbose=True)
        
        print(f"\n{'='*60}")
        print(f"RUN COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Final force residual: {result.wout.fsqr:.2e}")
        print(f"Number of iterations: {result.wout.iter}")
        print(f"Converged: {result.wout.fsqr < 1e-8}")
        
        return True
        
    except Exception as e:
        print(f"\nRun failed with error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    success = test_small_asymmetric()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")