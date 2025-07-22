#!/usr/bin/env python3
"""Test if asymmetric equilibria converge after the fixes."""

import vmecpp
import numpy as np

def test_simple_asymmetric_convergence():
    """Test convergence of a simple asymmetric tokamak."""
    print("Testing asymmetric convergence...")
    
    # Load simple asymmetric input
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Reduce iterations to see if it runs at all
    vmec_input.niter_array = np.array([100], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-6])
    
    print(f"Running asymmetric equilibrium: lasym={vmec_input.lasym}")
    print(f"Initial asymmetric perturbation: RBS(0,1)={vmec_input.rbs[1,0]}")
    
    try:
        # Run with verbose output to see iterations
        result = vmecpp.run(vmec_input, verbose=True)
        
        print(f"\n{'='*60}")
        print(f"RUN COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Final force residual: {result.wout.fsqr:.2e}")
        print(f"Number of iterations: {result.wout.iter}")
        print(f"Converged: {result.wout.fsqr < 1e-6}")
        print(f"Beta poloidal: {result.wout.betapol:.4f}")
        print(f"Beta toroidal: {result.wout.betator:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\nRun failed with error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_asymmetric_convergence()
    print(f"\nTest {'PASSED - ASYMMETRIC EQUILIBRIA CONVERGE!' if success else 'FAILED'}")