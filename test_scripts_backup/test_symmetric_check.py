#!/usr/bin/env python3
"""Check if symmetric equilibria still work."""

import vmecpp
import numpy as np

def test_symmetric():
    """Test symmetric equilibrium to ensure we didn't break it."""
    print("Testing symmetric equilibrium...")
    
    # Load asymmetric input but force it to be symmetric
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Force symmetric - create fresh input without asymmetric fields
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")
    
    # Reduce iterations for testing
    vmec_input.niter_array = np.array([200], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-8])
    
    print(f"Running symmetric equilibrium: lasym={vmec_input.lasym}")
    print(f"Input file: Solovev equilibrium (ntor={vmec_input.ntor}, mpol={vmec_input.mpol})")
    
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
    success = test_symmetric()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")