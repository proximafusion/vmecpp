#!/usr/bin/env python3
"""Check derivative computation for asymmetric case."""

import vmecpp
import numpy as np

def test_check_derivatives():
    """Check derivative values."""
    print("Checking derivative computation...")
    
    # Load Solovev input
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")
    
    # Enable asymmetric mode
    vmec_input.lasym = True
    
    # Set asymmetric perturbation
    vmec_input.RBS = np.zeros((vmec_input.mPol, 2 * vmec_input.nTor + 1))
    vmec_input.RBS[0, vmec_input.nTor] = 0.1  # Small perturbation
    
    # Run only 1 iteration to see derivatives
    vmec_input.niter_array = np.array([1], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-3])
    
    print(f"Running with RBS[0,{vmec_input.nTor}] = {vmec_input.RBS[0, vmec_input.nTor]}")
    
    try:
        result = vmecpp.run(vmec_input, verbose=True)
        print(f"\nCompleted 1 iteration")
        return True
    except Exception as e:
        print(f"\nError after 1 iteration: {e}")
        return False

if __name__ == "__main__":
    test_check_derivatives()