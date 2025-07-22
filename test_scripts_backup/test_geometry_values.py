#!/usr/bin/env python3
"""Test geometry values for asymmetric case."""

import vmecpp
import numpy as np

def test_geometry_values():
    """Check geometry values."""
    print("Checking geometry values for asymmetric case...")
    
    # Load Solovev input
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.solovev")
    
    # First test symmetric case
    print("\n=== SYMMETRIC CASE ===")
    vmec_input.lasym = False
    vmec_input.niter_array = np.array([2], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-3])
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        print("Symmetric case completed 2 iterations")
    except Exception as e:
        print(f"Symmetric case error: {e}")
    
    # Now test asymmetric case
    print("\n=== ASYMMETRIC CASE ===")
    vmec_input.lasym = True
    vmec_input.RBS = np.zeros((vmec_input.mPol, 2 * vmec_input.nTor + 1))
    vmec_input.RBS[0, vmec_input.nTor] = 0.05  # Small perturbation
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        print("Asymmetric case completed 2 iterations")
    except Exception as e:
        print(f"Asymmetric case error: {e}")

if __name__ == "__main__":
    test_geometry_values()