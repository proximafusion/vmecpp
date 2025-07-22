#!/usr/bin/env python3
"""Test asymmetric equilibrium initial values and geometry."""

import vmecpp
import numpy as np

def test_initial_values():
    # Set up an asymmetric tokamak with RBS coefficients  
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Debug: run just 1 iteration to see initial state
    vmec_input.niter_array = np.array([1], dtype=np.int64)
    vmec_input.return_outputs_even_if_not_converged = True
    
    print("=== TESTING ASYMMETRIC INITIAL VALUES ===")
    print(f"lasym = {vmec_input.lasym}")
    print(f"ns = {vmec_input.ns_array[0]}")
    print(f"mpol = {vmec_input.mpol}")
    print(f"ntor = {vmec_input.ntor}")
    
    # Show boundary coefficients
    print("\nSymmetric boundary coefficients:")
    for i, val in enumerate(vmec_input.rbc):
        if abs(val) > 1e-12:
            print(f"  RBC({i}) = {val}")
    for i, val in enumerate(vmec_input.zbs):
        if abs(val) > 1e-12:
            print(f"  ZBS({i}) = {val}")
            
    print("\nAsymmetric boundary coefficients:")
    for i, val in enumerate(vmec_input.rbs):
        if abs(val) > 1e-12:
            print(f"  RBS({i}) = {val}")
    for i, val in enumerate(vmec_input.zbc):
        if abs(val) > 1e-12:
            print(f"  ZBC({i}) = {val}")
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        print("\nVMEC++ completed (or returned partial results)")
        
        # Check if we have geometry output
        if hasattr(result, 'rmnc') and result.rmnc is not None:
            print("\nGeometry coefficients after 1 iteration:")
            # Show first few Fourier coefficients
            for j in range(min(3, result.rmnc.shape[0])):  # First 3 surfaces
                print(f"\n  Surface {j}:")
                for m in range(min(5, result.rmnc.shape[1])):  # First 5 modes
                    rmnc_val = result.rmnc[j, m] if j < result.rmnc.shape[0] and m < result.rmnc.shape[1] else 0.0
                    zmns_val = result.zmns[j, m] if j < result.zmns.shape[0] and m < result.zmns.shape[1] else 0.0
                    print(f"    m={m}: rmnc={rmnc_val:.6f}, zmns={zmns_val:.6f}")
                    
                    # Check asymmetric coefficients if available
                    if hasattr(result, 'rmns') and result.rmns is not None:
                        rmns_val = result.rmns[j, m] if j < result.rmns.shape[0] and m < result.rmns.shape[1] else 0.0
                        zmnc_val = result.zmnc[j, m] if j < result.zmnc.shape[0] and m < result.zmnc.shape[1] else 0.0
                        print(f"         rmns={rmns_val:.6f}, zmnc={zmnc_val:.6f}")
        
    except RuntimeError as e:
        print(f"\nFailed with error: {e}")
        print("\nThis indicates the initial guess has problems")

if __name__ == "__main__":
    test_initial_values()