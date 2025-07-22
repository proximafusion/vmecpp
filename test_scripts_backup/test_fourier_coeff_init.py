#!/usr/bin/env python3
"""Test Fourier coefficient initialization for asymmetric equilibria."""

import vmecpp
import numpy as np

def test_fourier_coeff_init():
    print("=== FOURIER COEFFICIENT INITIALIZATION TEST ===\n")
    
    # Load asymmetric tokamak configuration
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    print("Input boundary coefficients:")
    print(f"  lasym = {vmec_input.lasym}")
    print(f"  ntor = {vmec_input.ntor}")
    print(f"  mpol = {vmec_input.mpol}")
    
    print("\nSymmetric boundary (RBC, ZBS):")
    # For 2D (ntor=0), only n=0 column exists
    for m in range(min(5, len(vmec_input.rbc))):
        if abs(vmec_input.rbc[m, 0]) > 1e-12:
            print(f"  RBC({m},0) = {vmec_input.rbc[m, 0]}")
    for m in range(min(5, len(vmec_input.zbs))):
        if abs(vmec_input.zbs[m, 0]) > 1e-12:
            print(f"  ZBS({m},0) = {vmec_input.zbs[m, 0]}")
    
    print("\nAsymmetric boundary (RBS, ZBC):")
    for m in range(min(5, len(vmec_input.rbs))):
        if abs(vmec_input.rbs[m, 0]) > 1e-12:
            print(f"  RBS({m},0) = {vmec_input.rbs[m, 0]}")
    for m in range(min(5, len(vmec_input.zbc))):
        if abs(vmec_input.zbc[m, 0]) > 1e-12:
            print(f"  ZBC({m},0) = {vmec_input.zbc[m, 0]}")
    
    # Run just 1 iteration to see initial Fourier coefficients
    vmec_input.niter_array = np.array([1], dtype=np.int64)
    vmec_input.return_outputs_even_if_not_converged = True
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        
        if hasattr(result, 'rmnc') and result.rmnc is not None:
            print("\nInitialized Fourier coefficients:")
            
            # Check symmetric coefficients on first and last surface
            print("\nSymmetric coefficients (RMNC, ZMNS):")
            for surf_idx in [0, 1, result.rmnc.shape[0]-1]:
                print(f"\n  Surface {surf_idx}:")
                for m in range(min(5, result.rmnc.shape[1])):
                    rmnc = result.rmnc[surf_idx, m] if m < result.rmnc.shape[1] else 0.0
                    zmns = result.zmns[surf_idx, m] if m < result.zmns.shape[1] else 0.0
                    if abs(rmnc) > 1e-12 or abs(zmns) > 1e-12:
                        print(f"    m={m}: rmnc={rmnc:12.6f}, zmns={zmns:12.6f}")
            
            # Check asymmetric coefficients
            if hasattr(result, 'rmns') and result.rmns is not None:
                print("\nAsymmetric coefficients (RMNS, ZMNC):")
                for surf_idx in [0, 1, result.rmns.shape[0]-1]:
                    print(f"\n  Surface {surf_idx}:")
                    for m in range(min(5, result.rmns.shape[1])):
                        rmns = result.rmns[surf_idx, m] if m < result.rmns.shape[1] else 0.0
                        zmnc = result.zmnc[surf_idx, m] if m < result.zmnc.shape[1] else 0.0
                        if abs(rmns) > 1e-12 or abs(zmnc) > 1e-12:
                            print(f"    m={m}: rmns={rmns:12.6f}, zmnc={zmnc:12.6f}")
            
            # Check if asymmetric coefficients match boundary on last surface
            last_surf = result.rmnc.shape[0] - 1
            print(f"\nBoundary coefficient comparison (surface {last_surf}):")
            
            # For 2D (ntor=0), n=0 only
            # RBS(1,0) should map to RMNS(last_surf, m=1)
            if hasattr(result, 'rmns') and result.rmns is not None:
                rmns_1_0 = result.rmns[last_surf, 1] if 1 < result.rmns.shape[1] else 0.0
                print(f"  RBS(1,0) = 0.6, RMNS[{last_surf},1] = {rmns_1_0:.6f}")
                
                rmns_2_0 = result.rmns[last_surf, 2] if 2 < result.rmns.shape[1] else 0.0
                print(f"  RBS(2,0) = 0.12, RMNS[{last_surf},2] = {rmns_2_0:.6f}")
                
                # ZBS(1,0) should map to ZMNS (symmetric!)
                zmns_1_0 = result.zmns[last_surf, 1] if 1 < result.zmns.shape[1] else 0.0
                print(f"  ZBS(1,0) = 0.6, ZMNS[{last_surf},1] = {zmns_1_0:.6f}")
        
    except RuntimeError as e:
        print(f"\nVMEC++ failed: {e}")

if __name__ == "__main__":
    test_fourier_coeff_init()