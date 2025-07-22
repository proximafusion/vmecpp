#!/usr/bin/env python3
"""
Minimal test script to examine asymmetric Fourier coefficient setup
and early convergence behavior.
"""

import numpy as np
import h5py
import subprocess
import json
import os

def create_minimal_asymmetric_input():
    """Create a minimal asymmetric input file with small perturbations."""
    input_data = {
        "indata": {
            # Basic grid parameters
            "mpol": 5,
            "ntor": 2,
            "ns_array": [5],  # Just 5 radial surfaces for quick testing
            "niter": 10,      # Just 10 iterations to see initial behavior
            
            # Stellarator symmetry flags - explicitly set to enable asymmetry
            "lasym": 1,  # Use integer 1 instead of True
            "lfreeb": 0,  # Use integer 0 instead of False
            
            # Tokamak geometry
            "nfp": 1,
            "phiedge": 6.28318530718,  # 2*pi
            
            # Magnetic axis
            "raxis_c": [1.0],
            "raxis_s": [0.0],
            "zaxis_c": [0.0],
            "zaxis_s": [0.0],
            
            # Boundary shape - circular tokamak with small asymmetry
            "rbc": [
                {"n": 0, "m": 0, "value": 1.0},    # Major radius
                {"n": 0, "m": 1, "value": 0.3}     # Minor radius
            ],
            "rbs": [
                {"n": 0, "m": 1, "value": 0.01}    # Small asymmetric perturbation
            ],
            "zbs": [
                {"n": 0, "m": 1, "value": 0.3}     # Minor radius
            ],
            "zcc": [
                {"n": 0, "m": 1, "value": 0.01}    # Small asymmetric perturbation  
            ],
            
            # Pressure profile
            "am": [1.0],
            "pres_scale": 1000.0,
            
            # Current profile
            "ncurr": 1,
            "curtor": 1e5,
            
            # Numerical parameters
            "delt": 1.0,
            "ftol_array": [1e-12],
            "tcon0": 1.0
        }
    }
    
    with open('test_asymmetric_minimal.json', 'w') as f:
        json.dump(input_data, f, indent=2)
    
    return input_data

def extract_fourier_coefficients(output_file):
    """Extract and print key Fourier coefficients from output."""
    with h5py.File(output_file, 'r') as f:
        # Get dimensions
        ns = f['ns'][()]
        mpol = f['mpol'][()]
        ntor = f['ntor'][()]
        
        # Get mode numbers
        xm = f['xm'][:]
        xn = f['xn'][:]
        
        # Get Fourier coefficients
        rmncc = f['rmncc'][:]  # Cosine-cosine (symmetric)
        zmnsc = f['zmnsc'][:]  # Sine-cosine (symmetric)
        rmnsc = f['rmnsc'][:]  # Sine-cosine (asymmetric)
        zmncc = f['zmncc'][:]  # Cosine-cosine (asymmetric)
        
        print(f"\nOutput dimensions: ns={ns}, mpol={mpol}, ntor={ntor}")
        print(f"Number of modes: {len(xm)}")
        
        # Print coefficients for first few surfaces
        print("\n=== Fourier Coefficients for First 3 Radial Surfaces ===")
        
        for js in range(min(3, ns)):
            print(f"\n--- Surface {js+1} (s={(js/(ns-1)):.3f}) ---")
            
            # Find significant modes
            for i, (m, n) in enumerate(zip(xm, xn)):
                # Check if any coefficient is non-zero for this mode
                has_content = (
                    abs(rmncc[i, js]) > 1e-10 or
                    abs(zmnsc[i, js]) > 1e-10 or
                    abs(rmnsc[i, js]) > 1e-10 or
                    abs(zmncc[i, js]) > 1e-10
                )
                
                if has_content:
                    print(f"  Mode (m={int(m):2d}, n={int(n):2d}):")
                    print(f"    rmncc = {rmncc[i, js]:12.6e}  (symmetric R cos)")
                    print(f"    zmnsc = {zmnsc[i, js]:12.6e}  (symmetric Z sin)")
                    print(f"    rmnsc = {rmnsc[i, js]:12.6e}  (asymmetric R sin)")
                    print(f"    zmncc = {zmncc[i, js]:12.6e}  (asymmetric Z cos)")
        
        # Check force balance
        if 'fsqr' in f:
            fsqr = f['fsqr'][:]
            print(f"\n=== Force Balance (fsqr) ===")
            print(f"Initial: {fsqr[0]:.6e}")
            print(f"Final:   {fsqr[-1]:.6e}")
            print(f"Iterations: {len(fsqr)}")
            
            # Print first few iterations
            print("\nFirst 5 iterations:")
            for i in range(min(5, len(fsqr))):
                print(f"  Iter {i+1}: {fsqr[i]:.6e}")

def run_test():
    """Run the minimal asymmetric test."""
    print("Creating minimal asymmetric input...")
    input_data = create_minimal_asymmetric_input()
    
    # Print input perturbations
    print("\nInput asymmetric perturbations:")
    print(f"  rbs(0,1) = {input_data['indata']['rbs'][0]['value']}")
    print(f"  zcc(0,1) = {input_data['indata']['zcc'][0]['value']}")
    
    # Run VMEC++
    print("\nRunning VMEC++...")
    cmd = ["./build/vmec_standalone", "test_asymmetric_minimal.json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"VMEC++ failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return
    
    print("VMEC++ completed successfully")
    
    # Check if output file exists
    output_file = "test_asymmetric_minimal.out.h5"
    if not os.path.exists(output_file):
        print(f"Output file {output_file} not found!")
        return
    
    # Extract and analyze coefficients
    extract_fourier_coefficients(output_file)
    
    # Also check for NaN values
    print("\n=== Checking for NaN values ===")
    with h5py.File(output_file, 'r') as f:
        for key in ['rmncc', 'zmnsc', 'rmnsc', 'zmncc']:
            if key in f:
                data = f[key][:]
                nan_count = np.sum(np.isnan(data))
                if nan_count > 0:
                    print(f"WARNING: {key} contains {nan_count} NaN values!")
                else:
                    print(f"{key}: No NaN values")

if __name__ == "__main__":
    run_test()