#!/usr/bin/env python3
"""
Very simple test to check asymmetric Fourier coefficient behavior.
"""

import numpy as np
import h5py
import subprocess
import json
import os

# Create the simplest possible asymmetric input - axisymmetric tokamak with tiny asymmetry
input_data = {
    "lasym": True,  # Enable asymmetric mode
    "nfp": 1,    # Tokamak (1 field period)
    "mpol": 3,   # Low resolution
    "ntor": 0,   # Axisymmetric
    "ntheta": 8,
    "nzeta": 1,
    "ns_array": [3],  # Just 3 radial surfaces
    "ftol_array": [1e-10],
    "niter_array": [5],  # Just 5 iterations
    "delt": 0.5,
    "tcon0": 1.0,
    "phiedge": 6.28318530718,
    "pmass_type": "power_series",
    "am": [0.0],  # No pressure
    "pres_scale": 0.0,
    "gamma": 0.0,
    "ncurr": 0,
    "piota_type": "power_series", 
    "ai": [0.3],  # Simple iota profile
    "lfreeb": False,
    # Magnetic axis
    "raxis_c": [1.0],
    "raxis_s": [0.0],
    "zaxis_c": [0.0],
    "zaxis_s": [0.0],
    # Boundary - circular with tiny asymmetry
    "rbc": [
        {"n": 0, "m": 0, "value": 1.0},   # Major radius
        {"n": 0, "m": 1, "value": 0.2}    # Minor radius
    ],
    "rbs": [
        {"n": 0, "m": 1, "value": 0.001}  # Tiny asymmetric perturbation
    ],
    "zbs": [
        {"n": 0, "m": 1, "value": 0.2}    # Minor radius
    ],
    "zcc": [
        {"n": 0, "m": 1, "value": 0.001}  # Tiny asymmetric perturbation
    ]
}

# Write the input file
with open('test_simple_asymmetric.json', 'w') as f:
    json.dump(input_data, f, indent=2)

print("Created input file with:")
print(f"  lasym = {input_data['lasym']}")
print(f"  rbs(0,1) = {input_data['rbs'][0]['value']}")
print(f"  zcc(0,1) = {input_data['zcc'][0]['value']}")

# Run VMEC++
print("\nRunning VMEC++...")
cmd = ["./build/vmec_standalone", "test_simple_asymmetric.json"]
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"VMEC++ failed with return code {result.returncode}")
    print("STDOUT:", result.stdout[:2000] if result.stdout else "None")
    print("STDERR:", result.stderr[:2000] if result.stderr else "None")
else:
    print("VMEC++ completed successfully")
    
    # Check output
    output_file = "test_simple_asymmetric.out.h5"
    if os.path.exists(output_file):
        with h5py.File(output_file, 'r') as f:
            # Check if asymmetric arrays exist
            print("\nChecking for asymmetric arrays in output:")
            for key in ['rmnsc', 'zmncc']:
                if key in f:
                    arr = f[key][:]
                    non_zero = np.sum(np.abs(arr) > 1e-10)
                    print(f"  {key}: exists, shape={arr.shape}, non-zero elements={non_zero}")
                    if non_zero > 0:
                        print(f"    Max value: {np.max(np.abs(arr))}")
                else:
                    print(f"  {key}: NOT FOUND")
            
            # Check lasym flag
            if 'lasym' in f:
                print(f"\nlasym in output: {f['lasym'][()]}")