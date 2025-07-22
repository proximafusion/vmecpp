#!/usr/bin/env python3
"""
Minimal test to debug asymmetric Fourier coefficient issues.
"""

import numpy as np
import h5py
import subprocess
import json
import os

# Use the working symmetric case as a base, then add tiny asymmetry
input_data = {
    "lasym": True,  # Enable asymmetric mode
    "nfp": 1,
    "mpol": 2,  # Very low resolution
    "ntor": 0,
    "ntheta": 4,  # Very coarse
    "nzeta": 1,
    "ns_array": [3],  # Just 3 surfaces
    "ftol_array": [1e-8],  # Less stringent tolerance
    "niter_array": [20],  # More iterations
    "delt": 0.3,  # Smaller time step
    "tcon0": 1.0,
    "phiedge": 6.28318530718,
    "pmass_type": "power_series",
    "am": [0.0],  # No pressure
    "pres_scale": 0.0,
    "gamma": 0.0,
    "ncurr": 0,
    "piota_type": "power_series",
    "ai": [0.5],  # Simple iota
    "lfreeb": False,
    # Axis
    "raxis_c": [1.0],
    "raxis_s": [0.0],
    "zaxis_c": [0.0], 
    "zaxis_s": [0.0],
    # Boundary - start with pure symmetric then add tiny asymmetry
    "rbc": [
        {"n": 0, "m": 0, "value": 1.0},
        {"n": 0, "m": 1, "value": 0.3}
    ],
    "zbs": [
        {"n": 0, "m": 1, "value": 0.3}
    ],
    # Very small asymmetric components
    "rbs": [
        {"n": 0, "m": 1, "value": 0.0001}  # Very tiny
    ],
    "zcc": [
        {"n": 0, "m": 1, "value": 0.0001}  # Very tiny
    ]
}

# Write input
with open('test_minimal_debug.json', 'w') as f:
    json.dump(input_data, f, indent=2)

print("Running minimal debug test...")
print(f"  lasym = {input_data['lasym']}")
print(f"  rbs(0,1) = {input_data['rbs'][0]['value']}")
print(f"  zcc(0,1) = {input_data['zcc'][0]['value']}")

# First try: Run with asymmetry
print("\n1. Running WITH asymmetry...")
result = subprocess.run(["./build/vmec_standalone", "test_minimal_debug.json"], 
                       capture_output=True, text=True)
print(f"Return code: {result.returncode}")
if result.stdout:
    lines = result.stdout.split('\n')
    print("First 30 lines of stdout:")
    for line in lines[:30]:
        print(line)

# Second try: Run without asymmetry for comparison
print("\n2. Running WITHOUT asymmetry (for comparison)...")
input_data['lasym'] = False
input_data.pop('rbs', None)  # Remove asymmetric components
input_data.pop('zcc', None)
with open('test_minimal_debug_sym.json', 'w') as f:
    json.dump(input_data, f, indent=2)
    
result_sym = subprocess.run(["./build/vmec_standalone", "test_minimal_debug_sym.json"],
                           capture_output=True, text=True)
print(f"Return code: {result_sym.returncode}")
if result_sym.returncode == 0:
    print("Symmetric case succeeded!")
else:
    print("Symmetric case also failed")