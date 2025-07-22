#!/usr/bin/env python3
"""
Ultra-simple test case - should definitely converge.
"""

import subprocess
import json

# Create the absolute simplest case - large aspect ratio circular tokamak
input_data = {
    "lasym": False,  # Start with symmetric
    "nfp": 1,
    "mpol": 2,
    "ntor": 0,
    "ntheta": 8,
    "nzeta": 1,
    "ns_array": [3],
    "ftol_array": [1e-6],
    "niter_array": [100],
    "delt": 0.9,
    "tcon0": 1.0,
    "phiedge": 6.28318530718,
    "pmass_type": "power_series",
    "am": [0.0],  # Zero pressure
    "pres_scale": 0.0,
    "gamma": 0.0,
    "ncurr": 0,
    "piota_type": "power_series",
    "ai": [1.0],  # Constant iota
    "lfreeb": False,
    # Large aspect ratio
    "raxis_c": [10.0],  # R0 = 10
    "raxis_s": [0.0],
    "zaxis_c": [0.0],
    "zaxis_s": [0.0],
    # Small minor radius
    "rbc": [
        {"n": 0, "m": 0, "value": 10.0},
        {"n": 0, "m": 1, "value": 1.0}  # a = 1, so aspect ratio = 10
    ],
    "zbs": [
        {"n": 0, "m": 1, "value": 1.0}
    ]
}

print("1. Testing ultra-simple SYMMETRIC case...")
with open('test_ultra_simple.json', 'w') as f:
    json.dump(input_data, f, indent=2)

result = subprocess.run(["./build/vmec_standalone", "test_ultra_simple.json"],
                       capture_output=True, text=True)
print(f"Return code: {result.returncode}")

if result.returncode == 0:
    print("SUCCESS! Now adding asymmetry...")
    
    # Add tiny asymmetry
    input_data['lasym'] = True
    input_data['rbs'] = [{"n": 0, "m": 1, "value": 0.001}]
    input_data['zcc'] = [{"n": 0, "m": 1, "value": 0.001}]
    
    with open('test_ultra_simple_asym.json', 'w') as f:
        json.dump(input_data, f, indent=2)
    
    print("\n2. Testing ultra-simple ASYMMETRIC case...")
    result_asym = subprocess.run(["./build/vmec_standalone", "test_ultra_simple_asym.json"],
                                capture_output=True, text=True)
    print(f"Return code: {result_asym.returncode}")
    
    if result_asym.returncode != 0:
        print("\nAsymmetric failed. Error output:")
        print(result_asym.stderr[:1000])
else:
    print("Even the symmetric case failed!")
    print("Error:", result.stderr[:500])