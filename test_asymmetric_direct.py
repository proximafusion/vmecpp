#!/usr/bin/env python3
"""Direct test of asymmetric mode to see the specific error."""

import vmecpp
import json
import os

print("=== Testing Asymmetric Mode Directly ===")

# Create a simple symmetric configuration first
config = {
    "lasym": False,  # Start with symmetric
    "nfp": 1,
    "mpol": 4,
    "ntor": 0,
    "ns_array": [3, 5],
    "niter_array": [50, 100],
    "ftol_array": [1e-6, 1e-8],
    "delt": 0.9,
    "phiedge": 1.0,
    "pmass_type": "power_series",
    "am": [1.0, -1.0],
    "pres_scale": 0.1,
    "gamma": 0.0,
    "raxis_c": [10.0],
    "zaxis_s": [0.0],
    "rbc": [
        {"m": 0, "n": 0, "value": 10.0},
        {"m": 1, "n": 0, "value": 1.0}
    ],
    "zbs": [
        {"m": 1, "n": 0, "value": 1.0}
    ]
}

# Test symmetric case first
print("\n1. Testing symmetric case...")
try:
    result = vmecpp.run(json.dumps(config))
    print("✓ Symmetric case successful!")
except Exception as e:
    print(f"✗ Symmetric case failed: {e}")

# Now test asymmetric case
print("\n2. Testing asymmetric case...")
config["lasym"] = True
config["raxis_s"] = [0.0]  # Add asymmetric axis arrays
config["zaxis_c"] = [0.0]
config["rbs"] = [{"m": 1, "n": 0, "value": 0.01}]  # Add asymmetric perturbation
config["zbc"] = [{"m": 1, "n": 0, "value": 0.01}]

try:
    result = vmecpp.run(json.dumps(config))
    print("✓ Asymmetric case successful!")
except Exception as e:
    print(f"✗ Asymmetric case failed: {e}")
    print("\nThis is the error we need to fix!")