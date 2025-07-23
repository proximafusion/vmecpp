#!/usr/bin/env python3
"""Test simple symmetric case to debug convergence issue."""

import json
import subprocess
import sys

# Create a very simple symmetric test case
test_input = {
    "lasym": False,
    "nfp": 1,
    "mpol": 3,
    "ntor": 0,
    "ns_array": [3],
    "ftol_array": [1e-8],
    "niter_array": [50],
    "rbc": [
        {"m": 0, "n": 0, "value": 1.0},
        {"m": 1, "n": 0, "value": 0.1}
    ],
    "zbs": [
        {"m": 1, "n": 0, "value": 0.1}
    ],
    "phiedge": 0.1,
    "ncurr": 0,
    "curtor": 0.0,
    "pres_scale": 1.0,
    "am": [0.0],
    "ai": [0.0],
    "ac": [0.0]
}

# Write test input
with open("test_simple_symmetric.json", "w") as f:
    json.dump(test_input, f, indent=2)

print("Created test_simple_symmetric.json")
print("Running VMEC++ with simple symmetric test case...")

# Run VMEC++
result = subprocess.run([
    "bazel", "run", "//vmecpp/vmec/vmec_standalone:vmec_standalone", "--",
    "/home/ert/code/vmecpp/test_simple_symmetric.json"
], capture_output=True, text=True)

print("\nSTDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")