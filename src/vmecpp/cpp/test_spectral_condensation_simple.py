#!/usr/bin/env python3
"""Simple test of asymmetric spectral condensation."""

import vmecpp
import json

# Load test case
with open("test_symmetric_simple.json", "r") as f:
    data = json.load(f)

# Make it asymmetric
data["lasym"] = True

# Add asymmetric perturbations
for i, (m, n) in enumerate(zip(data["xm"], data["xn"])):
    if m == 1 and n == 0:
        data["rbs"][i] = 0.01  # R sin component  
        data["zbc"][i] = 0.01  # Z cos component
        break

# Save modified file
with open("test_asymmetric_spectral.json", "w") as f:
    json.dump(data, f, indent=2)

# Run VMEC
print("Running asymmetric VMEC...")
try:
    vmec = vmecpp.Vmec("test_asymmetric_spectral.json", verbose=5, checkpoint="forces")
    vmec.run()
    print("SUCCESS: VMEC completed without spectral condensation errors!")
except Exception as e:
    print(f"ERROR: {e}")
    if "spectral" in str(e).lower():
        print("SPECTRAL CONDENSATION ERROR DETECTED")