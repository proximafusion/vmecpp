#!/usr/bin/env python3
"""Minimal test to check if asymmetric case starts without azNorm error"""
from vmecpp.cpp import _vmecpp as vmec

# Test asymmetric case
indata = vmec.VmecINDATAPyWrapper.from_file(
    "src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak_simple.json"
)
assert indata.lasym, "Must be asymmetric"

# Run for just 1 iteration to check if it starts
indata.nstep = 1

print("Testing asymmetric equilibrium (1 iteration)...")
try:
    output = vmec.run(indata, verbose=True)
    print(f"✓ Asymmetric case runs without azNorm=0 error!")
    print(f"  Completed 1 iteration successfully")
except Exception as e:
    if "azNorm should never be 0.0" in str(e):
        print("✗ azNorm=0 error still occurs!")
    else:
        print(f"✗ Different error: {e}")
    raise