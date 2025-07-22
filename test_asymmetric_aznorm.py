#!/usr/bin/env python3
"""Test asymmetric case for azNorm error"""
from vmecpp.cpp import _vmecpp as vmec

# Test asymmetric case
indata = vmec.VmecINDATAPyWrapper.from_file(
    "src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak_simple.json"
)
assert indata.lasym, "Must be asymmetric"

print("Testing asymmetric equilibrium...")
try:
    output = vmec.run(indata, verbose=True)
    print("✓ Asymmetric case runs without azNorm=0 error!")
    print(f"  Final ier_flag = {output.wout.ier_flag}")
except Exception as e:
    if "azNorm should never be 0.0" in str(e):
        print("✗ azNorm=0 error still occurs!")
    else:
        print(f"✗ Different error: {e}")
    raise