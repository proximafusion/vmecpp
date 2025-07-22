#!/usr/bin/env python3
"""Test high-level vs low-level interfaces to isolate PT_TYPE issue."""

import sys
sys.path.insert(0, '/home/ert/code/vmecpp/src')

print("=== Testing high-level vmecpp interface (baseline) ===")

try:
    import vmecpp
    
    # Test using high-level interface first
    input_file = "src/vmecpp/cpp/vmecpp/test_data/input.solovev"
    vmec_input = vmecpp.VmecInput.from_file(input_file)
    
    print(f"High-level input loaded:")
    print(f"  MPOL = {vmec_input.mpol}")
    print(f"  NTOR = {vmec_input.ntor}")
    
    # Check if PT_TYPE fields exist in high-level interface
    has_pt_type = hasattr(vmec_input, 'bcrit') or hasattr(vmec_input, 'pt_type')
    print(f"  Has PT_TYPE fields: {has_pt_type}")
    
    if has_pt_type:
        print(f"  bcrit = {getattr(vmec_input, 'bcrit', 'N/A')}")
        print(f"  pt_type = '{getattr(vmec_input, 'pt_type', 'N/A')}'")
        print(f"  ph_type = '{getattr(vmec_input, 'ph_type', 'N/A')}'")
    
    print("\n🔥 Running VMEC++ via high-level interface...")
    result = vmecpp.run(vmec_input, verbose=False)
    
    print(f"✅ HIGH-LEVEL SUCCESS! Residual: {result.fsql:.2e}")
    
except Exception as e:
    print(f"❌ High-level interface failed: {e}")

print("\n=== Testing low-level _vmecpp interface ===")

try:
    sys.path.insert(0, '/home/ert/code/vmecpp/build')
    import _vmecpp
    import json
    
    # Load same input via JSON
    with open('src/vmecpp/cpp/vmecpp/test_data/solovev.json', 'r') as f:
        config = json.load(f)
    
    indata = _vmecpp.VmecINDATAPyWrapper.from_json(json.dumps(config))
    
    print(f"Low-level input loaded:")
    print(f"  mpol = {indata.mpol}")
    print(f"  ntor = {indata.ntor}")
    print(f"  bcrit = {indata.bcrit}")
    print(f"  pt_type = '{indata.pt_type}'")
    print(f"  ph_type = '{indata.ph_type}'")
    
    print("\n🔥 Running VMEC++ via low-level interface...")
    result = _vmecpp.run(indata)
    
    print(f"✅ LOW-LEVEL SUCCESS! Residual: {result.fsqr:.2e}")
    
except Exception as e:
    print(f"❌ Low-level interface failed: {e}")

print("\n=== Analysis ===")
print("If high-level works but low-level fails, the issue is with PT_TYPE implementation.")
print("If both fail, the issue is deeper in the physics solver.")