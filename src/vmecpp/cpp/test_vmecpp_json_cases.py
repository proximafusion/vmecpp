#!/usr/bin/env python3
"""
Test VMEC++ on JSON test cases, focusing on asymmetric cases
"""

import subprocess
import json
from pathlib import Path
import vmecpp

# Test cases with JSON files
test_cases = [
    {
        "name": "Solovev (symmetric)",
        "file": "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/solovev.json",
        "lasym": False
    },
    {
        "name": "Up-down asymmetric tokamak",
        "file": "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak.json",
        "lasym": True
    },
    {
        "name": "Up-down asymmetric tokamak (simple)",
        "file": "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak_simple.json",
        "lasym": True
    },
    {
        "name": "Circular tokamak",
        "file": "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json",
        "lasym": False
    },
    {
        "name": "CTH-like fixed boundary",
        "file": "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/cth_like_fixed_bdy.json",
        "lasym": False
    }
]

print("="*80)
print("VMEC++ Test Cases - Focus on Asymmetric Equilibria")
print("="*80)

for test_case in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {test_case['name']}")
    print(f"File: {Path(test_case['file']).name}")
    print(f"Asymmetric: {test_case['lasym']}")
    print("-"*60)
    
    try:
        # Load the input
        vmec_input = vmecpp.VmecInput.from_file(test_case['file'])
        print(f"✅ Loaded successfully")
        print(f"   mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
        print(f"   lasym={vmec_input.lasym}")
        
        # Check array shapes
        print(f"   rbc.shape={vmec_input.rbc.shape}")
        if vmec_input.lasym:
            print(f"   rbs.shape={vmec_input.rbs.shape}")
        
        # Quick run with minimal iterations
        vmec_input.nstep = 10
        vmec_input.niter_array = [10]
        vmec_input.ns_array = [3]
        vmec_input.ftol_array = [1e-8]
        
        print("\nRunning VMEC++...")
        output = vmecpp.run(vmec_input, verbose=False)
        
        print(f"✅ SUCCESS!")
        print(f"   Volume = {output.volume_p:.4f}")
        print(f"   Beta = {output.beta:.6f}")
        print(f"   Aspect ratio = {output.aspect:.4f}")
        
        if test_case['lasym']:
            print(f"\n   🎉 Asymmetric equilibrium ran successfully!")
            print(f"   The azNorm=0 fix is working!")
            
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ FAILED: {error_msg}")
        
        if "azNorm should never be 0.0" in error_msg:
            print("   ⚠️  azNorm=0 error detected - asymmetric transforms not working")
        elif "double free" in error_msg:
            print("   ⚠️  Memory corruption detected")
        elif "has wrong size" in error_msg:
            print("   ⚠️  Array size mismatch between Python and C++")

print("\n" + "="*80)
print("Test Summary")
print("="*80)
print("The azNorm=0 fix has been implemented in the C++ code.")
print("However, there's an array size mismatch issue between Python and C++")
print("that prevents loading some files through the Python interface.")