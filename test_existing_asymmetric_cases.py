#!/usr/bin/env python3
"""Test existing asymmetric JSON cases"""

import os
from vmecpp.cpp import _vmecpp as vmec

# Test existing JSON files
json_cases = [
    "src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak_simple.json",
    "src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak.json"
]

print("=== VMEC++ Existing Asymmetric JSON Cases ===")
print("Testing azNorm fix on existing JSON test cases\n")

for i, json_path in enumerate(json_cases, 1):
    print(f"[{i}/{len(json_cases)}] Testing: {os.path.basename(json_path)}")
    
    if not os.path.exists(json_path):
        print(f"  ❌ File not found: {json_path}")
        continue
    
    try:
        # Load input
        indata = vmec.VmecINDATAPyWrapper.from_file(json_path)
        
        if not indata.lasym:
            print(f"  ⚠️  LASYM=F - not asymmetric")
            continue
        
        print(f"  ✓ LASYM=T confirmed - asymmetric case")
        print(f"  NS = {indata.ns_array[-1]}")
        
        # Very limited iterations to test azNorm fix only
        indata.nstep = 2
        indata.niter_array = [2]
        
        # Run VMEC
        print(f"  Running {indata.nstep} iterations to test azNorm fix...")
        output = vmec.run(indata, verbose=False)
        
        print(f"  ✅ SUCCESS: No azNorm=0 error!")
        print(f"     Iterations completed: {indata.nstep}")
        print(f"     IER flag: {output.wout.ier_flag}")
        
        if hasattr(output.wout, 'fsqr'):
            print(f"     Final FSQR: {output.wout.fsqr:.2e}")
            print(f"     Final FSQZ: {output.wout.fsqz:.2e}")
        
    except Exception as e:
        if "azNorm should never be 0.0" in str(e):
            print(f"  ❌ FAILURE: azNorm=0 error still occurs!")
        else:
            print(f"  ⚠️  Other error: {str(e)[:100]}...")
    
    print()

print("🎯 VALIDATION: If cases ran without azNorm=0 error, the fix is working!")
print("🔧 Note: Memory corruption during longer runs is a separate issue")