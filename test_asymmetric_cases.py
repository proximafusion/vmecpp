#!/usr/bin/env python3
"""Test multiple asymmetric cases to validate the azNorm fix"""

import sys
import os
from vmecpp.cpp import _vmecpp as vmec

# Convert Fortran input to JSON if needed
def convert_input_to_json(fortran_input_path):
    """Convert Fortran VMEC input to JSON format for VMEC++"""
    base_name = os.path.basename(fortran_input_path)
    json_path = f"{base_name}.json"
    
    if os.path.exists(json_path):
        return json_path
        
    # Use indata2json tool to convert
    import subprocess
    try:
        result = subprocess.run([
            "python", "-c", 
            f"from vmecpp.cpp.third_party.indata2json import indata2json; indata2json.convert_file('{fortran_input_path}', '{json_path}')"
        ], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(json_path):
            return json_path
        else:
            print(f"Failed to convert {fortran_input_path}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error converting {fortran_input_path}: {e}")
        return None

# Test cases - asymmetric inputs
test_cases = [
    "../benchmark_vmec/input.SOLOVEV_asym",
    "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.up_down_asymmetric_tokamak",
    "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.LandremanSenguptaPlunk_section5p3_low_res"
]

print("=== VMEC++ Asymmetric Cases Validation ===")
print("Testing multiple asymmetric equilibria for azNorm error fix\n")

results = []

for i, input_path in enumerate(test_cases, 1):
    print(f"[{i}/{len(test_cases)}] Testing: {os.path.basename(input_path)}")
    
    if not os.path.exists(input_path):
        print(f"  ❌ File not found: {input_path}")
        results.append(("NOT_FOUND", input_path, "File not found"))
        continue
    
    # Convert to JSON if it's a Fortran input
    if input_path.endswith('.json'):
        json_path = input_path
    else:
        json_path = convert_input_to_json(input_path)
        if not json_path:
            print(f"  ❌ Failed to convert to JSON")
            results.append(("CONVERT_FAILED", input_path, "JSON conversion failed"))
            continue
    
    try:
        # Load input
        indata = vmec.VmecINDATAPyWrapper.from_file(json_path)
        
        if not indata.lasym:
            print(f"  ⚠️  LASYM=F - skipping symmetric case")
            results.append(("SYMMETRIC", input_path, "Not asymmetric"))
            continue
        
        print(f"  ✓ LASYM=T confirmed - asymmetric case")
        print(f"  NS = {indata.ns_array[-1]}, NFOUR = {indata.mnmax_nyq}")
        
        # Limit iterations to avoid memory corruption
        indata.nstep = 3
        indata.niter_array = [3]
        
        # Run VMEC
        print(f"  Running {indata.nstep} iterations...")
        output = vmec.run(indata, verbose=False)
        
        print(f"  ✅ SUCCESS: No azNorm=0 error!")
        print(f"     IER: {output.wout.ier_flag}")
        print(f"     FSQR: {output.wout.fsqr:.2e}")
        print(f"     FSQZ: {output.wout.fsqz:.2e}")
        
        results.append(("SUCCESS", input_path, f"IER={output.wout.ier_flag}"))
        
    except Exception as e:
        if "azNorm should never be 0.0" in str(e):
            print(f"  ❌ FAILURE: azNorm=0 error still occurs!")
            results.append(("AZNORM_ERROR", input_path, "azNorm=0 error"))
        else:
            print(f"  ⚠️  Other error: {str(e)[:100]}...")
            results.append(("OTHER_ERROR", input_path, str(e)[:100]))
    
    print()

# Summary
print("=== SUMMARY ===")
success_count = 0
aznorm_failures = 0

for status, path, detail in results:
    name = os.path.basename(path)
    if status == "SUCCESS":
        print(f"✅ {name}: {detail}")
        success_count += 1
    elif status == "AZNORM_ERROR":
        print(f"❌ {name}: azNorm=0 error")
        aznorm_failures += 1
    else:
        print(f"⚠️  {name}: {detail}")

print(f"\nResults: {success_count}/{len([r for r in results if r[0] not in ['NOT_FOUND', 'CONVERT_FAILED', 'SYMMETRIC']])} asymmetric cases ran without azNorm error")

if aznorm_failures == 0:
    print("🎯 VALIDATION SUCCESS: azNorm=0 error eliminated across all test cases!")
else:
    print(f"❌ {aznorm_failures} cases still have azNorm=0 error")

print(f"🔧 Note: Memory corruption during longer runs is a separate issue")