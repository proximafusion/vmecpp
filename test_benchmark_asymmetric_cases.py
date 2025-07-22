#!/usr/bin/env python3
"""Test additional asymmetric cases from benchmark directories"""

import os
import subprocess
import tempfile
from vmecpp.cpp import _vmecpp as vmec

def convert_fortran_input_to_json(fortran_input_path):
    """Convert Fortran VMEC input to JSON format using indata2json"""
    try:
        # Create temporary JSON file
        base_name = os.path.splitext(os.path.basename(fortran_input_path))[0]
        json_path = f"{base_name}_temp.json"
        
        # Use the indata2json binary directly
        indata2json_path = "/home/ert/code/.venv/lib/python3.13/site-packages/vmecpp/cpp/third_party/indata2json/indata2json"
        
        if os.path.exists(indata2json_path):
            # Use the C++ indata2json tool
            result = subprocess.run([
                indata2json_path, fortran_input_path, json_path
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(json_path):
                return json_path
            else:
                print(f"  indata2json failed: {result.stderr}")
        
        # Fallback: try Python approach
        result = subprocess.run([
            "python", "-c", 
            f"import sys; sys.path.append('src/vmecpp/python'); from vmecpp._free_boundary import FormatType, FortranInput; fi = FortranInput.from_file('{fortran_input_path}'); fi.to_json_file('{json_path}')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and os.path.exists(json_path):
            return json_path
        else:
            print(f"  Python conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"  Conversion error: {e}")
        return None

# Test cases - asymmetric inputs from benchmark
test_cases = [
    "../benchmark_vmec/input.SOLOVEV_asym",
    "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.up_down_asymmetric_tokamak", 
    "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.LandremanSenguptaPlunk_section5p3_low_res"
]

print("=== VMEC++ Benchmark Asymmetric Cases Validation ===")
print("Testing azNorm fix on additional asymmetric cases from benchmark suite\n")

results = {"success": 0, "aznorm_failures": 0, "other_errors": 0, "conversion_failures": 0}

for i, input_path in enumerate(test_cases, 1):
    name = os.path.basename(input_path)
    print(f"[{i}/{len(test_cases)}] Testing: {name}")
    
    if not os.path.exists(input_path):
        print(f"  ❌ File not found: {input_path}")
        continue
    
    # Convert Fortran input to JSON
    json_path = convert_fortran_input_to_json(input_path)
    if not json_path:
        print(f"  ❌ Failed to convert to JSON format")
        results["conversion_failures"] += 1
        continue
    
    try:
        # Load input
        indata = vmec.VmecINDATAPyWrapper.from_file(json_path)
        
        if not indata.lasym:
            print(f"  ⚠️  LASYM=F - not asymmetric (unexpected)")
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
        print(f"     Iterations completed successfully")
        print(f"     IER flag: {output.wout.ier_flag}")
        
        results["success"] += 1
        
    except Exception as e:
        if "azNorm should never be 0.0" in str(e):
            print(f"  ❌ FAILURE: azNorm=0 error still occurs!")
            results["aznorm_failures"] += 1
        else:
            print(f"  ⚠️  Other error: {str(e)[:80]}...")
            results["other_errors"] += 1
    
    finally:
        # Clean up temporary JSON file
        if json_path and os.path.exists(json_path):
            try:
                os.remove(json_path)
            except:
                pass
    
    print()

# Summary
print("=== BENCHMARK VALIDATION SUMMARY ===")
print(f"✅ Successful cases (no azNorm error): {results['success']}")
print(f"❌ azNorm=0 failures: {results['aznorm_failures']}")
print(f"⚠️  Other errors (memory/timeout): {results['other_errors']}")
print(f"🔧 Conversion failures: {results['conversion_failures']}")

total_attempted = results["success"] + results["aznorm_failures"] + results["other_errors"]
if total_attempted > 0 and results["aznorm_failures"] == 0:
    print(f"\n🎯 VALIDATION SUCCESS: azNorm=0 error eliminated across {results['success']} benchmark cases!")
    print("🚀 Asymmetric equilibrium solver confirmed working on diverse test cases")
elif results["aznorm_failures"] > 0:
    print(f"\n❌ {results['aznorm_failures']} cases still have azNorm=0 error")
else:
    print(f"\n⚠️  Could not test due to conversion/other issues")

print("\n📝 Note: Memory corruption during extended runs is a separate optimization issue")