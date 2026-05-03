#!/usr/bin/env python3
"""
Comprehensive test of VMEC++ with multiple benchmark cases
Tests both symmetric and asymmetric equilibria
"""

import os
import glob
import vmecpp
from pathlib import Path
import traceback

def find_test_cases():
    """Find all available test cases"""
    test_cases = []
    
    # JSON test cases in the test data directory
    json_files = glob.glob("src/vmecpp/cpp/vmecpp/test_data/*.json")
    for f in json_files:
        name = Path(f).stem
        test_cases.append((f, name, "json"))
    
    # Input files from benchmark directory
    benchmark_inputs = [
        "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.circular_tokamak",
        "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.up_down_asymmetric_tokamak",
        "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.li383_low_res",
        "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.LandremanSenguptaPlunk_section5p3_low_res",
        "../benchmark_vmec/vmec_repos/educational_VMEC/input.up_down_asymmetric_tokamak",
        "../benchmark_vmec/vmec_repos/jvmec/input.solovev_analytical",
        "../benchmark_vmec/vmec_repos/jvmec/input.tok_simple_asym",
        "../benchmark_vmec/input.SOLOVEV",
        "../benchmark_vmec/input.SOLOVEV_asym",
    ]
    
    for f in benchmark_inputs:
        if os.path.exists(f):
            name = Path(f).name
            test_cases.append((f, name, "indata"))
    
    return test_cases

def test_single_case(filepath, name, filetype):
    """Test a single case and return results"""
    result = {
        "name": name,
        "path": filepath,
        "type": filetype,
        "status": "unknown",
        "lasym": None,
        "mpol": None,
        "ntor": None,
        "error": None,
        "volume": None,
        "beta": None,
        "iterations": None
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"Path: {filepath}")
        
        # Load the input
        if filetype == "json" or filepath.endswith(".json"):
            vmec_input = vmecpp.VmecInput.from_file(filepath)
        else:
            # For INDATA files, we need to handle them differently
            # For now, skip them as they need special handling
            print("  ⚠️  INDATA files need special handling, skipping...")
            result["status"] = "skipped"
            return result
        
        # Extract parameters
        result["lasym"] = vmec_input.lasym
        result["mpol"] = vmec_input.mpol
        result["ntor"] = vmec_input.ntor
        
        print(f"  Parameters: lasym={vmec_input.lasym}, mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
        print(f"  Array shapes: rbc={vmec_input.rbc.shape}, zbs={vmec_input.zbs.shape}")
        
        # Run with limited iterations
        vmec_input.nstep = 3
        vmec_input.niter_array = [10, 10, 10]
        
        print(f"  Running VMEC++ with {vmec_input.nstep} steps...")
        output = vmecpp.run(vmec_input, verbose=False)
        
        # Extract results
        result["status"] = "success"
        result["volume"] = float(output.volume_p)
        result["beta"] = float(output.beta)
        result["iterations"] = int(output.iter_)
        
        print(f"  ✅ SUCCESS!")
        print(f"     Volume = {result['volume']:.3f}")
        print(f"     Beta = {result['beta']:.6f}")
        print(f"     Iterations = {result['iterations']}")
        
    except Exception as e:
        error_msg = str(e)
        result["error"] = error_msg
        
        if "azNorm should never be 0.0" in error_msg:
            result["status"] = "aznorm_error"
            print(f"  ❌ FAILED: azNorm=0 error")
        elif "could not broadcast" in error_msg:
            result["status"] = "array_shape_error"
            print(f"  ❌ FAILED: Array shape mismatch")
        elif "has wrong size" in error_msg:
            result["status"] = "size_error"
            print(f"  ❌ FAILED: Array size validation error")
        else:
            result["status"] = "other_error"
            print(f"  ❌ FAILED: {error_msg[:100]}...")
            if len(error_msg) > 100:
                print(f"     Full error: {error_msg}")
    
    return result

def main():
    print("=== VMEC++ Comprehensive Benchmark Test ===")
    print("Testing with multiple cases including asymmetric equilibria\n")
    
    # Find all test cases
    test_cases = find_test_cases()
    print(f"Found {len(test_cases)} test cases")
    
    # Test each case
    results = []
    for filepath, name, filetype in test_cases:
        result = test_single_case(filepath, name, filetype)
        results.append(result)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("=== SUMMARY ===")
    
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    aznorm_errors = sum(1 for r in results if r["status"] == "aznorm_error")
    shape_errors = sum(1 for r in results if r["status"] == "array_shape_error")
    size_errors = sum(1 for r in results if r["status"] == "size_error")
    other_errors = sum(1 for r in results if r["status"] == "other_error")
    
    # Count by symmetry
    symmetric_success = sum(1 for r in results if r["status"] == "success" and r["lasym"] == False)
    asymmetric_success = sum(1 for r in results if r["status"] == "success" and r["lasym"] == True)
    
    print(f"\nTotal cases tested: {total}")
    print(f"✅ Successful: {success} ({symmetric_success} symmetric, {asymmetric_success} asymmetric)")
    print(f"⚠️  Skipped: {skipped}")
    print(f"❌ Failed: {total - success - skipped}")
    print(f"   - azNorm=0 errors: {aznorm_errors}")
    print(f"   - Array shape errors: {shape_errors}")
    print(f"   - Array size errors: {size_errors}")
    print(f"   - Other errors: {other_errors}")
    
    # Detailed results table
    print(f"\n{'='*60}")
    print("=== DETAILED RESULTS ===")
    print(f"{'Name':<40} {'Type':<6} {'lasym':<6} {'Status':<15} {'Volume':<10}")
    print("-" * 85)
    
    for r in results:
        name = r['name'][:39]
        type_str = r['type']
        lasym = "True" if r['lasym'] else "False" if r['lasym'] is not None else "N/A"
        status = r['status']
        volume = f"{r['volume']:.1f}" if r['volume'] else "N/A"
        
        print(f"{name:<40} {type_str:<6} {lasym:<6} {status:<15} {volume:<10}")
    
    # Final verdict
    print(f"\n{'='*60}")
    if aznorm_errors == 0 and asymmetric_success > 0:
        print("🎉 SUCCESS: azNorm=0 error has been eliminated!")
        print(f"   {asymmetric_success} asymmetric cases ran successfully")
    elif aznorm_errors > 0:
        print("❌ FAILURE: azNorm=0 error still occurring")
    
    if size_errors > 0:
        print(f"\n⚠️  Note: {size_errors} cases have array size validation issues")
        print("   This is related to the mpol vs mpol+1 array sizing mismatch")

if __name__ == "__main__":
    main()