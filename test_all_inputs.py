#!/usr/bin/env python3
"""
Test script to run VMEC++ on all available input files in the codebase.
Excludes input.test as requested.
"""

import vmecpp
import sys
import os
from pathlib import Path

def test_input_file(input_path):
    """Test a single input file and return results."""
    try:
        print(f"Testing: {input_path}")
        vmec_input = vmecpp.VmecInput.from_file(str(input_path))
        output = vmecpp.run(vmec_input)
        
        # Extract key metrics
        wb = output.wout.wb
        betatotal = getattr(output.wout, 'betatotal', 0.0)
        aspect = getattr(output.wout, 'aspect', 0.0)
        volume_p = getattr(output.wout, 'volume_p', 0.0)
        
        print(f"  ✅ SUCCESS - MHD Energy: {wb:.6e}")
        return {
            'status': 'SUCCESS',
            'wb': wb,
            'betatotal': betatotal,
            'aspect': aspect,
            'volume_p': volume_p,
            'error': None
        }
        
    except Exception as e:
        print(f"  ❌ FAILED - {str(e)}")
        return {
            'status': 'FAILED',
            'wb': None,
            'betatotal': None,
            'aspect': None,
            'volume_p': None,
            'error': str(e)
        }

def main():
    # All input files found, excluding input.test
    input_files = [
        "examples/data/input.solovev",
        "examples/data/input.w7x", 
        "examples/data/input.nfp4_QH_warm_start",
        "examples/data/input.cth_like_fixed_bdy",
        "examples/data/input.up_down_asymmetric_tokamak",
        "src/vmecpp/cpp/vmecpp/test_data/input.cma",
        "src/vmecpp/cpp/vmecpp/test_data/input.cth_like_fixed_bdy",
        "src/vmecpp/cpp/vmecpp/test_data/input.cth_like_fixed_bdy_nzeta_37",
        "src/vmecpp/cpp/vmecpp/test_data/input.cth_like_free_bdy",
        "src/vmecpp/cpp/vmecpp/test_data/input.li383_low_res",
        "src/vmecpp/cpp/vmecpp/test_data/input.solovev",
        "src/vmecpp/cpp/vmecpp/test_data/input.solovev_analytical",
        "src/vmecpp/cpp/vmecpp/test_data/input.solovev_no_axis",
        "src/vmecpp/cpp/vmecpp/test_data/input.test_asymmetric",
        "src/vmecpp/cpp/vmecpp/test_data/input.up_down_asymmetric_tokamak.json",
        "build/_deps/indata2json-src/demo_inputs/input.HELIOTRON",
        "build/_deps/indata2json-src/demo_inputs/input.w7x_ref_167_12_12",
        "build/_deps/indata2json-src/demo_inputs/input.vmec",
        "build/_deps/indata2json-src/demo_inputs/input.W7X_s2048_M16_N16_f12_cpu8"
    ]
    
    print(f"Testing {len(input_files)} input files with VMEC++")
    print("=" * 60)
    
    results = {}
    success_count = 0
    
    for input_file in input_files:
        input_path = Path(input_file)
        if input_path.exists():
            result = test_input_file(input_path)
            results[input_file] = result
            if result['status'] == 'SUCCESS':
                success_count += 1
        else:
            print(f"❌ File not found: {input_file}")
            results[input_file] = {
                'status': 'NOT_FOUND',
                'wb': None,
                'betatotal': None, 
                'aspect': None,
                'volume_p': None,
                'error': 'File not found'
            }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files tested: {len(input_files)}")
    print(f"Successful runs: {success_count}")
    print(f"Failed runs: {len(input_files) - success_count}")
    print(f"Success rate: {success_count/len(input_files)*100:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 60)
    for input_file, result in results.items():
        status = result['status']
        if status == 'SUCCESS':
            print(f"✅ {input_file}: MHD Energy = {result['wb']:.6e}")
        else:
            error = result['error'][:50] + "..." if result['error'] and len(result['error']) > 50 else result['error']
            print(f"❌ {input_file}: {error}")

if __name__ == "__main__":
    main()