#!/usr/bin/env python3
"""
Test VMEC++ against educational_VMEC on various input cases
"""

import subprocess
import json
import os
import shutil
from pathlib import Path
import tempfile

# Test cases to run
test_cases = [
    {
        "name": "SOLOVEV symmetric",
        "input": "/home/ert/code/vmecpp/benchmark_vmec/input.SOLOVEV",
        "lasym": False
    },
    {
        "name": "SOLOVEV asymmetric", 
        "input": "/home/ert/code/vmecpp/benchmark_vmec/input.SOLOVEV_asym",
        "lasym": True
    },
    {
        "name": "Circular tokamak",
        "input": "/home/ert/code/vmecpp/benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.circular_tokamak",
        "lasym": False
    },
    {
        "name": "Up-down asymmetric tokamak",
        "input": "/home/ert/code/vmecpp/benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.up_down_asymmetric_tokamak", 
        "lasym": True
    }
]

# Path to educational_VMEC executable
edu_vmec = "/home/ert/code/educational_VMEC/build/bin/xvmec"

def run_educational_vmec(input_file, work_dir):
    """Run educational_VMEC on an input file"""
    # Copy input file to work directory
    input_name = Path(input_file).name
    work_input = work_dir / input_name
    shutil.copy(input_file, work_input)
    
    # Run educational_VMEC
    result = subprocess.run(
        [edu_vmec, input_name],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    return result

def convert_to_json(input_file):
    """Convert VMEC input file to JSON format for VMEC++"""
    # Use indata2json tool
    indata2json = "/home/ert/code/vmecpp/build/_deps/indata2json-build/indata2json"
    
    # Create temporary directory and copy input file there
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_name = Path(input_file).name
        temp_input = temp_path / input_name
        shutil.copy(input_file, temp_input)
        
        # Create JSON filename
        json_name = Path(input_name).stem + '.json'
        
        # Run conversion in the temp directory
        result = subprocess.run(
            [indata2json, input_name, json_name],
            cwd=temp_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"   indata2json error: {result.stderr}")
            return None
        
        # Copy JSON file to original directory
        temp_json = temp_path / json_name
        if temp_json.exists():
            json_file = Path(input_file).parent / json_name
            shutil.copy(temp_json, json_file)
            return json_file
            
    return None

def run_vmecpp(json_file):
    """Run VMEC++ on a JSON input file"""
    try:
        result = subprocess.run(
            ["python", "-m", "vmecpp", str(json_file)],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result
    except Exception as e:
        print(f"Error running VMEC++: {e}")
        return None

# Create results directory
results_dir = Path("comparison_results")
results_dir.mkdir(exist_ok=True)

print("="*80)
print("VMEC++ vs educational_VMEC Comparison")
print("="*80)

for test_case in test_cases:
    print(f"\nTesting: {test_case['name']}")
    print(f"Input file: {test_case['input']}")
    print(f"Asymmetric: {test_case['lasym']}")
    
    # Check if input file exists
    if not Path(test_case['input']).exists():
        print(f"❌ Input file not found, skipping")
        continue
    
    # Create work directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        edu_dir = temp_path / "educational_vmec"
        edu_dir.mkdir()
        vmecpp_dir = temp_path / "vmecpp"
        vmecpp_dir.mkdir()
        
        # Run educational_VMEC
        print("\nRunning educational_VMEC...")
        edu_result = run_educational_vmec(test_case['input'], edu_dir)
        
        if edu_result.returncode == 0:
            print("✅ educational_VMEC completed successfully")
            # Check for output file
            wout_files = list(edu_dir.glob("wout_*.nc"))
            if wout_files:
                print(f"   Output: {wout_files[0].name}")
        else:
            print(f"❌ educational_VMEC failed with code {edu_result.returncode}")
            if "azNorm should never be 0.0" in edu_result.stdout or "azNorm should never be 0.0" in edu_result.stderr:
                print("   Error: azNorm=0 detected")
        
        # Convert to JSON and run VMEC++
        print("\nConverting to JSON for VMEC++...")
        json_file = convert_to_json(test_case['input'])
        
        if json_file and json_file.exists():
            print(f"✅ Converted to {json_file.name}")
            
            # Copy JSON to work directory
            work_json = vmecpp_dir / json_file.name
            shutil.copy(json_file, work_json)
            
            print("Running VMEC++...")
            vmecpp_result = run_vmecpp(work_json)
            
            if vmecpp_result and vmecpp_result.returncode == 0:
                print("✅ VMEC++ completed successfully")
            elif vmecpp_result:
                print(f"❌ VMEC++ failed with code {vmecpp_result.returncode}")
                if "azNorm should never be 0.0" in vmecpp_result.stdout or "azNorm should never be 0.0" in vmecpp_result.stderr:
                    print("   Error: azNorm=0 detected")
                elif "double free" in vmecpp_result.stderr:
                    print("   Error: Double free detected")
                else:
                    # Show first few lines of error
                    error_lines = vmecpp_result.stderr.split('\n')[:5]
                    for line in error_lines:
                        if line.strip():
                            print(f"   {line}")
            
            # Clean up JSON file
            json_file.unlink()
        else:
            print("❌ Failed to convert to JSON")
        
        # Save results
        case_dir = results_dir / test_case['name'].replace(' ', '_')
        case_dir.mkdir(exist_ok=True)
        
        # Save educational_VMEC results
        if edu_result:
            (case_dir / "educational_vmec_stdout.txt").write_text(edu_result.stdout)
            (case_dir / "educational_vmec_stderr.txt").write_text(edu_result.stderr)
        
        # Save VMEC++ results
        if 'vmecpp_result' in locals() and vmecpp_result:
            (case_dir / "vmecpp_stdout.txt").write_text(vmecpp_result.stdout)
            (case_dir / "vmecpp_stderr.txt").write_text(vmecpp_result.stderr)

print("\n" + "="*80)
print("Comparison complete. Results saved in comparison_results/")
print("="*80)