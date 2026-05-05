#!/usr/bin/env python3
"""
Test VMEC++ standalone executable on JSON cases to verify azNorm fix
"""

import subprocess
import json
from pathlib import Path

# VMEC++ standalone executable
vmecpp_exe = "/home/ert/code/vmecpp/build/vmec_standalone"

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
    }
]

print("="*80)
print("VMEC++ Standalone Test - Verifying azNorm=0 Fix")
print("="*80)

for test_case in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {test_case['name']}")
    print(f"File: {Path(test_case['file']).name}")
    print(f"Asymmetric: {test_case['lasym']}")
    print("-"*60)
    
    # Check if file exists
    if not Path(test_case['file']).exists():
        print(f"❌ File not found")
        continue
    
    # Read JSON to check lasym setting
    with open(test_case['file']) as f:
        data = json.load(f)
        actual_lasym = data.get('lasym', False)
        print(f"JSON lasym setting: {actual_lasym}")
    
    # Run VMEC++ standalone
    print("\nRunning VMEC++ standalone...")
    try:
        result = subprocess.run(
            [vmecpp_exe, test_case['file']],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS! VMEC++ completed successfully")
            
            # Check for key output indicators
            if "Beta" in result.stdout:
                # Extract beta value if possible
                for line in result.stdout.split('\n'):
                    if "Beta" in line and ":" in line:
                        print(f"   {line.strip()}")
                        break
            
            if test_case['lasym']:
                print(f"\n   🎉 Asymmetric equilibrium ran successfully!")
                print(f"   The azNorm=0 fix is working in the C++ code!")
                
        else:
            print(f"❌ FAILED with exit code {result.returncode}")
            
            # Check for specific errors
            if "azNorm should never be 0.0" in result.stdout or "azNorm should never be 0.0" in result.stderr:
                print("   ⚠️  azNorm=0 error detected!")
                print("   The asymmetric Fourier transforms are not working correctly")
            else:
                # Show first few lines of error
                error_lines = result.stderr.split('\n') if result.stderr else result.stdout.split('\n')
                for i, line in enumerate(error_lines[:5]):
                    if line.strip():
                        print(f"   {line}")
                        
    except subprocess.TimeoutExpired:
        print("❌ Timeout after 30 seconds")
    except Exception as e:
        print(f"❌ Error running VMEC++: {e}")

print("\n" + "="*80)
print("Summary")
print("="*80)
print("This test runs the VMEC++ C++ executable directly to verify")
print("that the azNorm=0 fix is working for asymmetric equilibria.")