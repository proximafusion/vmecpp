#!/usr/bin/env python3
"""Simple test of key VMEC++ input files"""
import vmecpp
import subprocess
import sys
from pathlib import Path

def test_vmecpp_file(filepath):
    try:
        print(f"Testing VMEC++ on {filepath}:")
        vmec_input = vmecpp.VmecInput.from_file(filepath)
        output = vmecpp.run(vmec_input, verbose=False)
        print(f"  ✅ MHD Energy: {output.wout.wb:.6e}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:60]}...")
        return False

def test_vmec2000_available():
    try:
        result = subprocess.run(['python', '-c', 'import vmec; print("VMEC2000 available")'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def main():
    # Test just a few key files
    test_files = [
        "examples/data/input.solovev",
        "src/vmecpp/cpp/vmecpp/test_data/input.solovev", 
        "src/vmecpp/cpp/vmecpp/test_data/input.test_asymmetric"
    ]
    
    print("VMEC Testing Summary")
    print("=" * 40)
    
    # Check VMEC2000 availability
    vmec2000_avail = test_vmec2000_available()
    print(f"VMEC2000 available: {'Yes' if vmec2000_avail else 'No'}")
    print()
    
    vmecpp_success = 0
    total_files = 0
    
    for filepath in test_files:
        if Path(filepath).exists():
            total_files += 1
            if test_vmecpp_file(filepath):
                vmecpp_success += 1
            print()
    
    print("=" * 40)
    print(f"VMEC++ Success Rate: {vmecpp_success}/{total_files} ({vmecpp_success/total_files*100:.1f}%)")

if __name__ == "__main__":
    main()