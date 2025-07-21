#!/usr/bin/env python3
import vmecpp
import sys
from pathlib import Path

# Quick test of key input files
input_files = [
    "examples/data/input.solovev",
    "src/vmecpp/cpp/vmecpp/test_data/input.solovev", 
    "src/vmecpp/cpp/vmecpp/test_data/input.test_asymmetric",
    "examples/data/input.up_down_asymmetric_tokamak"
]

success = 0
total = 0

for input_file in input_files:
    total += 1
    try:
        if not Path(input_file).exists():
            print(f"❌ {input_file}: File not found")
            continue
            
        vmec_input = vmecpp.VmecInput.from_file(input_file)
        output = vmecpp.run(vmec_input, verbose=False)
        wb = output.wout.wb
        print(f"✅ {input_file}: {wb:.6e}")
        success += 1
        
    except Exception as e:
        print(f"❌ {input_file}: {str(e)[:50]}...")

print(f"\nSUMMARY: {success}/{total} successful ({success/total*100:.1f}%)")