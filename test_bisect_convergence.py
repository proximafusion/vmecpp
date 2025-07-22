#!/usr/bin/env python3
"""
Test VMEC++ convergence for bisection
"""

import vmecpp

print("Loading Solovev analytical tokamak input...")

# Load the existing Solovev input
input_file = "src/vmecpp/cpp/vmecpp/test_data/input.solovev"
vmec_input = vmecpp.VmecInput.from_file(input_file)

print("\nRunning VMEC++ with Solovev input...")
try:
    result = vmecpp.run(vmec_input, verbose=False)
    print("\n✓ VMEC++ RAN SUCCESSFULLY - GOOD")
    
    # Check various convergence indicators
    if hasattr(result, 'fsql'):
        print(f"Final force residual: {result.fsql:.2e}")
        if result.fsql < 1e-8:
            print("✓ Force residual indicates convergence")
        else:
            print("✗ Force residual too high")
    
    exit(0)  # Success
    
except Exception as e:
    print(f"\n✗ VMEC++ FAILED - BAD: {e}")
    exit(1)  # Failure