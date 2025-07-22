#!/usr/bin/env python3
"""
Test running VMEC++ with existing Solovev input file (baseline test)
"""

import vmecpp

print("Loading Solovev analytical tokamak input...")

# Load the existing Solovev input
input_file = "src/vmecpp/cpp/vmecpp/test_data/input.solovev"
vmec_input = vmecpp.VmecInput.from_file(input_file)

print(f"\nLoaded input parameters:")
print(f"  NFP = {vmec_input.nfp}")
print(f"  MPOL = {vmec_input.mpol}")
print(f"  NTOR = {vmec_input.ntor}")
print(f"  PHIEDGE = {vmec_input.phiedge}")
print(f"  NCURR = {vmec_input.ncurr}")
print(f"  Pressure: {vmec_input.pmass_type}, AM = {vmec_input.am}")
print(f"  Current: AI = {vmec_input.ai}")
print(f"  Grid levels: {vmec_input.ns_array}")

print("\nRunning VMEC++ with baseline Solovev input...")
try:
    result = vmecpp.run(vmec_input, verbose=False)
    
    print("\n✓ VMEC++ completed successfully!")
    print(f"\nConvergence flag: {result.r00_convergence_flag}")
    print(f"Final force residual: {result.fsql:.2e}")
    
    # Extract some key results
    print("\nEquilibrium properties:")
    print(f"  Volume-averaged beta: {result.betatot:.6f}")
    print(f"  On-axis rotational transform: {result.iota[0]:.6f}")
    print(f"  Edge rotational transform: {result.iota[-1]:.6f}")
    print(f"  Magnetic axis R: {result.raxis_symm[0]:.6f}")
    print(f"  Aspect ratio: {result.aspect:.4f}")
    
    print("\n✓ Baseline Solovev equilibrium runs successfully!")
    
except Exception as e:
    print(f"\nError running VMEC++: {e}")
    import traceback
    traceback.print_exc()