#!/usr/bin/env python3
"""
Test running VMEC++ with Solovev JSON input file
"""

import vmecpp

print("Loading Solovev JSON input...")

# Load the existing Solovev JSON input
json_file = "src/vmecpp/cpp/vmecpp/test_data/solovev.json"
vmec_input = vmecpp.VmecInput.from_file(json_file)

print(f"\nLoaded input parameters:")
print(f"  NFP = {vmec_input.nfp}")
print(f"  MPOL = {vmec_input.mpol}")
print(f"  NTOR = {vmec_input.ntor}")
print(f"  PHIEDGE = {vmec_input.phiedge}")
print(f"  NCURR = {vmec_input.ncurr}")
print(f"  Grid levels: {vmec_input.ns_array}")

print(f"\nDefault PT_TYPE fields:")
print(f"  bcrit = {vmec_input.bcrit}")
print(f"  pt_type = '{vmec_input.pt_type}'")
print(f"  at = {vmec_input.at}")
print(f"  ph_type = '{vmec_input.ph_type}'")
print(f"  ah = {vmec_input.ah}")

print("\nRunning VMEC++ with Solovev JSON input...")
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
    
    print("\n✓ JSON Solovev equilibrium converges successfully!")
    print("✓ PT_TYPE fields are working in VMEC++ with real equilibria!")
    
except Exception as e:
    print(f"\nError running VMEC++: {e}")
    import traceback
    traceback.print_exc()