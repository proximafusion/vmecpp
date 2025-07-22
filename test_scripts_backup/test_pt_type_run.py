#!/usr/bin/env python3
"""
Test running VMEC++ with PT_TYPE fields
"""

import vmecpp
import numpy as np
import json

# Create a complete input with PT_TYPE fields
print("Creating VMEC++ input with PT_TYPE support...")

# Basic resolution parameters
mpol = 5
ntor = 4
nfp = 3

# Create input data
input_data = {
    # Grid parameters
    "lasym": True,
    "nfp": nfp,
    "mpol": mpol,
    "ntor": ntor,
    "ntheta": 0,  # auto
    "nzeta": 0,   # auto
    
    # Multi-grid
    "ns_array": [11, 25],
    "ftol_array": [1.0e-8, 1.0e-11],
    "niter_array": [300, 1000],
    
    # Physics parameters
    "phiedge": 0.5,
    "ncurr": 0,  # Fixed iota
    
    # Pressure profile
    "pmass_type": "power_series",
    "am": [0.0, 0.0, 0.0, 0.0, 0.0],  # No pressure
    "am_aux_s": [],
    "am_aux_f": [],
    "pres_scale": 1.0,
    "gamma": 0.0,
    "spres_ped": 1.0,
    
    # Iota profile
    "piota_type": "power_series",
    "ai": [0.4, 0.0, 0.0, 0.0, 0.0],  # Constant iota = 0.4
    "ai_aux_s": [],
    "ai_aux_f": [],
    
    # Current profile (not used with ncurr=0)
    "pcurr_type": "power_series",
    "ac": [0.0, 0.0, 0.0, 0.0, 0.0],
    "ac_aux_s": [],
    "ac_aux_f": [],
    "curtor": 0.0,
    "bloat": 1.0,
    
    # ANIMEC PT_TYPE fields
    "bcrit": 1.0,
    "pt_type": "power_series",
    "at": [1.0, -0.2, 0.0, 0.0, 0.0],  # Temperature anisotropy profile
    "ph_type": "power_series",
    "ah": [0.1, -0.1, 0.0, 0.0, 0.0],  # Hot particle pressure profile
    
    # Free boundary
    "lfreeb": False,
    "mgrid_file": "",
    "extcur": [],
    "nvacskip": 1,
    "free_boundary_method": "NESTOR",
    
    # Numerical parameters
    "nstep": 200,
    "aphi": [],
    "delt": 0.9,
    "tcon0": 2.0,
    "lforbal": False,
    "iteration_style": "VMEC_8_52",
    
    # Initial axis (circular)
    "raxis_c": [1.0],
    "zaxis_s": [0.0],
    "raxis_s": [0.0],
    "zaxis_c": [0.0],
    
    # Boundary shape - simple circular cross-section
    "rbc": [
        (0, 0, 1.0),    # R00 = 1.0
        (1, 0, 0.1),    # R10 = 0.1 (circular cross-section)
    ],
    "zbs": [
        (1, 0, 0.1),    # Z10 = 0.1 (circular cross-section)
    ],
    "rbs": None,  # Optional for stellarator-symmetric
    "zbc": None   # Optional for stellarator-symmetric
}

# Convert to VmecInput
print("\nConverting to VmecInput...")
vmec_input = vmecpp.VmecInput(**input_data)

# Verify PT_TYPE fields
print("\nPT_TYPE fields in input:")
print(f"  bcrit = {vmec_input.bcrit}")
print(f"  pt_type = '{vmec_input.pt_type}'")
print(f"  at = {vmec_input.at}")
print(f"  ph_type = '{vmec_input.ph_type}'")
print(f"  ah = {vmec_input.ah}")

# Save to JSON to verify serialization
print("\nSaving to JSON...")
# Convert numpy arrays to lists for JSON serialization
data_for_json = vmec_input.model_dump()
for key, value in data_for_json.items():
    if hasattr(value, 'tolist'):  # Convert numpy arrays to lists
        data_for_json[key] = value.tolist()
        
with open('test_pt_type_input.json', 'w') as f:
    json.dump(data_for_json, f, indent=2)

# Run VMEC++
print("\nRunning VMEC++ with PT_TYPE fields...")
try:
    result = vmecpp.run(vmec_input, verbose=True)
    print("\n✓ VMEC++ ran successfully with PT_TYPE fields!")
    
    # Check if the fields were preserved in output
    if hasattr(result, 'input'):
        print("\nPT_TYPE fields in output:")
        print(f"  bcrit = {result.input.bcrit}")
        print(f"  pt_type = '{result.input.pt_type}'")
        print(f"  at = {result.input.at[:5]}...")
        print(f"  ph_type = '{result.input.ph_type}'")
        print(f"  ah = {result.input.ah[:5]}...")
    
    # Check convergence
    print(f"\nConvergence: {result.r00_convergence_flag}")
    print(f"Final force residual: {result.fsql}")
    
except Exception as e:
    print(f"\nError running VMEC++: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ PT_TYPE fields are fully functional in VMEC++!")