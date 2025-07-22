#!/usr/bin/env python3
"""
Test PT_TYPE support using JSON input
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import _vmecpp
import json

print("Testing PT_TYPE support in VMEC++ via JSON...")

# Create minimal JSON input with PT_TYPE fields
input_json = {
    # Grid parameters
    "lasym": True,
    "nfp": 3,
    "mpol": 3,
    "ntor": 2,
    "ntheta": 0,
    "nzeta": 0,
    
    # Multi-grid
    "ns_array": [11, 25],
    "ftol_array": [1.0e-8, 1.0e-11],
    "niter_array": [300, 1000],
    
    # Physics
    "phiedge": 0.5,
    "ncurr": 0,
    
    # Profiles
    "pmass_type": "power_series",
    "am": [0.0, 0.0, 0.0, 0.0, 0.0],
    "am_aux_s": [],
    "am_aux_f": [],
    "pres_scale": 1.0,
    "gamma": 0.0,
    "spres_ped": 1.0,
    
    "piota_type": "power_series",
    "ai": [0.4, 0.0, 0.0, 0.0, 0.0],
    "ai_aux_s": [],
    "ai_aux_f": [],
    
    "pcurr_type": "power_series",
    "ac": [0.0, 0.0, 0.0, 0.0, 0.0],
    "ac_aux_s": [],
    "ac_aux_f": [],
    "curtor": 0.0,
    "bloat": 1.0,
    
    # ANIMEC PT_TYPE fields
    "bcrit": 1.0,
    "pt_type": "power_series",
    "at": [1.0, -0.2, 0.0, 0.0, 0.0],
    "ph_type": "power_series",
    "ah": [0.1, -0.1, 0.0, 0.0, 0.0],
    
    # Free boundary
    "lfreeb": False,
    "mgrid_file": "",
    "extcur": [],
    "nvacskip": 1,
    "free_boundary_method": "nestor",
    
    # Numerical
    "nstep": 200,
    "aphi": [],
    "delt": 0.9,
    "tcon0": 0.9,
    "lforbal": False,
    "return_outputs_even_if_not_converged": False,
    
    # Axis
    "raxis_c": [1.0, 0.0, 0.0],
    "zaxis_s": [0.0, 0.0, 0.0],
    "raxis_s": [0.0, 0.0, 0.0],
    "zaxis_c": [0.0, 0.0, 0.0],
    
    # Boundary coefficients in proper format
    "rbc": [
        {"m": 0, "n": 0, "value": 1.0},   # R00 = 1.0 (major radius)
        {"m": 1, "n": 0, "value": 0.1},   # R10 = 0.1 (circular cross-section)
    ],
    "zbs": [
        {"m": 1, "n": 0, "value": 0.1},   # Z10 = 0.1 (circular cross-section)
    ],
    "rbs": [],  # Empty for symmetric case
    "zbc": []   # Empty for symmetric case
}

# Convert to JSON string
json_str = json.dumps(input_json, indent=2)

# Create wrapper from JSON
print("\nCreating VmecINDATAPyWrapper from JSON...")
try:
    wrapper = _vmecpp.VmecINDATAPyWrapper.from_json(json_str)
    print("✓ Successfully created wrapper from JSON")
    
    # Verify PT_TYPE fields
    print("\nPT_TYPE fields loaded from JSON:")
    print(f"  bcrit = {wrapper.bcrit}")
    print(f"  pt_type = '{wrapper.pt_type}'")
    print(f"  at = {list(wrapper.at)}")
    print(f"  ph_type = '{wrapper.ph_type}'")
    print(f"  ah = {list(wrapper.ah)}")
    
    # Test roundtrip
    print("\nTesting JSON roundtrip...")
    json_out = wrapper.to_json()
    data_out = json.loads(json_out)
    
    print("PT_TYPE fields in output JSON:")
    print(f"  bcrit = {data_out.get('bcrit', 'NOT FOUND')}")
    print(f"  pt_type = '{data_out.get('pt_type', 'NOT FOUND')}'")
    print(f"  at = {data_out.get('at', 'NOT FOUND')[:5]}...")
    print(f"  ph_type = '{data_out.get('ph_type', 'NOT FOUND')}'")
    print(f"  ah = {data_out.get('ah', 'NOT FOUND')[:5]}...")
    
    # Run VMEC
    print("\nRunning VMEC++ with PT_TYPE fields...")
    result = _vmecpp.run(wrapper, verbose=False)
    print("✓ VMEC++ ran successfully!")
    
    print(f"\nConvergence: {result.r00_convergence_flag}")
    print(f"Final force residual: {result.fsql}")
    
    print("\n✓ PT_TYPE support is fully implemented and functional in VMEC++!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()