#!/usr/bin/env python3
"""
Test VMEC++ with JSON preprocessing to fix array size mismatch
"""

import json
import vmecpp
from pathlib import Path
import tempfile

def preprocess_json_for_cpp(json_path):
    """Preprocess JSON to ensure arrays have mpol rows instead of mpol+1"""
    with open(json_path) as f:
        data = json.load(f)
    
    mpol = data.get("mpol", 0)
    ntor = data.get("ntor", 0)
    
    # Convert boundary coefficients from sparse to dense format with correct size
    for field in ["rbc", "zbs", "rbs", "zbc"]:
        if field in data and isinstance(data[field], list):
            # Create dense array with mpol rows (not mpol+1)
            dense = [[0.0] * (2*ntor + 1) for _ in range(mpol)]
            
            # Fill from sparse representation
            for entry in data[field]:
                m = entry["m"]
                n = entry["n"]
                value = entry["value"]
                if m < mpol:  # Only include modes up to mpol-1
                    dense[m][n + ntor] = value
            
            # Convert back to sparse for JSON
            sparse = []
            for m in range(mpol):
                for n in range(-ntor, ntor+1):
                    val = dense[m][n + ntor]
                    if val != 0.0:
                        sparse.append({"m": m, "n": n, "value": val})
            
            data[field] = sparse
    
    return data

# Test files
test_files = [
    "src/vmecpp/cpp/vmecpp/test_data/solovev.json",
    "src/vmecpp/cpp/vmecpp/test_data/input.up_down_asymmetric_tokamak.json",
]

for filepath in test_files:
    print(f"\nTesting: {filepath}")
    
    try:
        # Preprocess JSON
        data = preprocess_json_for_cpp(filepath)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            # Load with C++ loader
            vmec_input = vmecpp.VmecInput.from_file(temp_path)
            print(f"✅ Loaded successfully!")
            print(f"   lasym={vmec_input.lasym}, mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
            print(f"   rbc.shape={vmec_input.rbc.shape}")
            
            # Quick run
            vmec_input.nstep = 1
            vmec_input.niter_array = [5]
            vmec_input.ns_array = [3]
            vmec_input.ftol_array = [1e-8]
            
            output = vmecpp.run(vmec_input, verbose=False)
            print(f"✅ Run successful! Volume={output.volume_p:.3f}")
            
        finally:
            Path(temp_path).unlink()
            
    except Exception as e:
        print(f"❌ Error: {e}")