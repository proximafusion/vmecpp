#!/usr/bin/env python3
import vmecpp
import numpy as np

# Test asymmetric case using working solovev.json as base
print("Testing asymmetric case using working JSON file as base...")

try:
    # Load from working JSON file
    input_data = vmecpp.VmecInput.from_file("/home/ert/code/vmecpp/examples/data/solovev.json")
    
    # Enable asymmetric mode
    input_data.lasym = True
    
    # Add small asymmetric perturbation - need to match array dimensions
    mpol, two_ntor_plus_one = input_data.rbc.shape
    
    # Add asymmetric boundary coefficients with same shape as symmetric ones
    input_data.rbs = np.zeros((mpol, two_ntor_plus_one))
    input_data.rbs[1, 0] = 0.001  # Small R perturbation at m=1, n=0
    
    input_data.zbc = np.zeros((mpol, two_ntor_plus_one))
    # Leave zbc as zeros for minimal perturbation
    
    # Add asymmetric axis coefficients
    ntor_plus_one = len(input_data.raxis_c)
    input_data.raxis_s = np.zeros(ntor_plus_one)
    input_data.zaxis_c = np.zeros(ntor_plus_one)
    
    print("Running asymmetric test from JSON file...")
    result = vmecpp.run(input_data)
    print("SUCCESS: Asymmetric test passed")
    print(f"Beta = {result.beta:.6f}")
    
except Exception as e:
    print(f"ERROR: Asymmetric test failed: {e}")
    import traceback
    traceback.print_exc()