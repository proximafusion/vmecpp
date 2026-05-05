#!/usr/bin/env python3
import vmecpp
import numpy as np

# Complete asymmetric test with all required VmecInput fields
print("Creating complete asymmetric test...")

try:
    # Use VmecInput.default() to get base with all required fields
    input_data = vmecpp.VmecInput.default()
    
    # Override key fields for asymmetric test
    input_data.lasym = True  # Enable asymmetric mode
    input_data.mpol = 3
    input_data.ntor = 0
    input_data.nfp = 1
    input_data.ntheta = 24
    input_data.nzeta = 24
    input_data.delt = 0.9
    input_data.ncurr = 0
    input_data.nstep = 100
    input_data.ns_array = np.array([11])
    input_data.niter_array = np.array([1000])
    input_data.ftol_array = np.array([1.0e-12])
    
    # Simple tokamak-like equilibrium
    input_data.am = np.array([0.0, 0.5])
    input_data.ai = np.array([1.0])
    input_data.raxis_c = np.array([1.0])
    input_data.zaxis_s = np.array([0.0])
    
    # Set symmetric boundary coefficients 
    input_data.rbc = np.zeros((3, 1))  # Shape (mpol, 2*ntor+1)
    input_data.rbc[0, 0] = 1.3   # R00
    input_data.rbc[1, 0] = 0.3   # R10
    
    input_data.zbs = np.zeros((3, 1))  # Shape (mpol, 2*ntor+1)
    input_data.zbs[1, 0] = 0.3   # Z10
    
    # Add small asymmetric perturbation
    input_data.rbs = np.zeros((3, 1))  # Shape (mpol, 2*ntor+1)
    input_data.rbs[1, 0] = 0.001  # Small R perturbation
    
    input_data.zbc = np.zeros((3, 1))  # Shape (mpol, 2*ntor+1)
    # Leave zbc as zeros for minimal perturbation
    
    # Keep axis symmetric (most conservative)
    input_data.raxis_s = None
    input_data.zaxis_c = None
    
    print("Running complete asymmetric test...")
    result = vmecpp.run(input_data)
    print("SUCCESS: Asymmetric test passed")
    print(f"Beta = {result.beta:.6f}")
    
except Exception as e:
    print(f"ERROR: Asymmetric test failed: {e}")
    import traceback
    traceback.print_exc()