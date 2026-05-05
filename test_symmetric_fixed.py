#!/usr/bin/env python3
import vmecpp
import numpy as np

# Fixed symmetric test with correct array dimensions
print("Creating fixed symmetric test...")

try:
    # Use VmecInput.default() to get base with all required fields
    input_data = vmecpp.VmecInput.default()
    
    # Override key fields for symmetric test
    input_data.lasym = False
    input_data.mpol = 3  # Requires arrays of size mpol+1 = 4
    input_data.ntor = 0  # Requires arrays of size 2*ntor+1 = 1
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
    input_data.raxis_c = np.array([1.0])  # Size ntor+1 = 1
    input_data.zaxis_s = np.array([0.0])  # Size ntor+1 = 1
    
    # Set boundary coefficients with correct dimensions
    # Shape must be (mpol+1, 2*ntor+1) = (4, 1)
    input_data.rbc = np.zeros((4, 1))  # Shape (mpol+1, 2*ntor+1)
    input_data.rbc[0, 0] = 1.3   # R00 (major radius)
    input_data.rbc[1, 0] = 0.3   # R10 (ellipticity)
    
    input_data.zbs = np.zeros((4, 1))  # Shape (mpol+1, 2*ntor+1)
    input_data.zbs[1, 0] = 0.3   # Z10 (ellipticity)
    
    # Ensure asymmetric fields are None for symmetric case
    input_data.rbs = None
    input_data.zbc = None
    input_data.raxis_s = None
    input_data.zaxis_c = None
    
    print("Running fixed symmetric test...")
    result = vmecpp.run(input_data)
    print("SUCCESS: Symmetric test passed")
    print(f"Beta = {result.beta:.6f}")
    
except Exception as e:
    print(f"ERROR: Symmetric test failed: {e}")
    import traceback
    traceback.print_exc()