#!/usr/bin/env python3
import vmecpp
import numpy as np

# Fixed asymmetric test with proper array dimensions and axis settings
print("Creating fixed asymmetric test...")

try:
    # Use VmecInput.default() to get base with all required fields
    input_data = vmecpp.VmecInput.default()
    
    # Override key fields for asymmetric test
    input_data.lasym = True  # Enable asymmetric mode
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
    
    # Set symmetric boundary coefficients with correct dimensions
    # Shape must be (mpol+1, 2*ntor+1) = (4, 1)
    input_data.rbc = np.zeros((4, 1))  # Shape (mpol+1, 2*ntor+1)
    input_data.rbc[0, 0] = 1.3   # R00 (major radius)
    input_data.rbc[1, 0] = 0.3   # R10 (ellipticity)
    
    input_data.zbs = np.zeros((4, 1))  # Shape (mpol+1, 2*ntor+1)
    input_data.zbs[1, 0] = 0.3   # Z10 (ellipticity)
    
    # Add small asymmetric perturbation with correct dimensions
    input_data.rbs = np.zeros((4, 1))  # Shape (mpol+1, 2*ntor+1)
    input_data.rbs[1, 0] = 0.001  # Small R perturbation
    
    input_data.zbc = np.zeros((4, 1))  # Shape (mpol+1, 2*ntor+1)
    # Leave zbc as zeros for minimal perturbation
    
    # Set asymmetric axis coefficients properly (required when lasym=True)
    input_data.raxis_s = np.zeros(1)  # Size ntor+1 = 1
    input_data.zaxis_c = np.zeros(1)  # Size ntor+1 = 1
    
    print("Running fixed asymmetric test...")
    result = vmecpp.run(input_data)
    print("SUCCESS: Asymmetric test passed")
    print(f"Beta = {result.beta:.6f}")
    
except Exception as e:
    print(f"ERROR: Asymmetric test failed: {e}")
    import traceback
    traceback.print_exc()