#!/usr/bin/env python3
"""Test symmetric mode - MUST pass"""

from vmecpp import run, VmecInput

# Test symmetric SOLOVEV case
input_data = VmecInput()
input_data.mgrid_file = "NONE"
input_data.lasym = False  # SYMMETRIC mode
input_data.nfp = 1
input_data.ncurr = 0
input_data.niter = 2
input_data.ns_array = [11]
input_data.ftol_array = [1e-12]

# Boundary coefficients - symmetric only
input_data.rbc = {(0, 0): 1.3, (1, 0): 0.3}
input_data.zbs = {(1, 0): 0.3}

# Pressure profile
input_data.am = [0.0, 0.33, 0.67, 1.0]
input_data.ai = [0.0, 0.33, 0.67, 1.0]
input_data.ac = [0.0, 0.0, 0.0, 0.0]

print("Running symmetric SOLOVEV test...")
try:
    result = run(input_data, verbose=True)
    print(f"SUCCESS: Symmetric test passed with ier={result.ier}")
    print(f"Beta = {result.beta_total:.6f}")
    print(f"Aspect ratio = {result.aspect:.6f}")
except Exception as e:
    print(f"FAILED: Symmetric test failed: {e}")
    exit(1)