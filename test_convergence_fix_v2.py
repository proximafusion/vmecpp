#!/usr/bin/env python3
"""Test to find what's causing convergence failure"""

import vmecpp
import os

# Test Solovev convergence
print("Testing Solovev convergence...")

# Load the input
indata = vmecpp.VmecINDATA()
try:
    # Try JSON first
    if os.path.exists('src/vmecpp/cpp/vmecpp/test_data/input.solovev.json'):
        indata = vmecpp.VmecINDATA.from_file('src/vmecpp/cpp/vmecpp/test_data/input.solovev.json')
    else:
        # Try parsing text format
        print("JSON not found, trying to create from text...")
        # Read text file and set parameters manually
        indata.lasym = False  # Explicitly set to False
        indata.nfp = 1
        indata.mpol = 2
        indata.ntor = 0
        indata.ntheta = 32
        indata.nzeta = 1
        indata.ns_array = [31]
        indata.ftol_array = [1e-10]
        indata.niter_array = [1000]
        indata.phiedge = 0.16
        indata.ncurr = 0
        indata.pmass_type = "power_series"
        indata.am = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        indata.pres_scale = 20.0
        indata.gamma = 0.0
        indata.spres_ped = 1.0
        indata.piota_type = "power_series"
        indata.ai = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        indata.curtor = 0.0
        indata.delt = 0.5
        indata.tcon0 = 1.0
        indata.raxis_c = [1.0]
        indata.zaxis_s = [0.0]
        indata.rbc = [0.0, 1.0, 0.0, 0.0]  # R00=0, R10=1, R01=0, R11=0
        indata.zbs = [0.0, 0.0, 0.0, 1.0]  # Z00=0, Z10=0, Z01=0, Z11=1
        
except Exception as e:
    print(f"Error loading input: {e}")
    exit(1)

print(f"Input lasym value: {indata.lasym}")
print(f"Input type: {type(indata.lasym)}")

# Create and run VMEC++
print("\nCreating VMEC++ instance...")
try:
    vmec = vmecpp.Vmec(indata)
    print("VMEC++ created successfully")
    
    # Run it
    print("Running VMEC++...")
    result = vmec.run()
    
    if result.success:
        print(f"VMEC++ converged!")
        print(f"Final force residual: {result.fsqr:.6e}")
    else:
        print(f"VMEC++ failed to converge")
        print(f"Final force residual: {result.fsqr:.6e}")
        print(f"Number of iterations: {result.niter}")
        
except Exception as e:
    print(f"Error running VMEC++: {e}")
    import traceback
    traceback.print_exc()