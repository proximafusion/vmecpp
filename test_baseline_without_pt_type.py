#!/usr/bin/env python3
"""Test baseline convergence without PT_TYPE fields to isolate the issue."""

import sys
sys.path.insert(0, '/home/ert/code/vmecpp/build')
sys.path.insert(0, '/home/ert/code/vmecpp/src')

import _vmecpp
import json

print("Testing baseline Solovev convergence without PT_TYPE modifications...")

# Create a configuration without PT_TYPE fields - manually build simple input
indata = _vmecpp.VmecINDATAPyWrapper()

# Set all the basic properties manually to avoid PT_TYPE interference
indata.nfp = 1
indata.lasym = False
indata._set_mpol_ntor(6, 0)

# Basic arrays
indata.ns_array = [5, 11, 55]
indata.ftol_array = [1e-12, 1e-12, 1e-12] 
indata.niter_array = [1000, 2000, 2000]
indata.delt = 0.9
indata.tcon0 = 1.0
indata.aphi = [1.0]
indata.phiedge = 1.0
indata.nstep = 250

# Pressure profile
indata.pmass_type = "power_series"
indata.am = [0.125, -0.125]
indata.pres_scale = 1.0
indata.gamma = 0.0
indata.spres_ped = 1.0

# Current constraint
indata.ncurr = 0

# Iota profile
indata.piota_type = "power_series"
indata.ai = [1.0]

# Boundary
indata.lfreeb = False
indata.mgrid_file = "NONE"
indata.nvacskip = 1
indata.lforbal = False

# Check PT_TYPE default values
print(f"PT_TYPE fields in manually created input:")
print(f"  bcrit = {indata.bcrit}")
print(f"  pt_type = '{indata.pt_type}'")
print(f"  ph_type = '{indata.ph_type}'")

# Try modifying PT_TYPE to see if defaults are the issue
print(f"\nTrying with empty AT and AH arrays...")
indata.at = []
indata.ah = []

try:
    print("\n🔥 Running VMEC++ with manual input...")
    result = _vmecpp.run(indata)
    
    print(f"\n✅ SUCCESS! VMEC++ converged!")
    print(f"   Final residual: {result.fsqr:.3e}")
    print(f"   Iterations: {result.ier}")
    
except Exception as e:
    print(f"\n❌ FAILED: {str(e)}")
    
    # Try disabling PT_TYPE by setting to zero/empty
    print(f"\nTrying with PT_TYPE disabled...")
    indata.bcrit = 0.0
    indata.pt_type = ""
    indata.ph_type = ""
    indata.at = []
    indata.ah = []
    
    try:
        result = _vmecpp.run(indata)
        print(f"\n✅ SUCCESS after disabling PT_TYPE!")
        print(f"   Final residual: {result.fsqr:.3e}")
        print(f"   Iterations: {result.ier}")
        
    except Exception as e2:
        print(f"\n❌ Still FAILED even with PT_TYPE disabled: {str(e2)}")
        
        print(f"\nThe issue may be deeper than PT_TYPE fields...")
        sys.exit(1)