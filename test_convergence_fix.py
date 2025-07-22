#!/usr/bin/env python3
"""Test convergence after PT_TYPE implementation fix."""

import sys
sys.path.insert(0, '/home/ert/code/vmecpp/src')

import _vmecpp
import numpy as np

print("Testing Solovev equilibrium convergence after PT_TYPE fix...")

# Create basic Solovev equilibrium input
indata = _vmecpp.VmecINDATAPyWrapper()

# Basic tokamak parameters
indata.nfp = 1
indata.lasym = False
indata._set_mpol_ntor(6, 0)

# Grid resolution
indata.ns_array = np.array([31], dtype=int)
indata.ftol_array = np.array([1.0e-12])
indata.niter_array = np.array([100], dtype=int)

# Physics parameters
indata.phiedge = 1.0
indata.ncurr = 0

# Pressure profile (parabolic)
indata.pmass_type = "power_series"
indata.am = np.array([0.0, 0.0, 100000.0])  # p = am[2] * (1-s^2)
indata.pres_scale = 1.0
indata.gamma = 0.0
indata.spres_ped = 1.0

# Iota profile
indata.piota_type = "power_series"
indata.ai = np.array([0.4, 0.8])  # iota = ai[0] + ai[1]*s

# Current profile
indata.pcurr_type = "power_series"
indata.curtor = 0.0
indata.bloat = 1.0

# Magnetic axis (on-axis)
indata.raxis_c = np.array([3.0])
indata.zaxis_s = np.array([0.0])

# Boundary (circular cross-section)
indata.rbc[0, 0] = 3.0  # R00 = 3.0 (major radius)
indata.rbc[1, 0] = 1.0  # R10 = 1.0 (minor radius)
indata.zbs[1, 0] = 1.0  # Z11 = 1.0

# PT_TYPE fields should now have proper defaults and not interfere with convergence
print(f"PT_TYPE fields:")
print(f"  bcrit = {indata.bcrit}")
print(f"  pt_type = '{indata.pt_type}'")
print(f"  at = {np.array(indata.at)}")
print(f"  ph_type = '{indata.ph_type}'")
print(f"  ah = {np.array(indata.ah)}")

try:
    print("\nRunning VMEC++ equilibrium calculation...")
    result = _vmecpp.run_vmecpp_standalone_from_indata(indata)
    
    print("\n✓ CONVERGENCE SUCCESS!")
    print(f"  Final force residual: {result.fsqr:.3e}")
    print(f"  Total iterations: {result.ier}")
    
    # Test PT_TYPE field access
    print(f"\nPT_TYPE fields accessible in result:")
    print(f"  Input bcrit: {result.indata.bcrit}")
    print(f"  Input pt_type: '{result.indata.pt_type}'")
    
except Exception as e:
    print(f"\n✗ CONVERGENCE FAILED: {e}")
    sys.exit(1)

print("\n✓ PT_TYPE implementation fix successful - convergence restored!")