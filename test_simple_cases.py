#!/usr/bin/env python3
"""Simple test of symmetric and asymmetric cases using vmecpp module."""

import sys
sys.path.insert(0, 'src/vmecpp/python')

import vmecpp

print("=== Testing VMEC++ Symmetric and Asymmetric Cases ===\n")

# Test 1: Symmetric case
print("1. Testing SYMMETRIC case...")
try:
    # Simple tokamak configuration
    config_sym = vmecpp.VmecInput()
    config_sym.lasym = False
    config_sym.nfp = 1
    config_sym.mpol = 4
    config_sym.ntor = 0
    config_sym.ntheta = 18
    config_sym.nzeta = 1
    config_sym.ns_array = [3, 5]
    config_sym.niter_array = [50, 100]
    config_sym.ftol_array = [1e-6, 1e-8]
    config_sym.delt = 0.9
    config_sym.phiedge = 1.0
    config_sym.ncurr = 0
    config_sym.lfreeb = False
    
    # Pressure profile
    config_sym.pmass_type = "power_series"
    config_sym.am = [1.0, -1.0]
    config_sym.am_aux_s = []
    config_sym.am_aux_f = []
    config_sym.pres_scale = 0.1
    config_sym.gamma = 0.0
    config_sym.spres_ped = 1.0
    
    # Iota profile
    config_sym.piota_type = "power_series"
    config_sym.ai = []
    config_sym.ai_aux_s = []
    config_sym.ai_aux_f = []
    
    # Current profile
    config_sym.pcurr_type = "power_series"
    config_sym.ac = []
    config_sym.ac_aux_s = []
    config_sym.ac_aux_f = []
    config_sym.curtor = 0.0
    
    # Boundary
    config_sym.raxis_c = [10.0]
    config_sym.zaxis_s = [0.0]
    config_sym.rbc = [
        {"m": 0, "n": 0, "value": 10.0},
        {"m": 1, "n": 0, "value": 1.0}
    ]
    config_sym.zbs = [
        {"m": 1, "n": 0, "value": 1.0}
    ]
    
    # Other required fields
    config_sym.bloat = 1.0
    config_sym.mgrid_file = ""
    config_sym.extcur = []
    config_sym.nvacskip = 0
    config_sym.nstep = 200
    config_sym.aphi = []
    config_sym.tcon0 = 1.0
    config_sym.lforbal = False
    
    result = vmecpp.run(config_sym)
    print("✓ Symmetric case PASSED!")
    print(f"  Beta: {result.beta:.6f}")
    print(f"  Energy: {result.w_mhd:.6f}")
    print(f"  Aspect ratio: {result.aspect_ratio:.3f}")
    
except Exception as e:
    print(f"✗ Symmetric case FAILED: {e}")

# Test 2: Asymmetric case  
print("\n2. Testing ASYMMETRIC case...")
try:
    # Copy symmetric config and add asymmetric perturbation
    config_asym = vmecpp.VmecInput()
    config_asym.lasym = True  # Enable asymmetric mode
    config_asym.nfp = 1
    config_asym.mpol = 4
    config_asym.ntor = 0
    config_asym.ntheta = 36  # Double for asymmetric
    config_asym.nzeta = 1
    config_asym.ns_array = [3, 5]
    config_asym.niter_array = [50, 100]
    config_asym.ftol_array = [1e-6, 1e-8]
    config_asym.delt = 0.9
    config_asym.phiedge = 1.0
    config_asym.ncurr = 0
    config_asym.lfreeb = False
    
    # Pressure profile
    config_asym.pmass_type = "power_series"
    config_asym.am = [1.0, -1.0]
    config_asym.am_aux_s = []
    config_asym.am_aux_f = []
    config_asym.pres_scale = 0.1
    config_asym.gamma = 0.0
    config_asym.spres_ped = 1.0
    
    # Iota profile
    config_asym.piota_type = "power_series"
    config_asym.ai = []
    config_asym.ai_aux_s = []
    config_asym.ai_aux_f = []
    
    # Current profile
    config_asym.pcurr_type = "power_series"
    config_asym.ac = []
    config_asym.ac_aux_s = []
    config_asym.ac_aux_f = []
    config_asym.curtor = 0.0
    
    # Symmetric boundary
    config_asym.raxis_c = [10.0]
    config_asym.zaxis_s = [0.0]
    config_asym.rbc = [
        {"m": 0, "n": 0, "value": 10.0},
        {"m": 1, "n": 0, "value": 1.0}
    ]
    config_asym.zbs = [
        {"m": 1, "n": 0, "value": 1.0}
    ]
    
    # Asymmetric axis arrays (required for lasym=True)
    config_asym.raxis_s = [0.0]
    config_asym.zaxis_c = [0.0]
    
    # Add small asymmetric perturbation
    config_asym.rbs = [
        {"m": 1, "n": 0, "value": 0.01}  # 1% perturbation
    ]
    config_asym.zbc = [
        {"m": 1, "n": 0, "value": 0.01}
    ]
    
    # Other required fields
    config_asym.bloat = 1.0
    config_asym.mgrid_file = ""
    config_asym.extcur = []
    config_asym.nvacskip = 0
    config_asym.nstep = 200
    config_asym.aphi = []
    config_asym.tcon0 = 1.0
    config_asym.lforbal = False
    
    result = vmecpp.run(config_asym)
    print("✓ Asymmetric case PASSED!")
    print(f"  Beta: {result.beta:.6f}")
    print(f"  Energy: {result.w_mhd:.6f}")
    print(f"  Aspect ratio: {result.aspect_ratio:.3f}")
    
except Exception as e:
    print(f"✗ Asymmetric case FAILED: {e}")
    if "JACOBIAN" in str(e):
        print("  ➤ Jacobian sign error detected - transform algorithm issue")

print("\n=== Test Summary ===")
print("Symmetric mode should PASS (requirement)")
print("Asymmetric mode goal is to PASS after fixes")