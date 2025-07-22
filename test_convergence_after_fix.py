#!/usr/bin/env python3
"""Test if the buffer overflow fix resolves convergence"""

import sys
sys.path.insert(0, '.')
import vmecpp
import os

print("Testing convergence after fixing buffer overflow in symrzl_geometry...")

# Create simple symmetric input (Solovev)
indata = vmecpp.VmecINDATA()
indata.lasym = False  # Symmetric case
indata.nfp = 1
indata.mpol = 3
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
indata.rbc = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # R00=0, R10=1
indata.zbs = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # Z00=0, Z11=1

print(f"Created input with lasym={indata.lasym}")

# Run VMEC++
try:
    vmec = vmecpp.Vmec(indata)
    print("\nRunning VMEC++...")
    result = vmec.run()
    
    if result.success:
        print(f"\n✅ SUCCESS! VMEC++ converged!")
        print(f"Final force residual: {result.fsqr:.6e}")
        print(f"Number of iterations: {result.niter}")
        print("\nThe buffer overflow fix appears to have resolved the convergence issue!")
    else:
        print(f"\n❌ FAILED: VMEC++ did not converge")
        print(f"Final force residual: {result.fsqr:.6e}")
        print(f"Number of iterations: {result.niter}")
        print("\nThe fix may not be complete - further investigation needed.")
        
except Exception as e:
    print(f"\n❌ ERROR running VMEC++: {e}")
    import traceback
    traceback.print_exc()