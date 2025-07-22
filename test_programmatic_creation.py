#!/usr/bin/env python3
"""
Test VMEC++ by creating input programmatically instead of loading from JSON
"""

import numpy as np
import vmecpp

# Create a simple symmetric case programmatically
vmec_input = vmecpp.VmecInput()

# Basic parameters
vmec_input.lasym = False
vmec_input.nfp = 5
vmec_input.mpol = 4
vmec_input.ntor = 3

# Minimal runtime parameters
vmec_input.nstep = 1
vmec_input.niter_array = [5]
vmec_input.ns_array = [3]
vmec_input.ftol_array = [1e-8]

# Set required physics parameters
vmec_input.phiedge = 1.0
vmec_input.curtor = 0.0
vmec_input.pres_scale = 1.0
vmec_input.ncurr = 0

# Initialize axis arrays
vmec_input.raxis_c = np.array([1.0, 0.1, 0.0, 0.0])
vmec_input.zaxis_s = np.array([0.0, 0.0, 0.0, 0.0])

# Initialize boundary arrays with proper size
# For symmetric case, we only need rbc and zbs
vmec_input.rbc = np.zeros((vmec_input.mpol, 2*vmec_input.ntor + 1))
vmec_input.zbs = np.zeros((vmec_input.mpol, 2*vmec_input.ntor + 1))

# Set some boundary coefficients
# R(m=0,n=0) = 1.0 (major radius)
vmec_input.rbc[0, vmec_input.ntor] = 1.0
# R(m=1,n=0) = 0.3 (minor radius)
vmec_input.rbc[1, vmec_input.ntor] = 0.3
# Z(m=1,n=0) = 0.3 (elongation)
vmec_input.zbs[1, vmec_input.ntor] = 0.3

print(f"Created symmetric input: mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
print(f"rbc.shape={vmec_input.rbc.shape}")
print(f"zbs.shape={vmec_input.zbs.shape}")

try:
    output = vmecpp.run(vmec_input, verbose=False)
    print(f"✅ SUCCESS! Volume={output.volume_p:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Now test asymmetric case
print("\n" + "="*60 + "\n")

vmec_input_asym = vmecpp.VmecInput()

# Basic parameters
vmec_input_asym.lasym = True
vmec_input_asym.nfp = 1
vmec_input_asym.mpol = 3
vmec_input_asym.ntor = 2

# Minimal runtime parameters
vmec_input_asym.nstep = 1
vmec_input_asym.niter_array = [5]
vmec_input_asym.ns_array = [3]
vmec_input_asym.ftol_array = [1e-8]

# Set required physics parameters
vmec_input_asym.phiedge = 1.0
vmec_input_asym.curtor = 0.0
vmec_input_asym.pres_scale = 1.0
vmec_input_asym.ncurr = 0

# Initialize axis arrays
vmec_input_asym.raxis_c = np.array([1.0, 0.0, 0.0])
vmec_input_asym.zaxis_s = np.array([0.0, 0.0, 0.0])
vmec_input_asym.raxis_s = np.array([0.0, 0.0, 0.0])
vmec_input_asym.zaxis_c = np.array([0.0, 0.0, 0.0])

# Initialize boundary arrays with proper size
vmec_input_asym.rbc = np.zeros((vmec_input_asym.mpol, 2*vmec_input_asym.ntor + 1))
vmec_input_asym.zbs = np.zeros((vmec_input_asym.mpol, 2*vmec_input_asym.ntor + 1))
vmec_input_asym.rbs = np.zeros((vmec_input_asym.mpol, 2*vmec_input_asym.ntor + 1))
vmec_input_asym.zbc = np.zeros((vmec_input_asym.mpol, 2*vmec_input_asym.ntor + 1))

# Set some boundary coefficients
# R(m=0,n=0) = 1.0 (major radius)
vmec_input_asym.rbc[0, vmec_input_asym.ntor] = 1.0
# R(m=1,n=0) = 0.3 (minor radius)
vmec_input_asym.rbc[1, vmec_input_asym.ntor] = 0.3
# Z(m=1,n=0) = 0.3 (symmetric elongation)
vmec_input_asym.zbs[1, vmec_input_asym.ntor] = 0.3
# R(m=1,n=0) asymmetric = 0.05 (asymmetric shift)
vmec_input_asym.rbs[1, vmec_input_asym.ntor] = 0.05

print(f"Created asymmetric input: mpol={vmec_input_asym.mpol}, ntor={vmec_input_asym.ntor}")
print(f"rbc.shape={vmec_input_asym.rbc.shape}")
print(f"rbs.shape={vmec_input_asym.rbs.shape}")
print(f"zbs.shape={vmec_input_asym.zbs.shape}")
print(f"zbc.shape={vmec_input_asym.zbc.shape}")

try:
    output = vmecpp.run(vmec_input_asym, verbose=False)
    print(f"✅ SUCCESS! No azNorm=0 error!")
    print(f"   Volume={output.volume_p:.3f}")
    print(f"   Beta={output.beta:.6f}")
except Exception as e:
    error_msg = str(e)
    if "azNorm should never be 0.0" in error_msg:
        print(f"❌ FAILED: azNorm=0 error still present!")
    else:
        print(f"❌ Error: {error_msg}")