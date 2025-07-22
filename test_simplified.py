#!/usr/bin/env python3
"""
Simplified test to check if the azNorm fix works
"""

import vmecpp
import numpy as np

# Test 1: Load a symmetric case and resize arrays
print("TEST 1: Symmetric case with array resizing")
filepath = "src/vmecpp/cpp/vmecpp/test_data/solovev.json"

vmec_input = vmecpp.VmecInput.from_file(filepath)
print(f"Loaded: lasym={vmec_input.lasym}, mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
print(f"Original rbc.shape={vmec_input.rbc.shape}")

# Resize arrays to mpol rows
if vmec_input.rbc.shape[0] == vmec_input.mpol + 1:
    print(f"Resizing arrays from {vmec_input.mpol+1} to {vmec_input.mpol} rows...")
    vmec_input.rbc = vmec_input.rbc[:vmec_input.mpol]
    vmec_input.zbs = vmec_input.zbs[:vmec_input.mpol]
    
print(f"New rbc.shape={vmec_input.rbc.shape}")

# Minimal run
vmec_input.nstep = 1
vmec_input.niter_array = [5]
vmec_input.ns_array = [3]
vmec_input.ftol_array = [1e-8]

try:
    output = vmecpp.run(vmec_input, verbose=False)
    print(f"✅ SUCCESS! Volume={output.volume_p:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Create a simple asymmetric case from scratch
print("\n" + "="*60)
print("TEST 2: Simple asymmetric case")

# Start with the symmetric case and modify it
vmec_asym = vmecpp.VmecInput.from_file(filepath)
vmec_asym.lasym = True
vmec_asym.nfp = 1  # Tokamak
vmec_asym.mpol = 3
vmec_asym.ntor = 2

# Resize arrays to correct size
vmec_asym.rbc = np.zeros((vmec_asym.mpol, 2*vmec_asym.ntor + 1))
vmec_asym.zbs = np.zeros((vmec_asym.mpol, 2*vmec_asym.ntor + 1))
vmec_asym.rbs = np.zeros((vmec_asym.mpol, 2*vmec_asym.ntor + 1))
vmec_asym.zbc = np.zeros((vmec_asym.mpol, 2*vmec_asym.ntor + 1))

# Set basic geometry
ntor_idx = vmec_asym.ntor  # Index for n=0 mode
vmec_asym.rbc[0, ntor_idx] = 1.0  # R00 = 1.0 (major radius)
vmec_asym.rbc[1, ntor_idx] = 0.3  # R10 = 0.3 (minor radius)
vmec_asym.zbs[1, ntor_idx] = 0.3  # Z10 = 0.3 (elongation)
vmec_asym.rbs[1, ntor_idx] = 0.05 # R10_asym = 0.05 (asymmetric shift)

# Resize axis arrays
vmec_asym.raxis_c = np.array([1.0, 0.0, 0.0])
vmec_asym.zaxis_s = np.array([0.0, 0.0, 0.0])
vmec_asym.raxis_s = np.array([0.0, 0.0, 0.0])
vmec_asym.zaxis_c = np.array([0.0, 0.0, 0.0])

# Minimal run
vmec_asym.nstep = 1
vmec_asym.niter_array = [5]
vmec_asym.ns_array = [3]
vmec_asym.ftol_array = [1e-8]

print(f"Created asymmetric input: mpol={vmec_asym.mpol}, ntor={vmec_asym.ntor}")
print(f"rbc.shape={vmec_asym.rbc.shape}")
print(f"rbs.shape={vmec_asym.rbs.shape}")

try:
    output = vmecpp.run(vmec_asym, verbose=False)
    print(f"✅ SUCCESS! No azNorm=0 error!")
    print(f"   Volume={output.volume_p:.3f}")
    print(f"   Beta={output.beta:.6f}")
    print(f"   The azNorm=0 fix is working!")
except Exception as e:
    error_msg = str(e)
    if "azNorm should never be 0.0" in error_msg:
        print(f"❌ FAILED: azNorm=0 error still present!")
        print(f"   The asymmetric Fourier transform is not being used correctly")
    else:
        print(f"❌ Error: {error_msg}")