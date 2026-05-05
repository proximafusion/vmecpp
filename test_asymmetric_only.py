#!/usr/bin/env python3
"""
Test only asymmetric cases to verify azNorm fix
"""

import vmecpp

# Test the asymmetric case
filepath = "src/vmecpp/cpp/vmecpp/test_data/input.up_down_asymmetric_tokamak.json"

print(f"Testing asymmetric case: {filepath}")

try:
    # Load
    vmec_input = vmecpp.VmecInput.from_file(filepath)
    print(f"Loaded: lasym={vmec_input.lasym}, mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
    print(f"rbc.shape={vmec_input.rbc.shape}")
    
    # Set arrays to expected size (mpol rows)
    if vmec_input.rbc.shape[0] == vmec_input.mpol + 1:
        print(f"Resizing arrays from {vmec_input.mpol+1} to {vmec_input.mpol} rows...")
        vmec_input.rbc = vmec_input.rbc[:vmec_input.mpol]
        vmec_input.zbs = vmec_input.zbs[:vmec_input.mpol]
        if vmec_input.rbs is not None:
            vmec_input.rbs = vmec_input.rbs[:vmec_input.mpol]
        if vmec_input.zbc is not None:
            vmec_input.zbc = vmec_input.zbc[:vmec_input.mpol]
    
    # Minimal run
    vmec_input.nstep = 1
    vmec_input.niter_array = [5]
    vmec_input.ns_array = [3]
    vmec_input.ftol_array = [1e-8]
    
    # Run
    print("Running VMEC++...")
    output = vmecpp.run(vmec_input, verbose=False)
    
    print(f"✅ SUCCESS! No azNorm=0 error!")
    print(f"   Volume={output.volume_p:.3f}")
    print(f"   Beta={output.beta:.6f}")
    
except Exception as e:
    error_msg = str(e)
    if "azNorm should never be 0.0" in error_msg:
        print(f"❌ FAILED: azNorm=0 error still present!")
    else:
        print(f"❌ Error: {error_msg}")