#!/usr/bin/env python3
"""
Test a single file to debug issues
"""

import vmecpp

# Test the solovev case which should be stable
filepath = "src/vmecpp/cpp/vmecpp/test_data/solovev.json"

print(f"Testing: {filepath}")

try:
    # Load
    vmec_input = vmecpp.VmecInput.from_file(filepath)
    print(f"Loaded: lasym={vmec_input.lasym}, mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
    print(f"rbc.shape={vmec_input.rbc.shape}")
    
    # Minimal run
    vmec_input.nstep = 1
    vmec_input.niter_array = [5]
    vmec_input.ns_array = [3]
    vmec_input.ftol_array = [1e-8]
    
    print(f"nstep={vmec_input.nstep}")
    print(f"niter_array={vmec_input.niter_array}")
    print(f"ns_array={vmec_input.ns_array}")
    print(f"ftol_array={vmec_input.ftol_array}")
    
    # Run
    print("Running VMEC++...")
    output = vmecpp.run(vmec_input, verbose=False)
    
    print(f"✅ SUCCESS! Volume={output.volume_p:.3f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()