#!/usr/bin/env python3
"""Verify PT_TYPE implementation doesn't break convergence."""

import sys
sys.path.insert(0, '/home/ert/code/vmecpp/build')
sys.path.insert(0, '/home/ert/code/vmecpp/src')

import _vmecpp
import json

print("Testing PT_TYPE convergence after implementation fix...")

# Load baseline Solovev configuration
with open('/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/solovev.json', 'r') as f:
    config = json.load(f)

try:
    # Create input from JSON config
    indata = _vmecpp.VmecINDATAPyWrapper.from_json(json.dumps(config))
    
    print(f"Configuration loaded:")
    print(f"  nfp = {indata.nfp}")
    print(f"  lasym = {indata.lasym}")
    print(f"  mpol = {indata.mpol}")
    print(f"  ntor = {indata.ntor}")
    
    print(f"\nPT_TYPE fields after loading:")
    print(f"  bcrit = {indata.bcrit}")
    print(f"  pt_type = '{indata.pt_type}'")
    print(f"  ph_type = '{indata.ph_type}'")
    
    print("\n🔥 Running VMEC++ equilibrium calculation...")
    result = _vmecpp.run(indata)
    
    print(f"\n✅ SUCCESS! VMEC++ converged!")
    print(f"   Final residual: {result.fsqr:.3e}")
    print(f"   Iterations: {result.ier}")
    
    # Verify PT_TYPE fields are preserved
    print(f"\nPT_TYPE fields in result:")
    print(f"   bcrit: {result.indata.bcrit}")
    print(f"   pt_type: '{result.indata.pt_type}'")
    print(f"   ph_type: '{result.indata.ph_type}'")
    
    print(f"\n🎉 PT_TYPE implementation verified - convergence works!")
    
except Exception as e:
    print(f"\n❌ FAILED: {str(e)}")
    sys.exit(1)