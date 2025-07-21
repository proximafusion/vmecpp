#!/usr/bin/env python3
import vmecpp
import sys

try:
    # Load the circular tokamak (symmetric) input
    vmec_input = vmecpp.VmecInput.from_file('../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.circular_tokamak')
    print("Loaded input file successfully")
    print(f"LASYM = {vmec_input.lasym}")
    print(f"MPOL = {vmec_input.mpol}")
    print(f"NTOR = {vmec_input.ntor}")
    
    # Run VMEC
    print("\nRunning VMEC...")
    output = vmecpp.run(vmec_input)
    print("✅ VMEC converged successfully!")
    print(f"MHD Energy = {output.wout.wb}")
    sys.exit(0)
except Exception as e:
    print(f"❌ VMEC failed: {e}")
    sys.exit(1)