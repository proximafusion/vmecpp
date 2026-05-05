#!/usr/bin/env python3
"""
Debug the shape expectations
"""

import vmecpp
from vmecpp.cpp import _vmecpp

# Load a file
vmec_input = vmecpp.VmecInput.from_file("src/vmecpp/cpp/vmecpp/test_data/solovev.json")
print(f"VmecInput: mpol={vmec_input.mpol}, rbc.shape={vmec_input.rbc.shape}")

# Create a new C++ wrapper
cpp = _vmecpp.VmecINDATAPyWrapper()
print(f"\nNew wrapper before set: mpol={cpp.mpol}, rbc.shape={cpp.rbc.shape}")

# Set mpol/ntor
cpp._set_mpol_ntor(vmec_input.mpol, vmec_input.ntor)
print(f"After _set_mpol_ntor({vmec_input.mpol}, {vmec_input.ntor}): rbc.shape={cpp.rbc.shape}")

# What if we set other attributes first?
cpp2 = _vmecpp.VmecINDATAPyWrapper()
cpp2.lasym = vmec_input.lasym
cpp2.nfp = vmec_input.nfp
cpp2._set_mpol_ntor(vmec_input.mpol, vmec_input.ntor)
print(f"\nAfter setting lasym first: rbc.shape={cpp2.rbc.shape}")