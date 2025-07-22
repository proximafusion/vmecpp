#!/usr/bin/env python3
"""Test symmetric mode still works after changes"""
from vmecpp.cpp import _vmecpp as vmec

# Test symmetric Solovev
indata = vmec.VmecINDATAPyWrapper.from_file(
    "src/vmecpp/cpp/vmecpp/test_data/solovev.json"
)
assert not indata.lasym, "Must be symmetric"

# Run test
output = vmec.run(indata, verbose=False)

# Check convergence
assert output.wout.ier_flag == 0, "Symmetric must converge"
print("✓ Symmetric mode works")