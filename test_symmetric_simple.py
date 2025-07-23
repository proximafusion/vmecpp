#!/usr/bin/env python3
"""Simple symmetric test to verify it passes."""

import os
import tempfile
import subprocess

# Create symmetric VMEC input
symmetric_input = """&INDATA
  LASYM = F
  NFP = 1
  MPOL = 4
  NTOR = 0
  NTHETA = 18
  NZETA = 1
  NS_ARRAY = 3 5
  NITER_ARRAY = 50 100
  FTOL_ARRAY = 1.0E-6 1.0E-8
  DELT = 0.9
  PHIEDGE = 1.0
  NCURR = 0
  PMASS_TYPE = 'power_series'
  AM = 1.0 -1.0
  PRES_SCALE = 0.1
  GAMMA = 0.0
  SPRES_PED = 1.0
  BLOAT = 1.0
  TCON0 = 1.0
  LFORBAL = F
  RAXIS_CC = 10.0
  ZAXIS_CS = 0.0
  RBC(0,0) = 10.0
  RBC(1,0) = 1.0
  ZBS(1,0) = 1.0
/
"""

print("=== SYMMETRIC TEST ===")
print("Creating simple tokamak configuration...")

# Write to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.input', delete=False) as f:
    f.write(symmetric_input)
    input_file = f.name

print(f"Input file: {input_file}")

# Check if we can find a VMEC executable
exe_paths = [
    'build/vmec_standalone',
    'bazel-bin/vmecpp/vmec/vmec_standalone',
    '/home/ert/.cache/bazel/_bazel_ert/*/execroot/_main/bazel-bin/vmecpp/vmec/vmec_standalone'
]

vmec_exe = None
for path in exe_paths:
    if '*' in path:
        import glob
        matches = glob.glob(path)
        if matches:
            vmec_exe = matches[0]
            break
    elif os.path.exists(path):
        vmec_exe = path
        break

if vmec_exe:
    print(f"Found VMEC executable: {vmec_exe}")
    result = subprocess.run([vmec_exe, input_file], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ SYMMETRIC TEST PASSED")
        if "SUCCESSFUL" in result.stdout:
            print("  Converged successfully")
    else:
        print("✗ SYMMETRIC TEST FAILED")
        print("STDERR:", result.stderr[:500])
else:
    print("No VMEC executable found, trying Python module...")
    # Try Python module
    try:
        import vmecpp
        # Note: vmecpp Python module has different API
        print("✓ Python module imported")
    except ImportError:
        print("✗ Python module not available")

os.unlink(input_file)