#!/usr/bin/env python3
"""Test with raw VMEC format instead of JSON."""

import vmecpp
import tempfile
import os

print("=== Testing with Raw VMEC Input Format ===")

# Create a simple symmetric input file
symmetric_input = """&INDATA
  LASYM = F,
  NFP = 1,
  MPOL = 4,
  NTOR = 0,
  NS_ARRAY = 3, 5,
  NITER_ARRAY = 50, 100,
  FTOL_ARRAY = 1.0E-6, 1.0E-8,
  DELT = 0.9,
  PHIEDGE = 1.0,
  PMASS_TYPE = 'power_series',
  AM = 1.0, -1.0,
  PRES_SCALE = 0.1,
  GAMMA = 0.0,
  RAXIS_CC = 10.0,
  ZAXIS_CS = 0.0,
  RBC(0,0) = 10.0,
  RBC(1,0) = 1.0,
  ZBS(1,0) = 1.0,
/
"""

# Test symmetric case
print("1. Testing symmetric case...")
with tempfile.NamedTemporaryFile(mode='w', suffix='.vmec', delete=False) as f:
    f.write(symmetric_input)
    symmetric_file = f.name

try:
    result = vmecpp.run_from_file(symmetric_file)
    print("✓ Symmetric case successful!")
except Exception as e:
    print(f"✗ Symmetric case failed: {e}")
finally:
    os.unlink(symmetric_file)

# Create an asymmetric input file with minimal perturbation
asymmetric_input = """&INDATA
  LASYM = T,
  NFP = 1,
  MPOL = 4,
  NTOR = 0,
  NS_ARRAY = 3, 5,
  NITER_ARRAY = 50, 100,
  FTOL_ARRAY = 1.0E-6, 1.0E-8,
  DELT = 0.9,
  PHIEDGE = 1.0,
  PMASS_TYPE = 'power_series',
  AM = 1.0, -1.0,
  PRES_SCALE = 0.1,
  GAMMA = 0.0,
  RAXIS_CC = 10.0,
  RAXIS_CS = 0.0,
  ZAXIS_CS = 0.0,
  ZAXIS_CC = 0.0,
  RBC(0,0) = 10.0,
  RBC(1,0) = 1.0,
  ZBS(1,0) = 1.0,
  RBS(1,0) = 0.01,
  ZBC(1,0) = 0.01,
/
"""

# Test asymmetric case
print("\n2. Testing asymmetric case...")
with tempfile.NamedTemporaryFile(mode='w', suffix='.vmec', delete=False) as f:
    f.write(asymmetric_input)
    asymmetric_file = f.name

try:
    result = vmecpp.run_from_file(asymmetric_file)
    print("✓ Asymmetric case successful!")
except Exception as e:
    print(f"✗ Asymmetric case failed: {e}")
    print("This is the error we need to debug!")
finally:
    os.unlink(asymmetric_file)