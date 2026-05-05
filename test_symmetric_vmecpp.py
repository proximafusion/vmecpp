#!/usr/bin/env python3
"""Test symmetric mode to ensure it still works."""

import tempfile
import os
import subprocess

print("=== Testing Symmetric Mode ===")

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
print("Testing symmetric case...")
with tempfile.NamedTemporaryFile(mode='w', suffix='.vmec', delete=False) as f:
    f.write(symmetric_input)
    symmetric_file = f.name

try:
    # Run VMEC++ on symmetric case
    result = subprocess.run(
        ["./build/xvmec", symmetric_file],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("✓ Symmetric case PASSED!")
        # Look for key output
        if "MHD Energy" in result.stdout:
            for line in result.stdout.split('\n'):
                if "MHD Energy" in line or "BETA" in line:
                    print(f"  {line.strip()}")
    else:
        print("✗ Symmetric case FAILED!")
        print("STDOUT:", result.stdout[-500:] if result.stdout else "No stdout")
        print("STDERR:", result.stderr[-500:] if result.stderr else "No stderr")
        
except subprocess.TimeoutExpired:
    print("✗ Symmetric case TIMEOUT!")
except Exception as e:
    print(f"✗ Symmetric case ERROR: {e}")
finally:
    os.unlink(symmetric_file)