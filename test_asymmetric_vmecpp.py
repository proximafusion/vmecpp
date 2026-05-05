#!/usr/bin/env python3
"""Test asymmetric mode with minimal perturbation."""

import tempfile
import os
import subprocess

print("=== Testing Asymmetric Mode ===")

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
print("Testing asymmetric case...")
with tempfile.NamedTemporaryFile(mode='w', suffix='.vmec', delete=False) as f:
    f.write(asymmetric_input)
    asymmetric_file = f.name

try:
    # Run VMEC++ on asymmetric case
    result = subprocess.run(
        ["./build/xvmec", asymmetric_file],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("✓ Asymmetric case PASSED!")
        # Look for key output
        if "MHD Energy" in result.stdout:
            for line in result.stdout.split('\n'):
                if "MHD Energy" in line or "BETA" in line:
                    print(f"  {line.strip()}")
    else:
        print("✗ Asymmetric case FAILED!")
        print("STDOUT:", result.stdout[-500:] if result.stdout else "No stdout")
        print("STDERR:", result.stderr[-500:] if result.stderr else "No stderr")
        # Check for specific errors
        if "JACOBIAN CHANGED SIGN" in result.stdout:
            print("  ➤ Jacobian sign error detected")
        
except subprocess.TimeoutExpired:
    print("✗ Asymmetric case TIMEOUT!")
except Exception as e:
    print(f"✗ Asymmetric case ERROR: {e}")
finally:
    os.unlink(asymmetric_file)