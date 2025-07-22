#!/usr/bin/env python3
"""
Direct test to validate the azNorm=0 fix for asymmetric equilibria
"""

import subprocess
import json
import os
from pathlib import Path

# Test the asymmetric input file that was failing before
test_input = "/home/ert/code/vmecpp/benchmark_vmec/input.SOLOVEV_asym"

# Check if the file exists
if not Path(test_input).exists():
    print(f"Creating test asymmetric input file...")
    # Create a simple asymmetric test case
    test_input = "test_asymmetric_input.json"
    test_data = {
        "lasym": True,
        "nfp": 1,
        "mpol": 4,
        "ntor": 0,
        "ntheta": 16,
        "nzeta": 4,
        "phiedge": 0.1,
        "curtor": 0.0,
        "ncurr": 0,
        "pmass_type": "two_power",
        "am": [1.0, 1.0, 0.0],
        "am_aux_s": [0.0],
        "am_aux_f": [1.0],
        "pres_scale": 1.0,
        "gamma": 0.0,
        "spres_ped": 1.0,
        "piota_type": "power_series",
        "ai": [0.0],
        "ai_aux_s": [0.0],
        "ai_aux_f": [1.0],
        "pcurr_type": "power_series",
        "ac": [0.0],
        "ac_aux_s": [0.0],
        "ac_aux_f": [1.0],
        "bloat": 1.0,
        "lfreeb": False,
        "mgrid_file": "",
        "extcur": [0.0],
        "nvacskip": 0,
        "nstep": 10,
        "niter_array": [10],
        "ns_array": [3],
        "ftol_array": [1e-12],
        "aphi": [1.0],
        "delt": 0.9,
        "tcon0": 1.0,
        "lforbal": False,
        "raxis_c": [1.0],
        "zaxis_s": [0.0],
        "raxis_s": [0.0],  # Asymmetric axis
        "zaxis_c": [0.0],  # Asymmetric axis
        "rbc": [
            {"m": 0, "n": 0, "value": 1.0},
            {"m": 1, "n": 0, "value": 0.3}
        ],
        "zbs": [
            {"m": 1, "n": 0, "value": 0.3}
        ],
        "rbs": [
            {"m": 1, "n": 0, "value": 0.05}  # Asymmetric component
        ],
        "zbc": []
    }
    
    with open(test_input, "w") as f:
        json.dump(test_data, f, indent=2)

print(f"Testing VMEC++ with asymmetric input: {test_input}")

# Run VMEC++ with the test input
try:
    # Use the command line vmecpp if available
    result = subprocess.run(
        ["python", "-m", "vmecpp", test_input],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    print("\n" + "="*60)
    print("VMEC++ Output:")
    print("="*60)
    print(result.stdout)
    
    if result.stderr:
        print("\nErrors/Warnings:")
        print(result.stderr)
    
    # Check for azNorm error
    if "azNorm should never be 0.0" in result.stdout or "azNorm should never be 0.0" in result.stderr:
        print("\n❌ FAILED: azNorm=0 error is still present!")
        print("The asymmetric Fourier transform fix is not working correctly.")
    elif result.returncode != 0:
        print(f"\n❌ VMEC++ exited with error code: {result.returncode}")
    else:
        print("\n✅ SUCCESS: No azNorm=0 error detected!")
        print("The asymmetric equilibrium ran successfully.")
        
        # Try to check if output file was created
        if Path("wout_test_asymmetric_input.nc").exists():
            print("Output file created successfully.")
        
except subprocess.TimeoutExpired:
    print("\n❌ VMEC++ timed out after 30 seconds")
except Exception as e:
    print(f"\n❌ Error running VMEC++: {e}")

# Clean up
if Path("test_asymmetric_input.json").exists() and test_input == "test_asymmetric_input.json":
    os.unlink("test_asymmetric_input.json")