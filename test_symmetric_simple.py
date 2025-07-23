#\!/usr/bin/env python3
"""Simple symmetric test using vmecpy.vmecpp interface"""

import vmecpy.vmecpp as vmecpp
import json

# Create simple SOLOVEV input
input_dict = {
    "indata": {
        "lasym": False,
        "nfp": 1,
        "ncurr": 0,
        "niter": 2,
        "ns_array": [11],
        "ftol_array": [1e-12],
        "mgrid_file": "NONE",
        "rbc": {"(0,0)": 1.3, "(1,0)": 0.3},
        "zbs": {"(1,0)": 0.3},
        "am": [0.0, 0.33, 0.67, 1.0],
        "ai": [0.0, 0.33, 0.67, 1.0], 
        "ac": [0.0, 0.0, 0.0, 0.0]
    }
}

print("Running symmetric SOLOVEV test with vmecpy.vmecpp...")
try:
    result = vmecpp.run_vmec(input_dict)
    print(f"SUCCESS: Symmetric test passed")
    if hasattr(result, 'beta_total'):
        print(f"Beta = {result.beta_total:.6f}")
    if hasattr(result, 'aspect'):
        print(f"Aspect ratio = {result.aspect:.6f}")
except Exception as e:
    print(f"FAILED: Symmetric test failed: {e}")
    exit(1)
EOF < /dev/null
