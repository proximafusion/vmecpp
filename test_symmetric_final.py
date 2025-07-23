#!/usr/bin/env python3
import vmecpp

# SOLOVEV input
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

print("Running symmetric SOLOVEV test...")
result = vmecpp.run(vmecpp.VmecInput.from_dict(input_dict))
print("SUCCESS: Symmetric test passed")