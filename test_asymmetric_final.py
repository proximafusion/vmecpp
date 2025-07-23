#!/usr/bin/env python3
import vmecpp

# Asymmetric SOLOVEV input with small RBS perturbation
input_dict = {
    "indata": {
        "lasym": True,  # ASYMMETRIC mode
        "nfp": 1,
        "ncurr": 0,
        "niter": 10,
        "ns_array": [11],
        "ftol_array": [1e-12],
        "mgrid_file": "NONE",
        "rbc": {"(0,0)": 1.3, "(1,0)": 0.3},
        "zbs": {"(1,0)": 0.3},
        "rbs": {"(1,0)": 0.001},  # Small asymmetric perturbation
        "am": [0.0, 0.33, 0.67, 1.0],
        "ai": [0.0, 0.33, 0.67, 1.0], 
        "ac": [0.0, 0.0, 0.0, 0.0]
    }
}

print("Running asymmetric SOLOVEV test...")
try:
    result = vmecpp.run(vmecpp.VmecInput(**input_dict["indata"]))
    print("SUCCESS: Asymmetric test passed")
except Exception as e:
    print(f"EXPECTED: Asymmetric test failed (goal to pass after fixes): {e}")