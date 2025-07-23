#!/usr/bin/env python3
import vmecpp

# Minimal asymmetric test with small perturbation
input_data = {
    "lasym": True,  # ASYMMETRIC mode
    "mpol": 3,
    "ntor": 0,
    "delt": 0.9,
    "ncurr": 0,
    "nstep": 100,
    "ns_array": [11],
    "niter_array": [1000],
    "ftol_array": [1.0e-12],
    "am": [0.0, 0.5],
    "ai": [1.0],
    "raxis_c": [1.0],
    "zaxis_s": [0.0],
    "rbc": [
        {"n": 0, "m": 0, "value": 1.3},
        {"n": 0, "m": 1, "value": 0.3}
    ],
    "zbs": [
        {"n": 0, "m": 1, "value": 0.3}
    ],
    # Small asymmetric perturbation
    "rbs": [
        {"n": 0, "m": 1, "value": 0.001}
    ]
}

print("Running minimal asymmetric test...")
try:
    result = vmecpp.run(vmecpp.VmecInput(**input_data))
    print("SUCCESS: Asymmetric test passed")
    print(f"Beta = {result.beta:.6f}")
except Exception as e:
    print(f"ERROR: Asymmetric test failed: {e}")