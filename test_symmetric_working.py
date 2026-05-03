#\!/usr/bin/env python3
import vmecpp

# Simple symmetric SOLOVEV input - minimal working case
input_data = {
    "lasym": False,
    "nfp": 1,
    "mpol": 3,
    "ntor": 0, 
    "ntheta": 24,
    "nzeta": 24,
    "ns_array": [11],
    "ftol_array": [1e-12],
    "niter_array": [100],
    "delt": 0.9,
    "tcon0": 1.0,
    "aphi": [1.0],
    "phiedge": 1.0,
    "nstep": 200,
    "pmass_type": "power_series",
    "am": [0.0, 0.33, 0.67, 1.0],
    "pres_scale": 1.0,
    "gamma": 0.0,
    "spres_ped": 1.0,
    "ncurr": 0,
    "piota_type": "power_series",
    "ai": [1.0],
    "lfreeb": False,
    "mgrid_file": "NONE",
    "nvacskip": 6,
    "lforbal": True,
    "raxis_c": [1.0],
    "zaxis_s": [0.0],
    "rbc": [
        {"n": 0, "m": 0, "value": 1.3},
        {"n": 0, "m": 1, "value": 0.3}
    ],
    "zbs": [
        {"n": 0, "m": 1, "value": 0.3}
    ]
}

print("Running symmetric SOLOVEV test...")
try:
    result = vmecpp.run(vmecpp.VmecInput(**input_data))
    print("SUCCESS: Symmetric test passed")
    print(f"Beta = {result.beta:.6f}")
except Exception as e:
    print(f"ERROR: Symmetric test failed: {e}")
EOF < /dev/null
