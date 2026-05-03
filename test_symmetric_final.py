#!/usr/bin/env python3
import vmecpp

# SOLOVEV input
input_dict = {
    "indata": {
        "lasym": False,
        "nfp": 1,
        "mpol": 3,
        "ntor": 2,
        "ntheta": 24,
        "nzeta": 24,
        "ncurr": 0,
        "niter": 2,
        "niter_array": [100],
        "ns_array": [11],
        "ftol_array": [1e-12],
        "mgrid_file": "NONE",
        "phiedge": 1.0,
        "pmass_type": "power_series",
        "am_aux_s": [0.0],
        "am_aux_f": [0.0],
        "pres_scale": 1.0,
        "gamma": 0.0,
        "spres_ped": 1.0,
        "piota_type": "power_series",
        "ai_aux_s": [0.0],
        "ai_aux_f": [0.0],
        "pcurr_type": "power_series",
        "ac_aux_s": [0.0],
        "ac_aux_f": [0.0],
        "curtor": 0.0,
        "bloat": 1.0,
        "lfreeb": False,
        "extcur": [0.0],
        "nvacskip": 6,
        "nstep": 200,
        "delt": 1.0,
        "tcon0": 1.0,
        "lforbal": True,
        "raxis_c": [1.0, 0.0],
        "zaxis_s": [0.0, 0.0],
        "rbc": [[1.3, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        "zbs": [[0.0, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        "am": [0.0, 0.33, 0.67, 1.0],
        "ai": [0.0, 0.33, 0.67, 1.0], 
        "ac": [0.0, 0.0, 0.0, 0.0],
        "aphi": [0.0, 0.0, 0.0, 0.0]
    }
}

print("Running symmetric SOLOVEV test...")
result = vmecpp.run(vmecpp.VmecInput(**input_dict["indata"]))
print("SUCCESS: Symmetric test passed")