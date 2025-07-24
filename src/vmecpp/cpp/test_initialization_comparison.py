#!/usr/bin/env python3
"""
Test to compare initialization between symmetric and asymmetric cases
to find the root cause of dRdTheta being zero in asymmetric mode.
"""

from vmecpp.cpp._vmecpp import Solver
import numpy as np
import json

def run_single_iteration(config_dict, name):
    """Run exactly one iteration and capture state"""
    print(f"\n=== {name} ===")
    print(f"LASYM: {config_dict['LASYM']}")
    
    solver = Solver(config_dict, verbose=True)
    
    # Check initial state before any iteration
    print(f"\nBefore iteration:")
    hs = solver.handover_storage()
    
    # Check R and Z arrays
    r_check = np.any(hs.R != 0)
    z_check = np.any(hs.Z != 0)
    print(f"R non-zero: {r_check}, Z non-zero: {z_check}")
    
    # Check derivatives
    drdu_check = np.any(hs.dRdTheta != 0)
    dzdu_check = np.any(hs.dZdTheta != 0)
    print(f"dRdTheta non-zero: {drdu_check}, dZdTheta non-zero: {dzdu_check}")
    
    # Run one iteration
    try:
        result = solver.iterate(1)
        print(f"\nAfter 1 iteration:")
        print(f"Status: {result}")
        
        hs = solver.handover_storage()
        
        # Check derivatives again
        drdu_check = np.any(hs.dRdTheta != 0)
        dzdu_check = np.any(hs.dZdTheta != 0)
        print(f"dRdTheta non-zero: {drdu_check}, dZdTheta non-zero: {dzdu_check}")
        
        if not drdu_check or not dzdu_check:
            print("ERROR: Derivatives are zero!")
            # Print some values
            print(f"dRdTheta sample: {hs.dRdTheta.flatten()[:10]}")
            print(f"dZdTheta sample: {hs.dZdTheta.flatten()[:10]}")
            
    except Exception as e:
        print(f"ERROR during iteration: {e}")
        import traceback
        traceback.print_exc()

# Simple symmetric case
symmetric_config = {
    "LASYM": False,
    "NFP": 1,
    "MPOL": 4,
    "NTOR": 0,
    "NTHETA": 18,
    "NZETA": 1,
    "NS_ARRAY": [3],
    "NITER_ARRAY": [10],
    "FTOL_ARRAY": [1.0e-6],
    "DELT": 0.9,
    "PHIEDGE": 1.0,
    "NCURR": 0,
    "PMASS_TYPE": "power_series",
    "AM": [1.0, -1.0],
    "PRES_SCALE": 0.1,
    "GAMMA": 0.0,
    "SPRES_PED": 1.0,
    "BLOAT": 1.0,
    "TCON0": 1.0,
    "LFORBAL": False,
    "RAXIS_CC": [10.0],
    "ZAXIS_CS": [0.0],
    "RBC": {"0_0": 10.0, "1_0": 1.0},
    "ZBS": {"1_0": 1.0}
}

# Asymmetric version of the same
asymmetric_config = symmetric_config.copy()
asymmetric_config["LASYM"] = True
asymmetric_config["RAXIS_CS"] = [0.0]  # Required for asymmetric
asymmetric_config["ZAXIS_CC"] = [0.0]  # Required for asymmetric
asymmetric_config["RBS"] = {}  # Required for asymmetric
asymmetric_config["ZBC"] = {}  # Required for asymmetric

# Run tests
run_single_iteration(symmetric_config, "Symmetric")
run_single_iteration(asymmetric_config, "Asymmetric")