#!/usr/bin/env python3
"""Test asymmetric equilibrium with simpler configuration."""

import vmecpp
import numpy as np

def test_simple_asymmetric():
    # Start from the asymmetric tokamak configuration
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Smaller grid for faster testing
    vmec_input.ns_array = np.array([9], dtype=np.int64)
    vmec_input.niter_array = np.array([100], dtype=np.int64)
    vmec_input.ftol_array = np.array([1e-4], dtype=np.float64)
    
    # Reduce the asymmetry to make convergence easier
    # Find and reduce the RBS coefficients
    for rbs_coeff in vmec_input.rbs:
        if rbs_coeff['m'] == 1 and rbs_coeff['n'] == 0:
            rbs_coeff['value'] = 0.05  # Reduce from 0.6 to 0.05
        elif rbs_coeff['m'] == 2 and rbs_coeff['n'] == 0:
            rbs_coeff['value'] = 0.01  # Reduce from 0.12 to 0.01
    
    # Reduce ZBS coefficient
    for zbs_coeff in vmec_input.zbs:
        if zbs_coeff['m'] == 1 and zbs_coeff['n'] == 0:
            zbs_coeff['value'] = 0.05  # Reduce from 0.6 to 0.05
    
    print("=== SIMPLE ASYMMETRIC TEST ===")
    print(f"Grid size: ns={vmec_input.ns_array[0]}")
    print(f"Symmetric boundary: R0={vmec_input.rbc[0]['value']}, a={vmec_input.rbc[1]['value']}")
    # Find the RBS(1,0) coefficient
    rbs_1_0 = next((c['value'] for c in vmec_input.rbs if c['m'] == 1 and c['n'] == 0), 0.0)
    print(f"Asymmetric perturbation: RBS(1,0)={rbs_1_0}")
    
    try:
        result = vmecpp.run(vmec_input, verbose=True)
        print(f"\nVMEC++ converged successfully!")
        print(f"Final force residuals: fsqr={result.fsqr:.6e}, fsqz={result.fsqz:.6e}")
        
    except RuntimeError as e:
        print(f"\nFailed with error: {e}")
        
        # Try with return_outputs_even_if_not_converged
        vmec_input.return_outputs_even_if_not_converged = True
        try:
            result = vmecpp.run(vmec_input, verbose=False)
            print(f"\nPartial results: fsqr={result.fsqr:.6e}, fsqz={result.fsqz:.6e}")
        except:
            pass

if __name__ == "__main__":
    test_simple_asymmetric()