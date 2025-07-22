#!/usr/bin/env python3
"""Test asymmetric equilibria with the fixed theta-zeta grid handling."""

import vmecpp
import numpy as np

def test_simple_asymmetric():
    """Test simple asymmetric tokamak."""
    print("Testing simple asymmetric tokamak...")
    
    # Create simple asymmetric input
    vmec_input = vmecpp.VmecInput(
        mgrid="none",
        delt=0.1,
        ftol_array=[1e-6],
        niter=10000,
        ns_array=[3],
        nstep=200,
        nvacskip=1,
        gamma=0.0,
        phiedge=1.0,
        curtor=0.0,
        mpol=4,
        ntor=0,
        nfp=1,
        lasym=True,  # Asymmetric!
        rbc=np.zeros((5, 1)),
        zbs=np.zeros((5, 1)),
        rbs=np.zeros((5, 1)),  # Asymmetric arrays
        zbc=np.zeros((5, 1)),
        am=[1.0],
        ncurr=0,
        spres_ped=1.0,
        pmass_type="two_power",
        pcurr_type="sum_atan",
        pres_scale=0.0,
        raxis=np.array([1.0]),
        zaxis=np.array([0.0]),
        raxis_s=np.array([0.0]),  # Asymmetric axis
        zaxis_c=np.array([0.0]),
    )
    
    # Set up simple tokamak with asymmetry
    vmec_input.rbc[0, 0] = 1.0   # Major radius
    vmec_input.rbc[1, 0] = 0.3   # Minor radius
    vmec_input.zbs[1, 0] = 0.3   # Height
    vmec_input.rbs[1, 0] = 0.001 # Small asymmetric perturbation
    
    # Run VMEC
    try:
        vmec = vmecpp.Vmec(vmec_input, verbose=True)
        wout = vmec.run()
        
        print(f"Run completed! Final force residual: {wout.fsqr:.2e}")
        print(f"Number of iterations: {wout.iter}")
        print(f"Converged: {wout.fsqr < 1e-6}")
        
        return True
        
    except Exception as e:
        print(f"Run failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_asymmetric()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")