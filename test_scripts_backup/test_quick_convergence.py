#!/usr/bin/env python3
"""Quick convergence test for basic asymmetric case."""

import vmecpp
import numpy as np

def test_convergence():
    print("=== ASYMMETRIC CONVERGENCE CHECK ===")
    
    # Load the asymmetric tokamak configuration  
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Start with small case that works, then scale up
    vmec_input.ns_array = np.array([5], dtype=np.int64)  # Single grid
    vmec_input.niter_array = np.array([50], dtype=np.int64)  # Moderate iterations
    vmec_input.mpol = 3  # Moderate resolution
    vmec_input.return_outputs_even_if_not_converged = True
    
    # Fixed axis initialization 
    vmec_input.raxis_c = np.array([6.0])
    
    # Small asymmetry for convergence
    vmec_input.rbs[1, 0] = 0.01  
    vmec_input.zbs[1, 0] = 0.01
    
    print(f"Testing convergence: ns=5, mpol=3, 50 iterations")
    
    try:
        result = vmecpp.run(vmec_input, verbose=False)
        
        print(f"Final residual: {result.fsqr:.2e}")
        print(f"Converged: {'YES' if result.fsqr < 1e-10 else 'NO'}")
        
        if result.fsqr < 1e-10:
            print("🎉 EXCELLENT: Full convergence achieved!")
            return True
        elif result.fsqr < 1e-6:
            print("✅ GOOD: Physics working, partial convergence")
            return "partial"
        else:
            print("⚠️  HIGH RESIDUAL: Need more work")
            return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    test_convergence()