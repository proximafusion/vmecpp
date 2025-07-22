#!/usr/bin/env python3
"""Test asymmetric convergence and compare with jVMEC implementation details."""

import vmecpp
import numpy as np

def test_asymmetric_convergence():
    print("=== ASYMMETRIC CONVERGENCE DETAILED TESTING ===\n")
    
    # Load the asymmetric tokamak configuration  
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Configure for convergence testing
    vmec_input.ns_array = np.array([5, 11], dtype=np.int64)  # Multi-grid
    vmec_input.niter_array = np.array([400, 1000], dtype=np.int64)  # More iterations
    vmec_input.mpol = 4  # Higher resolution
    vmec_input.return_outputs_even_if_not_converged = True
    
    # Fixed axis initialization (from our successful fix)
    vmec_input.raxis_c = np.array([6.0])
    
    # Moderate asymmetry for better convergence
    vmec_input.rbs[1, 0] = 0.02  
    vmec_input.zbs[1, 0] = 0.02
    
    print(f"CONVERGENCE TEST CONFIGURATION:")
    print(f"  Multi-grid: ns = {vmec_input.ns_array}")
    print(f"  Iterations: niter = {vmec_input.niter_array}")
    print(f"  Resolution: mpol = {vmec_input.mpol}")
    print(f"  Asymmetry: RBS(1,0) = {vmec_input.rbs[1,0]}, ZBS(1,0) = {vmec_input.zbs[1,0]}")
    
    print(f"\nTEST: Does asymmetric equilibrium converge to tolerance?")
    print(f"Expected: Should converge with proper physics implementation\n")
    
    try:
        result = vmecpp.run(vmec_input, verbose=True)
        
        print(f"\n=== CONVERGENCE RESULTS ===")
        print(f"Final residual: {result.fsqr:.2e}")
        print(f"Converged: {'YES' if result.fsqr < 1e-11 else 'NO'}")
        
        if result.fsqr < 1e-11:
            print(f"🎉 EXCELLENT: Asymmetric equilibrium converged to tolerance!")
            
            print(f"\nASYMMETRIC COEFFICIENTS:")
            last_surf = result.rmns.shape[0] - 1
            print(f"  RMNS[{last_surf},1] = {result.rmns[last_surf, 1]:.6e}")
            print(f"  ZMNC[{last_surf},1] = {result.zmnc[last_surf, 1]:.6e}")
            
            print(f"\nMHD EQUILIBRIUM QUALITY:")
            print(f"  Magnetic axis: R = {result.raxis_cc[0]:.6f} m")
            print(f"  Volume: V = {result.volume:.3f} m³")
            print(f"  Beta: <beta> = {result.betatotal:.4f}")
            
        elif result.fsqr < 1e-6:
            print(f"⚠️  PARTIAL: Physics working, residual higher than ideal")
            print(f"   This suggests numerical convergence issue, not physics bug")
            
        else:
            print(f"❌ POOR: High residual suggests physics or numerical issue")
        
        return result.fsqr < 1e-11
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"\nERROR DURING CONVERGENCE: {error_msg}")
        
        if "BAD_JACOBIAN" in error_msg:
            print("❌ Physics issue: Jacobian problem")
        elif "matrix" in error_msg.lower() or "solver" in error_msg.lower():
            print("⚠️  Numerical issue: Matrix/solver problem")  
        else:
            print("❓ Other issue - needs investigation")
        
        return False

if __name__ == "__main__":
    success = test_asymmetric_convergence()
    if success:
        print("\n🌟 ASYMMETRIC IMPLEMENTATION COMPLETE AND WORKING!")
        print("Ready for detailed jVMEC comparison for optimization")
    else:
        print("\n🔧 ASYMMETRIC IMPLEMENTATION NEEDS FURTHER WORK")
        print("Continue detailed comparison with jVMEC")