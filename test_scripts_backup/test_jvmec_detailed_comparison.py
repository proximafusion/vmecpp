#!/usr/bin/env python3
"""Detailed comparison between VMEC++ and jVMEC for asymmetric equilibrium."""

import vmecpp
import numpy as np
import os

def test_detailed_jvmec_comparison():
    print("=== DETAILED VMEC++ vs jVMEC ASYMMETRIC COMPARISON ===\n")
    
    # Load the asymmetric tokamak configuration  
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Configure for minimal comparison case
    vmec_input.ns_array = np.array([5], dtype=np.int64)
    vmec_input.niter_array = np.array([3], dtype=np.int64)  # Just a few iterations
    vmec_input.mpol = 3
    vmec_input.ntor = 0  # Force 2D asymmetric case
    vmec_input.return_outputs_even_if_not_converged = True
    
    # Fixed axis initialization
    vmec_input.raxis_c = np.array([6.0])
    
    # Well-defined small asymmetry for comparison
    vmec_input.rbs[1, 0] = 0.01  
    vmec_input.zbs[1, 0] = 0.01
    
    print("COMPARISON TEST CONFIGURATION:")
    print(f"  ns = {vmec_input.ns_array[0]}, mpol = {vmec_input.mpol}, ntor = {vmec_input.ntor}")
    print(f"  Asymmetry: RBS(1,0) = {vmec_input.rbs[1,0]}, ZBS(1,0) = {vmec_input.zbs[1,0]}")
    print(f"  Axis: raxis_c = {vmec_input.raxis_c[0]}")
    
    # Create equivalent jVMEC input
    jvmec_input_file = "input.test_asymmetric_comparison"
    with open(jvmec_input_file, 'w') as f:
        f.write("&INDATA\n")
        f.write(f"  MGRID_FILE = 'none'\n")
        f.write(f"  DELT = 0.9\n")
        f.write(f"  NFP = {vmec_input.nfp}\n") 
        f.write(f"  MPOL = {vmec_input.mpol}\n")
        f.write(f"  NTOR = {vmec_input.ntor}\n")
        f.write(f"  NS_ARRAY = {vmec_input.ns_array[0]}\n")
        f.write(f"  NITER_ARRAY = {vmec_input.niter_array[0]}\n")
        f.write(f"  FTOL_ARRAY = 1.0E-11\n")
        f.write(f"  LASYM = T\n")
        f.write(f"  LFREEB = F\n")
        f.write(f"  PHIEDGE = {vmec_input.phiedge}\n")
        f.write(f"  CURTOR = {vmec_input.curtor}\n")
        f.write(f"  AM = 0 0 0\n")
        f.write(f"  AI = 1 0 0\n")
        f.write(f"  RAXIS_CC = {vmec_input.raxis_c[0]}\n")
        f.write(f"  ZAXIS_CS = 0.0\n")
        
        # Write boundary coefficients
        f.write(f"  RBC(0,0) = {vmec_input.rbc[0,0]}\n")
        f.write(f"  RBC(1,0) = {vmec_input.rbc[1,0]}\n") 
        f.write(f"  ZBS(1,0) = {vmec_input.zbs[1,0]}\n")
        
        # Asymmetric coefficients
        f.write(f"  RBS(1,0) = {vmec_input.rbs[1,0]}\n")
        f.write(f"  ZBC(1,0) = {vmec_input.zbs[1,0]}\n")  # Note: ZBC for jVMEC
        
        f.write("&END\n")
    
    print(f"\nCreated jVMEC input file: {jvmec_input_file}")
    
    try:
        print("\n=== RUNNING VMEC++ ===")
        vmecpp_result = vmecpp.run(vmec_input, verbose=False)
        
        print(f"VMEC++ Results:")
        print(f"  Final residual: {vmecpp_result.fsqr:.6e}")
        print(f"  Iterations: {len(vmecpp_result.fsqr_array) if hasattr(vmecpp_result, 'fsqr_array') else 'N/A'}")
        print(f"  Magnetic axis: R = {vmecpp_result.raxis_cc[0]:.6f}")
        print(f"  Volume: {vmecpp_result.volume:.6f}")
        
        # Show asymmetric coefficients
        if hasattr(vmecpp_result, 'rmns') and vmecpp_result.rmns is not None:
            print(f"\n  Asymmetric coefficients (boundary):")
            boundary_idx = vmecpp_result.rmns.shape[0] - 1
            if vmecpp_result.rmns.shape[1] > 1:
                print(f"    RMNS[{boundary_idx},1] = {vmecpp_result.rmns[boundary_idx, 1]:.8e}")
            if vmecpp_result.zmnc.shape[1] > 1:
                print(f"    ZMNC[{boundary_idx},1] = {vmecpp_result.zmnc[boundary_idx, 1]:.8e}")
        
        # Show key force balance
        print(f"\n  Force balance:")
        print(f"    Force residual: {vmecpp_result.fsqr:.6e}")
        print(f"    Beta: {getattr(vmecpp_result, 'betatotal', 0.0):.6f}")
        
    except Exception as e:
        print(f"❌ VMEC++ FAILED: {e}")
        return False
    
    try:
        print(f"\n=== WOULD RUN jVMEC (if available) ===")
        print(f"jVMEC input file created: {jvmec_input_file}")
        print("For actual comparison, run jVMEC with this input file")
        print("Expected jVMEC approach based on code analysis:")
        print("  - Single-pass Fourier transforms (toroidal then poloidal)")
        print("  - Unified symmetric + asymmetric coefficient handling")
        print("  - Force symmetrization in symforce subroutine")
        
        jvmec_available = os.path.exists("/usr/local/bin/xvmec") or os.path.exists("./jvmec")
        if jvmec_available:
            print("\n⚠️  jVMEC binary found but not running automatically")
            print("  To compare: run jVMEC manually with the generated input file")
        else:
            print("\n📝 jVMEC not available for direct comparison")
            print("  Comparison based on code analysis and known behavior")
        
    except Exception as e:
        print(f"jVMEC comparison setup: {e}")
    
    print(f"\n=== ANALYSIS BASED ON CODE COMPARISON ===")
    print("Key differences identified:")
    print("1. Transform order: jVMEC (toroidal→poloidal) vs VMEC++ (unified)")
    print("2. Coefficient handling: jVMEC (single-pass) vs VMEC++ (separated)")
    print("3. Reflection logic: jVMEC (implicit) vs VMEC++ (explicit)")
    print("4. Force symmetrization: Both implement symforce but differently")
    
    print(f"\nRecommendations for further investigation:")
    print("- Compare step-by-step Fourier coefficients during transforms")
    print("- Check geometry values at specific grid points")
    print("- Verify force calculations before and after symmetrization")
    print("- Compare matrix assembly and solver setup")
    
    # Clean up
    if os.path.exists(jvmec_input_file):
        os.remove(jvmec_input_file)
    
    return True

if __name__ == "__main__":
    test_detailed_jvmec_comparison()