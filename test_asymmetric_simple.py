#!/usr/bin/env python3
"""Simple test to understand the asymmetric equilibrium issue"""

from vmecpp.cpp import _vmecpp as vmec
import numpy as np

print("=" * 80)
print("Testing Asymmetric Equilibrium Issue")
print("=" * 80)

# Test file
test_file = "src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak.json"

try:
    print(f"\nLoading: {test_file}")
    indata = vmec.VmecINDATAPyWrapper.from_file(test_file)
    
    print(f"\nConfiguration:")
    print(f"  LASYM = {indata.lasym}")
    print(f"  NFP = {indata.nfp}")
    print(f"  MPOL = {indata.mpol}")
    print(f"  NTOR = {indata.ntor}")
    print(f"  NS = {list(indata.ns_array)}")
    
    # Check boundary arrays
    print(f"\nBoundary arrays:")
    print(f"  RBC shape: {indata.rbc.shape}")
    print(f"  ZBS shape: {indata.zbs.shape}")
    
    # Check for asymmetric boundary coefficients
    if hasattr(indata, 'rbs'):
        print(f"  RBS shape: {indata.rbs.shape}")
        # Check if any non-zero
        rbs_nonzero = np.count_nonzero(indata.rbs)
        print(f"  RBS non-zero elements: {rbs_nonzero}")
        if rbs_nonzero > 0:
            print(f"  First few RBS: {indata.rbs.flatten()[:10]}")
    
    if hasattr(indata, 'zbc'):
        print(f"  ZBC shape: {indata.zbc.shape}")
        zbc_nonzero = np.count_nonzero(indata.zbc)
        print(f"  ZBC non-zero elements: {zbc_nonzero}")
        if zbc_nonzero > 0:
            print(f"  First few ZBC: {indata.zbc.flatten()[:10]}")
    
    # Try to run with very limited iterations
    print(f"\nRunning VMEC++ with limited iterations...")
    indata.niter_array = [5]  # Very few iterations
    indata.ftol_array = [1e-6]  # Relaxed tolerance
    
    output = vmec.run(indata)
    
    print(f"\nRun completed (should not reach here if azNorm error occurs)")
    
except Exception as e:
    print(f"\nExpected ERROR: {e}")
    
    # Analyze the error
    error_str = str(e)
    if "azNorm should never be 0.0" in error_str:
        print("\nDIAGNOSIS: The error confirms that zuFull array is all zeros.")
        print("This happens because:")
        print("1. The 2D asymmetric transform is disabled (line 4176 in ideal_mhd_model.cc)")
        print("2. Without the transform, asymmetric geometry is not computed")
        print("3. This leaves zu (dZ/dtheta) arrays empty, causing azNorm = 0")
        
        print("\nSOLUTION: Fix the FourierToReal2DAsymmFastPoloidal function")
        print("The function was disabled due to 'buffer overflow issues'")
        print("Need to properly implement it following the jVMEC pattern")

print("\n" + "=" * 80)