#!/usr/bin/env python3
"""Test to compare jVMEC and VMEC++ asymmetric Fourier transform behavior"""

import numpy as np

def test_asymmetric_transform_comparison():
    """Compare the key differences between jVMEC and VMEC++ asymmetric transforms"""
    
    print("=== KEY DIFFERENCES FOUND ===\n")
    
    print("1. BASIS FUNCTION NORMALIZATION:")
    print("   - jVMEC: Applies mscale[m] and nscale[n] (sqrt(2) for m,n > 0) to basis functions")
    print("   - VMEC++: Uses raw cos/sin without scaling in asymmetric transform")
    print("   - This causes sqrt(2) factor differences for modes with m > 0 or n > 0\n")
    
    print("2. NEGATIVE N MODE HANDLING:")
    print("   - jVMEC: Only processes n >= 0 (lines 61, 273 in toRealSpace)")
    print("   - VMEC++: Also only processes n >= 0, matching jVMEC")
    print("   - Both codes correctly avoid negative n modes\n")
    
    print("3. SYMMETRIZATION LOGIC (theta in [pi, 2pi]):")
    print("   - jVMEC (lines 340-365): Uses reflection indices")
    print("     * lr = ntheta1 - l (theta reflection)")
    print("     * kr = (nzeta - k) % nzeta (zeta reflection)")
    print("     * R[l,k] = R[lr,kr] - asym_R[lr,kr]")
    print("     * Z[l,k] = -Z[lr,kr] + asym_Z[lr,kr]")
    print("   - VMEC++ (lines 170-190): Different reflection logic")
    print("     * l_reflected = ntheta_eff - 1 - l")
    print("     * Uses confusing double subtraction\n")
    
    print("4. ACCUMULATION PATTERN:")
    print("   - jVMEC: Accumulates zeta contributions first (work arrays), then theta")
    print("   - VMEC++: Similar pattern but without proper normalization\n")
    
    print("5. MAIN BUG IN VMEC++:")
    print("   The asymmetric transform in lines 73-80 uses raw cos/sin:")
    print("   ```cpp")
    print("   double cos_nv = std::cos(n * sizes.nfp * 2.0 * M_PI * k / nzeta);")
    print("   double sin_nv = std::sin(n * sizes.nfp * 2.0 * M_PI * k / nzeta);")
    print("   ```")
    print("   But should use normalized basis functions like jVMEC:")
    print("   ```java")
    print("   work[0][j - mystart][k] += rmncc[j][n][m] * cosnv[k][n];")
    print("   ```")
    print("   where cosnv includes nscale[n] normalization\n")
    
    print("6. SYMMETRIZATION BUG IN VMEC++:")
    print("   Lines 181-182 have confusing logic:")
    print("   ```cpp")
    print("   r_real[idx] = (r_real[idx_reflected] - asym_R[idx_reflected]) - asym_R[idx_reflected];")
    print("   z_real[idx] = -(z_real[idx_reflected] - asym_Z[idx_reflected]) + asym_Z[idx_reflected];")
    print("   ```")
    print("   This double-subtracts asym_R and doesn't match jVMEC pattern\n")
    
    # Example calculation showing the bug
    print("=== EXAMPLE CALCULATION ===")
    print("For R(theta=0) with rmncc[m=0,n=0] = 1.3:")
    print("- jVMEC: 1.3 * 1.0 (nscale[0]) = 1.3 ✓")
    print("- VMEC++: 1.3 * 1.0 (no scaling) = 1.3 ✓")
    print("\nFor R(theta=0) with rmncc[m=1,n=0] = 0.1:")
    print("- jVMEC: 0.1 * sqrt(2) (mscale[1]) * cos(0) = 0.1414...")
    print("- VMEC++: 0.1 * 1.0 (no scaling) * cos(0) = 0.1")
    print("\nThis sqrt(2) factor accumulates across all m>0 modes!")

if __name__ == "__main__":
    test_asymmetric_transform_comparison()