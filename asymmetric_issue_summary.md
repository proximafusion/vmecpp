# VMEC++ Asymmetric Equilibrium Issue Summary

## Problem
When running asymmetric equilibria (LASYM=true), VMEC++ fails with:
```
azNorm should never be 0.0
```

## Root Cause Analysis

### 1. Disabled 2D Asymmetric Transform
The `FourierToReal2DAsymmFastPoloidal` function is disabled in `ideal_mhd_model.cc` (lines 4176-4190):
```cpp
void IdealMhdModel::dft_FourierToReal_2d_asymm(
    const FourierGeometry& physical_x) {
  // FIX: The 2D asymmetric transform has bugs - using thread local storage incorrectly
  // This causes buffer overflows that affect convergence even for symmetric cases
  std::cerr << "WARNING: 2D asymmetric transforms not properly implemented\n";
  std::cerr << "         Skipping to prevent buffer overflow issues\n";
  return;
}
```

### 2. Consequence
Without the asymmetric transform:
- Asymmetric geometry arrays (`r1_e`, `z1_e`) are not populated
- Derivative arrays (`ru_o`, `zu_o`) remain zero
- `zuFull` (containing dZ/dtheta) becomes all zeros
- `azNorm = sum(zuFull^2)` = 0, triggering the error

### 3. jVMEC Comparison
jVMEC properly implements the asymmetric transform in `FourierTransformsJava.java`:
- `totzspa`: Transforms asymmetric Fourier coefficients to real space
- Computes derivatives: `ru = rmnsc * m * cos(m*theta)`, `zu = -zmncc * m * sin(m*theta)`
- Ensures `zuFull` is properly populated

## Solution Required

### 1. Fix FourierToReal2DAsymmFastPoloidal
- Remove the early return
- Fix the "buffer overflow issues" mentioned in the comment
- Ensure proper array bounds and indexing

### 2. Verify Derivative Calculation
- Asymmetric derivatives must be computed during the transform
- Follow jVMEC pattern for derivative calculation

### 3. Test Implementation
- Start with simple asymmetric test case
- Verify zuFull is non-zero after transform
- Check convergence for asymmetric equilibria

## Files to Modify
1. `fourier_asymmetric.cc` - Fix the 2D transform implementation
2. `ideal_mhd_model.cc` - Re-enable the transform call
3. Add proper array bounds checking and initialization

## Next Steps
1. Implement the fix in FourierToReal2DAsymmFastPoloidal
2. Add debug output to verify arrays are populated
3. Test with up_down_asymmetric_tokamak.json
4. Verify convergence matches jVMEC results