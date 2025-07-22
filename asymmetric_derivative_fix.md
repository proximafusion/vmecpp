# Asymmetric Derivative Fix for VMEC++

## Problem Analysis

The issue is that `azNorm` is becoming zero in asymmetric equilibria because `zuFull` contains zeros. This happens because the derivative arrays `ru_o` and `zu_o` are not being properly populated from the asymmetric Fourier coefficients.

## Key Differences: jVMEC vs VMEC++

### jVMEC Implementation (WORKING)

In jVMEC's `FourierTransformsJava.java`, the derivatives are computed during the transform:

```java
// Symmetric contributions (even parity)
dRdTheta[j][mParity][k][l] += work[0][j - mystart][k] * sinmum[l][m];
dZdTheta[j][mParity][k][l] += work[5][j - mystart][k] * cosmum[l][m];

// Asymmetric contributions (for lasym=true)
asym_dRdTheta[j - mystart][mParity][k][l] += work[0][j - mystart][k] * cosmum[l][m];
asym_dZdTheta[j - mystart][mParity][k][l] += work[5][j - mystart][k] * sinmum[l][m];
```

Then these are combined:
```java
dRdThetaCombined[j][lv][ku] = dRdTheta[j][0][lv][ku] + dRdTheta[j][1][lv][ku] * sqrtSFull[j];
dZdThetaCombined[j][lv][ku] = dZdTheta[j][0][lv][ku] + dZdTheta[j][1][lv][ku] * sqrtSFull[j];
```

### VMEC++ Implementation (BROKEN)

In VMEC++, the symmetric transform correctly populates derivatives:
```cpp
// In dft_FourierToReal_3d_symm (lines 1661-1669)
ru_e[idx_jl] += rnkcc_m[kEvenParity];  // Correctly computed
zu_e[idx_jl] += znksc_m[kEvenParity];  // Correctly computed
ru_o[idx_jl] += rnkcc_m[kOddParity];   // Correctly computed for symmetric
zu_o[idx_jl] += znksc_m[kOddParity];   // Correctly computed for symmetric
```

But for asymmetric contributions, there's a hacky implementation that doesn't compute derivatives properly:
```cpp
// Lines 1369-1382 - THIS IS THE PROBLEM!
if (s_.lasym && jF == 1 && kl >= 6 && kl <= 9) {
  // This is a hack just to test
  r1_o[idx] = 0.5;   // Test value for R odd
  z1_o[idx] = 0.5;   // Test value for Z odd
  ru_o[idx] = 0.1;   // Dummy derivative - NOT COMPUTED!
  zu_o[idx] = -0.1;  // Dummy derivative - NOT COMPUTED!
}
```

Later attempt (lines 1415-1445) doesn't compute derivatives either:
```cpp
// Transform asymmetric coefficients
for (int m = 0; m < num_m; ++m) {
  double sin_mu = sin(m * theta);
  double cos_mu = cos(m * theta);
  // ...
  // R asymmetric: sin basis
  r_asym += src_rsc[m] * sin_mu;
  ru_asym += src_rsc[m] * m * cos_mu;  // This computes derivative but...
  
  // Z asymmetric: cos basis  
  z_asym += src_zcc[m] * cos_mu;
  zu_asym -= src_zcc[m] * m * sin_mu;  // This computes derivative but...
}

// The problem: these store the wrong thing!
r1_o[idx] = r_symmetric - r_asym;  // Should be just r_asym contribution
ru_o[idx] = ru_symmetric - ru_asym; // Should be just ru_asym contribution
```

## The Fix

The asymmetric contributions need to be handled separately in the Fourier transform, just like jVMEC does:

1. The symmetric transform should only handle symmetric coefficients (rmncc, zmnsc)
2. A separate transform pass should handle asymmetric coefficients (rmnsc, zmncc) and properly compute their derivatives
3. The odd arrays should store ONLY the asymmetric contributions, not the full geometry

## Where to Fix

The fix should be implemented in `ideal_mhd_model.cc` in the `dft_FourierToReal_3d_symm` function:

1. Remove the hacky test code (lines 1369-1382)
2. Fix the asymmetric transform section (lines 1390-1448) to:
   - Compute derivatives correctly using the proper sin/cos factors
   - Store only the asymmetric contributions in the odd arrays
   - Follow the jVMEC pattern exactly

## Expected Result

After the fix:
- `ru_o` and `zu_o` will contain the proper asymmetric derivative contributions
- `ruFull` and `zuFull` will be properly computed as `ru_e + sqrt(s) * ru_o` and `zu_e + sqrt(s) * zu_o`
- `azNorm` will be non-zero, allowing the constraint force to be computed correctly
- The asymmetric equilibrium will converge properly