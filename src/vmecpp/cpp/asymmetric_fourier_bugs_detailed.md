# Detailed Line-by-Line Comparison: jVMEC vs VMEC++ Asymmetric Fourier Transforms

## Critical Bug #1: Missing Basis Function Normalization

### jVMEC (Correct)
```java
// Line 174: Uses pre-scaled basis functions
work[0][j - mystart][k] += rmncc[j][n][m] * cosnv[k][n];
work[5][j - mystart][k] += zmnsc[j][n][m] * cosnv[k][n];

// Where cosnv is defined in FourierBasis.java:
// Line 208: cosnv[k][n] = Math.cos(arg) * nscale[n];
// Line 174: cosmu[l][m] = Math.cos(arg) * mscale[m];
```

### VMEC++ (Incorrect)
```cpp
// Lines 77-80: Uses raw trigonometric functions without scaling
double cos_nv = (n <= sizes.nnyq2)
                    ? fourier_basis.cosnv[idx_nv]
                    : std::cos(n * sizes.nfp * 2.0 * M_PI * k / nzeta);
double sin_nv = (n <= sizes.nnyq2)
                    ? fourier_basis.sinnv[idx_nv]
                    : std::sin(n * sizes.nfp * 2.0 * M_PI * k / nzeta);
```

**Fix**: VMEC++ should consistently use the pre-scaled basis functions from `fourier_basis` for ALL n values, not just n <= nnyq2.

## Critical Bug #2: Incorrect Symmetrization Logic

### jVMEC (Correct, lines 340-365)
```java
// Clear reflection mapping
final int lr = ntheta1 - l;  // theta: pi+x -> pi-x
final int kr = (nzeta - k) % nzeta;  // zeta: v -> -v

// Apply symmetrization formulas
R[j][mParity][k][l] = R[j][mParity][kr][lr] - asym_R[j - mystart][mParity][kr][lr];
Z[j][mParity][k][l] = -Z[j][mParity][kr][lr] + asym_Z[j - mystart][mParity][kr][lr];
```

### VMEC++ (Incorrect, lines 181-183)
```cpp
// Confusing double subtraction
r_real[idx] = (r_real[idx_reflected] - asym_R[idx_reflected]) - asym_R[idx_reflected];
z_real[idx] = -(z_real[idx_reflected] - asym_Z[idx_reflected]) + asym_Z[idx_reflected];
```

**Issues**:
1. The formula `(r_real[idx_reflected] - asym_R[idx_reflected]) - asym_R[idx_reflected]` double-subtracts the asymmetric part
2. This doesn't match the mathematical symmetrization: `total = symmetric - antisymmetric`

## Bug #3: Missing Scaling in Theta Transform

### jVMEC (lines 198-204)
```java
// Uses pre-scaled basis functions
R[j][mParity][k][l] += work[0][j - mystart][k] * cosmu[l][m];
Z[j][mParity][k][l] += work[5][j - mystart][k] * sinmu[l][m];
// Where cosmu/sinmu include mscale[m]
```

### VMEC++ (lines 116-118)
```cpp
// Correctly uses scaled basis for symmetric part
r_real[idx] += rmkcc[k] * cos_mu;  // cos_mu from fourier_basis includes mscale
z_real[idx] += zmksc[k] * sin_mu;  // sin_mu from fourier_basis includes mscale
```

But the issue is in the zeta accumulation stage where raw trig functions are used for n > nnyq2.

## Summary of Required Fixes

1. **Fix zeta basis functions**: Always use `fourier_basis.cosnv/sinnv` which include `nscale[n]` normalization
2. **Fix symmetrization**: Implement proper reflection without double subtraction
3. **Ensure consistent scaling**: All basis functions must include mscale/nscale factors

The root cause is that VMEC++ incorrectly mixes scaled and unscaled basis functions, leading to incorrect amplitudes (missing sqrt(2) factors) for modes with m>0 or n>0.