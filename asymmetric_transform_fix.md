# Asymmetric Transform Fix Implementation

## Problem Analysis
The current `FourierToReal2DAsymmFastPoloidal` function:
1. Only transforms position arrays (r, z, lambda)
2. Does NOT compute derivatives (ru, zu)
3. This leaves zuFull = 0, causing azNorm = 0 error

## jVMEC Reference
From jVMEC's `totzspa` (lines 255-391):
```java
// Compute derivatives during transform
double cosmth = Math.cos(m * theta_v[l]);
double sinmth = Math.sin(m * theta_v[l]);

// Position
rmn += rmnsc * sinmth;  // R asymmetric (sin basis)
zmn += zmncc * cosmth;  // Z asymmetric (cos basis)

// Derivatives
rumns += m * rmnsc * cosmth;  // dR/dtheta
zumns -= m * zmncc * sinmth;  // dZ/dtheta
```

## Required Fix

### 1. Update Function Signature
Add derivative output arrays:
```cpp
void FourierToReal2DAsymmFastPoloidal(
    const Sizes& sizes,
    // Input coefficients...
    absl::Span<double> r_real,
    absl::Span<double> z_real, 
    absl::Span<double> lambda_real,
    absl::Span<double> ru_real,  // ADD: dR/dtheta
    absl::Span<double> zu_real   // ADD: dZ/dtheta
);
```

### 2. Compute Derivatives
In the transform loop:
```cpp
// Asymmetric contributions
r_real[idx] += rsc * sin_mu;
z_real[idx] += zcc * cos_mu;

// ADD: Asymmetric derivatives
ru_real[idx] += m * rsc * cos_mu;  // dR/dtheta
zu_real[idx] -= m * zcc * sin_mu;  // dZ/dtheta
```

### 3. Update ideal_mhd_model.cc
- Re-enable the 2D asymmetric transform
- Pass derivative arrays to the function
- Ensure derivatives are added to ru_o, zu_o arrays

## Implementation Steps
1. Modify fourier_asymmetric.h to add derivative parameters
2. Update fourier_asymmetric.cc to compute derivatives
3. Fix ideal_mhd_model.cc to pass and use derivative arrays
4. Remove the early return that disables the transform
5. Test with asymmetric equilibrium