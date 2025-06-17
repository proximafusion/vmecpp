# Fourier Basis Implementation in VMEC++

## Overview

This document describes the specific implementation of Fourier transformations and basis conversions in VMEC++. It complements the broader mathematical foundations covered in `the_numerics_of_vmecpp.pdf` by focusing on the computational implementation details that developers need when working with Fourier-related code.

**Note**: This document is referenced by `AGENTS.md` and `VMECPP_NAMING_GUIDE.md` as essential reading for understanding Fourier basis operations.

## Key Implementation Distinction: DFTs vs FFTs

### **VMEC++ Uses DFTs with Pre-computed Basis Arrays**

VMEC++ implements **Discrete Fourier Transforms (DFTs) with pre-computed basis arrays**, **NOT Fast Fourier Transforms (FFTs)**. This is a critical distinction that affects all Fourier-related operations.

#### **Pre-computed Basis Arrays**

```cpp
// Poloidal direction basis functions in FourierBasisFastPoloidal/ToroidalLayout:
// cosmu[l*(mnyq2+1) + m] = cos(m*\theta[l]) * mscale[m]
// sinmu[l*(mnyq2+1) + m] = sin(m*\theta[l]) * mscale[m]
std::vector<double> cosmu;    // Pre-computed cos(m*\theta) with scaling
std::vector<double> sinmu;    // Pre-computed sin(m*\theta) with scaling

// Integration weights for surface integrals:
// cosmui[l*(mnyq2+1) + m] = cosmu * integration_normalization
std::vector<double> cosmui;   // Integration weights: cosmu * d_theta_zeta
std::vector<double> sinmui;   // Integration weights: sinmu * d_theta_zeta

// Toroidal direction basis functions:
// cosnv[n*nZeta + k] = cos(n*\zeta[k]) * nscale[n]
// sinnv[n*nZeta + k] = sin(n*\zeta[k]) * nscale[n]
std::vector<double> cosnv;    // Pre-computed cos(n*\zeta) with scaling
std::vector<double> sinnv;    // Pre-computed sin(n*\zeta) with scaling
```

#### **Why DFTs Instead of FFTs?**

1. **Physics-driven grids**: VMEC uses specific angular discretizations
   - Poloidal: Reduced range [0,\pi] with nThetaReduced points
   - Toroidal: Full period [0,2\pi/nfp] with nZeta points
   - These grids are NOT power-of-2 sized

2. **Integration weight incorporation**: Surface integrals require specific quadrature weights pre-applied to basis functions

3. **Scaling factors**: Normalization factors (mscale, nscale) are pre-computed and applied once

4. **Derivative computation**: Pre-computed derivative arrays avoid repeated calculations

FFTs would require post-processing for all these physics-specific requirements, making DFTs more efficient.

## Two Fourier Basis Representations

### **1. Combined Basis (External Interface)**

**Mathematical Form**: cos(m*\theta - n*\zeta), sin(m*\theta - n*\zeta)

**Usage Context**:
- wout files (traditional VMEC output format)
- Python API input/output
- SIMSOPT compatibility
- Literature and research compatibility

**Storage Pattern**: Linear indexing by mode number mn
- mn=0: (m=0, n=0)
- mn=1: (m=0, n=1), ..., mn=ntor: (m=0, n=ntor)
- mn=ntor+1: (m=1, n=-ntor), ..., mn=ntor+1+2*ntor: (m=1, n=ntor)
- And so on for m=2,3,...,mpol-1

### **2. Product Basis (Internal Computation)**

**Mathematical Form**: cos(m*\theta)*cos(n*\zeta), sin(m*\theta)*sin(n*\zeta), etc.

**Usage Context**:
- Internal force calculations
- Energy evaluations
- Real-space transformations via DFTs
- Separable \theta/\zeta operations

**Storage Pattern**: 2D indexing by separate m,n indices
- Layout: fcCC[n*m_size + m] or fcCC[m*(n_size+1) + n] (varies by class)

### **Trigonometric Basis Function Relationships**

The mathematical identities underlying all coefficient conversions:

**Primary identities**:
```
cos(m*\theta - n*\zeta) = cos(m*\theta)*cos(n*\zeta) + sin(m*\theta)*sin(n*\zeta)
sin(m*\theta - n*\zeta) = sin(m*\theta)*cos(n*\zeta) - cos(m*\theta)*sin(n*\zeta)
```

**Inverse identities**:
```
cos(m*\theta)*cos(n*\zeta) = 0.5 * [cos(m*\theta - n*\zeta) + cos(m*\theta + n*\zeta)]
sin(m*\theta)*sin(n*\zeta) = 0.5 * [cos(m*\theta - n*\zeta) - cos(m*\theta + n*\zeta)]
sin(m*\theta)*cos(n*\zeta) = 0.5 * [sin(m*\theta - n*\zeta) + sin(m*\theta + n*\zeta)]
cos(m*\theta)*sin(n*\zeta) = 0.5 * [sin(m*\theta + n*\zeta) - sin(m*\theta - n*\zeta)]
```

These identities relate the **basis functions** themselves. The conversion functions in VMEC++ use these relationships to transform **coefficients** between representations, taking into account VMEC's symmetry conventions where only n >= 0 modes are stored.

## Conversion Function Implementation

### **Function Naming Convention**

Following VMECPP_NAMING_GUIDE.md conventions:

```cpp
// Convert external combined -> internal product basis
int cos_to_cc_ss(...)  // cos(m*\theta-n*\zeta) -> {CC, SS} components
int sin_to_sc_cs(...)  // sin(m*\theta-n*\zeta) -> {SC, CS} components

// Convert internal product -> external combined basis
int cc_ss_to_cos(...)  // {CC, SS} components -> cos(m*\theta-n*\zeta)
int sc_cs_to_sin(...)  // {SC, CS} components -> sin(m*\theta-n*\zeta)
```

### **Key Implementation Details**

#### **Scaling Factor Application**
All conversion functions apply pre-computed scaling factors:
```cpp
double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);
double normedFC = basis_norm * input_coefficient;
```

#### **Mode Symmetry Handling**
- **Positive n**: Direct coefficient assignment
- **Negative n**: Uses sign symmetry: cos(-n*\zeta) = cos(n*\zeta), sin(-n*\zeta) = -sin(n*\zeta)
- **n=0 modes**: Only CC and SC components (no SS or CS contributions)
- **m=0 modes**: Simplified indexing (only positive n allowed)

#### **Array Layout Differences**
- **FourierBasisFastPoloidal**: fcCC[m*(n_size+1) + n] (m-major ordering)
- **FourierBasisFastToroidal**: fcCC[n*m_size + m] (n-major ordering)

This difference reflects the optimization for different transformation directions.

#### **Standalone Operation**
Following recent improvements, conversion functions do NOT access class member state (like `s_.lthreed`). They always fill both output arrays, making them truly standalone and reusable.

## Physics Variable Naming

Following established VMEC++ conventions from VMECPP_NAMING_GUIDE.md:

### **Internal Product Basis Variables**
```cpp
// Cosine-based quantities (R coordinates, pressure, etc.)
std::vector<double> rmncc_;  // cos(m*\theta) * cos(n*\zeta) coefficients
std::vector<double> rmnss_;  // sin(m*\theta) * sin(n*\zeta) coefficients

// Sine-based quantities (Z coordinates, \lambda angles, etc.)
std::vector<double> zmnsc_;  // sin(m*\theta) * cos(n*\zeta) coefficients
std::vector<double> zmncs_;  // cos(m*\theta) * sin(n*\zeta) coefficients
```

**Suffix Meaning**:
- First letter: trigonometric function for m (\theta dependence)
- Second letter: trigonometric function for n (\zeta dependence)
- c = cos, s = sin

### **External Combined Basis Variables**
```cpp
// Traditional VMEC format - preserve historical names
RowMatrixXd rmnc;  // R * cos(m*\theta - n*\zeta) coefficients
RowMatrixXd zmns;  // Z * sin(m*\theta - n*\zeta) coefficients
RowMatrixXd lmns;  // \lambda * sin(m*\theta - n*\zeta) coefficients
```

## Performance Characteristics

### **Computational Complexity**
- **DFT operations**: O(N_m * N_n) for separable transforms
- **Basis pre-computation**: O(N_\theta * N_m + N_\zeta * N_n) done once
- **Conversion functions**: O(mnmax) linear in number of modes

### **Memory Access Patterns**
- **Product basis**: Optimized for separable operations
- **Combined basis**: Sequential access by mode number
- **Basis arrays**: Cache-friendly access during DFT operations

### **Integration Weight Application**
Surface integrals computed using pre-weighted basis arrays:
```cpp
// Integral: \int f(\theta,\zeta) d\theta d\zeta
// Implementation: \sum_{l,k} f[l,k] * cosmui[l,m] * cosnv[n,k]
// No runtime weight multiplication needed
```

## Development Guidelines

### **When to Use Each Basis**

**Use Combined Basis for**:
- Reading/writing wout files
- Python API interfaces
- SIMSOPT compatibility
- External data exchange

**Use Product Basis for**:
- Force calculations
- Energy evaluations
- Real-space DFT operations
- Internal computational kernels

### **Conversion Function Usage Patterns**

```cpp
// Typical external input -> internal computation flow
FourierBasisFastPoloidal fb(&sizes);
std::vector<double> rmncc_internal(sizes.mnsize);
std::vector<double> rmnss_internal(sizes.mnsize);
fb.cos_to_cc_ss(rmnc_external, rmncc_internal, rmnss_internal,
                 sizes.ntor, sizes.mpol);

// Typical internal computation -> external output flow
fb.cc_ss_to_cos(rmncc_internal, rmnss_internal, rmnc_external,
                 sizes.ntor, sizes.mpol);
```

### **Error Prevention**

1. **Array sizing**: Always allocate both output arrays (CC+SS or SC+CS)
2. **Mode indexing**: Use provided mnIdx() functions for correct linear indexing
3. **Layout consistency**: Respect the m-major vs n-major ordering differences
4. **Scaling application**: Let conversion functions handle all scaling

## Related Documentation

- **`the_numerics_of_vmecpp.pdf`**: Comprehensive mathematical foundations and physics background
- **`VMECPP_NAMING_GUIDE.md`**: Naming conventions for all Fourier-related variables and functions
- **`AGENTS.md`**: High-level architecture overview including Fourier basis system

This implementation documentation should be understood alongside these resources for complete comprehension of VMEC++'s Fourier basis architecture.
