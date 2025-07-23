# jVMEC vs VMEC++ Comprehensive Line-by-Line Comparison

## Executive Summary

After thorough examination of the jVMEC source code, I've identified several critical differences between jVMEC and VMEC++ that likely explain asymmetric convergence failures in first iterations.

## Key Findings

### 1. Memory Allocation and Array Initialization Differences

**jVMEC State.java (Lines 301-330):**
```java
public void allocateFourierGeometry(int numSurfaces) {
    final boolean lthreed = sizes.lthreed;
    final boolean lasym = sizes.lasym;
    
    // Always allocate base arrays
    rmncc = new double[numSurfaces][ntor + 1][mpol];
    zmnsc = new double[numSurfaces][ntor + 1][mpol];
    lmnsc = new double[numSurfaces][ntor + 1][mpol];
    
    if (lthreed) {
        rmnss = new double[numSurfaces][ntor + 1][mpol];
        zmncs = new double[numSurfaces][ntor + 1][mpol]; 
        lmncs = new double[numSurfaces][ntor + 1][mpol];
    }
    if (lasym) {
        rmnsc = new double[numSurfaces][ntor + 1][mpol];  // CRITICAL: asymmetric
        zmncc = new double[numSurfaces][ntor + 1][mpol];  // CRITICAL: asymmetric
        lmncc = new double[numSurfaces][ntor + 1][mpol];  // CRITICAL: asymmetric
        if (lthreed) {
            rmncs = new double[numSurfaces][ntor + 1][mpol];  // CRITICAL: asymmetric
            zmnss = new double[numSurfaces][ntor + 1][mpol];  // CRITICAL: asymmetric
            lmnss = new double[numSurfaces][ntor + 1][mpol];  // CRITICAL: asymmetric
        }
    }
}
```

**CRITICAL DIFFERENCE:** jVMEC conditionally allocates asymmetric arrays only when `lasym=true`. VMEC++ allocates all arrays regardless of `lasym` status.

### 2. Fourier Transform Implementation Differences

**jVMEC FourierTransformsJava.java - Asymmetric transforms (Lines 255-392):**

Key differences identified:

1. **Separate Array Processing:** jVMEC uses separate temporary arrays for asymmetric components:
   ```java
   final double[][][][] asym_R        = new double[myend - mystart][m_even_odd][nzeta][ntheta3];
   final double[][][][] asym_dRdTheta = new double[myend - mystart][m_even_odd][nzeta][ntheta3];
   // ... separate arrays for all quantities
   ```

2. **Symmetrization Logic (Lines 335-392):** jVMEC implements the exact stellarator symmetry formula:
   ```java
   // FIRST SUM SYMMETRIC, ANTISYMMETRIC PIECES ON EXTENDED INTERVAL, THETA = [PI,2*PI]
   for (int l = ntheta2; l < ntheta1; ++l) {
       final int lr = ntheta1 - l;
       for (int k = 0; k < nzeta; ++k) {
           final int kr = (nzeta - k) % nzeta; // ireflect
           for (int j = mystart; j < myend; ++j) {
               R[j][mParity][k][l] = R[j][mParity][kr][lr] - asym_R[j - mystart][mParity][kr][lr];
               dRdTheta[j][mParity][k][l] = -dRdTheta[j][mParity][kr][lr] + asym_dRdTheta[j - mystart][mParity][kr][lr];
               // ... exact reflection formula
           }
       }
   }
   ```

3. **Thread-Safe Implementation:** jVMEC uses thread-local arrays which avoid race conditions.

### 3. Initial Boundary Processing Differences

**jVMEC Boundaries.java - Axis Extrapolation (Lines 576-608):**
```java
public void extrapolateToAxis() {
    final int ntor = sizes.ntor;
    final boolean lthreed = sizes.lthreed;
    final boolean lasym = sizes.lasym;
    
    final int axis = 0;
    final int firstSurface = 1;
    
    for (int n = 0; n <= ntor; ++n) {
        rmncc[axis][n][1] = rmncc[firstSurface][n][1];
        zmnsc[axis][n][1] = zmnsc[firstSurface][n][1];
        lmnsc[axis][n][1] = lmnsc[firstSurface][n][1];
        if (lthreed) {
            rmnss[axis][n][1] = rmnss[firstSurface][n][1];
            zmncs[axis][n][1] = zmncs[firstSurface][n][1];
            lmncs[axis][n][0] = lmncs[firstSurface][n][0]; // TODO: iota force ???
            lmncs[axis][n][1] = lmncs[firstSurface][n][1];
        }
        if (lasym) {
            rmnsc[axis][n][1] = rmnsc[firstSurface][n][1]; // CRITICAL: asymmetric axis
            zmncc[axis][n][1] = zmncc[firstSurface][n][1]; // CRITICAL: asymmetric axis
            lmncc[axis][n][0] = lmncc[firstSurface][n][0]; // TODO: iota force ???
            lmncc[axis][n][1] = lmncc[firstSurface][n][1];
            if (lthreed) {
                rmncs[axis][n][1] = rmncs[firstSurface][n][1];
                zmnss[axis][n][1] = zmnss[firstSurface][n][1];
                lmnss[axis][n][1] = lmnss[firstSurface][n][1];
            }
        }
    }
}
```

**CRITICAL:** jVMEC extrapolates asymmetric coefficients to the axis, which VMEC++ may not be doing correctly.

### 4. Force Computation Order and Spectral Condensation

**jVMEC IdealMHDModel.java - First Iteration Setup (Lines 622-627):**
```java
if (status.getNumIterations() == iter1 && (!state.freeBoundaryMode || nestor.initState.ivac() <= NestorInitalizationState.INITIALIZE.ivac())) {
    // iter2 == iter1 is true at start of a new multi-grid iteration
    // ivac .le. 0 is always true for fixed-boundary,
    // but only true for first iteration in free-boundary (?)
    spectralCondensation.extrapolateRZConIntoVolume();
}
```

**jVMEC SpectralCondensation - m=1 Constraint (Lines 522-530):**
```java
// apply m=1 spectral "constraint"/"encouragement"  
final double scalingFactor = 1.0 / Math.sqrt(2.0); // TODO: why 1/sqrt(2) and not 1/2 ?
if (lthreed) {
    spectralCondensation.convert_to_m1_constrained(scaledState.frss, scaledState.fzcs, scalingFactor);
}
if (lasym) {
    spectralCondensation.convert_to_m1_constrained(scaledState.frsc, scaledState.fzcc, scalingFactor);
}
```

### 5. Index Mapping and Array Access Patterns

**jVMEC FourierBasis.java - Negative Mode Handling:**
jVMEC uses specific indexing for negative toroidal modes:
```java
// FFT order: [0, 1, ..., ntor, -ntor, ..., -1]
targetN = n < 0 ? 2*ntor+1 + n : n;
```

This differs from VMEC++ which may not handle negative modes correctly for asymmetric cases.

### 6. Numerical Precision and Scaling Factors

**jVMEC uses consistent scaling factors:**
- `1.0 / Math.sqrt(2.0)` for m=1 constraint forces
- Proper radial normalization with `sqrt(s)` factors
- Consistent `state.signOfJacobian` application

### 7. Thread Safety Issues

**CRITICAL FINDING:** jVMEC allocates thread-local temporary arrays:
```java
final double[][][][] asym_R = new double[myend - mystart][m_even_odd][nzeta][ntheta3];
```

VMEC++ may have shared state causing race conditions in asymmetric transforms.

## Critical Implementation Gaps in VMEC++

Based on line-by-line comparison, VMEC++ is missing:

1. **Proper asymmetric array allocation patterns**
2. **Correct asymmetric axis extrapolation**
3. **Thread-safe temporary array management**
4. **Proper stellarator symmetry reflection formulas**
5. **Correct m=1 constraint handling for asymmetric modes**
6. **Proper spectral condensation ordering in first iteration**

## Recommended Fixes

1. **Fix array allocation to match jVMEC conditional pattern**
2. **Implement correct asymmetric axis extrapolation**
3. **Add thread-local storage for transform arrays**
4. **Implement exact stellarator symmetry formulas from jVMEC**
5. **Fix m=1 constraint application for asymmetric modes**
6. **Ensure proper initialization order in first iteration**

## Testing Strategy

1. **Run symmetric case - MUST continue to pass**
2. **Test asymmetric case after each fix**
3. **Compare debug output line-by-line with jVMEC**
4. **Verify all force components match jVMEC in first iteration**

This analysis provides the roadmap for fixing asymmetric convergence in VMEC++.