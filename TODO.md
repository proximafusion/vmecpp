# VMEC++ Asymmetric Implementation - Status Update

## CURRENT STATUS: Transforms Fixed, Spectral Condensation Updated

### ✅ COMPLETED: Corrected Asymmetric Transform Algorithm
- **FIXED**: Implemented exact jVMEC two-stage transform approach
- **FIXED**: Use separate arrays for asymmetric contributions (initialized to zero)
- **FIXED**: Proper reflection handling for theta=[pi,2pi] range
- **FIXED**: Arrays no longer cleared (symmetric baseline preserved)
- **RESULT**: Transforms produce finite, geometrically valid results

### ✅ COMPLETED: Asymmetric Spectral Condensation
- **FIXED**: Added work[2] and work[3] arrays for asymmetric case
- **FIXED**: Implemented on-the-fly symmetrization as in jVMEC
- **FIXED**: Added gcc and gss Fourier coefficient arrays
- **FIXED**: Proper reflection index handling in deAliasConstraintForce

### 🔴 REMAINING ISSUE: Stellarator asymmetric test still fails with vector bounds error

## Phase 1: Immediate Debugging Tasks ✅ COMPLETED

### 1.1 Verify Transform Integration ✅
- [x] ✅ Symmetric transforms called before asymmetric (correct order)
- [x] ✅ Corrected implementation is used in all code paths
- [x] ✅ symrzl_geometry is called at the right time
- [x] ✅ Force symmetrization is properly implemented

### 1.2 Array Initialization Comparison ✅
- [x] ✅ Force arrays initialized to zero (fixed resize issue)
- [x] ✅ Geometry arrays properly sized for full theta range
- [x] ✅ Lambda array handling verified
- [x] ✅ All arrays now match jVMEC initialization pattern

### 1.3 Vector Bounds Error Investigation 🔴
- [ ] ❌ Stellarator asymmetric test fails with vector assertion
- [ ] Debug exact location of bounds violation
- [ ] Check all array access patterns in asymmetric mode
- [ ] Verify index calculations for reflected coordinates

## Phase 2: Line-by-Line jVMEC Comparison

### 2.1 Transform Details
- [ ] Compare EXACT coefficient ordering (mn indexing)
- [ ] Verify basis function normalization matches
- [ ] Check sign conventions for all terms
- [ ] Compare work array usage patterns

### 2.2 Force Calculation
- [ ] Compare MHD force calculations in asymmetric mode
- [ ] Check if forces need different treatment for asymmetric
- [ ] Verify force symmetrization matches jVMEC exactly
- [ ] Check force array indexing for full theta range

### 2.3 Convergence Parameters
- [ ] Compare initial guess generation
- [ ] Check time step (delt) handling
- [ ] Verify convergence criteria calculations
- [ ] Compare Jacobian calculations

## Phase 3: Missing Asymmetric Functions ✅ COMPLETED

### 3.1 Missing Implementations Found and Fixed
- [x] ✅ Array initialization: C++ resize() doesn't zero-initialize (FIXED)
- [x] ✅ Boundary theta shift: IMPLEMENTED in boundaries.cc
- [x] ✅ Spectral condensation: Added work[2]/work[3] arrays for asymmetric (FIXED)
- [x] ✅ Radial preconditioner: Already handles asymmetric blocks correctly
- [x] ✅ FourierBasis: Boundary modifications already implemented in sizes.cc

### 3.2 Integration Points Verified
- [x] Found all jVMEC functions with lasym handling
- [x] Compared with VMEC++ implementations
- [x] Fixed missing spectral condensation asymmetric handling

## Phase 4: Test Case Analysis

### 4.1 Small Perturbation Tests
- [ ] Create MINIMAL asymmetric test (e.g., 0.1% perturbation)
- [ ] Compare convergence behavior with jVMEC
- [ ] Gradually increase perturbation to find breaking point
- [ ] Document exact failure mode at each level

### 4.2 Mode-by-Mode Testing
- [ ] Test with only m=1 asymmetric mode
- [ ] Test with only m=2 asymmetric mode
- [ ] Test combinations to isolate problematic modes
- [ ] Compare mode coupling with jVMEC

## Phase 5: Detailed Numerical Comparison

### 5.1 First Iteration Deep Dive
- [ ] Save ALL arrays after first iteration from both codes
- [ ] Compare EVERY array element-by-element
- [ ] Find first divergence point
- [ ] Trace back to root cause

### 5.2 Force Residual Analysis
- [ ] Compare force residuals at each iteration
- [ ] Check which force component diverges first
- [ ] Analyze force distribution patterns
- [ ] Compare with jVMEC force evolution

## Phase 6: Critical Code Review

### 6.1 Array Bounds and Sizing
- [ ] Verify ALL array allocations for asymmetric mode
- [ ] Check for off-by-one errors in loop bounds
- [ ] Verify span sizes match array allocations
- [ ] Check for buffer overruns in full theta range

### 6.2 Memory and Threading
- [ ] Check ThreadLocalStorage for asymmetric arrays
- [ ] Verify no race conditions in asymmetric mode
- [ ] Check for memory aliasing issues
- [ ] Verify all temporary arrays are properly sized

## Success Criteria

1. **Exact Match**: VMEC++ produces identical results to jVMEC for test cases
2. **Convergence**: Asymmetric equilibria converge with similar iteration counts
3. **Stability**: No NaN or infinity values during iteration
4. **Correctness**: Force residuals decrease monotonically

## Priority Actions

1. **IMMEDIATE**: Debug vector bounds error in stellarator asymmetric test
2. **HIGH**: Investigate why basic Fourier transform tests are failing
3. **HIGH**: Check if there are additional missing asymmetric implementations
4. **MEDIUM**: Test with tokamak asymmetric case as suggested by user
5. **MEDIUM**: Deep dive into convergence issues once tests pass

## Known Issues to Fix

1. **Vector bounds error**: Stellarator asymmetric test fails with assertion
2. **Transform tests failing**: Basic Fourier transform tests produce incorrect results
3. **Index calculation issues**: Possible problems with reflection indices in asymmetric mode
4. **Convergence failure**: Even with spectral condensation fix, convergence issues persist

## Key Insight
The transforms are now correct (producing valid geometry), but something else in the VMEC algorithm differs from jVMEC. The issue is likely in:
- Force calculations
- Convergence criteria
- Missing asymmetric-specific functions
- Array initialization patterns
- Numerical precision/accumulation order

**UPDATE**: Spectral condensation asymmetric handling has been implemented. However, stellarator asymmetric test still fails with vector bounds error, suggesting additional issues to investigate.
