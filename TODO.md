# VMEC++ Asymmetric Implementation - Critical Debugging Plan

## CURRENT STATUS: Transforms Fixed, Convergence Issues Remain

### ✅ COMPLETED: Corrected Asymmetric Transform Algorithm
- **FIXED**: Implemented exact jVMEC two-stage transform approach
- **FIXED**: Use separate arrays for asymmetric contributions (initialized to zero)
- **FIXED**: Proper reflection handling for theta=[pi,2pi] range
- **FIXED**: Arrays no longer cleared (symmetric baseline preserved)
- **RESULT**: Transforms produce finite, geometrically valid results

### 🔴 CRITICAL ISSUE: Why doesn't VMEC++ converge when jVMEC and educational_VMEC do?

## Phase 1: Immediate Debugging Tasks

### 1.1 Verify Transform Integration
- [ ] Check if symmetric transforms are called BEFORE asymmetric (order matters!)
- [ ] Verify corrected implementation is used in ALL code paths
- [ ] Check if symrzl_geometry is called at the right time
- [ ] Verify force symmetrization is called when needed

### 1.2 Array Initialization Comparison
- [ ] Compare how jVMEC initializes ALL arrays (not just transform arrays)
- [ ] Check if force arrays need special initialization
- [ ] Verify geometry arrays are properly sized for full theta range
- [ ] Check lambda array handling (often overlooked)

### 1.3 Numerical Precision Issues
- [ ] Compare floating point operations order with jVMEC
- [ ] Check if accumulation order affects numerical stability
- [ ] Verify no uninitialized values propagate through calculations
- [ ] Check for division by zero or near-zero values

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

## Phase 3: Missing Asymmetric Functions

### 3.1 Potential Missing Implementations
- [ ] Check for asymmetric-specific preconditioner updates
- [ ] Verify asymmetric boundary condition handling
- [ ] Check for asymmetric-specific spectral condensation
- [ ] Look for asymmetric metric tensor calculations

### 3.2 Integration Points
- [ ] Search jVMEC for ALL uses of "lasym" flag
- [ ] List every function that has asymmetric-specific code
- [ ] Compare with VMEC++ to find missing implementations
- [ ] Check for subtle differences in shared code paths

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

1. **IMMEDIATE**: Compare array initialization patterns with jVMEC
2. **HIGH**: Check for missing asymmetric-specific functions
3. **HIGH**: Verify force calculation matches jVMEC exactly
4. **MEDIUM**: Test with minimal perturbations
5. **MEDIUM**: Deep dive first iteration comparison

## Known Issues to Fix

1. Vector assertion failure in stellarator test
2. Symmetric baseline fails in some tests (parameter issue?)
3. Convergence failure even with correct transforms
4. Possible missing asymmetric-specific calculations

## Key Insight
The transforms are now correct (producing valid geometry), but something else in the VMEC algorithm differs from jVMEC. The issue is likely in:
- Force calculations
- Convergence criteria
- Missing asymmetric-specific functions
- Array initialization patterns
- Numerical precision/accumulation order

**CRITICAL**: Must find why jVMEC and educational_VMEC converge but VMEC++ doesn't, despite having correct transforms.
