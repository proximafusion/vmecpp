# VMEC++ Asymmetric Mode Implementation Plan

Based on comprehensive analysis of jVMEC source code, here are the critical missing components:

## Priority 1: Jacobian Recovery Strategy (CRITICAL)

**Problem**: "INITIAL JACOBIAN CHANGED SIGN" with no recovery mechanism
**jVMEC Solution**: Automatic retry with axis optimization

### Implementation needed:
1. **GuessAxis algorithm** - grid search for optimal axis position
2. **BadJacobian detection and retry** - automatic 3-surface restart
3. **Boundary shape validation** - check expected vs actual Jacobian sign

## Priority 2: Complete Asymmetric Fourier Transforms

**Problem**: Our asymmetric transforms are incomplete stubs
**jVMEC Reference**: `totzspa`, `tomnspa`, `symrzl` methods

### Missing transforms:
1. **Asymmetric inverse DFT (totzspa)**:
   ```cpp
   // Missing coefficient combinations:
   asym_R += rmnsc * sin(m*theta)*cos(n*zeta)  // Not implemented
   asym_Z += zmncc * cos(m*theta)*cos(n*zeta)  // Not implemented
   asym_Lambda += lmncc * cos(m*theta)*cos(n*zeta)  // Not implemented
   ```

2. **Asymmetric forward DFT (tomnspa)**:
   ```cpp
   // Missing force projections:
   frsc[m][n] += force_R * sin(m*theta)*cos(n*zeta)  // Wrong target array
   fzcc[m][n] += force_Z * cos(m*theta)*cos(n*zeta)  // Wrong target array
   ```

3. **Symmetrization operation (symrzl)**:
   ```cpp
   // For theta ∈ [π, 2π]: total = symmetric - asymmetric
   // For theta ∈ [0, π]: total = symmetric + asymmetric
   ```

## Priority 3: Asymmetric Initial Guess Interpolation

**Problem**: Axis contributions handled incorrectly for asymmetric modes
**jVMEC Pattern**: Different sign handling for different coefficient types

### Fix needed in `ideal_mhd_model.cc` interpolation:
```cpp
if (s_.lasym) {
  // Some coefficients ADD axis contribution: rmncc, zmncc
  // Some coefficients SUBTRACT axis contribution: rmncs  
  // Some coefficients have NO axis contribution: rmnsc, zmnss
}
```

## Priority 4: Missing Components Summary

| Component | Status | Files to Modify |
|-----------|--------|-----------------|
| GuessAxis algorithm | ❌ Missing | `boundaries.cc` |
| BadJacobian retry | ❌ Missing | `vmec.cc` |
| Asymmetric totzspa | 🔄 Stub only | `ideal_mhd_model.cc` |
| Asymmetric tomnspa | 🔄 Stub only | `ideal_mhd_model.cc` |
| Symmetrization (symrzl) | ❌ Missing | `ideal_mhd_model.cc` |
| Asymmetric interpolation | ❌ Wrong | `ideal_mhd_model.cc` |

## Implementation Strategy

1. **Phase 1**: Implement basic Jacobian recovery (GuessAxis + retry)
2. **Phase 2**: Complete asymmetric Fourier transforms 
3. **Phase 3**: Fix asymmetric initial guess interpolation
4. **Phase 4**: Add comprehensive testing against jVMEC

## Key Insights from jVMEC Analysis

1. **Asymmetric arrays have different iteration patterns**: `mpol*(2*ntor+1)` vs `(mpol+1)*(2*ntor+1)`
2. **Axis contributions vary by coefficient type**: Some add, some subtract, some ignore axis
3. **Symmetrization is critical**: Must properly combine symmetric/asymmetric parts using stellarator symmetry
4. **m=1 mode constraints different**: Asymmetric case couples rbsc↔zbcc instead of rbss↔zbcs

## Next Steps

1. Start with GuessAxis implementation - this addresses the immediate "JACOBIAN CHANGED SIGN" failure
2. Complete the asymmetric transform stubs with proper coefficient targeting
3. Add comprehensive debug output matching jVMEC for validation
4. Test incrementally with simple asymmetric cases

The current boundary array size fix was the first critical step. Now we need the complete asymmetric physics implementation to make asymmetric mode work reliably.