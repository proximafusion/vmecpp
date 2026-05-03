# Missing Asymmetric Implementations in VMEC++

Based on comparison with jVMEC, the following asymmetric-specific features are MISSING:

## 1. Boundaries.cc - Theta Shift ✅ ACTUALLY IMPLEMENTED
**jVMEC**: Calculates delta angle shift to ensure RBS(n=0,m=1) = ZBC(n=0,m=1)
**VMEC++**: IMPLEMENTED in boundaries.cc lines 70-164, properly applies theta shift

## 2. SpectralCondensation.cc - Asymmetric Work Arrays
**jVMEC**: Uses work[2] and work[3] for asymmetric case, on-the-fly symmetrization
**VMEC++**: No special handling for lasym

## 3. RadialPreconditioner.cc - Asymmetric Coupling Blocks
**jVMEC**: Additional blocks for asymmetric coupling terms
**VMEC++**: No special handling for lasym

## 4. Array Initialization
**jVMEC**: Java arrays zero-initialized by default
**VMEC++**: resize() does NOT zero-initialize - FIXED in ideal_mhd_model.cc

## 5. Force Array Initialization
**jVMEC**: Relies on zero initialization
**VMEC++**: Was not initializing - FIXED

## 6. FourierBasis - Special Boundary Conditions
**jVMEC**: Modifies basis functions at theta=0 and theta=PI boundaries for lasym
**VMEC++**: Need to check

## 7. OutputQuantities - Asymmetric B-field Components
**jVMEC**: Additional calculations for asymmetric B-field
**VMEC++**: Need to verify

## Critical Functions to Check:
1. computeConstraintForce() in spectral_condensation
2. initBlocks() in radial_preconditioner
3. Boundary theta shift calculation
4. Basis function boundary modifications

These missing implementations likely explain why VMEC++ doesn't converge while jVMEC does.
