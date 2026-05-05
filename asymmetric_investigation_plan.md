# Asymmetric Code Investigation Plan

## Objective
Compare VMEC++ asymmetric implementation with jVMEC line-by-line to identify discrepancies causing the "azNorm should never be 0.0" error.

## Investigation Steps

### 1. Enable Selective Debug Output
- Add debug flags for asymmetric-specific code paths
- Focus on Jacobian calculation and azNorm computation
- Add matching debug output in jVMEC for comparison

### 2. Key Areas to Compare

#### A. Fourier Transform Implementation
- `geometryFromFourier` function for asymmetric case
- Check index calculations and array bounds
- Verify sine/cosine component handling

#### B. Jacobian Calculation
- `computeLocalJacobian` function
- Focus on azNorm calculation
- Check geometry derivatives (ru12, zu12, rs, zs)

#### C. Array Initialization
- Verify proper initialization of asymmetric arrays
- Check raxis_s, zaxis_s arrays
- Ensure proper boundary condition handling

#### D. Index Mapping
- Compare index calculations between VMEC++ and jVMEC
- Check for off-by-one errors
- Verify array sizing and bounds

### 3. Debug Output Points
1. Initial boundary coefficients (rbc, rbs, zbc, zbs)
2. Fourier transform results at key points
3. Jacobian components before azNorm calculation
4. Geometry derivatives at problematic grid points

### 4. Test Strategy
- Use minimal asymmetric test case
- Compare output at each step with jVMEC
- Identify exact location where outputs diverge

## Files to Focus On
1. VMEC++: `ideal_mhd_model.cc` - geometryFromFourier, computeLocalJacobian
2. jVMEC: Corresponding Fourier transform and Jacobian routines
3. Both: Array initialization and boundary handling code