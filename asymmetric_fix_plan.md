# Asymmetric Fix Implementation Plan

## Root Cause
The asymmetric 2D transform is disabled (lines 4176-4190 in ideal_mhd_model.cc) with a comment saying it has bugs. This causes:
1. Asymmetric geometry arrays (r1_e, z1_e) are not properly filled
2. Derivative arrays (ru_o, zu_o) remain zero
3. zuFull becomes zero, leading to azNorm = 0 error

## Fix Strategy

### 1. Re-enable and Fix the 2D Asymmetric Transform
The transform was disabled due to "buffer overflow issues" from incorrect use of thread local storage.

### 2. Key Issues to Address
- The asymmetric transform functions need proper implementation
- Array bounds and indexing must be verified
- Derivative calculations for asymmetric components are missing

### 3. Implementation Steps
1. Fix the FourierToReal2DAsymmFastPoloidal function
2. Ensure proper derivative calculation for asymmetric modes
3. Add proper array initialization and bounds checking
4. Test with simple asymmetric case

### 4. Comparison with jVMEC
jVMEC properly computes derivatives in the transform:
- For R (sin basis): ru = rmnsc * m * cos(m*theta)  
- For Z (cos basis): zu = -zmncc * m * sin(m*theta)

This ensures zuFull is populated correctly.