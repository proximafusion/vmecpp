# VMEC++ Asymmetric Equilibrium Fix Plan

## 🎯 CRITICAL OBJECTIVE: Fix azNorm=0 Error for Asymmetric Equilibria

### 🚨 CRITICAL CONSTRAINT: Test Symmetric Mode with EVERY Commit
**NEVER break symmetric mode** - test before and after each change!

## Current Status
- **Problem**: Asymmetric equilibria fail with "azNorm should never be 0.0" error
- **Root Cause**: `FourierToReal2DAsymmFastPoloidal` is disabled and doesn't compute derivatives
- **Impact**: `zuFull` array remains zero, causing `azNorm = sum(zuFull^2) = 0`

## Implementation Plan (Small Steps with Testing)

### Phase 1: Add Derivative Computation to Asymmetric Transform ✅ COMPLETED

#### Step 1.1: Update Function Signature ✅
- [x] Modify `fourier_asymmetric.h` to add derivative output parameters:
  ```cpp
  void FourierToReal2DAsymmFastPoloidal(
      // existing parameters...
      absl::Span<double> ru_real,  // ADD: dR/dtheta
      absl::Span<double> zu_real   // ADD: dZ/dtheta
  );
  ```
- [x] **TEST**: Compile and verify no errors
- [x] **TEST**: Run symmetric test suite - MUST PASS

#### Step 1.2: Implement Derivative Calculation ✅
- [x] In `fourier_asymmetric.cc`, add derivative computation:
  ```cpp
  // Asymmetric contributions
  r_real[idx] += rsc * sin_mu;
  z_real[idx] += zcc * cos_mu;
  
  // ADD: Asymmetric derivatives (following jVMEC pattern)
  ru_real[idx] += m * rsc * cos_mu;  // dR/dtheta
  zu_real[idx] -= m * zcc * sin_mu;  // dZ/dtheta
  ```
- [x] **TEST**: Unit test for derivative calculation (derivatives already implemented)
- [x] **TEST**: Symmetric mode still works

#### Step 1.3: Update ideal_mhd_model.cc ✅
- [x] Re-enable the 2D asymmetric transform (remove early return)
- [x] Pass derivative arrays to the function
- [x] Ensure derivatives are added to `ru_o`, `zu_o` arrays
- [x] **TEST**: Asymmetric case runs without azNorm=0 error
- [x] **TEST**: Symmetric convergence unchanged

### Phase 2: Fix Array Bounds and Buffer Overflow Issues ⏳ NEXT

#### Step 2.1: Identify Buffer Overflow
- [ ] Add bounds checking to asymmetric transform
- [ ] Use AddressSanitizer to catch overflow
- [ ] Document exact location of overflow
- [ ] **TEST**: No memory errors with sanitizer

#### Step 2.2: Fix Array Sizing
- [ ] Verify all array allocations match usage
- [ ] Check thread local storage sizing
- [ ] Fix any off-by-one errors
- [ ] **TEST**: Asymmetric runs without crashes
- [ ] **TEST**: Symmetric still converges

### Phase 3: Validate Against jVMEC ⏳

#### Step 3.1: Create Test Cases
- [ ] Use `up_down_asymmetric_tokamak.json` as test case
- [ ] Create minimal asymmetric test with small perturbation
- [ ] **TEST**: Both cases run without errors

#### Step 3.2: Compare with jVMEC
- [ ] Run same input through jVMEC
- [ ] Compare initial iterations
- [ ] Verify zuFull is populated correctly
- [ ] **TEST**: Results match within tolerance

### Phase 4: Integration Testing ⏳

#### Step 4.1: Convergence Tests
- [ ] Test asymmetric equilibrium convergence
- [ ] Monitor force residuals
- [ ] Verify physical results (beta, profiles)
- [ ] **TEST**: At least one asymmetric case converges

#### Step 4.2: Regression Testing
- [ ] Run full symmetric test suite
- [ ] Compare performance metrics
- [ ] Verify no numerical differences
- [ ] **TEST**: All symmetric tests pass

## Test Script for Every Commit

Create `test_symmetric_regression.py`:
```python
#!/usr/bin/env python3
"""Test symmetric mode still works after changes"""
from vmecpp.cpp import _vmecpp as vmec

# Test symmetric Solovev
indata = vmec.VmecINDATAPyWrapper.from_file(
    "src/vmecpp/cpp/vmecpp/test_data/solovev.json"
)
assert not indata.lasym, "Must be symmetric"

# Run with limited iterations
indata.niter_array = [50]
output = vmec.run(indata)

# Check convergence
assert output.wout.ier_flag == 0, "Symmetric must converge"
print("✓ Symmetric mode works")
```

Run before EVERY commit:
```bash
python test_symmetric_regression.py
```

## Success Criteria
1. ✅ Asymmetric equilibria run without azNorm=0 error
2. ✅ zuFull array is populated with non-zero values
3. ✅ At least one asymmetric case converges
4. ✅ NO regression in symmetric mode
5. ✅ All tests pass

## Current Focus
**NEXT STEP**: Implement Step 1.1 - Update function signature

Remember: Small steps, test after each change, NEVER break symmetric mode!