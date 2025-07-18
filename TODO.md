# VMEC++ Asymmetric Equilibria Implementation TODO

## Phase 1: Foundation - Fourier Transform Infrastructure

### 1.1 Create Basic Test Infrastructure
- [ ] Create src/vmecpp/cpp/vmecpp/vmec/fourier_asymmetric/fourier_asymmetric_test.cc
- [ ] Create src/vmecpp/cpp/vmecpp/vmec/fourier_asymmetric/BUILD.bazel
- [ ] Write test for FourierToReal3DAsymmFastPoloidal with simple 1-mode case
- [ ] Write test for RealToFourier3DAsymmFastPoloidal with simple 1-mode case
- [ ] Write test for round-trip transform accuracy

### 1.2 Implement Core Transform Functions
- [ ] Create src/vmecpp/cpp/vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h with function declarations
- [ ] Create src/vmecpp/cpp/vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.cc with stub implementations
- [ ] Implement FourierToReal3DAsymmFastPoloidal for single mode
- [ ] Implement RealToFourier3DAsymmFastPoloidal for single mode
- [ ] Verify tests pass for single mode

### 1.3 Extend to Multi-Mode Transforms
- [ ] Write test for multi-mode forward transform
- [ ] Write test for multi-mode inverse transform
- [ ] Extend FourierToReal3DAsymmFastPoloidal to handle all modes
- [ ] Extend RealToFourier3DAsymmFastPoloidal to handle all modes
- [ ] Verify multi-mode tests pass

### 1.4 Add 2D Transform Variants
- [ ] Write test for FourierToReal2DAsymmFastPoloidal
- [ ] Write test for RealToFourier2DAsymmFastPoloidal
- [ ] Implement 2D transform functions
- [ ] Verify 2D transform tests pass

## Phase 2: Symmetrization Operations

### 2.1 Geometry Symmetrization
- [ ] Write test for SymmetrizeRealSpaceGeometry basic case
- [ ] Compare expected output with educational_VMEC's symrzl
- [ ] Implement SymmetrizeRealSpaceGeometry
- [ ] Write test for edge cases (boundary points)
- [ ] Verify all symmetrization tests pass

### 2.2 Force Symmetrization
- [ ] Write test for SymmetrizeForces basic case
- [ ] Implement SymmetrizeForces
- [ ] Write test for force conservation properties
- [ ] Verify force symmetrization tests pass

## Phase 3: Integration with Core Algorithm

### 3.1 Prepare Integration Tests
- [ ] Create test/test_ideal_mhd_asymmetric.cc
- [ ] Write test comparing symmetric mode output (lasym=false baseline)
- [ ] Write test for simple asymmetric equilibrium

### 3.2 Modify ideal_mhd_model.cc
- [ ] Write test for geometryFromFourier with lasym=true
- [ ] Replace error placeholder in geometryFromFourier
- [ ] Add conditional call to asymmetric transforms
- [ ] Verify geometry computation test passes

### 3.3 Force Calculation Integration
- [ ] Write test for forcesToFourier with lasym=true
- [ ] Replace error placeholder in forcesToFourier
- [ ] Add conditional call to asymmetric force transforms
- [ ] Verify force calculation test passes

## Phase 4: Special Cases and Boundary Conditions

### 4.1 m=1 Mode Conversion
- [ ] Write test for m=1 mode conversion (compare with educational_VMEC)
- [ ] Implement convert_sym equivalent
- [ ] Implement convert_asym equivalent
- [ ] Verify m=1 mode tests pass

### 4.2 Boundary Condition Updates
- [ ] Write test for asymmetric boundary conditions
- [ ] Update boundary condition handling in relevant functions
- [ ] Verify boundary condition tests pass

## Phase 5: End-to-End Testing

### 5.1 Simple Asymmetric Test Case
- [ ] Create asymmetric tokamak test input
- [ ] Write test comparing output with educational_VMEC
- [ ] Debug any discrepancies using detailed output
- [ ] Verify test case passes

### 5.2 Complex Asymmetric Test Case
- [ ] Create stellarator with asymmetry test input
- [ ] Write test comparing key quantities with jVMEC
- [ ] Debug any discrepancies
- [ ] Verify test case passes

### 5.3 Performance and Numerical Tests
- [ ] Write test for force balance convergence
- [ ] Write test for energy conservation
- [ ] Verify numerical properties match reference codes

## Phase 6: Output and Diagnostics

### 6.1 Output Quantities
- [ ] Write test for asymmetric output quantities
- [ ] Update output routines to handle full u-range
- [ ] Verify output tests pass

### 6.2 Diagnostic Output
- [ ] Add debug output options for asymmetric transforms
- [ ] Create comparison scripts with educational_VMEC
- [ ] Document debug output usage

## Notes:
- Each test must be written BEFORE implementation (RED phase)
- Keep each implementation minimal to pass the test (GREEN phase)
- Refactor only after tests pass (REFACTOR phase)
- Run all tests after each step
- Use educational_VMEC and jVMEC as reference implementations
- **CRITICAL: Add debug output at EVERY TINY STEP comparing with educational_VMEC**
- **CRITICAL: Run tokamak input with debug prints matching educational_VMEC positions**
- **CRITICAL: Never implement any function without proper test coverage**
