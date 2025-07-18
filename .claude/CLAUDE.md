# VMEC++ Asymmetric Equilibria Implementation Guide

## Overview
This document contains critical information for implementing asymmetric equilibria support (lasym=true) in VMEC++. The implementation follows patterns from educational_VMEC and jVMEC.

## Key Concepts

### Fourier Basis Functions
**Symmetric mode (lasym=false):**
- R ~ rmncc*cos(m*u)*cos(n*v) + rmnss*sin(m*u)*sin(n*v)
- Z ~ zmnsc*sin(m*u)*cos(n*v) + zmncs*cos(m*u)*sin(n*v)
- Lambda ~ lmnsc*sin(m*u)*cos(n*v) + lmncs*cos(m*u)*sin(n*v)

**Asymmetric mode (lasym=true):**
- R gains: rmnsc*sin(m*u)*cos(n*v) + rmncs*cos(m*u)*sin(n*v)
- Z gains: zmncc*cos(m*u)*cos(n*v) + zmnss*sin(m*u)*sin(n*v)
- Lambda gains: lmncc*cos(m*u)*cos(n*v) + lmnss*sin(m*u)*sin(n*v)

### Array Ranges
- Symmetric: u ∈ [0, π], exploiting stellarator symmetry
- Asymmetric: u ∈ [0, 2π], full poloidal angle range

### Symmetrization Operation
For u in [0,π]: total = symmetric + anti-symmetric
For u in [π,2π]: total = symmetric - anti-symmetric

## Required Components

### 1. New File: fourier_asymmetric.cc/h
Must implement:
- `FourierToReal3DAsymmFastPoloidal()` - Equivalent to educational_VMEC's `totzspa`
- `FourierToReal2DAsymmFastPoloidal()` - For axisymmetric case
- `SymmetrizeRealSpaceGeometry()` - Equivalent to educational_VMEC's `symrzl`
- `RealToFourier3DAsymmFastPoloidal()` - Equivalent to educational_VMEC's `tomnspa`
- `RealToFourier2DAsymmFastPoloidal()` - For axisymmetric case
- `SymmetrizeForces()` - Apply symmetry to forces

### 2. Modifications to ideal_mhd_model.cc
- Replace error placeholders with calls to asymmetric transforms
- Handle full u-range in geometry arrays
- Ensure proper force symmetrization

### 3. Special m=1 Mode Handling
- Implement conversion routines for m=1 modes (see `convert_sym` and `convert_asym` in educational_VMEC)
- This affects boundary condition handling

## Reference Implementation Locations
- educational_VMEC: `totzspa.f90`, `tomnspa.f90`, `symrzl.f90`
- jVMEC: Similar functions with Java implementation
- Design docs: ../benchmark_vmec/design/

## Testing Strategy
1. **CRITICAL: Follow strict Test-Driven Development (TDD)**
   - Write failing tests FIRST before any implementation
   - Add debug output at EVERY TINY STEP comparing with educational_VMEC
   - Run tokamak input with debug prints matching educational_VMEC positions
2. Start with simple asymmetric test cases
3. Compare outputs with educational_VMEC and jVMEC
4. Verify conservation properties
5. Validate Jacobian calculations with full u-range

## Current State
- HandoverStorage already has asymmetric arrays allocated
- FourierGeometry has spans for asymmetric coefficients
- Sizes class has lasym flag
- Transform functions are missing (placeholders with errors)

## Build and Testing Information

### Build System
- VMEC++ uses Bazel for C++ builds and tests
- Python bindings use pytest for testing

### C++ Testing Structure
- Test files are named with `_test.cc` suffix
- Tests are located in the same directory as source files
- Tests use Google Test (gtest) framework
- Test rules in BUILD.bazel use `cc_test` with dependency on `@googletest//:gtest_main`

### Running Tests
- Navigate to `src/vmecpp/cpp` directory
- Run all tests: `bazel test //...`
- Run specific test: `bazel test //path/to:test_name`
- Run with specific config: `bazel test --config=opt //...` (opt, asan, ubsan)

### File Locations
- New asymmetric Fourier module should be in:
  - `src/vmecpp/cpp/vmecpp/vmec/fourier_asymmetric/`
  - Files: `fourier_asymmetric.h`, `fourier_asymmetric.cc`, `fourier_asymmetric_test.cc`, `BUILD.bazel`

### Integration Points
- Error placeholders in `src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc`:
  - Line ~387: "asymmetric inv-DFT not implemented yet"
  - Line ~389: "symrzl not implemented yet"
  - Line ~415: "asymmetric fwd-DFT not implemented yet"
  - Line ~417: "symforce not implemented yet"

## External Resources
- You can find general instructions on the codebase for agentic coding in AGENTS.md
