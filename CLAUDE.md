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

## Implementation Notes

### Negative Toroidal Mode Numbers
- We do not use negative toroidal mode numbers n due to the way one can compress a fourier series in 2d with half sided in one variable

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

## Critical Implementation Constraints

### NEVER CHANGE SYMMETRIC BEHAVIOR (CRITICAL CONSTRAINT)
- **🚨 ABSOLUTE REQUIREMENT: Any modifications to asymmetric Fourier transforms MUST NOT affect the behavior when lasym=false (symmetric case)**
- **🚨 The symmetric variant (lasym=F) is working correctly and MUST remain unchanged**
- Always test both lasym=true and lasym=false to ensure no regression in symmetric behavior
- Upstream has working symmetric transforms in `FourierToReal3DSymmFastPoloidal` - do not break this
- **Before making ANY changes, verify against symmetric baseline behavior**

### Reference Source Code Locations
- **jVMEC source code**: `../jVMEC/` - Use for exact negative n mode handling verification
- **educational_VMEC source code**: `../educational_VMEC/` - Use for detailed algorithm understanding
- **Design documentation**: `../benchmark_vmec/design/`

### Verification Requirements
- **All negative n mode handling and asymmetric transform behavior must match jVMEC exactly, not theoretical expectations**
- When implementing transforms, verify against actual jVMEC/educational_VMEC source code, not mathematical assumptions
- Use meticulous debug output comparing all three codes (VMEC++, jVMEC, educational_VMEC) line-by-line

## External Resources
- You can find general instructions on the codebase for agentic coding in AGENTS.md

## VMEC Benchmark Tool

A comprehensive benchmarking tool is available as a symlink `benchmark_vmec` in the VMEC++ directory for comparing VMEC++, educational_VMEC, jVMEC, and VMEC2000.

### Tool Overview
The benchmark tool (https://github.com/itpplasma/benchmark_vmec) provides automated comparison across multiple VMEC implementations:
- **VMEC++**: Modern C++ implementation (this repository)
- **Educational VMEC**: Reference Fortran implementation
- **jVMEC**: Java implementation (most up-to-date with bugfixes)
- **VMEC2000**: Python interface to VMEC

### Key Features
- Automated three-code comparison with detailed debug output
- Side-by-side execution of VMEC++, educational_VMEC, and jVMEC
- Support for both symmetric (lasym=F) and asymmetric (lasym=T) cases
- Timestamped debug directories for each run
- Comparison of physics quantities (beta, MHD energy, aspect ratio, etc.)
- Automated input file cleaning for compatibility

### Usage from VMEC++ Directory
The benchmark tool is available as a symlink in the VMEC++ directory:

```bash
# Run symmetric case comparison (SOLOVEV test case)
benchmark_vmec/compare_symmetric_debug.sh

# Run asymmetric case comparison
benchmark_vmec/compare_asymmetric_debug.sh

# View benchmark results
ls benchmark_vmec/benchmark_results/
cat benchmark_vmec/benchmark_results/comparison_report.md

# Access specific debug runs
ls benchmark_vmec/symmetric_debug_*/
ls benchmark_vmec/asymmetric_debug_*/
```

### Debug Output Structure
Each debug run creates a timestamped directory with outputs from all three codes:

```
symmetric_debug_20250719_090220/
├── educational_vmec/
│   ├── educational_vmec_output.log  # Full console output
│   ├── input.SOLOVEV               # Input file used
│   ├── wout_SOLOVEV.nc            # NetCDF output with solution
│   ├── jxbout_SOLOVEV.nc          # J×B force output
│   ├── mercier.SOLOVEV            # Mercier stability data
│   └── threed1.SOLOVEV            # 3D equilibrium data
├── jvmec/
│   ├── jvmec_output.log           # Console output
│   ├── input_cleaned.txt          # Cleaned input (comments removed)
│   └── wout_input_cleaned.nc      # NetCDF output
└── vmecpp/
    ├── input.SOLOVEV               # Input file
    ├── SOLOVEV.json               # JSON format (if available)
    └── vmecpp_output.log          # Console output
```

### Benchmark Results
The tool generates comprehensive comparison reports:
- `comparison_report.md`: Detailed comparison of physics quantities
- `comparison_table.csv`: Tabular data for analysis
- Per-case subdirectories with logs and outputs

### Reference Implementation Priority
- **Primary Reference**: jVMEC - most up-to-date with bugfixes and optimizations
- **Secondary Reference**: educational_VMEC - for additional insight
- **Target**: VMEC++ asymmetric implementation should match jVMEC exactly

### Building with FPM (Fortran Package Manager)
```bash
cd benchmark_vmec

# Build the tool
fpm build

# Build all VMEC implementations
fpm run vmec-build

# Run full benchmark suite
fpm run vmec-benchmark -- run

# Run with limited test cases
fpm run vmec-benchmark -- run --limit 5

# Other commands
fpm run vmec-benchmark -- setup         # Clone repositories
fpm run vmec-benchmark -- list-repos    # Check repository status
fpm run vmec-benchmark -- --help        # Show all options

# Run unit tests
fpm test
```

### Repository Structure
The benchmark tool manages VMEC repositories as symlinks:
```
benchmark_vmec/vmec_repos/
├── educational_VMEC -> ../../educational_VMEC
├── jVMEC -> ../../jVMEC
├── vmecpp -> ../../vmecpp
└── VMEC2000/  # Cloned directly
```

### Important Notes
- Input files are automatically cleaned for each implementation
- VMEC++ uses JSON format when available (automatic detection)
- jVMEC requires manual setup (private repository)
- Debug scripts provide the most reliable comparison method
- Results focus on physics validation, not performance
