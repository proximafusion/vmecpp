# VMEC++ Naming Guide

## Philosophy: Physics-Aware Modern C++ Style

VMEC++ follows a **domain-aware adaptation** of the Google C++ Style Guide that preserves critical physics knowledge while modernizing software engineering practices.

## Core Principles

### 1. **Preserve Physics Domain Knowledge**
Traditional physics variable names encode decades of research understanding and must be preserved.

### 2. **Modernize Infrastructure Code**
Non-physics infrastructure (utilities, I/O, error handling) follows Google C++ style.

### 3. **Make Data Flow Explicit**
Use naming conventions that immediately reveal computational intent and data dependencies.

---

## Naming Conventions

### **Functions**: CamelCase (Google Style)

```cpp
// ✅ All functions use CamelCase with capital first letter
class IdealMhdModel {
  void ComputeGeometry();           // Infrastructure function
  void UpdatePreconditioner();      // Infrastructure function
  void FourierToReal();            // Physics computation (descriptive name)
};

// ✅ Legacy physics function names acceptable when well-established
void funct3d();    // Historical VMEC routine name - keep for consistency
void forces();     // Core physics function - widely understood
```

### **Variables**: Context-Dependent Naming

#### **Local Variables**: `snake_case` for infrastructure, traditional names for physics
```cpp
void SomeFunction() {
  // ✅ Infrastructure variables: descriptive snake_case
  int iteration_count = 0;
  bool convergence_achieved = false;
  double tolerance_threshold = 1e-6;

  // ✅ Physics variables: preserve traditional names
  double iotas = 0.0;         // Rotational transform (stellarator physics)
  double presf = 0.0;         // Pressure on full grid
  double phips = 0.0;         // Toroidal flux derivative
}
```

#### **Member Variables**: Traditional physics names + trailing underscore
```cpp
class IdealMhdModel {
private:
  // ✅ Core physics variables: preserve names, add trailing underscore
  std::vector<double> bsupu_;     // B^θ contravariant component
  std::vector<double> bsupv_;     // B^ζ contravariant component
  std::vector<double> iotaf_;     // Rotational transform, full grid

  // ✅ Infrastructure variables: descriptive names + trailing underscore
  bool convergence_achieved_;     // Clear intent
  int iteration_count_;           // Descriptive
  FlowControl flow_control_;      // Modernized from m_fc
};
```

### **Function Parameters**: Mutable Parameter Convention

**CRITICAL**: Use `m_` prefix for parameters that **WILL BE MODIFIED** by the function:

```cpp
// ✅ Crystal clear data flow intent
void ComputeMagneticField(
    // INPUTS (read-only):
    const std::vector<double>& iotaf,           // Rotational transform
    const std::vector<double>& presf,           // Pressure profile
    const Sizes& grid_sizes,                    // Grid configuration

    // OUTPUTS (will be modified):
    std::vector<double>& m_bsupu,               // B^θ - MODIFIED
    std::vector<double>& m_bsupv,               // B^ζ - MODIFIED
    FourierGeometry& m_geometry);               // Geometry - MODIFIED

// ✅ Mixed input/output clearly identified
void UpdateEquilibrium(
    const VmecConstants& constants,             // INPUT: Physical constants
    RadialProfiles& m_profiles,                 // INPUT/OUTPUT: Modified
    FourierGeometry& m_fourier_geometry,        // OUTPUT: Computed geometry
    bool& m_convergence_flag);                  // OUTPUT: Convergence status
```

### **Constants**: kCamelCase with Physics Context

VMEC++ consolidates algorithmic and physical constants in `vmec_algorithm_constants.h` to replace magic numbers and improve code readability:

```cpp
namespace vmecpp::vmec_algorithm_constants {
  // ✅ Poloidal mode parity constants with descriptive names (replaces m_evn/m_odd)
  static constexpr int kEvenParity = 0;    // Even poloidal mode numbers (m=0,2,4,...)
  static constexpr int kOddParity = 1;     // Odd poloidal mode numbers (m=1,3,5,...)

  // ✅ Algorithm constants with clear purpose
  static constexpr int kMinIterationsForM1Constraint = 2;
  static constexpr double kFastConvergenceThreshold = 1.0e-6;
  static constexpr int kMaxIterationDeltaForEdgeForces = 50;

  // ✅ Physics constants with historical context
  static constexpr int kSignOfJacobian = -1;              // "signgs" in Fortran VMEC
  static constexpr double kMagneticFieldBlendingFactor = 0.05;  // "pdamp" in Fortran
  static constexpr double kToroidalNormalizationFactor = 2.0 * M_PI;

  // ✅ Convergence and tolerance constants
  static constexpr double kDefaultForceTolerance = 1.0e-10;
  static constexpr double kVacuumPressureThreshold = 1.0e-3;
}
```

### **Using Declarations**: File-Level Import for Readability

Follow Google C++ Style Guide by using file-level `using` declarations for frequently used symbols:

```cpp
// ✅ File-level using declarations for constants
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kOddParity;

// ✅ Now use descriptive names directly in code
void ProcessFourierModes() {
  for (int parity = kEvenParity; parity <= kOddParity; ++parity) {
    // Clear poloidal mode number parity vs cryptic m_evn/m_odd
    if (parity == kEvenParity) {
      // Process Fourier harmonics with even poloidal mode numbers (m=0,2,4,...)
    } else {
      // Process Fourier harmonics with odd poloidal mode numbers (m=1,3,5,...)
    }
  }
}

// ❌ Avoid global using directives
using namespace vmecpp::vmec_algorithm_constants;  // DON'T do this

// ❌ Avoid long namespace qualifiers in frequently-used code
vmecpp::vmec_algorithm_constants::kEvenParity;     // Too verbose for frequent use
```

---

## Fourier Basis Naming: Critical Domain Knowledge

**Important**: This section discusses **product basis parity** (even/odd trigonometric functions), which is distinct from **poloidal mode number parity** (kEvenParity/kOddParity for even/odd values of m).

### **The Two Fourier Representations in VMEC++**

VMEC++ uses **different Fourier bases** for internal computation vs external interface:

#### **Internal Product Basis** (Computational Efficiency)
```cpp
/**
 * INTERNAL Fourier coefficients using product basis cos(mθ) * cos(nζ).
 *
 * Mathematical form: R(θ,ζ) = Σ rmncc_[m,n] * cos(mθ) * cos(nζ)
 *
 * CRITICAL: This is NOT the combined basis cos(mθ-nζ) used externally.
 *
 * Computational advantage: Enables separable FFTs (θ and ζ independent)
 * Conversion: cos(mθ-nζ) = cos(mθ)cos(nζ) + sin(mθ)sin(nζ) = rmncc + rmnss
 * External equivalent: rmnc (combined basis)
 * Physics: Even-even trigonometric parity component of R boundary
 */
std::vector<double> rmncc_;

/**
 * INTERNAL Fourier coefficients using product basis sin(mθ) * sin(nζ).
 * Mathematical form: R(θ,ζ) = Σ rmnss_[m,n] * sin(mθ) * sin(nζ)
 * Physics: Odd-odd trigonometric parity component of R boundary
 */
std::vector<double> rmnss_;
```

#### **External Combined Basis** (Researcher Interface)
```cpp
/**
 * EXTERNAL Fourier coefficients using combined basis cos(mθ - nζ).
 *
 * Mathematical form: R(θ,ζ) = Σ rmnc[m,n] * cos(mθ - nζ)
 *
 * Traditional VMEC format: Used in wout files, researcher-familiar
 * Conversion from internal: rmnc = rmncc + rmnss (3D case)
 * Stellarator symmetry: cos(mθ-nζ) terms (stellarator-symmetric harmonics)
 */
RowMatrixXd rmnc;    // External interface - preserve traditional name

/**
 * EXTERNAL Z coefficients using combined basis sin(mθ - nζ).
 * Traditional VMEC format for Z-coordinate stellarator-symmetric terms
 */
RowMatrixXd zmns;    // External interface - preserve traditional name
```

#### **Suffix Convention for Internal Variables**
```cpp
// Internal product basis suffix meanings:
// NOTE: This "parity" refers to trigonometric function parity, NOT poloidal mode number parity
std::vector<double> rmncc_;  // cos(mθ) * cos(nζ)  [even-even trig parity]
std::vector<double> rmnss_;  // sin(mθ) * sin(nζ)  [odd-odd trig parity]
std::vector<double> rmnsc_;  // sin(mθ) * cos(nζ)  [odd-even trig parity]
std::vector<double> rmncs_;  // cos(mθ) * sin(nζ)  [even-odd trig parity]

std::vector<double> zmnsc_;  // Z: sin(mθ) * cos(nζ)
std::vector<double> zmncs_;  // Z: cos(mθ) * sin(nζ)
std::vector<double> lmnsc_;  // λ: sin(mθ) * cos(nζ)
std::vector<double> lmncs_;  // λ: cos(mθ) * sin(nζ)
```

### **Conversion Function Naming**
```cpp
// ✅ Basis transformation functions use descriptive names
class FourierBasisFastPoloidal {
  // Convert external combined → internal product basis
  int CosToProductBasis(...)  // cos(mθ-nζ) → {cc, ss}
  int SinToProductBasis(...)  // sin(mθ-nζ) → {sc, cs}

  // Convert internal product → external combined basis
  int ProductBasisToCos(...)  // {cc, ss} → cos(mθ-nζ)
  int ProductBasisToSin(...)  // {sc, cs} → sin(mθ-nζ)
};
```

---

## Documentation Strategy

### **Physics Variables**: Comprehensive Context
```cpp
/**
 * Contravariant magnetic field component B^θ in VMEC flux coordinates.
 *
 * Physics context:
 * - Represents magnetic field strength in poloidal direction
 * - Computed from equilibrium force balance ∇p = J × B
 * - Used in energy and force calculations
 *
 * Computational details:
 * - Grid: Half-grid in radial direction, full grid in angular directions
 * - Units: Tesla
 * - Memory layout: [radial_index * angular_size + angular_index]
 *
 * Historical reference: "bsupu" in Fortran VMEC
 * Related variables: bsupv_ (B^ζ component), bsubv_ (covariant B_ζ)
 */
std::vector<double> bsupu_;
```

### **Infrastructure Variables**: Clear Intent
```cpp
/**
 * Iteration counter for main equilibrium solver loop.
 *
 * Tracks progress through force-balance iterations until convergence.
 * Used for:
 * - Convergence criteria evaluation
 * - Checkpoint timing decisions
 * - Diagnostic output frequency
 *
 * Range: [0, maximum_iterations]
 */
int iteration_count_;
```

---

## Implementation Examples

### **Before and After: Function Signatures**
```cpp
// ❌ BEFORE: Unclear data flow, mixed conventions
absl::StatusOr<bool> update(
    FourierGeometry& m_decomposed_x,     // Modified? Unclear!
    FourierForces& m_physical_f,         // Modified? Unclear!
    bool& m_need_restart,                // Modified? Unclear!
    const int iter2);                    // Magic variable name

// ✅ AFTER: Crystal clear data flow and intent
absl::StatusOr<bool> Update(
    const int iteration_count,                    // INPUT: Clear name
    FourierGeometry& m_decomposed_geometry,       // OUTPUT: Modified
    FourierForces& m_physical_forces,             // OUTPUT: Modified
    bool& m_restart_required);                    // OUTPUT: Modified
```

### **Before and After: Class Members**
```cpp
// ❌ BEFORE: Mixed conventions, unclear purposes
class IdealMhdModel {
  bool m_liter_flag;           // Hungarian notation + unclear
  std::vector<double> bsupu;   // No context, unclear if member
  FlowControl m_fc;            // Cryptic abbreviation
};

// ✅ AFTER: Consistent conventions, clear intent
class IdealMhdModel {
  bool convergence_achieved_;           // Clear infrastructure naming
  std::vector<double> bsupu_;           // Physics name + member convention
  FlowControl flow_control_;            // Descriptive infrastructure naming
};
```

---

## File Organization: Google Style

```cpp
// ✅ Header files: snake_case
ideal_mhd_model.h
fourier_basis_fast_poloidal.h
vmec_algorithm_constants.h

// ✅ Include guards: UPPER_SNAKE_CASE with full path
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_

// ✅ Class names: CamelCase
class IdealMhdModel {
class FourierBasisFastPoloidal {
class VmecAlgorithmConstants {
```

---

## Recent Improvements: Constants Migration ✅

### **Completed: m_evn/m_odd → kEvenParity/kOddParity Migration**

Successfully migrated all 64 occurrences of cryptic `m_evn`/`m_odd` constants to descriptive `kEvenParity`/`kOddParity` throughout the VMEC++ codebase:

```cpp
// ❌ BEFORE: Cryptic parity indexing
if (parity == m_evn) {
  // Process even modes...
}
rmncc[idx] += fourier_data[m_odd];

// ✅ AFTER: Self-documenting poloidal mode parity operations
if (parity == kEvenParity) {
  // Process Fourier harmonics with even poloidal mode numbers...
}
rmncc[idx] += fourier_data[kOddParity];
```

**Benefits achieved:**
- **Immediate comprehension**: Code readers instantly understand poloidal mode number parity
- **Reduced cognitive load**: No need to remember arbitrary integer mappings
- **Enhanced maintainability**: Self-documenting code reduces debugging time
- **Physics clarity**: Links code directly to Fourier mode classification by poloidal mode number

### **Completed: Comprehensive Constants Consolidation**

Created `vmec_algorithm_constants.h` as central repository for 50+ algorithmic, physical, and mathematical constants previously scattered across the codebase:

```cpp
// ✅ Physics constants with full context documentation
static constexpr double kVacuumPermeability = 4.0e-7 * M_PI;  // Matches Fortran VMEC
static constexpr double kIonLarmorRadiusCoefficient = 3.2e-3;  // Used in output_quantities

// ✅ Algorithm constants with usage locations documented
static constexpr int kMinIterationsForM1Constraint = 2;        // From ideal_mhd_model.cc
static constexpr double kFastConvergenceThreshold = 1.0e-6;    // Convergence acceleration

// ✅ Gauss-Legendre quadrature arrays for high-accuracy integration
static constexpr std::array<double, 10> kGaussLegendreWeights10 = {...};
```

## Implementation Strategy

### **Phase 1: High-Impact, Low-Risk Changes ✅ COMPLETED**
1. ✅ **Replace magic numbers** with named constants from `vmec_algorithm_constants.h`
2. ✅ **Migrate parity constants** from `m_evn/m_odd` to `kEvenParity/kOddParity`
3. **Standardize function parameter signatures** with `m_` prefix for mutable parameters
4. **Add comprehensive documentation** to Fourier basis variables
5. **Update new code** to follow conventions consistently

### **Phase 2: Gradual Infrastructure Modernization**
1. **Member variable naming** in utility classes
2. **Function naming** for infrastructure code
3. **File organization** improvements

### **Phase 3: Enhanced Documentation**
1. **Cross-reference documentation** between internal and external representations
2. **Conversion function documentation** with mathematical relationships
3. **Developer onboarding guides** explaining basis choices

---

## Key Takeaways

1. **Preserve Physics Wisdom**: Traditional names like `bsupu`, `iotaf`, `presf` encode domain knowledge
2. **Modernize Infrastructure**: Error handling, utilities, I/O use Google C++ style
3. **Make Data Flow Explicit**: `m_` prefix immediately shows what gets modified
4. **Document Basis Distinctions**: Critical for understanding VMEC++ architecture
5. **Respect Computational Choices**: Product basis enables performance, combined basis enables compatibility
6. **Use Descriptive Constants**: `kEvenParity` refers to Fourier harmonics with even poloidal mode number m, and `kOddParity` refers to Fourier harmonics with odd poloidal mode number m
7. **Consolidate Magic Numbers**: Central `vmec_algorithm_constants.h` improves maintainability
8. **Follow Google C++ Style**: File-level `using` declarations for readability without namespace pollution
9. **Eliminate Deprecated Headers**: Remove C++17-deprecated includes like `<cstdbool>`

**Recent Achievements**: Successfully migrated 64 occurrences of cryptic parity constants and consolidated 50+ algorithmic constants, demonstrating that systematic modernization can preserve physics domain knowledge while improving code clarity.

This naming guide bridges the gap between modern software engineering practices and deep physics domain knowledge, making VMEC++ both maintainable and scientifically accessible.
