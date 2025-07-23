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

## Source Code Encoding Standards

### **ASCII-Only Requirement**
All VMEC++ source code files (`.cc`, `.h`, `.cpp`, `.hpp`) must contain only ASCII characters (ord 0-127).

**Rationale:**
- Ensures consistent behavior across different editors, compilers, and operating systems
- Prevents encoding-related build failures in international development environments
- Maintains compatibility with legacy build systems and tools

**Mathematical Symbols in Comments:**
Use LaTeX notation for mathematical expressions in comments:

```cpp
// [CORRECT] LaTeX notation for mathematical symbols
/**
 * Vacuum magnetic permeability \mu_0 in Vs/Am.
 * Traditional definition: \mu_0 = 4\pi \times 10^{-7} Vs/Am
 * Volume integral: \iiint d\rho d\theta d\zeta over torus
 * Constraint scaling: 1/\sqrt{2}
 */

// [INCORRECT] Unicode mathematical symbols
```

---

## Naming Conventions

### **Functions**: CamelCase (Google Style)

```cpp
// [CORRECT] All functions use CamelCase with capital first letter
class IdealMhdModel {
  void ComputeGeometry();           // Infrastructure function
  void UpdatePreconditioner();      // Infrastructure function
  void FourierToReal();            // Physics computation (descriptive name)
};

// [CORRECT] Legacy physics function names acceptable when well-established
void funct3d();    // Historical VMEC routine name - keep for consistency
void forces();     // Core physics function - widely understood
```

### **Variables**: Context-Dependent Naming

#### **Local Variables**: `snake_case` for infrastructure, traditional names for physics
```cpp
void SomeFunction() {
  // [CORRECT] Infrastructure variables: descriptive snake_case
  int iteration_count = 0;
  bool convergence_achieved = false;
  double tolerance_threshold = 1e-6;

  // [CORRECT] Physics variables: preserve traditional names
  double iotas = 0.0;         // Rotational transform (stellarator physics)
  double presf = 0.0;         // Pressure on full grid
  double phips = 0.0;         // Toroidal flux derivative
}
```

#### **Member Variables**: Traditional physics names + trailing underscore
```cpp
class IdealMhdModel {
private:
  // [CORRECT] Core physics variables: preserve names, add trailing underscore
  std::vector<double> bsupu_;     // B^\theta contravariant component
  std::vector<double> bsupv_;     // B^\zeta contravariant component
  std::vector<double> iotaf_;     // Rotational transform, full grid

  // [CORRECT] Infrastructure variables: descriptive names + trailing underscore
  bool convergence_achieved_;     // Clear intent
  int iteration_count_;           // Descriptive
  FlowControl flow_control_;      // Modernized from m_fc
};
```

### **Function Parameters**: Mutable Parameter Convention

**CRITICAL**: Use `m_` prefix for parameters that **WILL BE MODIFIED** by the function:

```cpp
// [CORRECT] Crystal clear data flow intent
void ComputeMagneticField(
    // INPUTS (read-only):
    const std::vector<double>& iotaf,           // Rotational transform
    const std::vector<double>& presf,           // Pressure profile
    const Sizes& grid_sizes,                    // Grid configuration

    // OUTPUTS (will be modified):
    std::vector<double>& m_bsupu,               // B^\theta - MODIFIED
    std::vector<double>& m_bsupv,               // B^\zeta - MODIFIED
    FourierGeometry& m_geometry);               // Geometry - MODIFIED

// [CORRECT] Mixed input/output clearly identified
void UpdateEquilibrium(
    const VmecConstants& constants,             // INPUT: Physical constants
    RadialProfiles& m_profiles,                 // INPUT/OUTPUT: Modified
    FourierGeometry& m_fourier_geometry,        // OUTPUT: Computed geometry
    bool& m_convergence_flag);                  // OUTPUT: Convergence status
```

### **Constants**: Domain-Organized kCamelCase

#### **Organization by Purpose in vmec_algorithm_constants.h**

Organize constants into logical domains with comprehensive documentation:

```cpp
namespace vmecpp::vmec_algorithm_constants {

// ========== Physical Constants ==========
/**
 * Vacuum magnetic permeability \mu_0 in Vs/Am.
 * Value matches Fortran VMEC for 1:1 comparison rather than CODATA-2022.
 * Traditional definition: \mu_0 = 4\pi \times 10^{-7} Vs/Am
 */
static constexpr double kVacuumPermeability = 4.0e-7 * M_PI;

// ========== Mathematical Constants ==========
/**
 * Toroidal volume normalization factor: 2\pi
 * Used throughout for volume integrals, flux surface calculations
 */
static constexpr double kToroidalNormalizationFactor = 2.0 * M_PI;

// ========== Convergence and Tolerance Constants ==========
/**
 * Force residual threshold for iteration decisions.
 * Context: ||F_residual|| < threshold \rightarrow good convergence
 */
static constexpr double kForceResidualThreshold = 1.0e-2;

// ========== Symmetry and Parity Constants ==========
/**
 * Even parity index for Fourier harmonics with even poloidal mode number m.
 * Context: Harmonics with m=0,2,4,6,... (even poloidal mode numbers)
 */
static constexpr int kEvenParity = 0;

} // namespace vmec_algorithm_constants
```

```cpp
// [CORRECT] Replace magic numbers systematically with documented constants
using vmecpp::vmec_algorithm_constants::kJacobianIterationThreshold;
using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kOddParity;

if (iteration_count > kJacobianIterationThreshold) { /* change approach */ }
scalxc[idx * 2 + kEvenParity] = 1.0;  // even poloidal mode numbers
scalxc[idx * 2 + kOddParity] = factor;  // odd poloidal mode numbers
```

### **Using Declarations**: File-Level Import for Readability

Follow Google C++ Style Guide by using file-level `using` declarations for frequently used symbols:

```cpp
// [CORRECT] File-level using declarations for constants
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kOddParity;

// [CORRECT] Now use descriptive names directly in code
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

// [INCORRECT] Avoid global using directives
using namespace vmecpp::vmec_algorithm_constants;  // DON'T do this

// [INCORRECT] Avoid long namespace qualifiers in frequently-used code
vmecpp::vmec_algorithm_constants::kEvenParity;     // Too verbose for frequent use
```

---

## Header Inclusion Patterns

### **Constants Headers**

Include algorithm constants header and use file-level using declarations:

```cpp
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

// [CORRECT] File-level using declarations for frequently used constants
using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kOddParity;
using vmecpp::vmec_algorithm_constants::kDefaultForceTolerance;

namespace vmecpp {

void ProcessEquilibrium() {
  // [CORRECT] Use descriptive constant names directly
  for (int parity = kEvenParity; parity <= kOddParity; ++parity) {
    // Process Fourier harmonics...
  }

  if (force_residual < kDefaultForceTolerance) {
    // Convergence achieved...
  }
}

} // namespace vmecpp
```


---

## Fourier Basis Naming: Critical Domain Knowledge

**Essential Reading**: For complete understanding of VMEC++'s Fourier basis implementation, including DFT vs FFT distinctions, mathematical foundations, and conversion algorithms, see `docs/fourier_basis_implementation.md`.

**Important**: This section discusses **product basis parity** (even/odd trigonometric functions), which is distinct from **poloidal mode number parity** (kEvenParity/kOddParity for even/odd values of m).

### **The Two Fourier Representations in VMEC++**

VMEC++ uses **different Fourier bases** for internal computation vs external interface:

#### **Internal Product Basis** (Computational Efficiency)
```cpp
/**
 * INTERNAL Fourier coefficients using product basis cos(m\theta) * cos(n\zeta).
 *
 * Mathematical form: R(\theta,\zeta) = \sum_{m,n} rmncc_[m,n] * cos(m\theta) * cos(n\zeta)
 *
 * CRITICAL: This is NOT the combined basis cos(m\theta-n\zeta) used externally.
 *
 * Computational advantage: Enables separable DFT operations (\theta and \zeta independent)
 * Basis function identity: cos(m\theta-n\zeta) = cos(m\theta)cos(n\zeta) + sin(m\theta)sin(n\zeta)
 * Coefficient conversion: rmnc coefficients -> rmncc + rmnss coefficients
 * External equivalent: rmnc (combined basis)
 * Physics: Even-even trigonometric parity component of R boundary
 */
std::vector<double> rmncc_;

/**
 * INTERNAL Fourier coefficients using product basis sin(m\theta) * sin(n\zeta).
 * Mathematical form: R(\theta,\zeta) = \sum_{m,n} rmnss_[m,n] * sin(m\theta) * sin(n\zeta)
 * Physics: Odd-odd trigonometric parity component of R boundary
 */
std::vector<double> rmnss_;
```

#### **External Combined Basis** (Researcher Interface)
```cpp
/**
 * EXTERNAL Fourier coefficients using combined basis cos(m\theta - n\zeta).
 *
 * Mathematical form: R(\theta,\zeta) = \sum_{m,n} rmnc[m,n] * cos(m\theta - n\zeta)
 *
 * Traditional VMEC format: Used in wout files, researcher-familiar
 * Conversion from internal: rmnc = rmncc + rmnss (3D case)
 * Stellarator symmetry: cos(m\theta-n\zeta) terms (stellarator-symmetric harmonics)
 */
RowMatrixXd rmnc;    // External interface - preserve traditional name

/**
 * EXTERNAL Z coefficients using combined basis sin(m\theta - n\zeta).
 * Traditional VMEC format for Z-coordinate stellarator-symmetric terms
 */
RowMatrixXd zmns;    // External interface - preserve traditional name
```

#### **Suffix Convention for Internal Variables**
```cpp
// Internal product basis suffix meanings:
// NOTE: This "parity" refers to trigonometric function parity, NOT poloidal mode number parity
std::vector<double> rmncc_;  // cos(m\theta) * cos(n\zeta)  [even-even trig parity]
std::vector<double> rmnss_;  // sin(m\theta) * sin(n\zeta)  [odd-odd trig parity]
std::vector<double> rmnsc_;  // sin(m\theta) * cos(n\zeta)  [odd-even trig parity]
std::vector<double> rmncs_;  // cos(m\theta) * sin(n\zeta)  [even-odd trig parity]

std::vector<double> zmnsc_;  // Z: sin(m\theta) * cos(n\zeta)
std::vector<double> zmncs_;  // Z: cos(m\theta) * sin(n\zeta)
std::vector<double> lmnsc_;  // \lambda: sin(m\theta) * cos(n\zeta)
std::vector<double> lmncs_;  // \lambda: cos(m\theta) * sin(n\zeta)
```

### **Conversion Function Naming**
```cpp
// [CORRECT] Basis transformation functions use descriptive names
class FourierBasisFastPoloidal {
  // Convert external combined -> internal product basis
  int CosToProductBasis(...)  // cos(m\theta-n\zeta) -> {cc, ss}
  int SinToProductBasis(...)  // sin(m\theta-n\zeta) -> {sc, cs}

  // Convert internal product -> external combined basis
  int ProductBasisToCos(...)  // {cc, ss} -> cos(m\theta-n\zeta)
  int ProductBasisToSin(...)  // {sc, cs} -> sin(m\theta-n\zeta)
};
```

---

## Documentation Strategy

### **Physics Variables**: Comprehensive Context
```cpp
/**
 * Contravariant magnetic field component B^\theta in VMEC flux coordinates.
 *
 * Physics context:
 * - Represents magnetic field strength in poloidal direction
 * - Computed from equilibrium force balance \nabla p = J \times B
 * - Used in energy and force calculations
 *
 * Computational details:
 * - Grid: Half-grid in radial direction, full grid in angular directions
 * - Units: Tesla
 * - Memory layout: [radial_index * angular_size + angular_index]
 *
 * Historical reference: "bsupu" in Fortran VMEC
 * Related variables: bsupv_ (B^\zeta component), bsubv_ (covariant B_\zeta)
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

### **Mathematical Documentation Standards**

#### **LaTeX Notation in Comments**
Use consistent LaTeX notation for mathematical expressions:

```cpp
/**
 * Gauss-Legendre quadrature integration.
 * Mathematical basis: \int_{-1}^1 f(x)dx \approx \sum_i w_i f(x_i)
 *
 * Integration limits: [-1, 1]
 * Accuracy: Exact for polynomials of degree \leq 2n-1
 * Implementation: 10-point quadrature for pressure profiles
 */
static constexpr std::array<double, 10> kGaussLegendreWeights10 = {...};

/**
 * Magnetic field components in flux coordinates.
 * Force balance: J \times B = \nabla p
 * Contravariant components: B^\theta, B^\zeta
 * Units: Tesla
 */
std::vector<double> bsupu_;  // B^\theta component
```

#### **Physical Context Documentation**
Always include physical meaning alongside mathematical notation:

```cpp
/**
 * Rotational transform profile \iota(s) on full radial grid.
 *
 * Physics: \iota = d\chi/d\phi (change in poloidal angle per toroidal angle)
 * Magnetic surfaces: \iota determines field line winding
 * Stability: \iota profiles affect MHD stability boundaries
 * Units: Dimensionless (radians/radian)
 */
std::vector<double> iotaf_;
```

---

## Implementation Examples

### **Before and After: Function Signatures**
```cpp
// [INCORRECT] BEFORE: Unclear data flow, mixed conventions
absl::StatusOr<bool> update(
    FourierGeometry& m_decomposed_x,     // Modified? Unclear!
    FourierForces& m_physical_f,         // Modified? Unclear!
    bool& m_need_restart,                // Modified? Unclear!
    const int iter2);                    // Magic variable name

// [CORRECT] AFTER: Crystal clear data flow and intent
absl::StatusOr<bool> Update(
    const int iteration_count,                    // INPUT: Clear name
    FourierGeometry& m_decomposed_geometry,       // OUTPUT: Modified
    FourierForces& m_physical_forces,             // OUTPUT: Modified
    bool& m_restart_required);                    // OUTPUT: Modified
```

### **Before and After: Class Members**
```cpp
// [INCORRECT] BEFORE: Mixed conventions, unclear purposes
class IdealMhdModel {
  bool m_liter_flag;           // Hungarian notation + unclear
  std::vector<double> bsupu;   // No context, unclear if member
  FlowControl m_fc;            // Cryptic abbreviation
};

// [CORRECT] AFTER: Consistent conventions, clear intent
class IdealMhdModel {
  bool convergence_achieved_;           // Clear infrastructure naming
  std::vector<double> bsupu_;           // Physics name + member convention
  FlowControl flow_control_;            // Descriptive infrastructure naming
};
```

---

## File Organization: Google Style

```cpp
// [CORRECT] Header files: snake_case
ideal_mhd_model.h
fourier_basis_fast_poloidal.h
vmec_algorithm_constants.h

// [CORRECT] Include guards: UPPER_SNAKE_CASE with full path
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_

// [CORRECT] Class names: CamelCase
class IdealMhdModel {
class FourierBasisFastPoloidal {
class VmecAlgorithmConstants {
```

---

## Recent Improvements: Constants Migration [DONE]

### **Completed: m_evn/m_odd -> kEvenParity/kOddParity Migration**

Successfully migrated all 64 occurrences of cryptic `m_evn`/`m_odd` constants to descriptive `kEvenParity`/`kOddParity` throughout the VMEC++ codebase:

```cpp
// [INCORRECT] BEFORE: Cryptic parity indexing
if (parity == m_evn) {
  // Process even modes...
}
rmncc[idx] += fourier_data[m_odd];

// [CORRECT] AFTER: Self-documenting poloidal mode parity operations
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
9. **Enforce ASCII Compliance**: All source code must use ASCII-only characters with LaTeX notation for mathematical symbols
10. **Organize Constants by Domain**: Group related constants with comprehensive documentation in appropriate sections
11. **Use Mathematical Documentation Standards**: Combine LaTeX notation with physical context for clarity

This naming guide bridges the gap between modern software engineering practices and deep physics domain knowledge, making VMEC++ both maintainable and scientifically accessible.
