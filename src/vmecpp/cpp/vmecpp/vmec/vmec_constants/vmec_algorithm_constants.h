// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_VMEC_CONSTANTS_VMEC_ALGORITHM_CONSTANTS_H_
#define VMECPP_VMEC_VMEC_CONSTANTS_VMEC_ALGORITHM_CONSTANTS_H_

#include <array>
#include <cmath>
#include <numbers>

#include "vmecpp/common/util/real_type.h"

namespace vmecpp {

/**
 * Comprehensive collection of numerical, algorithmic, and physical constants
 * used throughout VMEC++ to replace magic numbers and improve code readability.
 */
namespace vmec_algorithm_constants {

// ========== Physical Constants ==========

/**
 * Sign of Jacobian between cylindrical and flux coordinates.
 * This defines the orientation convention for coordinate transformations.
 * Historical name: signgs from Fortran VMEC.
 * Also defined in vmec.h as kSignOfJacobian.
 */
static constexpr int kSignOfJacobian = -1;

/**
 * Scaling factor for blending between different B^zeta computation methods.
 * This damping parameter controls the mixing of two different algorithms
 * for computing the contravariant magnetic field component B^zeta.
 * Historical name: kPDamp from Fortran VMEC.
 * Also defined in vmec.h as kPDamp.
 */
static constexpr real_t kMagneticFieldBlendingFactor = 0.05L;

/**
 * Vacuum magnetic permeability \mu_0 in Vs/Am.
 *
 * Value matches Fortran VMEC for 1:1 comparison rather than CODATA-2022.
 * Used in: Biot-Savart law calculations, magnetic field computations
 * Files: magnetic_field_provider_lib.cc, external_magnetic_field.cc
 * Traditional definition: \mu_0 = 4\pi \times 10^{-7} Vs/Am
 */
static constexpr real_t kVacuumPermeability = 4.0e-7L * M_PI;

/**
 * Ion Larmor radius calculation coefficient.
 * Used in: output_quantities.cc for plasma parameter calculations
 * Context: Relates ion gyroradius to plasma parameters
 */
static constexpr real_t kIonLarmorRadiusCoefficient = 3.2e-3L;

/**
 * Eigenvalue avoidance factor for numerical stability.
 * Used in: ideal_mhd_model.cc to avoid singular matrix systems
 * Context: Prevents division by zero in eigenvalue computations
 */
static constexpr real_t kEigenvalueAvoidanceFactor = -1.0e-10L;

// ========== Mathematical Constants ==========

/**
 * Toroidal volume normalization factor: 2\pi
 * Used throughout for: volume integrals, flux surface calculations
 * Appears 40+ times across the codebase for torus geometry
 */
static constexpr real_t kToroidalNormalizationFactor = 2.0L * M_PI;

/**
 * Toroidal volume factor: (2\pi)^2
 * Used in: volume calculations requiring full torus integration
 * Context: Volume = \iiint d\rho d\theta d\zeta over torus
 */
static constexpr real_t kToroidalVolumeFactor = 4.0L * M_PI * M_PI;

/**
 * Constraint scaling factor: 1/\sqrt{2}
 * Used in: DFT normalization and constraint scaling
 * Context: Preserves orthogonality in Fourier transformations
 */
static constexpr real_t kConstraintScalingFactor = 1.0L / std::numbers::sqrt2;

// ========== Convergence and Tolerance Constants ==========

/**
 * Default force tolerance for equilibrium convergence.
 * Used in: vmec_indata.h as kFTolDefault
 * Context: ||F|| < kDefaultForceTolerance indicates equilibrium
 */
static constexpr real_t kDefaultForceTolerance = 1.0e-10L;

/**
 * Fast convergence detection threshold.
 * Used in: ideal_mhd_model.cc for convergence acceleration
 * Context: When ||F|| < threshold, apply special techniques
 */
static constexpr real_t kFastConvergenceThreshold = 1.0e-6L;

/**
 * Vacuum pressure activation threshold.
 * Used in: ideal_mhd_model.cc for free-boundary calculations
 * Context: Gradual activation of vacuum pressure forces
 */
static constexpr real_t kVacuumPressureThreshold = 1.0e-3L;

/**
 * Current density scaling factor.
 * Used in: ideal_mhd_model.cc for current profile normalization
 * Context: J \cdot B force balance computations
 */
static constexpr real_t kCurrentScalingFactor = 1.0e-6L;

/**
 * Force residual threshold for iteration decisions.
 * Used in: vmec.cc for determining when to continue iterations
 * Context: ||F_residual|| < threshold \rightarrow good convergence
 */
static constexpr real_t kForceResidualThreshold = 1.0e-2L;

/**
 * Tangential epsilon for numerical stability in surface integrals.
 * Used in: regularized_integrals.cc for singular integral regularization
 * Context: Prevents division by zero in surface-surface interactions
 */
static constexpr real_t kTangentialEpsilon = 1.0e-15L;

// ========== Iteration Control Constants ==========

/**
 * Default maximum iteration count.
 * Used in: vmec_indata.h as kNIterDefault
 * Context: Safety limit for equilibrium solver iterations
 */
static constexpr int kDefaultMaxIterations = 100;

/**
 * Default multigrid step iteration limit.
 * Used in: vmec.cc for multigrid progression control
 * Context: Maximum iterations per radial resolution level
 */
static constexpr int kDefaultMultigridIterations = 500;

/**
 * Minimum iterations before m=1 constraint enforcement.
 * Used in: ideal_mhd_model.cc for constraint stability
 * Context: Ensures constraint system is well-conditioned
 */
static constexpr int kMinIterationsForM1Constraint = 2;

/**
 * Maximum iteration difference for edge force inclusion.
 * Used in: ideal_mhd_model.cc for fast convergence detection
 * Context: (iter2 - iter1) < threshold \rightarrow include edge forces
 */
static constexpr int kMaxIterationDeltaForEdgeForces = 50;

/**
 * Jacobian iteration threshold for convergence assessment.
 * Used in: vmec.cc for determining when to restart with different strategy
 * Context: After 75 bad Jacobian iterations \rightarrow change approach
 */
static constexpr int kJacobianIterationThreshold = 75;

/**
 * Preconditioner update frequency.
 * Used in: flow_control.h for preconditioner refresh timing
 * Context: Update preconditioner every N iterations for efficiency
 */
static constexpr int kPreconditionerUpdateInterval = 25;

// ========== Array Size Constants ==========

/**
 * Standard damping array size for numerical stability operations.
 * Used in various damping and smoothing algorithms throughout VMEC++.
 * Also defined in vmec.h as kNDamp.
 */
static constexpr int kNDamp = 10;

/**
 * Default radial grid resolution.
 * Used in: vmec_indata.h as kNsDefault
 * Context: Number of flux surfaces for equilibrium computation
 */
static constexpr int kDefaultRadialResolution = 31;

/**
 * Magnetic axis grid resolution for axis finding.
 * Used in: guess_magnetic_axis.cc for axis position determination
 * Context: Number of grid points for magnetic axis search
 */
static constexpr int kMagneticAxisGridPoints = 61;

/**
 * String buffer size for file I/O operations.
 * Used in: makegrid_lib.cc for safe string handling
 * Context: Prevents buffer overflows in file name operations
 */
static constexpr int kStringBufferSize = 30;

// ========== Scaling and Damping Factors ==========

/**
 * General reduction/scaling factor used throughout VMEC++.
 * Used in: constraint reduction, parameter adjustments
 * Context: Conservative scaling to maintain numerical stability
 */
static constexpr real_t kGeneralScalingFactor = 0.9L;

/**
 * Jacobian scaling factors for specific iteration ranges.
 * Used in: vmec.cc for adaptive scaling based on iteration count
 * Context: Scale factors applied at iterations 25 and 50
 */
static constexpr real_t kJacobianScaling25 = 0.98L;
static constexpr real_t kJacobianScaling50 = 0.96L;

/**
 * Edge pedestal factor for boundary layer physics.
 * Used in: ideal_mhd_model.cc for edge physics modeling
 * Context: Controls edge gradient steepness in H-mode-like profiles
 */
static constexpr real_t kEdgePedestalFactor = 0.05L;

/**
 * Mode damping parameters for numerical stability.
 * Used in: ideal_mhd_model.cc for suppressing unstable modes
 * Context: Prevents numerical instabilities in Fourier space
 */
static constexpr real_t kModeDampingLarge = 16.0L * 16.0L;
static constexpr real_t kModeDampingSmall = 8.0L;

/**
 * Vacuum frequency adjustment factors.
 * Used in: ideal_mhd_model.cc for free-boundary force balance
 * Context: Balances plasma and vacuum contributions
 */
static constexpr real_t kVacuumFrequencyLow = 0.1L;
static constexpr real_t kVacuumFrequencyHigh = 1.0e11L;

// ========== Symmetry and Parity Constants ==========

/**
 * Even parity index for Fourier harmonics with even poloidal mode number m.
 * Used in: Fourier mode classification by poloidal mode number parity
 * Context: Harmonics with m=0,2,4,6,... (even poloidal mode numbers)
 */
static constexpr int kEvenParity = 0;

/**
 * Odd parity index for Fourier harmonics with odd poloidal mode number m.
 * Used in: Fourier mode classification by poloidal mode number parity
 * Context: Harmonics with m=1,3,5,7,... (odd poloidal mode numbers)
 */
static constexpr int kOddParity = 1;

// ========== Gauss-Legendre Quadrature Constants ==========

/**
 * 10-point Gauss-Legendre quadrature weights.
 * Used in: radial_profiles.cc for accurate pressure integration
 * Context: High-accuracy integration of radial pressure profiles
 * Mathematical basis: \int_{-1}^1 f(x)dx \approx \sum_i w_i f(x_i)
 */
static constexpr std::array<real_t, 10> kGaussLegendreWeights10 = {
    0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
    0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
    0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
    0.0666713443086881};

/**
 * 10-point Gauss-Legendre quadrature abscissae.
 * Used in: radial_profiles.cc for accurate pressure integration
 * Context: Optimal sampling points for polynomial integration
 */
static constexpr std::array<real_t, 10> kGaussLegendreAbscissae10 = {
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
    0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
    0.9739065285171717};

}  // namespace vmec_algorithm_constants

}  // namespace vmecpp

#endif  // VMECPP_VMEC_VMEC_CONSTANTS_VMEC_ALGORITHM_CONSTANTS_H_
