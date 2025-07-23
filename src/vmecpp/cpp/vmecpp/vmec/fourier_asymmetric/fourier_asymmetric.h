// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_FOURIER_ASYMMETRIC_FOURIER_ASYMMETRIC_H_
#define VMECPP_VMEC_FOURIER_ASYMMETRIC_FOURIER_ASYMMETRIC_H_

#include <vector>

#include "absl/types/span.h"
#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

// Forward transform: Fourier space to real space for asymmetric equilibria
// Equivalent to educational_VMEC's totzspa
void FourierToReal3DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> rmncc,
    absl::Span<const double> rmnss,
    absl::Span<const double> rmnsc,  // Asymmetric
    absl::Span<const double> rmncs,  // Asymmetric
    absl::Span<const double> zmnsc, absl::Span<const double> zmncs,
    absl::Span<const double> zmncc,  // Asymmetric
    absl::Span<const double> zmnss,  // Asymmetric  
    absl::Span<const double> lmnsc, absl::Span<const double> lmncs,
    absl::Span<const double> lmncc,  // Asymmetric CRITICAL
    absl::Span<const double> lmnss,  // Asymmetric CRITICAL
    absl::Span<double> r_real, absl::Span<double> z_real,
    absl::Span<double> lambda_real,
    absl::Span<double> ru_real,  // ADD: dR/dtheta
    absl::Span<double> zu_real);  // ADD: dZ/dtheta

// NEW: Forward transform that outputs separate symmetric and antisymmetric
// arrays This is the key to fixing the array combination issue
void FourierToReal3DAsymmFastPoloidalSeparated(
    const Sizes& sizes, absl::Span<const double> rmncc,
    absl::Span<const double> rmnss,
    absl::Span<const double> rmnsc,  // Asymmetric
    absl::Span<const double> rmncs,  // Asymmetric
    absl::Span<const double> zmnsc, absl::Span<const double> zmncs,
    absl::Span<const double> zmncc,   // Asymmetric
    absl::Span<const double> zmnss,   // Asymmetric
    absl::Span<double> r_sym,         // SEPARATE symmetric output [0, π]
    absl::Span<double> r_asym,        // SEPARATE antisymmetric output [0, π]
    absl::Span<double> z_sym,         // SEPARATE symmetric output [0, π]
    absl::Span<double> z_asym,        // SEPARATE antisymmetric output [0, π]
    absl::Span<double> lambda_sym,    // SEPARATE symmetric output [0, π]
    absl::Span<double> lambda_asym,  // SEPARATE antisymmetric output [0, π]
    absl::Span<double> ru_sym,        // SEPARATE symmetric dR/dtheta [0, π]
    absl::Span<double> ru_asym,       // SEPARATE antisymmetric dR/dtheta [0, π]
    absl::Span<double> zu_sym,        // SEPARATE symmetric dZ/dtheta [0, π]
    absl::Span<double> zu_asym);      // SEPARATE antisymmetric dZ/dtheta [0, π]

// 2D version for axisymmetric case
void FourierToReal2DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> rmncc,
    absl::Span<const double> rmnss, absl::Span<const double> rmnsc,
    absl::Span<const double> rmncs, absl::Span<const double> zmnsc,
    absl::Span<const double> zmncs, absl::Span<const double> zmncc,
    absl::Span<const double> zmnss, 
    absl::Span<const double> lmnsc, absl::Span<const double> lmncs,
    absl::Span<const double> lmncc,  // Asymmetric CRITICAL
    absl::Span<const double> lmnss,  // Asymmetric CRITICAL
    absl::Span<double> r_real, absl::Span<double> z_real, 
    absl::Span<double> lambda_real,
    absl::Span<double> ru_real,  // ADD: dR/dtheta
    absl::Span<double> zu_real);  // ADD: dZ/dtheta

// Symmetrize real space geometry - FIXED VERSION
// Equivalent to educational_VMEC's symrzl
// Takes separate symmetric and antisymmetric arrays and combines them properly
void SymmetrizeRealSpaceGeometry(
    const absl::Span<const double> r_sym, const absl::Span<const double> r_asym,
    const absl::Span<const double> z_sym, const absl::Span<const double> z_asym,
    const absl::Span<const double> lambda_sym,
    const absl::Span<const double> lambda_asym, absl::Span<double> r_full,
    absl::Span<double> z_full, absl::Span<double> lambda_full,
    const Sizes& sizes);

// OLD VERSION - DEPRECATED - Remove after testing
void SymmetrizeRealSpaceGeometry(const Sizes& sizes, absl::Span<double> r_real,
                                 absl::Span<double> z_real,
                                 absl::Span<double> lambda_real);

// Inverse transform: Real space to Fourier space for asymmetric equilibria
// Equivalent to educational_VMEC's tomnspa
void RealToFourier3DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> r_real,
    absl::Span<const double> z_real, absl::Span<const double> lambda_real,
    absl::Span<double> rmncc, absl::Span<double> rmnss,
    absl::Span<double> rmnsc,  // Asymmetric
    absl::Span<double> rmncs,  // Asymmetric
    absl::Span<double> zmnsc, absl::Span<double> zmncs,
    absl::Span<double> zmncc,  // Asymmetric
    absl::Span<double> zmnss,  // Asymmetric
    absl::Span<double> lmnsc, absl::Span<double> lmncs,
    absl::Span<double> lmncc,   // Asymmetric
    absl::Span<double> lmnss);  // Asymmetric

// 2D version for axisymmetric case
void RealToFourier2DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> r_real,
    absl::Span<const double> z_real, absl::Span<const double> lambda_real,
    absl::Span<double> rmncc, absl::Span<double> rmnss,
    absl::Span<double> rmnsc, absl::Span<double> rmncs,
    absl::Span<double> zmnsc, absl::Span<double> zmncs,
    absl::Span<double> zmncc, absl::Span<double> zmnss,
    absl::Span<double> lmnsc, absl::Span<double> lmncs,
    absl::Span<double> lmncc, absl::Span<double> lmnss);

// Symmetrize forces
// Equivalent to educational_VMEC's symforce
void SymmetrizeForces(const Sizes& sizes, absl::Span<double> force_r,
                      absl::Span<double> force_z,
                      absl::Span<double> force_lambda);

// M=1 constraint coupling functions - PRIORITY 3 IMPLEMENTATION
// Equivalent to jVMEC's Boundaries.ensureM1Constrained()
void EnsureM1Constrained(const Sizes& sizes,
                         absl::Span<double> rbss, absl::Span<double> zbcs,
                         absl::Span<double> rbsc, absl::Span<double> zbcc);

// Equivalent to jVMEC's SpectralCondensation.convert_to_m1_constrained()
void ConvertToM1Constrained(const Sizes& sizes, 
                            int num_surfaces,
                            absl::Span<double> rss_rsc,
                            absl::Span<double> zcs_zcc,
                            double scaling_factor);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_ASYMMETRIC_FOURIER_ASYMMETRIC_H_
