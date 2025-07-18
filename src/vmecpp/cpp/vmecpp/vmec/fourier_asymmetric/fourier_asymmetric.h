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
    absl::Span<double> r_real, absl::Span<double> z_real,
    absl::Span<double> lambda_real);

// 2D version for axisymmetric case
void FourierToReal2DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> rmncc,
    absl::Span<const double> rmnss, absl::Span<const double> rmnsc,
    absl::Span<const double> rmncs, absl::Span<const double> zmnsc,
    absl::Span<const double> zmncs, absl::Span<const double> zmncc,
    absl::Span<const double> zmnss, absl::Span<double> r_real,
    absl::Span<double> z_real, absl::Span<double> lambda_real);

// Symmetrize real space geometry
// Equivalent to educational_VMEC's symrzl
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

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_ASYMMETRIC_FOURIER_ASYMMETRIC_H_
