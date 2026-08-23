// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_SPECTRAL_WIDTH_SPECTRAL_WIDTH_H_
#define VMECPP_COMMON_SPECTRAL_WIDTH_SPECTRAL_WIDTH_H_

#include <span>

#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

// R and Z Fourier coefficients of a single flux surface in the VMEC-internal
// product basis, indexed as [m * (ntor + 1) + n]. The components excluded by
// the symmetry flags are left empty and are never read.
struct SurfaceFourierGeometry {
  // contrib to R ~ cos(m * theta) * cos(n * zeta)
  std::span<const double> rmncc;

  // contrib to R ~ sin(m * theta) * sin(n * zeta)
  std::span<const double> rmnss;

  // contrib to R ~ sin(m * theta) * cos(n * zeta)
  std::span<const double> rmnsc;

  // contrib to R ~ cos(m * theta) * sin(n * zeta)
  std::span<const double> rmncs;

  // contrib to Z ~ sin(m * theta) * cos(n * zeta)
  std::span<const double> zmnsc;

  // contrib to Z ~ cos(m * theta) * sin(n * zeta)
  std::span<const double> zmncs;

  // contrib to Z ~ cos(m * theta) * cos(n * zeta)
  std::span<const double> zmncc;

  // contrib to Z ~ sin(m * theta) * sin(n * zeta)
  std::span<const double> zmnss;
};

// Spectral width <M> of a single flux surface,
//
//   <M> = \sum_{m \ge 1, n} |c_{mn}|^2 m^{p+q} / \sum_{m \ge 1, n} |c_{mn}|^2
//   m^p ,
//
// where |c_{mn}|^2 is the summed squared R and Z amplitude of mode (m, n).
// m = 0 is excluded because it carries the average position of the surface
// rather than its poloidal shape, and the m = 1 coefficients are unpacked from
// the internal R+/R- representation before they enter the sums.
//
// `mscale` and `nscale` undo the normalization of the internal Fourier basis:
// pass FourierBasisFastPoloidal::mscale and ::nscale for coefficients held in
// that basis, and arrays of ones for plain Fourier amplitudes.
double SpectralWidth(const SurfaceFourierGeometry& geometry, const Sizes& sizes,
                     std::span<const double> mscale,
                     std::span<const double> nscale, int p = 4, int q = 1);

}  // namespace vmecpp

#endif  // VMECPP_COMMON_SPECTRAL_WIDTH_SPECTRAL_WIDTH_H_
