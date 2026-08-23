// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/spectral_width/spectral_width.h"

#include <Eigen/Dense>
#include <cmath>

namespace vmecpp {

double SpectralWidth(const SurfaceFourierGeometry& geometry, const Sizes& sizes,
                     std::span<const double> mscale,
                     std::span<const double> nscale, const int p, const int q) {
  double spectral_width_numerator = 0.0;
  double spectral_width_denominator = 0.0;

  // note that we exclude m = 0
  for (int m = 1; m < sizes.mpol; ++m) {
    for (int n = 0; n < sizes.ntor + 1; ++n) {
      const int fourier_index = m * (sizes.ntor + 1) + n;

      const double basis_norm = mscale[m] * nscale[n];

      // Use Eigen for vectorized norm computation
      Eigen::Vector4d r_coefficients = Eigen::Vector4d::Zero();
      Eigen::Vector4d z_coefficients = Eigen::Vector4d::Zero();
      int basis_dimension = 0;

      r_coefficients[basis_dimension] = geometry.rmncc[fourier_index];
      z_coefficients[basis_dimension] = geometry.zmnsc[fourier_index];
      basis_dimension++;

      // CONVERT FROM INTERNAL XC REPRESENTATION FOR m=1 MODES,
      // R+(at rsc) = .5(rsc + zcc),
      // R-(at zcc) = .5(rsc - zcc),
      // TO REQUIRED rsc, zcc FORMS
      if (sizes.lthreed) {
        if (m == 1) {
          const double r_plus = geometry.rmnss[fourier_index];
          const double r_minus = geometry.zmncs[fourier_index];
          // rmnss
          r_coefficients[basis_dimension] = r_plus + r_minus;
          // zmncs
          z_coefficients[basis_dimension] = r_plus - r_minus;
        } else {
          r_coefficients[basis_dimension] = geometry.rmnss[fourier_index];
          z_coefficients[basis_dimension] = geometry.zmncs[fourier_index];
        }
        basis_dimension++;
      }
      if (sizes.lasym) {
        if (m == 1) {
          const double r_plus = geometry.rmnsc[fourier_index];
          const double r_minus = geometry.zmncc[fourier_index];
          // rmnsc
          r_coefficients[basis_dimension] = r_plus + r_minus;
          // zmncc
          z_coefficients[basis_dimension] = r_plus - r_minus;
        } else {
          r_coefficients[basis_dimension] = geometry.rmnsc[fourier_index];
          z_coefficients[basis_dimension] = geometry.zmncc[fourier_index];
        }
        basis_dimension++;
      }

      if (sizes.lasym && sizes.lthreed) {
        r_coefficients[basis_dimension] = geometry.rmncs[fourier_index];
        z_coefficients[basis_dimension] = geometry.zmnss[fourier_index];
        basis_dimension++;
      }

      // Vectorized squared norm computation
      double coefficient_norm =
          r_coefficients.head(basis_dimension).squaredNorm() +
          z_coefficients.head(basis_dimension).squaredNorm();
      coefficient_norm *= basis_norm * basis_norm;

      spectral_width_numerator += coefficient_norm * std::pow(m, p + q);
      spectral_width_denominator += coefficient_norm * std::pow(m, p);
    }  // n
  }  // m

  return spectral_width_numerator / spectral_width_denominator;
}  // SpectralWidth

}  // namespace vmecpp
