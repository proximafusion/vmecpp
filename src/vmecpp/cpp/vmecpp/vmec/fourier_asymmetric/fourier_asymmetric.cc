// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

#include <cmath>
#include <iostream>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"

namespace vmecpp {

void FourierToReal3DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> rmncc,
    absl::Span<const double> rmnss, absl::Span<const double> rmnsc,
    absl::Span<const double> rmncs, absl::Span<const double> zmnsc,
    absl::Span<const double> zmncs, absl::Span<const double> zmncc,
    absl::Span<const double> zmnss, absl::Span<double> r_real,
    absl::Span<double> z_real, absl::Span<double> lambda_real) {
  // Implementation based on educational_VMEC's totzspa.f90
  // This is the asymmetric transform that handles both symmetric and asymmetric
  // modes

  const double PI = 3.14159265358979323846;

  // Initialize output arrays
  std::fill(r_real.begin(), r_real.end(), 0.0);
  std::fill(z_real.begin(), z_real.end(), 0.0);
  std::fill(lambda_real.begin(), lambda_real.end(), 0.0);

  // Create FourierBasisFastPoloidal to get proper mode indexing
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // Compute scaling factors like educational_VMEC
  std::vector<double> mscale(sizes.mpol + 1);
  std::vector<double> nscale(sizes.ntor + 1);
  mscale[0] = 1.0;
  nscale[0] = 1.0;
  for (int m = 1; m <= sizes.mpol; ++m) {
    mscale[m] = 1.0 / sqrt(2.0);
  }
  for (int n = 1; n <= sizes.ntor; ++n) {
    nscale[n] = 1.0 / sqrt(2.0);
  }

  for (int i = 0; i < sizes.nThetaEff; ++i) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      if (idx >= static_cast<int>(r_real.size())) continue;

      double u = 2.0 * PI * i / sizes.nThetaEff;
      double v = 2.0 * PI * k / sizes.nZeta;

      double r_val = 0.0;
      double z_val = 0.0;
      double lambda_val = 0.0;

      // Process all modes using proper indexing
      for (int mn = 0; mn < sizes.mnmax; ++mn) {
        // Use FourierBasisFastPoloidal to decode m,n from linear index
        int m = fourier_basis.xm[mn];
        int n = fourier_basis.xn[mn] / sizes.nfp;  // xn includes nfp factor

        // For forward transform, don't apply normalization to achieve exact
        // recovery
        double cos_mu = cos(m * u);
        double sin_mu = sin(m * u);
        double cos_nv = cos(n * v);
        double sin_nv = sin(n * v);

        // Asymmetric transform: Handle both symmetric and asymmetric modes
        // For axisymmetric case (n=0), simplify expressions
        if (n == 0) {
          r_val += rmncc[mn] * cos_mu;  // cos(m*u) for R
          r_val += rmnsc[mn] * sin_mu;  // sin(m*u) for R

          z_val += zmnsc[mn] * sin_mu;  // sin(m*u) for Z
          z_val += zmncc[mn] * cos_mu;  // cos(m*u) for Z
        } else {
          // For n!=0, use full 3D expressions
          r_val += rmncc[mn] * cos_mu * cos_nv;  // cos(m*u - n*v) term
          r_val += rmnss[mn] * sin_mu * sin_nv;  // sin(m*u - n*v) term
          r_val += rmnsc[mn] * sin_mu * cos_nv;  // asymmetric terms
          r_val += rmncs[mn] * cos_mu * sin_nv;

          z_val += zmnsc[mn] * sin_mu * cos_nv;  // sin(m*u - n*v) term
          z_val += zmncs[mn] * cos_mu * sin_nv;  // cos(m*u - n*v) term
          z_val += zmncc[mn] * cos_mu * cos_nv;  // asymmetric terms
          z_val += zmnss[mn] * sin_mu * sin_nv;
        }
      }

      r_real[idx] = r_val;
      z_real[idx] = z_val;
      lambda_real[idx] = lambda_val;
    }
  }
}

void FourierToReal2DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> rmncc,
    absl::Span<const double> rmnss, absl::Span<const double> rmnsc,
    absl::Span<const double> rmncs, absl::Span<const double> zmnsc,
    absl::Span<const double> zmncs, absl::Span<const double> zmncc,
    absl::Span<const double> zmnss, absl::Span<double> r_real,
    absl::Span<double> z_real, absl::Span<double> lambda_real) {
  // TODO: Implement 2D asymmetric forward transform
  std::cerr << "FourierToReal2DAsymmFastPoloidal not implemented yet\n";
}

void SymmetrizeRealSpaceGeometry(const Sizes& sizes, absl::Span<double> r_real,
                                 absl::Span<double> z_real,
                                 absl::Span<double> lambda_real) {
  // TODO: Implement symmetrization
  std::cerr << "SymmetrizeRealSpaceGeometry not implemented yet\n";
}

void RealToFourier3DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> r_real,
    absl::Span<const double> z_real, absl::Span<const double> lambda_real,
    absl::Span<double> rmncc, absl::Span<double> rmnss,
    absl::Span<double> rmnsc, absl::Span<double> rmncs,
    absl::Span<double> zmnsc, absl::Span<double> zmncs,
    absl::Span<double> zmncc, absl::Span<double> zmnss,
    absl::Span<double> lmnsc, absl::Span<double> lmncs,
    absl::Span<double> lmncc, absl::Span<double> lmnss) {
  // Inverse transform from real space to Fourier coefficients
  // Based on discrete Fourier transform with trapezoidal rule integration

  const double PI = 3.14159265358979323846;

  // Initialize output arrays
  std::fill(rmncc.begin(), rmncc.end(), 0.0);
  std::fill(rmnss.begin(), rmnss.end(), 0.0);
  std::fill(rmnsc.begin(), rmnsc.end(), 0.0);
  std::fill(rmncs.begin(), rmncs.end(), 0.0);
  std::fill(zmnsc.begin(), zmnsc.end(), 0.0);
  std::fill(zmncs.begin(), zmncs.end(), 0.0);
  std::fill(zmncc.begin(), zmncc.end(), 0.0);
  std::fill(zmnss.begin(), zmnss.end(), 0.0);
  std::fill(lmnsc.begin(), lmnsc.end(), 0.0);
  std::fill(lmncs.begin(), lmncs.end(), 0.0);
  std::fill(lmncc.begin(), lmncc.end(), 0.0);
  std::fill(lmnss.begin(), lmnss.end(), 0.0);

  // Integration weights for discrete Fourier transform
  // (not used in current implementation)

  // Create FourierBasisFastPoloidal to get proper mode indexing
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // Compute scaling factors like educational_VMEC
  std::vector<double> mscale(sizes.mpol + 1);
  std::vector<double> nscale(sizes.ntor + 1);
  mscale[0] = 1.0;
  nscale[0] = 1.0;
  for (int m = 1; m <= sizes.mpol; ++m) {
    mscale[m] = 1.0 / sqrt(2.0);
  }
  for (int n = 1; n <= sizes.ntor; ++n) {
    nscale[n] = 1.0 / sqrt(2.0);
  }

  // For each mode
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    // Use FourierBasisFastPoloidal to decode m,n from linear index
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;  // xn includes nfp factor

    // Integrate over theta and zeta
    double sum_rmncc = 0.0, sum_rmnss = 0.0, sum_rmnsc = 0.0, sum_rmncs = 0.0;
    double sum_zmnsc = 0.0, sum_zmncs = 0.0, sum_zmncc = 0.0, sum_zmnss = 0.0;

    for (int i = 0; i < sizes.nThetaEff; ++i) {
      for (int k = 0; k < sizes.nZeta; ++k) {
        int idx = i * sizes.nZeta + k;
        if (idx >= static_cast<int>(r_real.size())) continue;

        double u = 2.0 * PI * i / sizes.nThetaEff;
        double v = 2.0 * PI * k / sizes.nZeta;

        double cos_mu = cos(m * u);
        double sin_mu = sin(m * u);
        double cos_nv = cos(n * v);
        double sin_nv = sin(n * v);

        // Integration using basis functions
        // For n=0, simplify to 1D integration
        if (n == 0) {
          sum_rmncc += r_real[idx] * cos_mu;
          sum_rmnsc += r_real[idx] * sin_mu;

          sum_zmnsc += z_real[idx] * sin_mu;
          sum_zmncc += z_real[idx] * cos_mu;
        } else {
          // For n!=0, use full 3D integration
          sum_rmncc += r_real[idx] * cos_mu * cos_nv;
          sum_rmnss += r_real[idx] * sin_mu * sin_nv;
          sum_rmnsc += r_real[idx] * sin_mu * cos_nv;
          sum_rmncs += r_real[idx] * cos_mu * sin_nv;

          sum_zmnsc += z_real[idx] * sin_mu * cos_nv;
          sum_zmncs += z_real[idx] * cos_mu * sin_nv;
          sum_zmncc += z_real[idx] * cos_mu * cos_nv;
          sum_zmnss += z_real[idx] * sin_mu * sin_nv;
        }
      }
    }

    // Normalize using discrete Fourier transform approach
    // Base normalization for integration over discrete grid
    double norm_factor = 1.0 / (sizes.nZeta * sizes.nThetaEff);

    // Apply inverse normalization factors for orthogonality
    // Since forward transform has no normalization, inverse must compensate
    norm_factor /= (mscale[m] * nscale[n]);

    // Store normalized coefficients
    rmncc[mn] = sum_rmncc * norm_factor;
    rmnss[mn] = sum_rmnss * norm_factor;
    rmnsc[mn] = sum_rmnsc * norm_factor;
    rmncs[mn] = sum_rmncs * norm_factor;

    zmnsc[mn] = sum_zmnsc * norm_factor;
    zmncs[mn] = sum_zmncs * norm_factor;
    zmncc[mn] = sum_zmncc * norm_factor;
    zmnss[mn] = sum_zmnss * norm_factor;
  }
}

void RealToFourier2DAsymmFastPoloidal(
    const Sizes& sizes, absl::Span<const double> r_real,
    absl::Span<const double> z_real, absl::Span<const double> lambda_real,
    absl::Span<double> rmncc, absl::Span<double> rmnss,
    absl::Span<double> rmnsc, absl::Span<double> rmncs,
    absl::Span<double> zmnsc, absl::Span<double> zmncs,
    absl::Span<double> zmncc, absl::Span<double> zmnss,
    absl::Span<double> lmnsc, absl::Span<double> lmncs,
    absl::Span<double> lmncc, absl::Span<double> lmnss) {
  // TODO: Implement 2D asymmetric inverse transform
  std::cerr << "RealToFourier2DAsymmFastPoloidal not implemented yet\n";
}

void SymmetrizeForces(const Sizes& sizes, absl::Span<double> force_r,
                      absl::Span<double> force_z,
                      absl::Span<double> force_lambda) {
  // TODO: Implement force symmetrization
  std::cerr << "SymmetrizeForces not implemented yet\n";
}

}  // namespace vmecpp
