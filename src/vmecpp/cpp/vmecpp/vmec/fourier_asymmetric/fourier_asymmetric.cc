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

  // DEBUG: Compare with educational_VMEC at same position
  std::cout << "DEBUG FourierToReal3DAsymmFastPoloidal: mnmax=" << sizes.mnmax
            << ", nThetaEff=" << sizes.nThetaEff << ", nZeta=" << sizes.nZeta
            << std::endl;
  if (sizes.mnmax > 0) {
    std::cout << "DEBUG input coefficients: rmncc[0]=" << rmncc[0]
              << ", rmncc[1]=" << (sizes.mnmax > 1 ? rmncc[1] : 0.0)
              << std::endl;
  }

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

        // Get pre-normalized basis functions from FourierBasisFastPoloidal
        // These already include mscale and nscale factors

        // For theta: FourierBasisFastPoloidal only stores values for [0,pi]
        // For asymmetric case, we need values for [0,2pi]
        double cos_mu, sin_mu;
        if (i < sizes.nThetaReduced) {
          // Direct lookup for [0,pi]
          cos_mu = fourier_basis.cosmu[m * sizes.nThetaReduced + i];
          sin_mu = fourier_basis.sinmu[m * sizes.nThetaReduced + i];
        } else {
          // For [pi,2pi], use symmetry: cos(m*(2pi-theta)) = cos(m*theta),
          // sin(m*(2pi-theta)) = -sin(m*theta)
          int i_sym = sizes.nThetaEff - i;
          cos_mu = fourier_basis.cosmu[m * sizes.nThetaReduced + i_sym];
          sin_mu = -fourier_basis.sinmu[m * sizes.nThetaReduced + i_sym];
        }

        // For zeta: FourierBasisFastPoloidal stores basis functions indexed by
        // (k,n) The storage format is: idx = k * (nnyq2 + 1) + n for n >= 0
        double cos_nv, sin_nv;

        // Check if n is within valid range for precomputed basis
        int abs_n = std::abs(n);
        if (abs_n <= sizes.nnyq2) {
          // Use precomputed basis functions
          int idx_nv = k * (sizes.nnyq2 + 1) + abs_n;
          cos_nv = fourier_basis.cosnv[idx_nv];
          sin_nv = fourier_basis.sinnv[idx_nv];

          // Apply symmetry for negative n: cos(-nv) = cos(nv), sin(-nv) =
          // -sin(nv)
          if (n < 0) {
            sin_nv = -sin_nv;
          }
        } else {
          // Compute basis functions directly for out-of-range n
          double nv = n * sizes.nfp * v;
          double nscale = (n == 0) ? 1.0 : 1.0 / sqrt(2.0);
          cos_nv = cos(nv) * nscale;
          sin_nv = sin(nv) * nscale;
        }

        // DEBUG: Check basis functions for negative n
        if (idx < 3 && mn < 5) {
          std::cout << "DEBUG basis: idx=" << idx << ", mn=" << mn
                    << ", m=" << m << ", n=" << n << ", cos_nv=" << cos_nv
                    << ", sin_nv=" << sin_nv << std::endl;
        }

        // Asymmetric transform: Handle both symmetric and asymmetric modes
        // Using VMEC conventions: cos(mu-nv) and sin(mu-nv) expansions
        if (n == 0) {
          // For n=0: cos(mu-0) = cos(mu), sin(mu-0) = sin(mu)
          r_val += rmncc[mn] * cos_mu;  // cos(m*u) for R symmetric
          r_val += rmnsc[mn] * sin_mu;  // sin(m*u) for R asymmetric

          z_val += zmnsc[mn] * sin_mu;  // sin(m*u) for Z symmetric
          z_val += zmncc[mn] * cos_mu;  // cos(m*u) for Z asymmetric
        } else {
          // For n!=0, use proper trigonometric expansions
          // cos(mu-nv) = cos(mu)*cos(nv) + sin(mu)*sin(nv)
          // sin(mu-nv) = sin(mu)*cos(nv) - cos(mu)*sin(nv)

          // R symmetric terms
          double cos_mu_nv = cos_mu * cos_nv + sin_mu * sin_nv;  // cos(mu-nv)
          double sin_mu_nv = sin_mu * cos_nv - cos_mu * sin_nv;  // sin(mu-nv)
          r_val += rmncc[mn] * cos_mu_nv;  // Rmncc * cos(mu-nv)
          r_val += rmnss[mn] * sin_mu_nv;  // Rmnss * sin(mu-nv)

          // R asymmetric terms
          r_val += rmnsc[mn] * sin_mu * cos_nv;  // Rmnsc * sin(mu)*cos(nv)
          r_val += rmncs[mn] * cos_mu * sin_nv;  // Rmncs * cos(mu)*sin(nv)

          // Z symmetric terms
          z_val += zmnsc[mn] * sin_mu_nv;  // Zmnsc * sin(mu-nv)
          z_val += zmncs[mn] * cos_mu_nv;  // Zmncs * cos(mu-nv)

          // Z asymmetric terms
          z_val += zmncc[mn] * cos_mu * cos_nv;  // Zmncc * cos(mu)*cos(nv)
          z_val += zmnss[mn] * sin_mu * sin_nv;  // Zmnss * sin(mu)*sin(nv)
        }
      }

      r_real[idx] = r_val;
      z_real[idx] = z_val;
      lambda_real[idx] = lambda_val;

      // DEBUG: Output first few points for comparison with educational_VMEC
      if (idx < 3) {
        std::cout << "DEBUG FourierToReal3D: idx=" << idx << ", u=" << u
                  << ", v=" << v << ", R=" << r_val << ", Z=" << z_val
                  << std::endl;
      }
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
  // 2D asymmetric forward transform (axisymmetric case, ntor=0)
  // Optimized version that only processes m modes (n=0)

  // DEBUG: Compare with educational_VMEC 2D case
  std::cout << "DEBUG FourierToReal2DAsymmFastPoloidal: 2D transform, mnmax="
            << sizes.mnmax << std::endl;

  const double PI = 3.14159265358979323846;

  // Initialize output arrays
  std::fill(r_real.begin(), r_real.end(), 0.0);
  std::fill(z_real.begin(), z_real.end(), 0.0);
  std::fill(lambda_real.begin(), lambda_real.end(), 0.0);

  // Create FourierBasisFastPoloidal to get proper mode indexing
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // Compute scaling factors like 3D case
  std::vector<double> mscale(sizes.mpol + 1);
  mscale[0] = 1.0;
  for (int m = 1; m <= sizes.mpol; ++m) {
    mscale[m] = 1.0 / sqrt(2.0);
  }

  for (int i = 0; i < sizes.nThetaEff; ++i) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      if (idx >= static_cast<int>(r_real.size())) continue;

      // double u = 2.0 * PI * i / sizes.nThetaEff;  // Not needed when using
      // pre-computed basis

      double r_val = 0.0;
      double z_val = 0.0;
      double lambda_val = 0.0;

      // Process only n=0 modes for 2D case
      for (int mn = 0; mn < sizes.mnmax; ++mn) {
        int m = fourier_basis.xm[mn];
        int n = fourier_basis.xn[mn] / sizes.nfp;

        // Skip non-axisymmetric modes
        if (n != 0) continue;

        // Get pre-normalized basis functions (2D case)
        double cos_mu, sin_mu;
        if (i < sizes.nThetaReduced) {
          cos_mu = fourier_basis.cosmu[m * sizes.nThetaReduced + i];
          sin_mu = fourier_basis.sinmu[m * sizes.nThetaReduced + i];
        } else {
          // For [pi,2pi], use symmetry
          int i_sym = sizes.nThetaEff - i;
          cos_mu = fourier_basis.cosmu[m * sizes.nThetaReduced + i_sym];
          sin_mu = -fourier_basis.sinmu[m * sizes.nThetaReduced + i_sym];
        }

        // 2D asymmetric transform: only theta dependence
        r_val += rmncc[mn] * cos_mu;  // symmetric cos(mu)
        r_val += rmnsc[mn] * sin_mu;  // asymmetric sin(mu)

        z_val += zmnsc[mn] * sin_mu;  // symmetric sin(mu)
        z_val += zmncc[mn] * cos_mu;  // asymmetric cos(mu)
      }

      r_real[idx] = r_val;
      z_real[idx] = z_val;
      lambda_real[idx] = lambda_val;
    }
  }
}

void SymmetrizeRealSpaceGeometry(const Sizes& sizes, absl::Span<double> r_real,
                                 absl::Span<double> z_real,
                                 absl::Span<double> lambda_real) {
  // Symmetrize real space geometry for asymmetric equilibria
  // Equivalent to educational_VMEC's symrzl subroutine
  // Only called when lasym=true to combine symmetric and antisymmetric parts

  // DEBUG: Compare with educational_VMEC symrzl.f90
  std::cout << "DEBUG SymmetrizeRealSpaceGeometry: nThetaEff="
            << sizes.nThetaEff << ", nThetaReduced=" << sizes.nThetaReduced
            << ", nZeta=" << sizes.nZeta << std::endl;

  // This function should only be called for asymmetric equilibria
  if (!sizes.lasym) {
    return;
  }

  // Build reflection index for zeta -> -zeta mapping
  std::vector<int> ireflect(sizes.nZeta);
  for (int k = 0; k < sizes.nZeta; ++k) {
    ireflect[k] = (sizes.nZeta - k) % sizes.nZeta;
  }

  // Process extended interval [π, 2π] using symmetry relations
  for (int i = sizes.nThetaReduced; i < sizes.nThetaEff; ++i) {
    // Map theta to pi-theta: ir = ntheta1 + 2 - i
    int ir = sizes.nThetaReduced + (sizes.nThetaReduced - 1 - i);

    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      int idx_r = ir * sizes.nZeta + ireflect[k];

      if (idx >= static_cast<int>(r_real.size()) ||
          idx_r >= static_cast<int>(r_real.size())) {
        continue;
      }

      // For R: even parity R(u,v) = R(π-u,-v) - R_antisym(π-u,-v)
      // In practice, this means R_total = R_symmetric + R_antisymmetric
      // The extended interval gets: R_symmetric(reflected) +
      // R_antisymmetric(reflected)
      r_real[idx] = r_real[idx_r];

      // For Z: odd parity Z(u,v) = -Z(π-u,-v) + Z_antisym(π-u,-v)
      // The extended interval gets: -Z_symmetric(reflected) +
      // Z_antisymmetric(reflected)
      z_real[idx] = -z_real[idx_r];

      // For lambda: similar to R (even parity)
      lambda_real[idx] = lambda_real[idx_r];
    }
  }
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

  // DEBUG: Compare with educational_VMEC tomnspa.f90
  std::cout
      << "DEBUG RealToFourier3DAsymmFastPoloidal: inverse transform, mnmax="
      << sizes.mnmax << std::endl;

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

    // DEBUG: Show which modes are being processed
    std::cout << "DEBUG RealToFourier3D: Processing mode mn=" << mn
              << ", m=" << m << ", n=" << n << std::endl;

    // Integrate over theta and zeta
    double sum_rmncc = 0.0, sum_rmnss = 0.0, sum_rmnsc = 0.0, sum_rmncs = 0.0;
    double sum_zmnsc = 0.0, sum_zmncs = 0.0, sum_zmncc = 0.0, sum_zmnss = 0.0;

    for (int i = 0; i < sizes.nThetaEff; ++i) {
      for (int k = 0; k < sizes.nZeta; ++k) {
        int idx = i * sizes.nZeta + k;
        if (idx >= static_cast<int>(r_real.size())) continue;

        double u = 2.0 * PI * i / sizes.nThetaEff;
        double v = 2.0 * PI * k / sizes.nZeta;

        // Use plain trigonometric functions for inverse transform
        // The forward transform uses normalized basis functions, but inverse
        // should use plain cos/sin
        double cos_mu = cos(m * u);
        double sin_mu = sin(m * u);
        double cos_nv = cos(n * v);
        double sin_nv = sin(n * v);

        // Integration using basis functions
        if (n == 0) {
          // For n=0: project onto cos(mu) and sin(mu)
          sum_rmncc += r_real[idx] * cos_mu;
          sum_rmnsc += r_real[idx] * sin_mu;

          sum_zmnsc += z_real[idx] * sin_mu;
          sum_zmncc += z_real[idx] * cos_mu;
        } else {
          // For n!=0, project onto cos(mu-nv) and sin(mu-nv) basis
          double cos_mu_nv = cos_mu * cos_nv + sin_mu * sin_nv;  // cos(mu-nv)
          double sin_mu_nv = sin_mu * cos_nv - cos_mu * sin_nv;  // sin(mu-nv)

          // R symmetric: project onto cos(mu-nv) and sin(mu-nv)
          sum_rmncc += r_real[idx] * cos_mu_nv;
          sum_rmnss += r_real[idx] * sin_mu_nv;

          // R asymmetric: project onto sin(mu)*cos(nv) and cos(mu)*sin(nv)
          sum_rmnsc += r_real[idx] * sin_mu * cos_nv;
          sum_rmncs += r_real[idx] * cos_mu * sin_nv;

          // Z symmetric: project onto sin(mu-nv) and cos(mu-nv)
          sum_zmnsc += z_real[idx] * sin_mu_nv;
          sum_zmncs += z_real[idx] * cos_mu_nv;

          // Z asymmetric: project onto cos(mu)*cos(nv) and sin(mu)*sin(nv)
          sum_zmncc += z_real[idx] * cos_mu * cos_nv;
          sum_zmnss += z_real[idx] * sin_mu * sin_nv;
        }
      }
    }

    // Normalize by grid size (standard discrete Fourier transform
    // normalization)
    double norm_factor = 1.0 / (sizes.nZeta * sizes.nThetaEff);

    // DEBUG: Show normalization factors
    std::cout << "DEBUG RealToFourier3D: mn=" << mn
              << ", norm_factor=" << norm_factor << std::endl;

    // Store coefficients with standard DFT normalization only
    // The forward transform already applies mscale/nscale through the basis
    // functions
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
  // 2D asymmetric inverse transform (axisymmetric case, ntor=0)
  // Optimized version that only processes m modes (n=0)

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

  // Create FourierBasisFastPoloidal to get proper mode indexing
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // Compute scaling factors for normalization
  std::vector<double> mscale(sizes.mpol + 1);
  mscale[0] = 1.0;
  for (int m = 1; m <= sizes.mpol; ++m) {
    mscale[m] = 1.0 / sqrt(2.0);
  }

  // For each mode (only n=0 modes for 2D)
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;

    // Skip non-axisymmetric modes
    if (n != 0) continue;

    // Integrate over theta (zeta integration is trivial for 2D)
    double sum_rmncc = 0.0, sum_rmnsc = 0.0;
    double sum_zmnsc = 0.0, sum_zmncc = 0.0;

    for (int i = 0; i < sizes.nThetaEff; ++i) {
      for (int k = 0; k < sizes.nZeta; ++k) {
        int idx = i * sizes.nZeta + k;
        if (idx >= static_cast<int>(r_real.size())) continue;

        // Use plain trigonometric functions for inverse transform
        double u = 2.0 * PI * i / sizes.nThetaEff;
        double cos_mu = cos(m * u);
        double sin_mu = sin(m * u);

        // 2D integration: only theta dependence
        sum_rmncc += r_real[idx] * cos_mu;
        sum_rmnsc += r_real[idx] * sin_mu;

        sum_zmnsc += z_real[idx] * sin_mu;
        sum_zmncc += z_real[idx] * cos_mu;
      }
    }

    // Normalize by grid size (standard discrete Fourier transform
    // normalization)
    double norm_factor = 1.0 / (sizes.nZeta * sizes.nThetaEff);

    // Store coefficients with standard DFT normalization only
    rmncc[mn] = sum_rmncc * norm_factor;
    rmnsc[mn] = sum_rmnsc * norm_factor;
    zmnsc[mn] = sum_zmnsc * norm_factor;
    zmncc[mn] = sum_zmncc * norm_factor;
  }
}

void SymmetrizeForces(const Sizes& sizes, absl::Span<double> force_r,
                      absl::Span<double> force_z,
                      absl::Span<double> force_lambda) {
  // Symmetrize forces for asymmetric equilibria
  // Equivalent to educational_VMEC's symforce subroutine
  // Decomposes forces into symmetric and antisymmetric parts for Fourier
  // integration

  // DEBUG: Compare with educational_VMEC symforce.f90
  std::cout << "DEBUG SymmetrizeForces: force symmetrization started"
            << std::endl;

  // This function should only be called for asymmetric equilibria
  if (!sizes.lasym) {
    return;
  }

  // Build reflection index for zeta -> -zeta mapping
  std::vector<int> ireflect(sizes.nZeta);
  for (int k = 0; k < sizes.nZeta; ++k) {
    ireflect[k] = (sizes.nZeta - k) % sizes.nZeta;
  }

  // Create temporary arrays to store original forces
  std::vector<double> force_r_temp(force_r.begin(), force_r.end());
  std::vector<double> force_z_temp(force_z.begin(), force_z.end());
  std::vector<double> force_lambda_temp(force_lambda.begin(),
                                        force_lambda.end());

  // Process the full theta interval [0, π] to decompose forces
  for (int i = 0; i < sizes.nThetaReduced; ++i) {
    // Map theta to pi-theta: ir = ntheta1 + 2 - i
    int ir = sizes.nThetaReduced + (sizes.nThetaReduced - 1 - i);

    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      int idx_r = ir * sizes.nZeta + ireflect[k];

      if (idx >= static_cast<int>(force_r.size()) ||
          idx_r >= static_cast<int>(force_r.size())) {
        continue;
      }

      // Force decomposition into symmetric and antisymmetric parts:
      // F_symmetric = 0.5 * (F(u,v) + F(π-u,-v))     [for cos(mu-nv) terms]
      // F_antisymmetric = 0.5 * (F(u,v) - F(π-u,-v)) [for sin(mu-nv) terms]

      // Force_R has standard parity (even) - use symmetric part
      force_r[idx] = 0.5 * (force_r_temp[idx] + force_r_temp[idx_r]);

      // Force_Z has reverse parity (odd) - use antisymmetric part
      force_z[idx] = 0.5 * (force_z_temp[idx] - force_z_temp[idx_r]);

      // Force_Lambda has standard parity (even) - use symmetric part
      force_lambda[idx] =
          0.5 * (force_lambda_temp[idx] + force_lambda_temp[idx_r]);
    }
  }

  // Fill the extended interval [π, 2π] with the symmetrized values
  for (int i = sizes.nThetaReduced; i < sizes.nThetaEff; ++i) {
    int ir = sizes.nThetaReduced + (sizes.nThetaReduced - 1 - i);

    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      int idx_r = ir * sizes.nZeta + ireflect[k];

      if (idx >= static_cast<int>(force_r.size()) ||
          idx_r >= static_cast<int>(force_r.size())) {
        continue;
      }

      // Apply parity relations for the extended interval
      force_r[idx] = force_r[idx_r];            // Even parity
      force_z[idx] = -force_z[idx_r];           // Odd parity
      force_lambda[idx] = force_lambda[idx_r];  // Even parity
    }
  }
}

}  // namespace vmecpp
