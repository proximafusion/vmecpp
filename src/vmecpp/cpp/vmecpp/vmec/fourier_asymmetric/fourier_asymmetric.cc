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
  const int nzeta = sizes.nZeta;
  const int ntheta2 = sizes.nThetaReduced;  // [0, pi]
  const int ntheta1 = 2 * ntheta2;          // full range [0, 2pi]
  const int nznt = ntheta1 * nzeta;

  // CRITICAL: Create SEPARATE arrays for asymmetric contributions
  // These start at zero, not the symmetric baseline!
  std::vector<double> asym_R(nznt, 0.0);
  std::vector<double> asym_Z(nznt, 0.0);
  std::vector<double> asym_L(nznt, 0.0);

  // Get basis functions
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // Work arrays for intermediate results
  std::vector<std::vector<double>> work(12, std::vector<double>(nzeta, 0.0));

  // Process each poloidal mode m
  for (int m = 0; m < sizes.mpol; ++m) {
    // Clear work arrays for this m
    for (int idx = 0; idx < 12; ++idx) {
      std::fill(work[idx].begin(), work[idx].end(), 0.0);
    }

    // STAGE 1: Accumulate zeta contributions
    for (int k = 0; k < nzeta; ++k) {
      for (int n = 0; n <= sizes.ntor; ++n) {
        // Find mode (m,n)
        int mn = -1;
        for (int mn_candidate = 0; mn_candidate < sizes.mnmax; ++mn_candidate) {
          if (fourier_basis.xm[mn_candidate] == m &&
              fourier_basis.xn[mn_candidate] / sizes.nfp == n) {
            mn = mn_candidate;
            break;
          }
        }
        if (mn < 0) continue;

        // Get basis functions
        int idx_nv = k * (sizes.nnyq2 + 1) + n;
        double cos_nv = (n <= sizes.nnyq2)
                            ? fourier_basis.cosnv[idx_nv]
                            : std::cos(n * sizes.nfp * 2.0 * M_PI * k / nzeta);
        double sin_nv = (n <= sizes.nnyq2)
                            ? fourier_basis.sinnv[idx_nv]
                            : std::sin(n * sizes.nfp * 2.0 * M_PI * k / nzeta);

        // Accumulate asymmetric coefficients
        work[0][k] += rmnsc[mn] * cos_nv;
        work[5][k] += zmncc[mn] * cos_nv;

        if (sizes.lthreed) {
          work[1][k] += rmncs[mn] * sin_nv;
          work[4][k] += zmnss[mn] * sin_nv;
        }
      }
    }

    // STAGE 2: Transform in theta for [0,pi]
    for (int l = 0; l < ntheta2; ++l) {
      double sin_mu = fourier_basis.sinmu[m * sizes.nThetaReduced + l];
      double cos_mu = fourier_basis.cosmu[m * sizes.nThetaReduced + l];

      for (int k = 0; k < nzeta; ++k) {
        int idx = l * nzeta + k;

        // Accumulate into separate asymmetric arrays
        asym_R[idx] += work[0][k] * sin_mu;
        asym_Z[idx] += work[5][k] * cos_mu;

        if (sizes.lthreed) {
          asym_R[idx] += work[1][k] * cos_mu;
          asym_Z[idx] += work[4][k] * sin_mu;
        }
      }
    }
  }

  // STEP 1: Apply asymmetric contributions for theta=[0,pi]
  for (int l = 0; l < ntheta2; ++l) {
    for (int k = 0; k < nzeta; ++k) {
      int idx = l * nzeta + k;
      if (idx >= r_real.size()) continue;

      r_real[idx] += asym_R[idx];
      z_real[idx] += asym_Z[idx];
      lambda_real[idx] += asym_L[idx];
    }
  }

  // STEP 2: Handle theta=[pi,2pi] using reflection
  for (int l = ntheta2; l < ntheta1; ++l) {
    int lr = ntheta1 - l;

    for (int k = 0; k < nzeta; ++k) {
      int kr = (nzeta - k) % nzeta;

      int idx = l * nzeta + k;
      int idx_reflect = lr * nzeta + kr;

      if (idx >= r_real.size() || idx_reflect >= r_real.size()) continue;

      r_real[idx] = r_real[idx_reflect] - asym_R[idx_reflect];
      z_real[idx] = -z_real[idx_reflect] + asym_Z[idx_reflect];
      lambda_real[idx] = lambda_real[idx_reflect] - asym_L[idx_reflect];
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
  const int nzeta = sizes.nZeta;
  const int ntheta2 = sizes.nThetaReduced;  // [0, pi]
  const int ntheta1 = 2 * ntheta2;          // full range [0, 2pi]

  // CRITICAL: Create SEPARATE arrays for asymmetric contributions
  // These start at zero, not the symmetric baseline!
  std::vector<double> asym_R(ntheta1 * nzeta, 0.0);
  std::vector<double> asym_Z(ntheta1 * nzeta, 0.0);
  std::vector<double> asym_L(ntheta1 * nzeta, 0.0);

  // Get basis functions
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // For 2D case (ntor=0), only n=0 modes exist
  // cosnv=1, sinnv=0 for all n=0

  // Process each poloidal mode m
  for (int m = 0; m < sizes.mpol; ++m) {
    // Find mode mn for (m,n=0)
    int mn = -1;
    for (int mn_candidate = 0; mn_candidate < sizes.mnmax; ++mn_candidate) {
      if (fourier_basis.xm[mn_candidate] == m &&
          fourier_basis.xn[mn_candidate] / sizes.nfp == 0) {
        mn = mn_candidate;
        break;
      }
    }
    if (mn < 0) continue;  // mode not found

    // Get coefficients for this mode
    double rsc = (mn < rmnsc.size()) ? rmnsc[mn] : 0.0;
    double zcc = (mn < zmncc.size()) ? zmncc[mn] : 0.0;

    if (std::abs(rsc) < 1e-12 && std::abs(zcc) < 1e-12) continue;

    // Compute asymmetric contributions for theta=[0,pi]
    for (int l = 0; l < ntheta2; ++l) {
      double sin_mu = fourier_basis.sinmu[m * sizes.nThetaReduced + l];
      double cos_mu = fourier_basis.cosmu[m * sizes.nThetaReduced + l];

      for (int k = 0; k < nzeta; ++k) {
        int idx = l * nzeta + k;

        // Following jVMEC lines 302-305 exactly
        asym_R[idx] += rsc * sin_mu;  // rmnsc * sinmu
        asym_Z[idx] += zcc * cos_mu;  // zmncc * cosmu
      }
    }
  }

  // STEP 1: Apply asymmetric contributions for theta=[0,pi]
  // Following jVMEC lines 368-390
  for (int l = 0; l < ntheta2; ++l) {
    for (int k = 0; k < nzeta; ++k) {
      int idx = l * nzeta + k;
      if (idx >= r_real.size()) continue;

      // ADD asymmetric to existing symmetric values
      r_real[idx] += asym_R[idx];
      z_real[idx] += asym_Z[idx];
      lambda_real[idx] += asym_L[idx];
    }
  }

  // STEP 2: Handle theta=[pi,2pi] using reflection
  // Following jVMEC lines 340-365
  for (int l = ntheta2; l < ntheta1; ++l) {
    int lr = ntheta1 - l;  // reflection index

    for (int k = 0; k < nzeta; ++k) {
      int kr = (nzeta - k) % nzeta;  // zeta reflection (ireflect)

      int idx = l * nzeta + k;
      int idx_reflect = lr * nzeta + kr;

      if (idx >= r_real.size() || idx_reflect >= r_real.size()) continue;

      // Following jVMEC reflection formulas exactly:
      // R[pi,2pi] = R_sym[reflected] - asym_R[reflected]
      // Z[pi,2pi] = -Z_sym[reflected] + asym_Z[reflected]
      r_real[idx] = r_real[idx_reflect] - asym_R[idx_reflect];
      z_real[idx] = -z_real[idx_reflect] + asym_Z[idx_reflect];
      lambda_real[idx] = lambda_real[idx_reflect] - asym_L[idx_reflect];
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

  // Debug: Print values before symmetrization
  std::cout << "DEBUG: Values before symmetrization (first 3 theta points):"
            << std::endl;
  for (int i = 0; i < std::min(3, sizes.nThetaEff); ++i) {
    for (int k = 0; k < std::min(1, sizes.nZeta); ++k) {
      int idx = i * sizes.nZeta + k;
      if (idx < static_cast<int>(r_real.size())) {
        std::cout << "  idx=" << idx << " (i=" << i << ", k=" << k << "): "
                  << "R=" << r_real[idx] << ", Z=" << z_real[idx] << std::endl;
      }
    }
  }

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
    // In educational_VMEC: i = ntheta2+1 to ntheta1, ir = ntheta1 + 2 - i
    // For i = nThetaReduced (=ntheta2), ir should be nThetaReduced-1
    // For i = nThetaEff-1 (=ntheta1-1), ir should be 1
    int ir = sizes.nThetaReduced + (sizes.nThetaReduced - 1 - i);

    // DEBUG: Verify ir calculation
    if (ir < 0 || ir >= sizes.nThetaEff) {
      std::cout << "ERROR: Invalid ir=" << ir << " for i=" << i
                << ", nThetaReduced=" << sizes.nThetaReduced
                << ", nThetaEff=" << sizes.nThetaEff << std::endl;
      continue;
    }

    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      int idx_r = ir * sizes.nZeta + ireflect[k];

      // DEBUG: Additional bounds checking
      if (ireflect[k] < 0 || ireflect[k] >= sizes.nZeta) {
        std::cout << "ERROR: Invalid ireflect[" << k << "]=" << ireflect[k]
                  << ", nZeta=" << sizes.nZeta << std::endl;
        continue;
      }

      if (idx >= static_cast<int>(r_real.size()) ||
          idx_r >= static_cast<int>(r_real.size())) {
        std::cout << "ERROR: Index out of bounds: idx=" << idx
                  << ", idx_r=" << idx_r << ", array size=" << r_real.size()
                  << std::endl;
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

      if (i < sizes.ntheta + 3 && k == 0) {
        std::cout << "DEBUG: Symmetrization at i=" << i << ", ir=" << ir
                  << ": copying from idx_r=" << idx_r << " to idx=" << idx
                  << ", R[" << idx << "]=" << r_real[idx] << ", Z[" << idx
                  << "]=" << z_real[idx] << std::endl;

        // Check for NaN/inf values
        if (!std::isfinite(r_real[idx]) || !std::isfinite(z_real[idx])) {
          std::cout << "ERROR: Non-finite values detected at symmetrization i="
                    << i << ", R=" << r_real[idx] << ", Z=" << z_real[idx]
                    << std::endl;
        }
      }
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

  // const double PI = 3.14159265358979323846;  // unused

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

        const double PI = 3.14159265358979323846;
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

  // const double PI = 3.14159265358979323846;  // unused

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
        const double PI = 3.14159265358979323846;
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

  // DEBUG: Check input forces for NaN
  bool found_nan_forces = false;
  for (int i = 0; i < std::min(10, static_cast<int>(force_r.size())); ++i) {
    if (!std::isfinite(force_r[i]) || !std::isfinite(force_z[i]) ||
        !std::isfinite(force_lambda[i])) {
      std::cout << "ERROR: Non-finite force at i=" << i << ", fr=" << force_r[i]
                << ", fz=" << force_z[i] << ", fl=" << force_lambda[i]
                << std::endl;
      found_nan_forces = true;
    }
  }
  if (!found_nan_forces) {
    std::cout << "DEBUG: All input forces are finite (first 10 checked)"
              << std::endl;
  }

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
