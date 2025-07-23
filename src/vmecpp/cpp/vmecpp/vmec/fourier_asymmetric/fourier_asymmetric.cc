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
    absl::Span<double> z_real, absl::Span<double> lambda_real,
    absl::Span<double> ru_real, absl::Span<double> zu_real) {
  const int nzeta = sizes.nZeta;
  const int ntheta2 = sizes.nThetaReduced;  // [0, pi] including endpoint
  const int ntheta_eff = sizes.nThetaEff;   // effective theta grid size
  const int nznt = ntheta_eff * nzeta;

  // Initialize output arrays
  std::fill(r_real.begin(), r_real.end(), 0.0);
  std::fill(z_real.begin(), z_real.end(), 0.0);
  std::fill(lambda_real.begin(), lambda_real.end(), 0.0);
  std::fill(ru_real.begin(), ru_real.end(), 0.0);
  std::fill(zu_real.begin(), zu_real.end(), 0.0);

  // Create arrays for asymmetric contributions
  std::vector<double> asym_R(nznt, 0.0);
  std::vector<double> asym_Z(nznt, 0.0);
  std::vector<double> asym_L(nznt, 0.0);
  
  // Create arrays for asymmetric derivative contributions
  std::vector<double> asym_Ru(nznt, 0.0);
  std::vector<double> asym_Zu(nznt, 0.0);

  // Get basis functions
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // Process each poloidal mode m
  for (int m = 0; m < sizes.mpol; ++m) {
    // Work arrays for this m mode
    std::vector<double> rmkcc(nzeta, 0.0);
    std::vector<double> rmkss(nzeta, 0.0);
    std::vector<double> zmksc(nzeta, 0.0);
    std::vector<double> zmkcs(nzeta, 0.0);
    std::vector<double> rmksc_asym(nzeta, 0.0);
    std::vector<double> rmkcs_asym(nzeta, 0.0);
    std::vector<double> zmkcc_asym(nzeta, 0.0);
    std::vector<double> zmkss_asym(nzeta, 0.0);

    // STAGE 1: Accumulate zeta contributions for both symmetric and asymmetric
    // Only process n >= 0 (jVMEC uses n ∈ [0, ntor] exactly)
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

        // Get basis functions (n >= 0 always, following jVMEC)
        // All basis functions from fourier_basis already include nscale normalization
        int idx_nv = k * (sizes.nnyq2 + 1) + n;
        double cos_nv = fourier_basis.cosnv[idx_nv];
        double sin_nv = fourier_basis.sinnv[idx_nv];

        // Accumulate symmetric coefficients
        rmkcc[k] += rmncc[mn] * cos_nv;
        zmksc[k] += zmnsc[mn] * cos_nv;

        if (sizes.lthreed) {
          rmkss[k] += rmnss[mn] * sin_nv;
          zmkcs[k] += zmncs[mn] * sin_nv;
        }

        // Accumulate asymmetric coefficients
        rmksc_asym[k] += rmnsc[mn] * cos_nv;
        zmkcc_asym[k] += zmncc[mn] * cos_nv;

        if (sizes.lthreed) {
          rmkcs_asym[k] += rmncs[mn] * sin_nv;
          zmkss_asym[k] += zmnss[mn] * sin_nv;
        }
      }
    }

    // STAGE 2: Transform in theta for [0,pi]
    for (int l = 0; l < ntheta2; ++l) {
      int idx_basis = m * sizes.nThetaReduced + l;
      if (idx_basis >= fourier_basis.sinmu.size()) {
        // Debug: skip invalid access
        continue;
      }
      double sin_mu = fourier_basis.sinmu[idx_basis];
      double cos_mu = fourier_basis.cosmu[idx_basis];

      for (int k = 0; k < nzeta; ++k) {
        int idx = l * nzeta + k;

        // Symmetric contributions
        r_real[idx] += rmkcc[k] * cos_mu;
        z_real[idx] += zmksc[k] * sin_mu;
        
        // Symmetric derivatives (dR/dtheta, dZ/dtheta)
        ru_real[idx] += -m * rmkcc[k] * sin_mu;  // d(cos(m*theta))/dtheta = -m*sin(m*theta)
        zu_real[idx] += m * zmksc[k] * cos_mu;   // d(sin(m*theta))/dtheta = m*cos(m*theta)

        if (sizes.lthreed) {
          r_real[idx] += rmkss[k] * sin_mu;
          z_real[idx] += zmkcs[k] * cos_mu;
          
          // Additional symmetric derivative contributions
          ru_real[idx] += m * rmkss[k] * cos_mu;   // d(sin(m*theta))/dtheta = m*cos(m*theta)
          zu_real[idx] += -m * zmkcs[k] * sin_mu;  // d(cos(m*theta))/dtheta = -m*sin(m*theta)
        }

        // Asymmetric contributions (stored separately for reflection)
        asym_R[idx] += rmksc_asym[k] * sin_mu;
        asym_Z[idx] += zmkcc_asym[k] * cos_mu;
        
        // Asymmetric derivatives (following jVMEC pattern)
        asym_Ru[idx] += m * rmksc_asym[k] * cos_mu;  // d(sin(m*theta))/dtheta = m*cos(m*theta)
        asym_Zu[idx] += -m * zmkcc_asym[k] * sin_mu; // d(cos(m*theta))/dtheta = -m*sin(m*theta)

        if (sizes.lthreed) {
          asym_R[idx] += rmkcs_asym[k] * cos_mu;
          asym_Z[idx] += zmkss_asym[k] * sin_mu;
          
          // Additional asymmetric derivative contributions
          asym_Ru[idx] += -m * rmkcs_asym[k] * sin_mu; // d(cos(m*theta))/dtheta = -m*sin(m*theta)
          asym_Zu[idx] += m * zmkss_asym[k] * cos_mu;  // d(sin(m*theta))/dtheta = m*cos(m*theta)
        }
      }
    }
  }

  // STEP 1: Add asymmetric contributions for theta=[0,pi]
  for (int l = 0; l < ntheta2; ++l) {
    for (int k = 0; k < nzeta; ++k) {
      int idx = l * nzeta + k;
      if (idx >= r_real.size()) continue;

      r_real[idx] += asym_R[idx];
      z_real[idx] += asym_Z[idx];
      lambda_real[idx] += asym_L[idx];
      
      // Add asymmetric derivative contributions
      ru_real[idx] += asym_Ru[idx];
      zu_real[idx] += asym_Zu[idx];
    }
  }

  // STEP 2: Symmetrization to fill theta=[pi,2pi] using jVMEC pattern
  // For u in [π, 2π]: total = symmetric - asymmetric
  const int ntheta1 = 2 * ntheta2;  // full range size
  for (int l = ntheta2; l < ntheta_eff; ++l) {
    for (int k = 0; k < nzeta; ++k) {
      int idx = l * nzeta + k;
      if (idx >= r_real.size()) continue;
      
      // Find corresponding point using jVMEC reflection formula
      int lr = ntheta1 - l;  // jVMEC: theta reflection
      if (lr >= ntheta2) lr = ntheta2 - 1;  // ensure within [0, pi] range
      int kr = (nzeta - k) % nzeta;  // jVMEC: zeta reflection
      int idx_reflected = lr * nzeta + kr;
      
      if (idx_reflected >= 0 && idx_reflected < ntheta2 * nzeta) {
        // Apply symmetrization following jVMEC pattern:
        // For theta in [pi, 2pi]: total = symmetric - asymmetric
        // Get symmetric part (before asymmetric was added)
        double r_sym = r_real[idx_reflected] - asym_R[idx_reflected];
        double z_sym = z_real[idx_reflected] - asym_Z[idx_reflected];
        double lambda_sym = lambda_real[idx_reflected] - asym_L[idx_reflected];
        double ru_sym = ru_real[idx_reflected] - asym_Ru[idx_reflected];
        double zu_sym = zu_real[idx_reflected] - asym_Zu[idx_reflected];
        
        // Apply reflection with sign changes
        r_real[idx] = r_sym - asym_R[idx_reflected];
        z_real[idx] = -z_sym + asym_Z[idx_reflected];
        lambda_real[idx] = lambda_sym - asym_L[idx_reflected];
        
        // Derivative symmetrization
        ru_real[idx] = -ru_sym - asym_Ru[idx_reflected];
        zu_real[idx] = zu_sym + asym_Zu[idx_reflected];
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
    absl::Span<double> z_real, absl::Span<double> lambda_real,
    absl::Span<double> ru_real, absl::Span<double> zu_real) {
  const int nzeta = sizes.nZeta;
  const int ntheta2 = sizes.nThetaReduced;  // [0, pi]
  const int ntheta1 = 2 * ntheta2;          // full range [0, 2pi]

  // Initialize output arrays
  std::fill(r_real.begin(), r_real.end(), 0.0);
  std::fill(z_real.begin(), z_real.end(), 0.0);
  std::fill(lambda_real.begin(), lambda_real.end(), 0.0);
  std::fill(ru_real.begin(), ru_real.end(), 0.0);
  std::fill(zu_real.begin(), zu_real.end(), 0.0);

  // Create arrays for asymmetric contributions
  std::vector<double> asym_R(ntheta1 * nzeta, 0.0);
  std::vector<double> asym_Z(ntheta1 * nzeta, 0.0);
  std::vector<double> asym_L(ntheta1 * nzeta, 0.0);
  
  // Create arrays for asymmetric derivative contributions
  std::vector<double> asym_Ru(ntheta1 * nzeta, 0.0);
  std::vector<double> asym_Zu(ntheta1 * nzeta, 0.0);

  // Get basis functions
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // For 2D case (ntor=0), only n=0 modes exist
  // cosnv=1, sinnv=0 for all n=0

  // Process each surface
  const int num_surfaces = rmncc.size() / sizes.mnmax;
  for (int js = 0; js < num_surfaces; ++js) {
    // Clear temporary arrays for this surface
    std::fill(asym_R.begin(), asym_R.end(), 0.0);
    std::fill(asym_Z.begin(), asym_Z.end(), 0.0);
    std::fill(asym_L.begin(), asym_L.end(), 0.0);
    std::fill(asym_Ru.begin(), asym_Ru.end(), 0.0);
    std::fill(asym_Zu.begin(), asym_Zu.end(), 0.0);
    
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

      // Get coefficients for this surface
      int coeff_idx = js * sizes.mnmax + mn;
      
      // Get symmetric coefficients
      double rcc = (coeff_idx < rmncc.size()) ? rmncc[coeff_idx] : 0.0;
      double zsc = (coeff_idx < zmnsc.size()) ? zmnsc[coeff_idx] : 0.0;

      // Get asymmetric coefficients
      double rsc = (coeff_idx < rmnsc.size()) ? rmnsc[coeff_idx] : 0.0;
      double zcc = (coeff_idx < zmncc.size()) ? zmncc[coeff_idx] : 0.0;

      // Compute both symmetric and asymmetric contributions for full theta range
      for (int l = 0; l < sizes.nThetaEff; ++l) {
        // Map to basis functions (only defined for [0,pi])
        int l_basis = (l < ntheta2) ? l : (ntheta1 - l);
        int idx_basis = m * sizes.nThetaReduced + l_basis;
        if (idx_basis >= fourier_basis.sinmu.size()) {
          continue;
        }
        double sin_mu = fourier_basis.sinmu[idx_basis];
        double cos_mu = fourier_basis.cosmu[idx_basis];
        
        // Sign adjustments for second half
        if (l >= ntheta2) {
          sin_mu = -sin_mu;  // sin(2π - θ) = -sin(θ)
        }

        for (int k = 0; k < nzeta; ++k) {
          int idx_surf = js * sizes.nZnT + l * nzeta + k;  // Index in output arrays
          
          if (idx_surf >= r_real.size()) continue;

          // Symmetric contributions
          r_real[idx_surf] += rcc * cos_mu;  // rmncc * cosmu
          z_real[idx_surf] += zsc * sin_mu;  // zmnsc * sinmu
          
          // Symmetric derivatives
          ru_real[idx_surf] += -m * rcc * sin_mu;  // d(cos(m*theta))/dtheta = -m*sin(m*theta)
          zu_real[idx_surf] += m * zsc * cos_mu;   // d(sin(m*theta))/dtheta = m*cos(m*theta)
          
          // For asymmetric part in second half, apply reflection
          if (l < ntheta2) {
            // First half: add asymmetric directly
            r_real[idx_surf] += rsc * sin_mu;  // rmnsc * sinmu
            z_real[idx_surf] += zcc * cos_mu;  // zmncc * cosmu
            ru_real[idx_surf] += m * rsc * cos_mu;   // d(sin(m*theta))/dtheta
            zu_real[idx_surf] += -m * zcc * sin_mu;  // d(cos(m*theta))/dtheta
          } else {
            // Second half: subtract asymmetric (reflection)
            int kr = (nzeta - k) % nzeta;
            r_real[idx_surf] -= rsc * sin_mu;  // subtract for reflection
            z_real[idx_surf] -= zcc * cos_mu;  // note: z also gets negated
            ru_real[idx_surf] -= m * rsc * cos_mu;   // derivative sign change
            zu_real[idx_surf] += m * zcc * sin_mu;   // derivative sign change
          }
        }
      }
    }
  }  // end surface loop
}

// FIXED VERSION - Following educational_VMEC pattern exactly
void SymmetrizeRealSpaceGeometry(
    const absl::Span<const double> r_sym, const absl::Span<const double> r_asym,
    const absl::Span<const double> z_sym, const absl::Span<const double> z_asym,
    const absl::Span<const double> lambda_sym,
    const absl::Span<const double> lambda_asym, absl::Span<double> r_full,
    absl::Span<double> z_full, absl::Span<double> lambda_full,
    const Sizes& sizes) {
  // This implements the correct educational_VMEC symrzl logic
  // Key insight: Keep symmetric and antisymmetric arrays separate
  // and combine them with proper reflection, NOT division by tau

  const int ntheta_reduced = sizes.nThetaReduced;  // [0, π] range
  const int ntheta_eff = sizes.nThetaEff;          // [0, 2π] range
  const int nzeta = sizes.nZeta;

  // Calculate number of surfaces from array sizes
  const int reduced_slice_size = ntheta_reduced * nzeta;
  const int full_slice_size = ntheta_eff * nzeta;
  const int nsurfaces = r_sym.size() / reduced_slice_size;

  std::cout << "DEBUG SymmetrizeRealSpaceGeometry FIXED: ntheta_reduced="
            << ntheta_reduced << ", ntheta_eff=" << ntheta_eff
            << ", nzeta=" << nzeta << ", nsurfaces=" << nsurfaces << std::endl;

  // Process each surface separately
  for (int surface = 0; surface < nsurfaces; ++surface) {
    // First half: theta in [0, π] - direct addition (NO zeta reflection)
    for (int k = 0; k < nzeta; ++k) {
      for (int j = 0; j < ntheta_reduced; ++j) {
        const int idx_half =
            j + k * ntheta_reduced + surface * reduced_slice_size;
        const int idx_full_first =
            j + k * ntheta_eff + surface * full_slice_size;

        // Bounds checking
        if (idx_half >= r_sym.size() || idx_full_first >= r_full.size()) {
          std::cout << "ERROR: Bounds error in first half at surface="
                    << surface << ", j=" << j << ", k=" << k << std::endl;
          continue;
        }

        // First half: symmetric + antisymmetric (direct mapping)
        r_full[idx_full_first] = r_sym[idx_half] + r_asym[idx_half];
        z_full[idx_full_first] = z_sym[idx_half] + z_asym[idx_half];
        lambda_full[idx_full_first] =
            lambda_sym[idx_half] + lambda_asym[idx_half];
      }
    }

    // Second half: theta in [π, 2π] - reflection with sign change
    // Following educational_VMEC/jVMEC pattern: theta -> 2π - theta, zeta ->
    // -zeta
    for (int k = 0; k < nzeta; ++k) {
      // Critical fix: limit j to ensure idx_full_second stays in bounds
      const int max_j_second = ntheta_eff - ntheta_reduced;
      for (int j = 0; j < max_j_second; ++j) {
        const int idx_full_second =
            (j + ntheta_reduced) + k * ntheta_eff + surface * full_slice_size;

        // Reflection mapping: theta -> 2π - theta, zeta -> -zeta
        const int j_reflected = ntheta_reduced - 1 - j;
        const int k_reflected =
            (nzeta - k) % nzeta;  // zeta -> -zeta reflection
        const int idx_reflected = j_reflected + k_reflected * ntheta_reduced +
                                  surface * reduced_slice_size;

        // Detailed bounds checking with debug info
        if (idx_reflected >= r_sym.size() || idx_full_second >= r_full.size()) {
          std::cout << "ERROR: Bounds error in second half at surface="
                    << surface << ", j=" << j << ", k=" << k
                    << ", j_reflected=" << j_reflected
                    << ", k_reflected=" << k_reflected
                    << ", idx_reflected=" << idx_reflected
                    << " (max=" << r_sym.size() << ")"
                    << ", idx_full_second=" << idx_full_second
                    << " (max=" << r_full.size() << ")"
                    << ", max_j_second=" << max_j_second << std::endl;
          continue;
        }

        // Second half: symmetric - antisymmetric (with reflection)
        r_full[idx_full_second] = r_sym[idx_reflected] - r_asym[idx_reflected];
        z_full[idx_full_second] = z_sym[idx_reflected] - z_asym[idx_reflected];
        lambda_full[idx_full_second] =
            lambda_sym[idx_reflected] - lambda_asym[idx_reflected];
      }
    }
  }

  std::cout << "DEBUG: Symmetrization completed successfully" << std::endl;
}

// OLD VERSION - DEPRECATED - Keep temporarily for compatibility
void SymmetrizeRealSpaceGeometry(const Sizes& sizes, absl::Span<double> r_real,
                                 absl::Span<double> z_real,
                                 absl::Span<double> lambda_real) {
  // This function should only be called for asymmetric equilibria
  if (!sizes.lasym) {
    return;
  }

  std::cout << "WARNING: Using old SymmetrizeRealSpaceGeometry - should be "
               "replaced with fixed version"
            << std::endl;

  // Build reflection index for zeta -> -zeta mapping
  std::vector<int> ireflect(sizes.nZeta);
  for (int k = 0; k < sizes.nZeta; ++k) {
    ireflect[k] = (sizes.nZeta - k) % sizes.nZeta;
  }

  // Process extended interval [π, 2π] using symmetry relations
  for (int i = sizes.nThetaReduced; i < sizes.nThetaEff; ++i) {
    int ir = sizes.nThetaReduced + (sizes.nThetaReduced - 1 - i);

    if (ir < 0 || ir >= sizes.nThetaEff) {
      continue;
    }

    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      int idx_r = ir * sizes.nZeta + ireflect[k];

      if (ireflect[k] < 0 || ireflect[k] >= sizes.nZeta) {
        continue;
      }

      if (idx >= static_cast<int>(r_real.size()) ||
          idx_r >= static_cast<int>(r_real.size())) {
        continue;
      }

      r_real[idx] = r_real[idx_r];
      z_real[idx] = -z_real[idx_r];
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

  // Compute scaling factors for inverse transform
  // Since forward transform applies sqrt(2) for m>0, n>0,
  // inverse transform must also apply sqrt(2) to recover coefficients
  // (due to symmetric normalization convention)
  std::vector<double> mscale(sizes.mpol + 1);
  std::vector<double> nscale(sizes.ntor + 1);
  mscale[0] = 1.0;
  nscale[0] = 1.0;
  for (int m = 1; m <= sizes.mpol; ++m) {
    mscale[m] = sqrt(2.0);  // Match forward transform normalization
  }
  for (int n = 1; n <= sizes.ntor; ++n) {
    nscale[n] = sqrt(2.0);  // Match forward transform normalization
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

    // Apply normalization factors to match forward transform
    // Forward transform applies sqrt(2) for m>0 and n>0 modes
    // Inverse must apply 1/sqrt(2) to recover original coefficients
    double mode_scale = mscale[m];
    if (n != 0) {
      mode_scale *= nscale[std::abs(n)];
    }

    // Store coefficients with DFT normalization and mode scaling
    rmncc[mn] = sum_rmncc * norm_factor * mode_scale;
    rmnss[mn] = sum_rmnss * norm_factor * mode_scale;
    rmnsc[mn] = sum_rmnsc * norm_factor * mode_scale;
    rmncs[mn] = sum_rmncs * norm_factor * mode_scale;

    zmnsc[mn] = sum_zmnsc * norm_factor * mode_scale;
    zmncs[mn] = sum_zmncs * norm_factor * mode_scale;
    zmncc[mn] = sum_zmncc * norm_factor * mode_scale;
    zmnss[mn] = sum_zmnss * norm_factor * mode_scale;
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
  // Match forward transform normalization convention
  std::vector<double> mscale(sizes.mpol + 1);
  mscale[0] = 1.0;
  for (int m = 1; m <= sizes.mpol; ++m) {
    mscale[m] = sqrt(2.0);  // Same as 3D case
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

    // Apply mode scaling to match forward transform normalization
    double mode_scale = mscale[m];

    // Store coefficients with DFT normalization and mode scaling
    rmncc[mn] = sum_rmncc * norm_factor * mode_scale;
    rmnsc[mn] = sum_rmnsc * norm_factor * mode_scale;
    zmnsc[mn] = sum_zmnsc * norm_factor * mode_scale;
    zmncc[mn] = sum_zmncc * norm_factor * mode_scale;
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

void FourierToReal3DAsymmFastPoloidalSeparated(
    const Sizes& sizes, absl::Span<const double> rmncc,
    absl::Span<const double> rmnss, absl::Span<const double> rmnsc,
    absl::Span<const double> rmncs, absl::Span<const double> zmnsc,
    absl::Span<const double> zmncs, absl::Span<const double> zmncc,
    absl::Span<const double> zmnss, absl::Span<double> r_sym,
    absl::Span<double> r_asym, absl::Span<double> z_sym,
    absl::Span<double> z_asym, absl::Span<double> lambda_sym,
    absl::Span<double> lambda_asym, absl::Span<double> ru_sym,
    absl::Span<double> ru_asym, absl::Span<double> zu_sym,
    absl::Span<double> zu_asym) {
  const int nzeta = sizes.nZeta;
  const int ntheta2 = sizes.nThetaReduced;   // [0, pi] including endpoint
  const int nznt_reduced = ntheta2 * nzeta;  // Size for separate arrays

  // Initialize output arrays - these are for [0, π] range only
  std::fill(r_sym.begin(), r_sym.end(), 0.0);
  std::fill(r_asym.begin(), r_asym.end(), 0.0);
  std::fill(z_sym.begin(), z_sym.end(), 0.0);
  std::fill(z_asym.begin(), z_asym.end(), 0.0);
  std::fill(lambda_sym.begin(), lambda_sym.end(), 0.0);
  std::fill(lambda_asym.begin(), lambda_asym.end(), 0.0);
  std::fill(ru_sym.begin(), ru_sym.end(), 0.0);
  std::fill(ru_asym.begin(), ru_asym.end(), 0.0);
  std::fill(zu_sym.begin(), zu_sym.end(), 0.0);
  std::fill(zu_asym.begin(), zu_asym.end(), 0.0);

  // Get basis functions
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // Process each poloidal mode m
  for (int m = 0; m < sizes.mpol; ++m) {
    // Work arrays for this m mode
    std::vector<double> rmkcc(nzeta, 0.0);       // Symmetric R
    std::vector<double> rmkss(nzeta, 0.0);       // Symmetric R
    std::vector<double> zmksc(nzeta, 0.0);       // Symmetric Z
    std::vector<double> zmkcs(nzeta, 0.0);       // Symmetric Z
    std::vector<double> rmksc_asym(nzeta, 0.0);  // Antisymmetric R
    std::vector<double> rmkcs_asym(nzeta, 0.0);  // Antisymmetric R
    std::vector<double> zmkcc_asym(nzeta, 0.0);  // Antisymmetric Z
    std::vector<double> zmkss_asym(nzeta, 0.0);  // Antisymmetric Z

    // STAGE 1: Accumulate zeta contributions for both symmetric and asymmetric
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

        // Calculate zeta angle
        const double zeta = 2.0 * M_PI * k / nzeta;
        const double arg = n * zeta;
        const double cos_arg = cos(arg);
        const double sin_arg = sin(arg);

        // Symmetric coefficients (same as original VMEC)
        double rcc = (mn < static_cast<int>(rmncc.size())) ? rmncc[mn] : 0.0;
        double rss = (mn < static_cast<int>(rmnss.size())) ? rmnss[mn] : 0.0;
        double zsc = (mn < static_cast<int>(zmnsc.size())) ? zmnsc[mn] : 0.0;
        double zcs = (mn < static_cast<int>(zmncs.size())) ? zmncs[mn] : 0.0;

        // Antisymmetric coefficients (new for asymmetric mode)
        double rsc = (mn < static_cast<int>(rmnsc.size())) ? rmnsc[mn] : 0.0;
        double rcs = (mn < static_cast<int>(rmncs.size())) ? rmncs[mn] : 0.0;
        double zcc = (mn < static_cast<int>(zmncc.size())) ? zmncc[mn] : 0.0;
        double zss = (mn < static_cast<int>(zmnss.size())) ? zmnss[mn] : 0.0;

        // Accumulate symmetric contributions
        rmkcc[k] += rcc * cos_arg;
        rmkss[k] += rss * sin_arg;
        zmksc[k] += zsc * sin_arg;
        zmkcs[k] += zcs * cos_arg;

        // Accumulate antisymmetric contributions
        rmksc_asym[k] += rsc * sin_arg;
        rmkcs_asym[k] += rcs * cos_arg;
        zmkcc_asym[k] += zcc * cos_arg;
        zmkss_asym[k] += zss * sin_arg;
      }
    }

    // STAGE 2: Transform to real space for [0, π] range only
    for (int j = 0; j < ntheta2; ++j) {
      // Calculate theta angle for [0, π] range
      const double theta = M_PI * j / (ntheta2 - 1);

      // Basis function index
      const int idx_basis = m * ntheta2 + j;
      if (idx_basis >= static_cast<int>(fourier_basis.sinmu.size())) {
        continue;
      }

      const double cosmu = fourier_basis.cosmu[idx_basis];
      const double sinmu = fourier_basis.sinmu[idx_basis];

      for (int k = 0; k < nzeta; ++k) {
        const int idx = j + k * ntheta2;  // Index for [0, π] arrays
        if (idx >= static_cast<int>(r_sym.size())) continue;

        // Add symmetric contributions
        r_sym[idx] += rmkcc[k] * cosmu + rmkss[k] * sinmu;
        z_sym[idx] += zmksc[k] * sinmu + zmkcs[k] * cosmu;
        // Lambda symmetric (for now, set to zero - can be added later)
        // lambda_sym[idx] += ...;
        
        // Add symmetric derivatives
        ru_sym[idx] += -m * rmkcc[k] * sinmu + m * rmkss[k] * cosmu;
        zu_sym[idx] += m * zmksc[k] * cosmu - m * zmkcs[k] * sinmu;

        // Add antisymmetric contributions
        r_asym[idx] += rmksc_asym[k] * sinmu + rmkcs_asym[k] * cosmu;
        z_asym[idx] += zmkcc_asym[k] * cosmu + zmkss_asym[k] * sinmu;
        // Lambda antisymmetric (for now, set to zero - can be added later)
        // lambda_asym[idx] += ...;
        
        // Add antisymmetric derivatives
        ru_asym[idx] += m * rmksc_asym[k] * cosmu - m * rmkcs_asym[k] * sinmu;
        zu_asym[idx] += -m * zmkcc_asym[k] * sinmu + m * zmkss_asym[k] * cosmu;
      }
    }
  }
}

}  // namespace vmecpp
