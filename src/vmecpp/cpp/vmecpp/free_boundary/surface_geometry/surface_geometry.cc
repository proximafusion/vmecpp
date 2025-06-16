// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"

#include "absl/algorithm/container.h"  // c_fill_n

namespace vmecpp {

SurfaceGeometry::SurfaceGeometry(const Sizes* s,
                                 const FourierBasisFastToroidal* fb,
                                 const TangentialPartitioning* tp)
    : s_(*s), fb_(*fb), tp_(*tp) {
  cos_per.resize(s_.nfp);
  sin_per.resize(s_.nfp);

  cos_phi.resize(s_.nZeta);
  sin_phi.resize(s_.nZeta);

  // -----------------

  // full surface
  r1b.resize(s_.nThetaEven * s_.nZeta);
  z1b.resize(s_.nThetaEven * s_.nZeta);
  rcosuv.resize(s_.nThetaEven * s_.nZeta);
  rsinuv.resize(s_.nThetaEven * s_.nZeta);
  rzb2.resize(s_.nThetaEven * s_.nZeta);

  // thread-local tangential grid point range
  int numLocal = tp_.ztMax - tp_.ztMin;

  rub.resize(numLocal);
  rvb.resize(numLocal);
  zub.resize(numLocal);
  zvb.resize(numLocal);

  ruu.resize(numLocal);
  ruv.resize(numLocal);
  rvv.resize(numLocal);
  zuu.resize(numLocal);
  zuv.resize(numLocal);
  zvv.resize(numLocal);

  snr.resize(numLocal);
  snv.resize(numLocal);
  snz.resize(numLocal);

  guu.resize(numLocal);
  guv.resize(numLocal);
  gvv.resize(numLocal);

  auu.resize(numLocal);
  auv.resize(numLocal);
  avv.resize(numLocal);

  drv.resize(numLocal);

  // -----------------

  computeConstants();
}

void SurfaceGeometry::computeConstants() {
  double omega_per = 2.0 * M_PI / s_.nfp;
  for (int p = 0; p < s_.nfp; ++p) {
    double phi_per = omega_per * p;
    cos_per[p] = cos(phi_per);
    sin_per[p] = sin(phi_per);
  }

  double omega_phi = 2.0 * M_PI / (s_.nfp * s_.nZeta);
  for (int k = 0; k < s_.nZeta; ++k) {
    double phi = omega_phi * k;
    cos_phi[k] = cos(phi);
    sin_phi[k] = sin(phi);
  }
}

// Evaluate the Fourier series for the surface geometry
// and compute quantities depending on it.
void SurfaceGeometry::update(
    const std::span<const double> rCC, const std::span<const double> rSS,
    const std::span<const double> rSC, const std::span<const double> rCS,
    const std::span<const double> zSC, const std::span<const double> zCS,
    const std::span<const double> zCC, const std::span<const double> zSS,
    int signOfJacobian, bool fullUpdate) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  inverseDFT(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  derivedSurfaceQuantities(signOfJacobian, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

// Perform inverse Fourier transform from Fourier coefficients of surface
// geometry into realspace and compute 1st- and 2nd-order tangential
// derivatives.
void SurfaceGeometry::inverseDFT(
    const std::span<const double> rCC, const std::span<const double> rSS,
    const std::span<const double> rSC, const std::span<const double> rCS,
    const std::span<const double> zSC, const std::span<const double> zCS,
    const std::span<const double> zCC, const std::span<const double> zSS,
    bool fullUpdate) {
  // TODO(jons): implement lasym-related code
  (void)rSC;
  (void)rCS;
  (void)zCC;
  (void)zSS;

  // ----------------

  absl::c_fill_n(r1b, s_.nThetaEven * s_.nZeta, 0);
  absl::c_fill_n(z1b, s_.nThetaEven * s_.nZeta, 0);

  // ----------------
  int numLocal = tp_.ztMax - tp_.ztMin;

  absl::c_fill_n(rub, numLocal, 0);
  absl::c_fill_n(rvb, numLocal, 0);
  absl::c_fill_n(zub, numLocal, 0);
  absl::c_fill_n(zvb, numLocal, 0);

  if (fullUpdate) {
    absl::c_fill_n(ruu, numLocal, 0);
    absl::c_fill_n(ruv, numLocal, 0);
    absl::c_fill_n(rvv, numLocal, 0);
    absl::c_fill_n(zuu, numLocal, 0);
    absl::c_fill_n(zuv, numLocal, 0);
    absl::c_fill_n(zvv, numLocal, 0);
  }

  for (int n = 0; n < s_.ntor + 1; ++n) {
    // needed for second-order toroidal derivatives
    int nSq = n * s_.nfp * n * s_.nfp;

    int lMin = tp_.ztMin / s_.nZeta;
    int lMax = tp_.ztMax / s_.nZeta;

    for (int l = 0; l < s_.nThetaReduced; ++l) {
      double rmkcc = 0.0;
      double rmkss = 0.0;
      double zmksc = 0.0;
      double zmkcs = 0.0;

      // ----------------

      double rmkcc_m = 0.0;
      double rmkcc_mm = 0.0;
      double rmkss_m = 0.0;
      double rmkss_mm = 0.0;

      double zmksc_m = 0.0;
      double zmksc_mm = 0.0;
      double zmkcs_m = 0.0;
      double zmkcs_mm = 0.0;

      for (int m = 0; m < s_.mpol; ++m) {
        int idx_mn = n * s_.mpol + m;

        // needed for second-order poloidal derivatives
        int mSq = m * m;

        double cosmu = fb_.cosmu[l * (s_.mnyq2 + 1) + m];
        double sinmu = fb_.sinmu[l * (s_.mnyq2 + 1) + m];

        rmkcc += rCC[idx_mn] * cosmu;
        rmkss += rSS[idx_mn] * sinmu;
        zmksc += zSC[idx_mn] * sinmu;
        zmkcs += zCS[idx_mn] * cosmu;

        // ----------------

        if (lMin <= l && l <= lMax) {
          // TODO(jons): in asymmetric case, some processors will have local
          // poloidal ranges outside the first half-module
          // --> these would be excluded here, but they still need to do some
          // work here!

          double cosmum = fb_.cosmum[l * (s_.mnyq2 + 1) + m];
          double sinmum = fb_.sinmum[l * (s_.mnyq2 + 1) + m];
          double cosmumm = -mSq * fb_.cosmu[l * (s_.mnyq2 + 1) + m];
          double sinmumm = -mSq * fb_.sinmu[l * (s_.mnyq2 + 1) + m];

          rmkcc_m += rCC[idx_mn] * sinmum;
          rmkcc_mm += rCC[idx_mn] * cosmumm;
          rmkss_m += rSS[idx_mn] * cosmum;
          rmkss_mm += rSS[idx_mn] * sinmumm;

          zmksc_m += zSC[idx_mn] * cosmum;
          zmksc_mm += zSC[idx_mn] * sinmumm;
          zmkcs_m += zCS[idx_mn] * sinmum;
          zmkcs_mm += zCS[idx_mn] * cosmumm;
        }
      }  // m

      for (int k = 0; k < s_.nZeta; ++k) {
        int idx_kl = l * s_.nZeta + k;

        double cosnv = fb_.cosnv[n * s_.nZeta + k];
        double sinnv = fb_.sinnv[n * s_.nZeta + k];

        r1b[idx_kl] += rmkcc * cosnv + rmkss * sinnv;
        z1b[idx_kl] += zmksc * cosnv + zmkcs * sinnv;

        // ----------------

        if (tp_.ztMin <= idx_kl && idx_kl < tp_.ztMax) {
          double cosnvn = fb_.cosnvn[n * s_.nZeta + k];
          double sinnvn = fb_.sinnvn[n * s_.nZeta + k];

          rub[idx_kl - tp_.ztMin] += rmkcc_m * cosnv + rmkss_m * sinnv;
          rvb[idx_kl - tp_.ztMin] += rmkcc * sinnvn + rmkss * cosnvn;
          zub[idx_kl - tp_.ztMin] += zmksc_m * cosnv + zmkcs_m * sinnv;
          zvb[idx_kl - tp_.ztMin] += zmksc * sinnvn + zmkcs * cosnvn;

          if (fullUpdate) {
            double cosnvnn = -nSq * fb_.cosnv[n * s_.nZeta + k];
            double sinnvnn = -nSq * fb_.sinnv[n * s_.nZeta + k];

            ruu[idx_kl - tp_.ztMin] += rmkcc_mm * cosnv + rmkss_mm * sinnv;
            ruv[idx_kl - tp_.ztMin] += rmkcc_m * sinnvn + rmkss_m * cosnvn;
            rvv[idx_kl - tp_.ztMin] += rmkcc * cosnvnn + rmkss * sinnvnn;
            zuu[idx_kl - tp_.ztMin] += zmksc_mm * cosnv + zmkcs_mm * sinnv;
            zuv[idx_kl - tp_.ztMin] += zmksc_m * sinnvn + zmkcs_m * cosnvn;
            zvv[idx_kl - tp_.ztMin] += zmksc * cosnvnn + zmkcs * sinnvnn;
          }
        }
      }  // k
    }  // l
  }  // n

  if (s_.lasym) {
    // mirror quantities into respective
    // non-symmetric other half of poloidal interval ]pi,2pi[

    // TODO(jons)
  }
}

void SurfaceGeometry::derivedSurfaceQuantities(int signOfJacobian,
                                               bool fullUpdate) {
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    // surface normal vector components
    snr[kl - tp_.ztMin] = signOfJacobian * r1b[kl] * zub[kl - tp_.ztMin];
    snv[kl - tp_.ztMin] =
        signOfJacobian * (rub[kl - tp_.ztMin] * zvb[kl - tp_.ztMin] -
                          zub[kl - tp_.ztMin] * rvb[kl - tp_.ztMin]);
    snz[kl - tp_.ztMin] = -signOfJacobian * r1b[kl] * rub[kl - tp_.ztMin];

    // metric elements; used in Imn and Kmn
    guu[kl - tp_.ztMin] = rub[kl - tp_.ztMin] * rub[kl - tp_.ztMin] +
                          zub[kl - tp_.ztMin] * zub[kl - tp_.ztMin];
    guv[kl - tp_.ztMin] = 2.0 *
                          (rub[kl - tp_.ztMin] * rvb[kl - tp_.ztMin] +
                           zub[kl - tp_.ztMin] * zvb[kl - tp_.ztMin]) /
                          s_.nfp;
    gvv[kl - tp_.ztMin] =
        (rvb[kl - tp_.ztMin] * rvb[kl - tp_.ztMin] + r1b[kl] * r1b[kl] +
         zvb[kl - tp_.ztMin] * zvb[kl - tp_.ztMin]) /
        (s_.nfp * s_.nfp);

    if (fullUpdate) {
      // d^2X/d(ij) . N (used in Kmn)
      auu[kl - tp_.ztMin] = (ruu[kl - tp_.ztMin] * snr[kl - tp_.ztMin] +
                             zuu[kl - tp_.ztMin] * snz[kl - tp_.ztMin]) /
                            2;
      auv[kl - tp_.ztMin] = (ruv[kl - tp_.ztMin] * snr[kl - tp_.ztMin] +
                             rub[kl - tp_.ztMin] * snv[kl - tp_.ztMin] +
                             zuv[kl - tp_.ztMin] * snz[kl - tp_.ztMin]) /
                            s_.nfp;
      avv[kl - tp_.ztMin] =
          (rvb[kl - tp_.ztMin] * snv[kl - tp_.ztMin] +
           ((rvv[kl - tp_.ztMin] - r1b[kl]) * snr[kl - tp_.ztMin] +
            zvv[kl - tp_.ztMin] * snz[kl - tp_.ztMin]) /
               2) /
          (s_.nfp * s_.nfp);

      // -(R N^R + Z N^Z)
      drv[kl - tp_.ztMin] =
          -(r1b[kl] * snr[kl - tp_.ztMin] + z1b[kl] * snz[kl - tp_.ztMin]);
    }
  }  // kl

  if (fullUpdate) {
    // R^2 + Z^2
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      rzb2[kl] = r1b[kl] * r1b[kl] + z1b[kl] * z1b[kl];
    }

    if (!s_.lasym) {
      // mirror into non-stellarator-symmetric half of poloidal range
      for (int l = 1; l < s_.nThetaReduced - 1; ++l) {
        int lRev = (s_.nThetaEven - l) % s_.nThetaEven;
        for (int k = 0; k < s_.nZeta; ++k) {
          int kRev = (s_.nZeta - k) % s_.nZeta;

          int kl = l * s_.nZeta + k;
          int klRev = lRev * s_.nZeta + kRev;

          r1b[klRev] = r1b[kl];
          z1b[klRev] = -z1b[kl];

          rzb2[klRev] = rzb2[kl];
        }  // k
      }  // l
    }

    // x and y
    for (int kl = 0; kl < s_.nThetaEven * s_.nZeta; ++kl) {
      int k = kl % s_.nZeta;
      rcosuv[kl] = r1b[kl] * cos_phi[k];
      rsinuv[kl] = r1b[kl] * sin_phi[k];
    }  // kl
  }
}

}  // namespace vmecpp
