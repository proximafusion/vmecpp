// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/regularized_integrals/regularized_integrals.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"

namespace vmecpp {

RegularizedIntegrals::RegularizedIntegrals(const Sizes* s,
                                           const TangentialPartitioning* tp,
                                           const SurfaceGeometry* sg)
    : s_(*s), tp_(*tp), sg_(*sg) {
  gsave.resize(s_.nThetaEven * s_.nZeta);
  dsave.resize(s_.nThetaEven * s_.nZeta);

  tanu.resize(s_.nThetaEven);
  tanv.resize(s_.nZeta * s_.nVacuumPeriods);

  // thread-local tangential grid point range
  int numLocal = tp_.ztMax - tp_.ztMin;
  greenp.resize(numLocal * s_.nThetaEven * s_.nZeta);

  gstore.resize(s_.nThetaEven * s_.nZeta);

  // --------------

  computeConstants();
}

void RegularizedIntegrals::computeConstants() {
  const double epsTan = 1.0e-15;
  const double bigNo = 1.0e50;

  for (int l = 0; l < s_.nThetaEven; ++l) {
    const double argu = M_PI / s_.nThetaEven * l;
    if (std::abs(argu - 0.5 * M_PI) < epsTan) {
      // mask singularities at pi/2
      tanu[l] = bigNo;
    } else {
      tanu[l] = 2.0 * std::tan(argu);
    }
  }  // l

  // Fortran: argv = pi*(kv-1)/nv + pi*(kp-1)/nvper
  // For 3D (nVacuumPeriods == nfp), only p==0 block is used in the full branch,
  // giving tanv[delta_k] = 2*tan(pi*delta_k/nZeta) as before.
  // For axisymmetric (nZeta==1), all 64 blocks are used.
  for (int p = 0; p < s_.nVacuumPeriods; ++p) {
    for (int k = 0; k < s_.nZeta; ++k) {
      const double argv =
          M_PI * k / s_.nZeta + M_PI * p / s_.nVacuumPeriods;
      if (std::abs(argv - 0.5 * M_PI) < epsTan) {
        tanv[k + p * s_.nZeta] = bigNo;
      } else {
        tanv[k + p * s_.nZeta] = 2.0 * std::tan(argv);
      }
    }
  }
}

void RegularizedIntegrals::update(const std::vector<double>& bDotN) {
  // thread-local tangential grid point range
  const int numLocal = tp_.ztMax - tp_.ztMin;
  const int theta_by_nzeta = s_.nThetaEven * s_.nZeta;
  const double twopidivnfp = 2.0 * M_PI / s_.nVacuumPeriods;

  absl::c_fill_n(greenp, numLocal * theta_by_nzeta, 0);
  absl::c_fill_n(gstore, theta_by_nzeta, 0);

  // storage for intermediate results
  std::vector<double> ga1_buf(s_.nZeta);
  std::vector<double> ga2_buf(s_.nZeta);
  std::vector<double> htemp_buf(s_.nZeta);
  std::vector<double> ftemp_buf(s_.nZeta);

  for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
    const int ip_idx_base = (klp - tp_.ztMin) * theta_by_nzeta;
    int lp = klp / s_.nZeta;
    int kp = klp % s_.nZeta;

    double bexni = bDotN[klp - tp_.ztMin] * s_.wInt[lp];

    double xp = sg_.rcosuv[klp];
    double yp = sg_.rsinuv[klp];

    for (int kl = 0; kl < theta_by_nzeta; ++kl) {
      gsave[kl] = sg_.rzb2[klp] + sg_.rzb2[kl] - 2 * sg_.z1b[kl] * sg_.z1b[klp];
    }

    for (int kl = 0; kl < theta_by_nzeta; ++kl) {
      dsave[kl] =
          sg_.drv[klp - tp_.ztMin] + sg_.z1b[kl] * sg_.snz[klp - tp_.ztMin];
    }

    // SUM OVER FIELD-PERIODS (NVPER=NFPER) OR INTEGRATE OVER NV (NVPER=64) IF
    // NV == 1. Matches Fortran greenf.f90 loop structure.
    for (int p = 0; p < s_.nVacuumPeriods; ++p) {
      double xper = xp * sg_.cos_per[p] - yp * sg_.sin_per[p];
      double yper = xp * sg_.sin_per[p] + yp * sg_.cos_per[p];

      double sxsave =
          (sg_.snr[klp - tp_.ztMin] * xper -
           sg_.snv[klp - tp_.ztMin] * yper) /
          sg_.r1b[klp];
      double sysave =
          (sg_.snr[klp - tp_.ztMin] * yper +
           sg_.snv[klp - tp_.ztMin] * xper) /
          sg_.r1b[klp];

      // Fortran: IF (kp.EQ.1 .OR. nv.EQ.1) THEN
      if (p == 0 || s_.nZeta == 1) {
        // Full branch: compute analytic approximation ga1, ga2
        // and subtract from exact Green's function.
        // Skips diagonal kl==klp only for p==0.
        auto full_branch = [&](int kl_start, int kl_end) {
          for (int kl = kl_start; kl < kl_end; ++kl) {
            const int l = kl / s_.nZeta;
            const int k = kl % s_.nZeta;
            const int delta_l = (l - lp + s_.nThetaEven) % s_.nThetaEven;
            const int delta_k = (k - kp + s_.nZeta) % s_.nZeta;

            // Precomputed tanv value for combined toroidal angle difference
            // (within-period grid offset delta_k + field-period offset p).
            const double tanv_val = tanv[delta_k + p * s_.nZeta];

            double ga1 =
                sg_.guu[klp - tp_.ztMin] * tanu[delta_l] * tanu[delta_l] +
                sg_.guv[klp - tp_.ztMin] * tanu[delta_l] * tanv_val +
                sg_.gvv[klp - tp_.ztMin] * tanv_val * tanv_val;
            double ga2 =
                sg_.auu[klp - tp_.ztMin] * tanu[delta_l] * tanu[delta_l] +
                sg_.auv[klp - tp_.ztMin] * tanu[delta_l] * tanv_val +
                sg_.avv[klp - tp_.ztMin] * tanv_val * tanv_val;
            ga2 /= ga1;
            ga1 = 1.0 / sqrt(ga1);

            double ftemp =
                1.0 /
                (gsave[kl] -
                 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
            double htemp = sqrt(ftemp);

            greenp[ip_idx_base + kl] +=
                twopidivnfp *
                (htemp * ftemp *
                     (sg_.rcosuv[kl] * sxsave + sg_.rsinuv[kl] * sysave +
                      dsave[kl]) -
                 ga1 * ga2);
            const double g = twopidivnfp * (htemp - ga1);
            gstore[kl] += bexni * g;
          }
        };

        if (p == 0) {
          // Skip diagonal kl==klp (self-singularity)
          full_branch(0, klp);
          full_branch(klp + 1, theta_by_nzeta);
        } else {
          // No diagonal skip (different toroidal angle)
          full_branch(0, theta_by_nzeta);
        }

      } else {
        // Simplified branch: Fortran ELSE (kp>1 AND nv>1)
        // No analytic subtraction needed; other field periods are smooth.
        for (int kl = 0; kl < theta_by_nzeta; ++kl) {
          double ftemp =
              1.0 /
              (gsave[kl] -
               2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
          double htemp = sqrt(ftemp);

          greenp[ip_idx_base + kl] +=
              twopidivnfp * htemp * ftemp *
              (sg_.rcosuv[kl] * sxsave + sg_.rsinuv[kl] * sysave +
               dsave[kl]);
          const double g = twopidivnfp * htemp;
          gstore[kl] += bexni * g;
        }  // kl
      }
    }  // p
  }  // klp
}

}  // namespace vmecpp
