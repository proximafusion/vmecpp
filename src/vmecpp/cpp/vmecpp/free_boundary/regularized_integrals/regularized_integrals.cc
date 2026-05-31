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
  tanv.resize(s_.nZeta);

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

  for (int k = 0; k < s_.nZeta; ++k) {
    const double argv = M_PI / s_.nZeta * k;
    if (std::abs(argv - 0.5 * M_PI) < epsTan) {
      // mask singularity at pi/2
      tanv[k] = bigNo;
    } else {
      tanv[k] = 2.0 * std::tan(argv);
    }
  }  // k

  // For an axisymmetric (nZeta == 1) plasma the toroidal direction is resolved
  // by nvper_ toroidal images rather than the surface grid; precompute the
  // analytic-approximation toroidal-angle factor at each image angle.
  nvper_ = (s_.nZeta == 1) ? kAxisymmetricToroidalImages : s_.nfp;
  if (s_.nZeta == 1) {
    tanv_per_.resize(nvper_);
    for (int p = 0; p < nvper_; ++p) {
      const double argv = M_PI * p / nvper_;
      if (std::abs(argv - 0.5 * M_PI) < epsTan) {
        // mask singularity at pi/2
        tanv_per_[p] = bigNo;
      } else {
        tanv_per_[p] = 2.0 * std::tan(argv);
      }
    }  // p
  }
}

void RegularizedIntegrals::update(const Eigen::VectorXd& bDotN) {
  if (s_.nZeta == 1) {
    // Axisymmetric plasma: the surface grid has a single toroidal plane, so the
    // toroidal integral is performed over nvper_ toroidal images instead.
    updateAxisymmetric(bDotN);
    return;
  }

  // thread-local tangential grid point range
  const int theta_by_nzeta = s_.nThetaEven * s_.nZeta;
  const double twopidivnfp = 2.0 * M_PI / s_.nfp;

  greenp.setZero();
  gstore.setZero();

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

    // index of toridal field period; 0, 1, ..., (nfp-1)
    int p = 0;

    // xper == xp in first period
    // yper == yp in first period
    // sxsave == snr in first period (TODO(jons): really?)
    // sysave == snv in first period (TODO(jons): really?)
    double xper = xp * sg_.cos_per[p] - yp * sg_.sin_per[p];
    double yper = xp * sg_.sin_per[p] + yp * sg_.cos_per[p];

    double sxsave =
        (sg_.snr[klp - tp_.ztMin] * xper - sg_.snv[klp - tp_.ztMin] * yper) /
        sg_.r1b[klp];
    double sysave =
        (sg_.snr[klp - tp_.ztMin] * yper + sg_.snv[klp - tp_.ztMin] * xper) /
        sg_.r1b[klp];

    for (int kl = 0; kl < theta_by_nzeta; ++kl) {
      gsave[kl] = sg_.rzb2[klp] + sg_.rzb2[kl] - 2 * sg_.z1b[kl] * sg_.z1b[klp];
    }

    for (int kl = 0; kl < theta_by_nzeta; ++kl) {
      dsave[kl] =
          sg_.drv[klp - tp_.ztMin] + sg_.z1b[kl] * sg_.snz[klp - tp_.ztMin];
    }

    auto do_loop = [&](int kl_start, int kl_end) {
      // NOTE: kl is incremented by the loop body
      for (int kl = kl_start; kl < kl_end;) {
        // we can compute these once at the beginning
        const int l = kl / s_.nZeta;
        const int delta_l = (l - lp + s_.nThetaEven) % s_.nThetaEven;

        // In the following, loops over k are split to help auto-vectorization
        // kl is increased together with k and reset between loops as needed.
        const int k_start = kl % s_.nZeta;
        const int k_end = std::min(s_.nZeta, kl_end - l * s_.nZeta);
        const int kl_backup = kl;

        for (int k = k_start; k < k_end; ++k, ++kl) {
          const int delta_k = (k - kp + s_.nZeta) % s_.nZeta;
          ga1_buf[k] =
              sg_.guu[klp - tp_.ztMin] * tanu[delta_l] * tanu[delta_l] +
              sg_.guv[klp - tp_.ztMin] * tanu[delta_l] * tanv[delta_k] +
              sg_.gvv[klp - tp_.ztMin] * tanv[delta_k] * tanv[delta_k];

          ga2_buf[k] =
              sg_.auu[klp - tp_.ztMin] * tanu[delta_l] * tanu[delta_l] +
              sg_.auv[klp - tp_.ztMin] * tanu[delta_l] * tanv[delta_k] +
              sg_.avv[klp - tp_.ztMin] * tanv[delta_k] * tanv[delta_k];
        }

        for (int k = k_start; k < k_end; ++k, ++kl) {
          ga2_buf[k] /= ga1_buf[k];
          ga1_buf[k] = 1.0 / sqrt(ga1_buf[k]);
        }

        kl = kl_backup;
        for (int k = k_start; k < k_end; ++k, ++kl) {
          ftemp_buf[k] =
              1.0 /
              (gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
          htemp_buf[k] = sqrt(ftemp_buf[k]);
        }

        kl = kl_backup;
        for (int k = k_start; k < k_end; ++k, ++kl) {
          const int ip = ip_idx_base + kl;

          // 2 pi from Laplace equation
          // 1/nfp to make the toroidal integral below over the whole machine
          greenp[ip] +=
              twopidivnfp * (htemp_buf[k] * ftemp_buf[k] *
                                 (sg_.rcosuv[kl] * sxsave +
                                  sg_.rsinuv[kl] * sysave + dsave[kl]) -
                             ga1_buf[k] * ga2_buf[k]);
          const double g = twopidivnfp * (htemp_buf[k] - ga1_buf[k]);
          gstore[kl] += bexni * g;
        }
      }  // kl
    };

    // first field period
    // 0 < kl < klp
    do_loop(/*kl_start*/ 0, /*kl_end*/ klp);
    // klp < kl < theta_by_nzeta
    do_loop(/*kl_start*/ klp + 1, /*kl_end*/ theta_by_nzeta);

    // all following field periods
    for (int p = 1; p < s_.nfp; ++p) {
      double xper = xp * sg_.cos_per[p] - yp * sg_.sin_per[p];
      double yper = xp * sg_.sin_per[p] + yp * sg_.cos_per[p];
      double sxsave =
          (sg_.snr[klp - tp_.ztMin] * xper - sg_.snv[klp - tp_.ztMin] * yper) /
          sg_.r1b[klp];
      double sysave =
          (sg_.snr[klp - tp_.ztMin] * yper + sg_.snv[klp - tp_.ztMin] * xper) /
          sg_.r1b[klp];

      for (int kl = 0; kl < theta_by_nzeta; ++kl) {
        double ftemp =
            1.0 /
            (gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
        double htemp = sqrt(ftemp);

        // 2 pi from Laplace equation (TODO(jons): really?)
        // 1/nfp to make the toroidal integral below over the whole machine
        greenp[ip_idx_base + kl] +=
            twopidivnfp * htemp * ftemp *
            (sg_.rcosuv[kl] * sxsave + sg_.rsinuv[kl] * sysave + dsave[kl]);
        const double g = twopidivnfp * htemp;
        gstore[kl] += bexni * g;
      }  // kl
    }  // p
  }  // klp
}

void RegularizedIntegrals::updateAxisymmetric(
    const Eigen::VectorXd& bDotN) {
  // Axisymmetric (nZeta == 1) Green's-function regularization. The single
  // toroidal plane of the surface grid does not resolve the toroidal angle, so
  // the toroidal integral of the regularized Green's function is performed by
  // summing over nvper_ equally-spaced toroidal images of the evaluation point
  // (educational_VMEC greenf with nvper = 64 for the tokamak). The analytic
  // approximation is subtracted at every image; its closed-form toroidal
  // integral is added back in SingularIntegrals.
  const int numLocal = tp_.ztMax - tp_.ztMin;
  const int nThetaEven = s_.nThetaEven;  // == theta_by_nzeta since nZeta == 1

  absl::c_fill_n(greenp, numLocal * nThetaEven, 0);
  absl::c_fill_n(gstore, nThetaEven, 0);

  // 2 pi from the Laplace equation; 1/nvper_ turns the toroidal image sum into
  // a toroidal integral over the whole machine.
  const double toroidal_measure = 2.0 * M_PI / nvper_;

  for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
    const int ip_idx_base = (klp - tp_.ztMin) * nThetaEven;
    const int klpRel = klp - tp_.ztMin;
    const int lp = klp;  // k == 0 since nZeta == 1

    const double bexni = bDotN[klpRel] * s_.wInt[lp];

    // source slice at zeta = 0: rcosuv == r1b, rsinuv == 0
    const double xp = sg_.rcosuv[klp];
    const double yp = sg_.rsinuv[klp];

    for (int kl = 0; kl < nThetaEven; ++kl) {
      gsave[kl] = sg_.rzb2[klp] + sg_.rzb2[kl] - 2 * sg_.z1b[kl] * sg_.z1b[klp];
      dsave[kl] = sg_.drv[klpRel] + sg_.z1b[kl] * sg_.snz[klpRel];
    }

    // integrate over the toroidal direction by summing the rotated images of
    // the evaluation point around the whole torus
    for (int p = 0; p < nvper_; ++p) {
      const double cosper = std::cos(toroidal_measure * p);
      const double sinper = std::sin(toroidal_measure * p);

      const double xper = xp * cosper - yp * sinper;
      const double yper = xp * sinper + yp * cosper;

      const double sxsave =
          (sg_.snr[klpRel] * xper - sg_.snv[klpRel] * yper) / sg_.r1b[klp];
      const double sysave =
          (sg_.snr[klpRel] * yper + sg_.snv[klpRel] * xper) / sg_.r1b[klp];

      const double tanv_p = tanv_per_[p];

      for (int kl = 0; kl < nThetaEven; ++kl) {
        // The exact singularity (image coincides with the source point) is
        // handled analytically in SingularIntegrals; skip it here.
        if (p == 0 && kl == klp) {
          continue;
        }

        const int delta_l = (kl - lp + nThetaEven) % nThetaEven;

        double ga1 = sg_.guu[klpRel] * tanu[delta_l] * tanu[delta_l] +
                     sg_.guv[klpRel] * tanu[delta_l] * tanv_p +
                     sg_.gvv[klpRel] * tanv_p * tanv_p;
        double ga2 = sg_.auu[klpRel] * tanu[delta_l] * tanu[delta_l] +
                     sg_.auv[klpRel] * tanu[delta_l] * tanv_p +
                     sg_.avv[klpRel] * tanv_p * tanv_p;
        ga2 /= ga1;
        ga1 = 1.0 / std::sqrt(ga1);

        const double ftemp =
            1.0 /
            (gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
        const double htemp = std::sqrt(ftemp);

        greenp[ip_idx_base + kl] +=
            toroidal_measure * (htemp * ftemp *
                                    (sg_.rcosuv[kl] * sxsave +
                                     sg_.rsinuv[kl] * sysave + dsave[kl]) -
                                ga1 * ga2);
        gstore[kl] += bexni * toroidal_measure * (htemp - ga1);
      }  // kl
    }  // p
  }  // klp
}

}  // namespace vmecpp
