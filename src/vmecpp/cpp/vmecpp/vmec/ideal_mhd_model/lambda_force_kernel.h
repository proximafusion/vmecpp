// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_LAMBDA_FORCE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_LAMBDA_FORCE_KERNEL_H_

namespace vmecpp {

// Hybrid lambda force on the full grid: blmn (and clmn in 3D). The covariant
// field bsubv is interpolated from the half grid two ways (a plain average and
// an alternative from gvv/gsqrt times lambda plus, in 3D, guv*bsupu) and
// blended with the radialBlending profile. The radial sweep carries the inside
// half-grid point in scratch (bsubu_i, bsubv_i, gvv_gsqrt_i, guv_bsupu_i, each
// nZnT) and shifts it outward each surface. Allocation-free over flat buffers,
// shared between IdealMhdModel::hybridLambdaForce and the Enzyme autodiff path.
//
// Half-grid inputs (index (jH-nsMinH)*nZnT): bsubu, bsubv, gvv, gsqrt, guv,
// bsupu. Full-grid lambda (index (jF-nsMinF1)*nZnT): lu_e, lu_o. Profiles:
// sqrtSH (half), sqrtSF and radialBlending (full, index jF-nsMinF1). Outputs
// (index (jF-nsMinF)*nZnT): blmn_e/o, clmn_e/o.
inline void ComputeHybridLambdaForce(
    const double* bsubu, const double* bsubv, const double* gvv,
    const double* gsqrt, const double* guv, const double* bsupu,
    const double* lu_e, const double* lu_o, const double* sqrtSH,
    const double* sqrtSF, const double* radialBlending, double lamscale,
    bool lthreed, int nZnT, int nsMinF, int nsMinF1, int nsMinH, int nsMaxH,
    int nsMaxFIncludingLcfs, double* bsubu_i, double* bsubv_i,
    double* gvv_gsqrt_i, double* guv_bsupu_i, double* blmn_e, double* blmn_o,
    double* clmn_e, double* clmn_o) {
  // first inside point
  int j0 = nsMinF;
  double sqrtSHi = 0.0;
  if (j0 > 0) {
    sqrtSHi = sqrtSH[j0 - 1 - nsMinH];
  }
  for (int kl = 0; kl < nZnT; ++kl) {
    if (j0 == 0) {
      // no contribution from the half-grid point inside the axis
      bsubu_i[kl] = 0.0;
      bsubv_i[kl] = 0.0;
      gvv_gsqrt_i[kl] = 0.0;
      guv_bsupu_i[kl] = 0.0;
    } else {
      int iHalf = (j0 - 1 - nsMinH) * nZnT + kl;
      bsubu_i[kl] = bsubu[iHalf];
      bsubv_i[kl] = bsubv[iHalf];
      gvv_gsqrt_i[kl] = gvv[iHalf] / gsqrt[iHalf];
      if (lthreed) {
        guv_bsupu_i[kl] = guv[iHalf] * bsupu[iHalf];
      }
    }
  }  // kl

  for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
    double sqrtSHo = 0.0;
    if (jF < nsMaxH) {
      sqrtSHo = sqrtSH[jF - nsMinH];
    }

    for (int kl = 0; kl < nZnT; ++kl) {
      // next outside point (defaults to 0 outside the LCFS)
      double bsubv_o = 0.0;
      double gvv_gsqrt_o = 0.0;
      double guv_bsupu_o = 0.0;
      if (jF < nsMaxH) {
        int iHalf = (jF - nsMinH) * nZnT + kl;
        bsubv_o = bsubv[iHalf];
        gvv_gsqrt_o = gvv[iHalf] / gsqrt[iHalf];
        if (lthreed) {
          guv_bsupu_o = guv[iHalf] * bsupu[iHalf];
        }
      }

      double gvv_gsqrt_lu_e = 0.5 * (gvv_gsqrt_i[kl] + gvv_gsqrt_o) *
                              lu_e[(jF - nsMinF1) * nZnT + kl];
      double gvv_gsqrt_lu_o =
          0.5 * (gvv_gsqrt_i[kl] * sqrtSHi + gvv_gsqrt_o * sqrtSHo) *
          lu_o[(jF - nsMinF1) * nZnT + kl];

      double gvv_gsqrt_lu = gvv_gsqrt_lu_e + gvv_gsqrt_lu_o;
      double bsubv_alternative = gvv_gsqrt_lu;
      if (lthreed) {
        double guv_bsupu = 0.5 * (guv_bsupu_i[kl] + guv_bsupu_o);
        bsubv_alternative += guv_bsupu;
      }

      const double bsubv_average = 0.5 * (bsubv_o + bsubv_i[kl]);

      // blend the two interpolations of bsubv
      double _blmn = bsubv_average * (1.0 - radialBlending[jF - nsMinF1]) +
                     bsubv_alternative * radialBlending[jF - nsMinF1];

      if (jF > 0) {
        // MINUS SIGN => HESSIAN DIAGONALS ARE POSITIVE
        _blmn *= -lamscale;
      }

      blmn_e[(jF - nsMinF) * nZnT + kl] = _blmn;
      blmn_o[(jF - nsMinF) * nZnT + kl] = _blmn * sqrtSF[jF - nsMinF1];

      if (lthreed) {
        double bsubu_o = 0.0;
        if (jF < nsMaxH) {
          bsubu_o = bsubu[(jF - nsMinH) * nZnT + kl];
        }

        double _clmn = 0.5 * (bsubu_o + bsubu_i[kl]);

        if (jF > 0) {
          _clmn *= -lamscale;
        }

        clmn_e[(jF - nsMinF) * nZnT + kl] = _clmn;
        clmn_o[(jF - nsMinF) * nZnT + kl] = _clmn * sqrtSF[jF - nsMinF1];

        bsubu_i[kl] = bsubu_o;
      }  // lthreed

      bsubv_i[kl] = bsubv_o;
      gvv_gsqrt_i[kl] = gvv_gsqrt_o;
      if (lthreed) {
        guv_bsupu_i[kl] = guv_bsupu_o;
      }
    }  // kl
    sqrtSHi = sqrtSHo;
  }  // jF
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_LAMBDA_FORCE_KERNEL_H_
