// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_METRIC_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_METRIC_KERNEL_H_

namespace vmecpp {

// Half-grid metric kernel: gsqrt = tau * r12 and the metric elements guu, guv,
// gvv from the full-grid geometry (and the Jacobian tau, r12 from
// ComputeHalfGridJacobian). guv and the 3D part of gvv are computed only when
// lthreed. Shared, allocation-free over flat buffers, between
// IdealMhdModel::computeMetricElements and the Enzyme autodiff path. Same
// indexing conventions as jacobian_kernel.h. sqrtSF is indexed jF - nsMinF1.
inline void ComputeMetricElements(
    const double* __restrict r1e, const double* __restrict r1o,
    const double* __restrict rue, const double* __restrict ruo,
    const double* __restrict zue, const double* __restrict zuo,
    const double* __restrict rve, const double* __restrict rvo,
    const double* __restrict zve, const double* __restrict zvo,
    const double* __restrict tau, const double* __restrict r12,
    const double* __restrict sqrtSF, const double* __restrict sqrtSH,
    bool lthreed, int nZnT, int nsMinF1, int nsMinH, int nsMaxH,
    double* __restrict gsqrt, double* __restrict guu, double* __restrict guv,
    double* __restrict gvv) {
  for (int jH = nsMinH; jH < nsMaxH; ++jH) {
    const double sF_i = sqrtSF[jH - nsMinF1] * sqrtSF[jH - nsMinF1];
    const double sF_o = sqrtSF[jH + 1 - nsMinF1] * sqrtSF[jH + 1 - nsMinF1];
    const double sH = sqrtSH[jH - nsMinH];
    for (int kl = 0; kl < nZnT; ++kl) {
      const int i_in = (jH - nsMinF1) * nZnT + kl;
      const int i_out = (jH + 1 - nsMinF1) * nZnT + kl;
      const int ih = (jH - nsMinH) * nZnT + kl;

      const double r1e_i = r1e[i_in], r1e_o = r1e[i_out];
      const double r1o_i = r1o[i_in], r1o_o = r1o[i_out];
      const double rue_i = rue[i_in], rue_o = rue[i_out];
      const double ruo_i = ruo[i_in], ruo_o = ruo[i_out];
      const double zue_i = zue[i_in], zue_o = zue[i_out];
      const double zuo_i = zuo[i_in], zuo_o = zuo[i_out];

      gsqrt[ih] = tau[ih] * r12[ih];

      guu[ih] = 0.5 * ((rue_i * rue_i + zue_i * zue_i) +
                       (rue_o * rue_o + zue_o * zue_o) +
                       sF_i * (ruo_i * ruo_i + zuo_i * zuo_i) +
                       sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
                sH * ((rue_i * ruo_i + zue_i * zuo_i) +
                      (rue_o * ruo_o + zue_o * zuo_o));

      gvv[ih] = 0.5 * (r1e_i * r1e_i + r1e_o * r1e_o + sF_i * r1o_i * r1o_i +
                       sF_o * r1o_o * r1o_o) +
                sH * (r1e_i * r1o_i + r1e_o * r1o_o);

      if (lthreed) {
        const double rve_i = rve[i_in], rve_o = rve[i_out];
        const double rvo_i = rvo[i_in], rvo_o = rvo[i_out];
        const double zve_i = zve[i_in], zve_o = zve[i_out];
        const double zvo_i = zvo[i_in], zvo_o = zvo[i_out];

        guv[ih] = 0.5 * ((rue_i * rve_i + zue_i * zve_i) +
                         (rue_o * rve_o + zue_o * zve_o) +
                         sF_i * (ruo_i * rvo_i + zuo_i * zvo_i) +
                         sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
                         sH * ((rue_i * rvo_i + zue_i * zvo_i) +
                               (rue_o * rvo_o + zue_o * zvo_o) +
                               (rve_i * ruo_i + zve_i * zuo_i) +
                               (rve_o * ruo_o + zve_o * zuo_o)));

        gvv[ih] += 0.5 * ((rve_i * rve_i + zve_i * zve_i) +
                          (rve_o * rve_o + zve_o * zve_o) +
                          sF_i * (rvo_i * rvo_i + zvo_i * zvo_i) +
                          sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
                   sH * ((rve_i * rvo_i + zve_i * zvo_i) +
                         (rve_o * rvo_o + zve_o * zvo_o));
      }
    }
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_METRIC_KERNEL_H_
