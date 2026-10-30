// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_JACOBIAN_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_JACOBIAN_KERNEL_H_

namespace vmecpp {

// Half-grid Jacobian kernel: maps full-grid geometry (R, Z and their poloidal
// derivatives, even/odd parity) to the half-grid metric quantities r12, ru12,
// zu12, rs, zs and the Jacobian tau.
//
// This is the single source of truth shared by IdealMhdModel::computeJacobian
// (the solver) and the Enzyme autodiff test. It is written allocation-free over
// flat double buffers (scalar locals, no Eigen temporaries), which is the form
// Enzyme differentiates in both forward and reverse mode. tau is nonlinear in
// the geometry (ru12*zs - rs*zu12, ...), so its Jacobian is a building block of
// the exact MHD force Hessian.
//
// Geometry inputs are indexed (jF - nsMinF1) * nZnT + kl over the full-grid
// radial partition; outputs are indexed (jH - nsMinH) * nZnT + kl over the
// half-grid; sqrtSH is indexed jH - nsMinH. The half-grid point jH sits between
// full-grid surfaces jH (inside) and jH + 1 (outside).
inline void ComputeHalfGridJacobian(
    const double* __restrict r1e, const double* __restrict r1o,
    const double* __restrict z1e, const double* __restrict z1o,
    const double* __restrict rue, const double* __restrict ruo,
    const double* __restrict zue, const double* __restrict zuo,
    const double* __restrict sqrtSH, double deltaS, double dSHalfDsInterp,
    int nZnT, int nsMinF1, int nsMinH, int nsMaxH, double* __restrict r12,
    double* __restrict ru12, double* __restrict zu12, double* __restrict rs,
    double* __restrict zs, double* __restrict tau) {
  for (int jH = nsMinH; jH < nsMaxH; ++jH) {
    const double sH = sqrtSH[jH - nsMinH];
    for (int kl = 0; kl < nZnT; ++kl) {
      const int i_in = (jH - nsMinF1) * nZnT + kl;
      const int i_out = (jH + 1 - nsMinF1) * nZnT + kl;
      const int ih = (jH - nsMinH) * nZnT + kl;

      const double r1e_i = r1e[i_in], r1e_o = r1e[i_out];
      const double r1o_i = r1o[i_in], r1o_o = r1o[i_out];
      const double z1e_i = z1e[i_in], z1e_o = z1e[i_out];
      const double z1o_i = z1o[i_in], z1o_o = z1o[i_out];
      const double rue_i = rue[i_in], rue_o = rue[i_out];
      const double ruo_i = ruo[i_in], ruo_o = ruo[i_out];
      const double zue_i = zue[i_in], zue_o = zue[i_out];
      const double zuo_i = zuo[i_in], zuo_o = zuo[i_out];

      r12[ih] = 0.5 * ((r1e_i + r1e_o) + sH * (r1o_i + r1o_o));
      ru12[ih] = 0.5 * ((rue_i + rue_o) + sH * (ruo_i + ruo_o));
      zu12[ih] = 0.5 * ((zue_i + zue_o) + sH * (zuo_i + zuo_o));
      rs[ih] = ((r1e_o - r1e_i) + sH * (r1o_o - r1o_i)) / deltaS;
      zs[ih] = ((z1e_o - z1e_i) + sH * (z1o_o - z1o_i)) / deltaS;

      const double tau1 = ru12[ih] * zs[ih] - rs[ih] * zu12[ih];
      const double tau2 =
          ruo_o * z1o_o + ruo_i * z1o_i - zuo_o * r1o_o - zuo_i * r1o_i +
          (rue_o * z1o_o + rue_i * z1o_i - zue_o * r1o_o - zue_i * r1o_i) / sH;
      tau[ih] = tau1 + dSHalfDsInterp * tau2;
    }
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_JACOBIAN_KERNEL_H_
