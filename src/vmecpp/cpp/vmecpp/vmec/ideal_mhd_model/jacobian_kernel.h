// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_JACOBIAN_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_JACOBIAN_KERNEL_H_

// __host__ __device__ under a CUDA compiler, plain inline otherwise. This lets
// the per-point kernel below be shared verbatim by the CPU solver loop and a
// CUDA device kernel (see cuda_enzyme_jacobian_jvp.cu), so the primal Jacobian
// formula is maintained in exactly one place.
#if defined(__CUDACC__)
#define VMECPP_HD __host__ __device__
#else
#define VMECPP_HD
#endif

namespace vmecpp {

// Per-point inputs for the half-grid Jacobian: the even/odd-parity geometry
// (R, Z and their poloidal derivatives) at the inside (_i) and outside (_o)
// full-grid surfaces bracketing one half-grid point, plus sqrt(s) at that point
// (sH) and the two scalar step constants (deltaS, dSHalfDsInterp). A flat
// struct so a CUDA kernel can carry one point per thread and Enzyme can
// differentiate HalfGridJacobianPoint as a pure device function over device
// buffers.
struct JacobianPointInput {
  double r1e_i, r1e_o, r1o_i, r1o_o;
  double z1e_i, z1e_o, z1o_i, z1o_o;
  double rue_i, rue_o, ruo_i, ruo_o;
  double zue_i, zue_o, zuo_i, zuo_o;
  double sH, deltaS, dSHalfDsInterp;
};

// Per-point outputs: the half-grid metric quantities and the Jacobian tau.
struct JacobianPointOutput {
  double r12, ru12, zu12, rs, zs, tau;
};

// One half-grid point of the Jacobian kernel. Pure scalar arithmetic, no
// allocation and no Eigen: the form Enzyme differentiates in forward and
// reverse mode, on host and (via VMECPP_HD) on the GPU. tau is nonlinear in the
// geometry (ru12*zs - rs*zu12, ...), so its Jacobian is a building block of the
// exact MHD force Hessian.
VMECPP_HD inline void HalfGridJacobianPoint(const JacobianPointInput& x,
                                            JacobianPointOutput* y) {
  y->r12 = 0.5 * ((x.r1e_i + x.r1e_o) + x.sH * (x.r1o_i + x.r1o_o));
  y->ru12 = 0.5 * ((x.rue_i + x.rue_o) + x.sH * (x.ruo_i + x.ruo_o));
  y->zu12 = 0.5 * ((x.zue_i + x.zue_o) + x.sH * (x.zuo_i + x.zuo_o));
  y->rs = ((x.r1e_o - x.r1e_i) + x.sH * (x.r1o_o - x.r1o_i)) / x.deltaS;
  y->zs = ((x.z1e_o - x.z1e_i) + x.sH * (x.z1o_o - x.z1o_i)) / x.deltaS;

  const double tau1 = y->ru12 * y->zs - y->rs * y->zu12;
  const double tau2 = x.ruo_o * x.z1o_o + x.ruo_i * x.z1o_i -
                      x.zuo_o * x.r1o_o - x.zuo_i * x.r1o_i +
                      (x.rue_o * x.z1o_o + x.rue_i * x.z1o_i -
                       x.zue_o * x.r1o_o - x.zue_i * x.r1o_i) /
                          x.sH;
  y->tau = tau1 + x.dSHalfDsInterp * tau2;
}

// Half-grid Jacobian kernel: maps full-grid geometry (R, Z and their poloidal
// derivatives, even/odd parity) to the half-grid metric quantities r12, ru12,
// zu12, rs, zs and the Jacobian tau.
//
// This is the single source of truth shared by IdealMhdModel::computeJacobian
// (the solver) and the Enzyme autodiff tests. It is written allocation-free
// over flat double buffers, delegating each point to HalfGridJacobianPoint so
// the CPU loop and the CUDA device kernel run identical arithmetic.
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

      const JacobianPointInput x{
          r1e[i_in],  r1e[i_out], r1o[i_in],  r1o[i_out],    z1e[i_in],
          z1e[i_out], z1o[i_in],  z1o[i_out], rue[i_in],     rue[i_out],
          ruo[i_in],  ruo[i_out], zue[i_in],  zue[i_out],    zuo[i_in],
          zuo[i_out], sH,         deltaS,     dSHalfDsInterp};
      JacobianPointOutput y;
      HalfGridJacobianPoint(x, &y);

      r12[ih] = y.r12;
      ru12[ih] = y.ru12;
      zu12[ih] = y.zu12;
      rs[ih] = y.rs;
      zs[ih] = y.zs;
      tau[ih] = y.tau;
    }
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_JACOBIAN_KERNEL_H_
