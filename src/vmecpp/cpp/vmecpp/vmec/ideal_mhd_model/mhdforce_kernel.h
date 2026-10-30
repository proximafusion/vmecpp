// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_MHDFORCE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_MHDFORCE_KERNEL_H_

#include <Eigen/Dense>

namespace vmecpp {

// Real-space MHD force-density assembly (armn/azmn/brmn/bzmn and, in 3D,
// crmn/czmn, even+odd parity) from the half-grid pressure, Jacobian, metric and
// contravariant field. This is the sixth and last force-chain kernel; together
// with the Jacobian/metric/B^contra/B_cov/pressure kernels it forms the local
// map geometry -> force density whose Jacobian (composed with the linear
// spectral transforms) is the exact MHD force Hessian.
//
// Shared between IdealMhdModel::computeMHDForces and the Enzyme autodiff path.
// The Eigen arithmetic is preserved verbatim from the solver; only storage is
// flat buffers (Eigen::Map), so it is allocation-free and the result is
// bit-for-bit identical. The half-grid "inside/outside" handover scratch
// (X_i/X_o) and the surface averages (X_avg/X_wavg) are caller-owned buffers of
// length nZnT, reused across surfaces.
inline void ComputeMHDForceDensity(
    // full-grid geometry, indexed (jF - nsMinF1) * nZnT + kl
    const double* r1_e, const double* r1_o, const double* ru_e,
    const double* ru_o, const double* zu_e, const double* zu_o,
    const double* z1_o, const double* rv_e, const double* rv_o,
    const double* zv_e, const double* zv_o,
    // half-grid quantities, indexed (jH - nsMinH) * nZnT + kl
    const double* r12, const double* ru12, const double* zu12, const double* rs,
    const double* zs, const double* tau, const double* totalPressure,
    const double* gsqrt, const double* bsupu, const double* bsupv,
    // profiles
    const double* sqrtSF, const double* sqrtSH,
    // inside-handover scratch (length nZnT)
    double* P_i, double* rup_i, double* zup_i, double* rsp_i, double* zsp_i,
    double* taup_i, double* gbubu_i, double* gbubv_i, double* gbvbv_i,
    // outside + average scratch (length nZnT)
    double* P_o, double* rup_o, double* zup_o, double* rsp_o, double* zsp_o,
    double* taup_o, double* gbubu_o, double* gbubv_o, double* gbvbv_o,
    double* P_avg, double* P_wavg, double* gbubu_avg, double* gbubu_wavg,
    double* gbvbv_avg, double* gbvbv_wavg, double* gbubv_avg,
    double* gbubv_wavg,
    // sizes / flags
    double deltaS, int nZnT, int nsMinF, int nsMinF1, int nsMinH, int nsMaxH,
    int jMaxRZ, bool lthreed,
    // outputs, indexed (jF - nsMinF) * nZnT + kl
    double* armn_e, double* armn_o, double* azmn_e, double* azmn_o,
    double* brmn_e, double* brmn_o, double* bzmn_e, double* bzmn_o,
    double* crmn_e, double* crmn_o, double* czmn_e, double* czmn_o) {
  using V = Eigen::VectorXd;
  using Map = Eigen::Map<V>;
  using CMap = Eigen::Map<const V>;

  Map vP_i(P_i, nZnT), vrup_i(rup_i, nZnT), vzup_i(zup_i, nZnT);
  Map vrsp_i(rsp_i, nZnT), vzsp_i(zsp_i, nZnT), vtaup_i(taup_i, nZnT);
  Map vgbubu_i(gbubu_i, nZnT), vgbubv_i(gbubv_i, nZnT), vgbvbv_i(gbvbv_i, nZnT);
  Map vP_o(P_o, nZnT), vrup_o(rup_o, nZnT), vzup_o(zup_o, nZnT);
  Map vrsp_o(rsp_o, nZnT), vzsp_o(zsp_o, nZnT), vtaup_o(taup_o, nZnT);
  Map vgbubu_o(gbubu_o, nZnT), vgbubv_o(gbubv_o, nZnT), vgbvbv_o(gbvbv_o, nZnT);
  Map vP_avg(P_avg, nZnT), vP_wavg(P_wavg, nZnT);
  Map vgbubu_avg(gbubu_avg, nZnT), vgbubu_wavg(gbubu_wavg, nZnT);
  Map vgbvbv_avg(gbvbv_avg, nZnT), vgbvbv_wavg(gbvbv_wavg, nZnT);
  Map vgbubv_avg(gbubv_avg, nZnT), vgbubv_wavg(gbubv_wavg, nZnT);

  double sqrtSHi = 1.0;
  if (nsMinF > 0) {
    const int j0 = nsMinH;
    for (int kl = 0; kl < nZnT; ++kl) {
      const int iHalf = (j0 - nsMinH) * nZnT + kl;
      P_i[kl] = r12[iHalf] * totalPressure[iHalf];
      rup_i[kl] = ru12[iHalf] * P_i[kl];
      zup_i[kl] = zu12[iHalf] * P_i[kl];
      rsp_i[kl] = rs[iHalf] * P_i[kl];
      zsp_i[kl] = zs[iHalf] * P_i[kl];
      taup_i[kl] = tau[iHalf] * totalPressure[iHalf];
      gbubu_i[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupu[iHalf];
      gbubv_i[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupv[iHalf];
      gbvbv_i[kl] = gsqrt[iHalf] * bsupv[iHalf] * bsupv[iHalf];
    }
    sqrtSHi = sqrtSH[j0 - nsMinH];
  } else {
    vP_i.setZero();
    vrup_i.setZero();
    vzup_i.setZero();
    vrsp_i.setZero();
    vzsp_i.setZero();
    vtaup_i.setZero();
    vgbubu_i.setZero();
    vgbubv_i.setZero();
    vgbvbv_i.setZero();
  }

  vP_o.setZero();
  vrup_o.setZero();
  vzup_o.setZero();
  vrsp_o.setZero();
  vzsp_o.setZero();
  vtaup_o.setZero();
  vgbubu_o.setZero();
  vgbubv_o.setZero();
  vgbvbv_o.setZero();

  for (int jF = nsMinF; jF < jMaxRZ; ++jF) {
    const double sFull = sqrtSF[jF - nsMinF1] * sqrtSF[jF - nsMinF1];
    double sqrtSHo = 1.0;
    if (jF < nsMaxH) {
      sqrtSHo = sqrtSH[jF - nsMinH];
    }

    if (jF < nsMaxH) {
      const int iHalf_base = (jF - nsMinH) * nZnT;
      for (int kl = 0; kl < nZnT; ++kl) {
        const int iHalf = iHalf_base + kl;
        P_o[kl] = r12[iHalf] * totalPressure[iHalf];
        rup_o[kl] = ru12[iHalf] * P_o[kl];
        zup_o[kl] = zu12[iHalf] * P_o[kl];
        rsp_o[kl] = rs[iHalf] * P_o[kl];
        zsp_o[kl] = zs[iHalf] * P_o[kl];
        taup_o[kl] = tau[iHalf] * totalPressure[iHalf];
      }
      for (int kl = 0; kl < nZnT; ++kl) {
        const int iHalf = iHalf_base + kl;
        gbubu_o[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupu[iHalf];
        gbubv_o[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupv[iHalf];
        gbvbv_o[kl] = gsqrt[iHalf] * bsupv[iHalf] * bsupv[iHalf];
      }
    } else {
      vP_o.setZero();
      vrup_o.setZero();
      vzup_o.setZero();
      vrsp_o.setZero();
      vzsp_o.setZero();
      vtaup_o.setZero();
      vgbubu_o.setZero();
      vgbubv_o.setZero();
      vgbvbv_o.setZero();
    }

    const int g_off = (jF - nsMinF1) * nZnT;
    const int f_off = (jF - nsMinF) * nZnT;
    const CMap r1e(r1_e + g_off, nZnT), r1o(r1_o + g_off, nZnT);
    const CMap rue(ru_e + g_off, nZnT), ruo(ru_o + g_off, nZnT);
    const CMap zue(zu_e + g_off, nZnT), zuo(zu_o + g_off, nZnT);
    const CMap z1o(z1_o + g_off, nZnT);

    const double invDS = 1.0 / deltaS;
    const double invSHo = 1.0 / sqrtSHo;
    const double invSHi = 1.0 / sqrtSHi;
    vP_avg = 0.5 * (vP_o + vP_i);
    vP_wavg = 0.5 * (vP_o * invSHo + vP_i * invSHi);
    vgbubu_avg = 0.5 * (vgbubu_o + vgbubu_i);
    vgbubu_wavg = 0.5 * (vgbubu_o * sqrtSHo + vgbubu_i * sqrtSHi);
    vgbvbv_avg = 0.5 * (vgbvbv_o + vgbvbv_i);
    vgbvbv_wavg = 0.5 * (vgbvbv_o * sqrtSHo + vgbvbv_i * sqrtSHi);

    Map(armn_e + f_off, nZnT) =
        (vzup_o - vzup_i) * invDS + 0.5 * (vtaup_o + vtaup_i) -
        vgbvbv_avg.cwiseProduct(r1e) - vgbvbv_wavg.cwiseProduct(r1o);
    Map(armn_o + f_off, nZnT) =
        (vzup_o * sqrtSHo - vzup_i * sqrtSHi) * invDS -
        0.5 * vP_wavg.cwiseProduct(zue) - 0.5 * vP_avg.cwiseProduct(zuo) +
        0.5 * (vtaup_o * sqrtSHo + vtaup_i * sqrtSHi) -
        vgbvbv_wavg.cwiseProduct(r1e) - vgbvbv_avg.cwiseProduct(r1o) * sFull;

    Map(azmn_e + f_off, nZnT) = -(vrup_o - vrup_i) * invDS;
    Map(azmn_o + f_off, nZnT) = -(vrup_o * sqrtSHo - vrup_i * sqrtSHi) * invDS +
                                0.5 * vP_wavg.cwiseProduct(rue) +
                                0.5 * vP_avg.cwiseProduct(ruo);

    Map(brmn_e + f_off, nZnT) =
        0.5 * (vzsp_o + vzsp_i) + 0.5 * vP_wavg.cwiseProduct(z1o) -
        vgbubu_avg.cwiseProduct(rue) - vgbubu_wavg.cwiseProduct(ruo);
    Map(brmn_o + f_off, nZnT) = 0.5 * (vzsp_o * sqrtSHo + vzsp_i * sqrtSHi) +
                                0.5 * vP_avg.cwiseProduct(z1o) -
                                vgbubu_wavg.cwiseProduct(rue) -
                                vgbubu_avg.cwiseProduct(ruo) * sFull;

    Map(bzmn_e + f_off, nZnT) =
        -0.5 * (vrsp_o + vrsp_i) - 0.5 * vP_wavg.cwiseProduct(r1o) -
        vgbubu_avg.cwiseProduct(zue) - vgbubu_wavg.cwiseProduct(zuo);
    Map(bzmn_o + f_off, nZnT) = -0.5 * (vrsp_o * sqrtSHo + vrsp_i * sqrtSHi) -
                                0.5 * vP_avg.cwiseProduct(r1o) -
                                vgbubu_wavg.cwiseProduct(zue) -
                                vgbubu_avg.cwiseProduct(zuo) * sFull;

    if (lthreed) {
      vgbubv_avg = 0.5 * (vgbubv_o + vgbubv_i);
      vgbubv_wavg = 0.5 * (vgbubv_o * sqrtSHo + vgbubv_i * sqrtSHi);
      const CMap rve(rv_e + g_off, nZnT), rvo(rv_o + g_off, nZnT);
      const CMap zve(zv_e + g_off, nZnT), zvo(zv_o + g_off, nZnT);

      Map(brmn_e + f_off, nZnT) -=
          vgbubv_avg.cwiseProduct(rve) + vgbubv_wavg.cwiseProduct(rvo);
      Map(brmn_o + f_off, nZnT) -=
          vgbubv_wavg.cwiseProduct(rve) + vgbubv_avg.cwiseProduct(rvo) * sFull;
      Map(bzmn_e + f_off, nZnT) -=
          vgbubv_avg.cwiseProduct(zve) + vgbubv_wavg.cwiseProduct(zvo);
      Map(bzmn_o + f_off, nZnT) -=
          vgbubv_wavg.cwiseProduct(zve) + vgbubv_avg.cwiseProduct(zvo) * sFull;

      Map(crmn_e + f_off, nZnT) =
          vgbubv_avg.cwiseProduct(rue) + vgbubv_wavg.cwiseProduct(ruo) +
          vgbvbv_avg.cwiseProduct(rve) + vgbvbv_wavg.cwiseProduct(rvo);
      Map(crmn_o + f_off, nZnT) =
          vgbubv_wavg.cwiseProduct(rue) + vgbubv_avg.cwiseProduct(ruo) * sFull +
          vgbvbv_wavg.cwiseProduct(rve) + vgbvbv_avg.cwiseProduct(rvo) * sFull;

      Map(czmn_e + f_off, nZnT) =
          vgbubv_avg.cwiseProduct(zue) + vgbubv_wavg.cwiseProduct(zuo) +
          vgbvbv_avg.cwiseProduct(zve) + vgbvbv_wavg.cwiseProduct(zvo);
      Map(czmn_o + f_off, nZnT) =
          vgbubv_wavg.cwiseProduct(zue) + vgbubv_avg.cwiseProduct(zuo) * sFull +
          vgbvbv_wavg.cwiseProduct(zve) + vgbvbv_avg.cwiseProduct(zvo) * sFull;
    }

    vP_i = vP_o;
    vrup_i = vrup_o;
    vzup_i = vzup_o;
    vrsp_i = vrsp_o;
    vzsp_i = vzsp_o;
    vtaup_i = vtaup_o;
    vgbubu_i = vgbubu_o;
    vgbubv_i = vgbubv_o;
    vgbvbv_i = vgbvbv_o;
    sqrtSHi = sqrtSHo;
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_MHDFORCE_KERNEL_H_
