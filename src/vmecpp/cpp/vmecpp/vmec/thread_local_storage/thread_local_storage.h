// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_
#define VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_

#include <Eigen/Dense>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"

namespace vmecpp {

class ThreadLocalStorage {
  const Sizes& s_;

 public:
  explicit ThreadLocalStorage(const Sizes* s);

  // inv-DFT of geometry
  Eigen::VectorXd r1e_i;
  Eigen::VectorXd r1o_i;
  Eigen::VectorXd rue_i;
  Eigen::VectorXd ruo_i;
  Eigen::VectorXd rve_i;
  Eigen::VectorXd rvo_i;
  Eigen::VectorXd z1e_i;
  Eigen::VectorXd z1o_i;
  Eigen::VectorXd zue_i;
  Eigen::VectorXd zuo_i;
  Eigen::VectorXd zve_i;
  Eigen::VectorXd zvo_i;
  Eigen::VectorXd lue_i;
  Eigen::VectorXd luo_i;
  Eigen::VectorXd lve_i;
  Eigen::VectorXd lvo_i;

  // hybrid lambda forces
  Eigen::VectorXd bsubu_i;
  Eigen::VectorXd bsubv_i;
  Eigen::VectorXd gvv_gsqrt_i;  // gvv / gsqrt
  Eigen::VectorXd guv_bsupu_i;  // guv * bsupu

  // R, Z MHD forces
  Eigen::VectorXd P_i;      // r12 * totalPressure = P
  Eigen::VectorXd rup_i;    // ru12 * P
  Eigen::VectorXd zup_i;    // zu12 * P
  Eigen::VectorXd rsp_i;    //   rs * P
  Eigen::VectorXd zsp_i;    //   zs * P
  Eigen::VectorXd taup_i;   //  tau * P
  Eigen::VectorXd gbubu_i;  // gsqrt * bsupu * bsupu
  Eigen::VectorXd gbubv_i;  // gsqrt * bsupu * bsupv
  Eigen::VectorXd gbvbv_i;  // gsqrt * bsupv * bsupv

  // Outboard ("_o") counterparts of the half-grid quantities above, plus the
  // surface averages used in computeMHDForces.
  Eigen::VectorXd P_o;
  Eigen::VectorXd rup_o;
  Eigen::VectorXd zup_o;
  Eigen::VectorXd rsp_o;
  Eigen::VectorXd zsp_o;
  Eigen::VectorXd taup_o;
  Eigen::VectorXd gbubu_o;
  Eigen::VectorXd gbubv_o;
  Eigen::VectorXd gbvbv_o;
  Eigen::VectorXd P_avg;       // 0.5 (P_o + P_i)
  Eigen::VectorXd P_wavg;      // 0.5 (P_o/sqrtSHo + P_i/sqrtSHi)
  Eigen::VectorXd gbubu_avg;   // 0.5 (gbubu_o + gbubu_i)
  Eigen::VectorXd gbubu_wavg;  // 0.5 (gbubu_o sqrtSHo + gbubu_i sqrtSHi)
  Eigen::VectorXd gbvbv_avg;   // 0.5 (gbvbv_o + gbvbv_i)
  Eigen::VectorXd gbvbv_wavg;  // 0.5 (gbvbv_o sqrtSHo + gbvbv_i sqrtSHi)
  Eigen::VectorXd gbubv_avg;   // 0.5 (gbubv_o + gbubv_i)   [3D only]
  Eigen::VectorXd gbubv_wavg;  // 0.5 (gbubv_o sqrtSHo + gbubv_i sqrtSHi) [3D]
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_
