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
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_
