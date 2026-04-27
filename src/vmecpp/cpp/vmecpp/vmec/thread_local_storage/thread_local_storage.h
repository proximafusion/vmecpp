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
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> r1e_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> r1o_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rue_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ruo_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rve_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rvo_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> z1e_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> z1o_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zue_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zuo_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zve_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zvo_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> lue_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> luo_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> lve_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> lvo_i;

  // hybrid lambda forces
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bsubu_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bsubv_i;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gvv_gsqrt_i;  // gvv / gsqrt
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> guv_bsupu_i;  // guv * bsupu

  // R, Z MHD forces
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> P_i;      // r12 * totalPressure = P
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rup_i;    // ru12 * P
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zup_i;    // zu12 * P
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rsp_i;    //   rs * P
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zsp_i;    //   zs * P
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> taup_i;   //  tau * P
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gbubu_i;  // gsqrt * bsupu * bsupu
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gbubv_i;  // gsqrt * bsupu * bsupv
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gbvbv_i;  // gsqrt * bsupv * bsupv
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_
