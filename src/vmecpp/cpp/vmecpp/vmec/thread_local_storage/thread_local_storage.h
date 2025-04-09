// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_
#define VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_

#include <vector>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"

namespace vmecpp {

class ThreadLocalStorage {
  const Sizes& s_;

 public:
  explicit ThreadLocalStorage(const Sizes* s);

  // inv-DFT of geometry
  std::vector<double> r1e_i;
  std::vector<double> r1o_i;
  std::vector<double> rue_i;
  std::vector<double> ruo_i;
  std::vector<double> rve_i;
  std::vector<double> rvo_i;
  std::vector<double> z1e_i;
  std::vector<double> z1o_i;
  std::vector<double> zue_i;
  std::vector<double> zuo_i;
  std::vector<double> zve_i;
  std::vector<double> zvo_i;
  std::vector<double> lue_i;
  std::vector<double> luo_i;
  std::vector<double> lve_i;
  std::vector<double> lvo_i;

  // hybrid lambda forces
  std::vector<double> bsubu_i;
  std::vector<double> bsubv_i;
  std::vector<double> gvv_gsqrt_i;  // gvv / gsqrt
  std::vector<double> guv_bsupu_i;  // guv * bsupu

  // R, Z MHD forces
  std::vector<double> P_i;      // r12 * totalPressure = P
  std::vector<double> rup_i;    // ru12 * P
  std::vector<double> zup_i;    // zu12 * P
  std::vector<double> rsp_i;    //   rs * P
  std::vector<double> zsp_i;    //   zs * P
  std::vector<double> taup_i;   //  tau * P
  std::vector<double> gbubu_i;  // gsqrt * bsupu * bsupu
  std::vector<double> gbubv_i;  // gsqrt * bsupu * bsupv
  std::vector<double> gbvbv_i;  // gsqrt * bsupv * bsupv
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_THREAD_LOCAL_STORAGE_THREAD_LOCAL_STORAGE_H_
