// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/thread_local_storage/thread_local_storage.h"

namespace vmecpp {

ThreadLocalStorage::ThreadLocalStorage(const Sizes* s) : s_(*s) {
  const int nZnT = s_.nZnT;
  const int nZnT3d = s_.lthreed ? nZnT : 0;

  r1e_i.setZero(nZnT);
  r1o_i.setZero(nZnT);
  rue_i.setZero(nZnT);
  ruo_i.setZero(nZnT);
  rve_i.setZero(nZnT3d);
  rvo_i.setZero(nZnT3d);
  z1e_i.setZero(nZnT);
  z1o_i.setZero(nZnT);
  zue_i.setZero(nZnT);
  zuo_i.setZero(nZnT);
  zve_i.setZero(nZnT3d);
  zvo_i.setZero(nZnT3d);
  lue_i.setZero(nZnT);
  luo_i.setZero(nZnT);
  lve_i.setZero(nZnT3d);
  lvo_i.setZero(nZnT3d);
  bsubu_i.setZero(nZnT);
  bsubv_i.setZero(nZnT);
  gvv_gsqrt_i.setZero(nZnT);
  guv_bsupu_i.setZero(nZnT);
  P_i.setZero(nZnT);
  rup_i.setZero(nZnT);
  zup_i.setZero(nZnT);
  rsp_i.setZero(nZnT);
  zsp_i.setZero(nZnT);
  taup_i.setZero(nZnT);
  gbubu_i.setZero(nZnT);
  gbubv_i.setZero(nZnT);
  gbvbv_i.setZero(nZnT);
  P_o.setZero(nZnT);
  rup_o.setZero(nZnT);
  zup_o.setZero(nZnT);
  rsp_o.setZero(nZnT);
  zsp_o.setZero(nZnT);
  taup_o.setZero(nZnT);
  gbubu_o.setZero(nZnT);
  gbubv_o.setZero(nZnT);
  gbvbv_o.setZero(nZnT);
  P_avg.setZero(nZnT);
  P_wavg.setZero(nZnT);
  gbubu_avg.setZero(nZnT);
  gbubu_wavg.setZero(nZnT);
  gbvbv_avg.setZero(nZnT);
  gbvbv_wavg.setZero(nZnT);
  gbubv_avg.setZero(nZnT);
  gbubv_wavg.setZero(nZnT);
}
}  // namespace vmecpp
