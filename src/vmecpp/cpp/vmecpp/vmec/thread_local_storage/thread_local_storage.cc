// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/thread_local_storage/thread_local_storage.h"

namespace vmecpp {

ThreadLocalStorage::ThreadLocalStorage(const Sizes* s)
    : s_(*s),
      r1e_i(s_.nZnT),
      r1o_i(s_.nZnT),
      rue_i(s_.nZnT),
      ruo_i(s_.nZnT),
      rve_i(s_.lthreed ? s_.nZnT : 0),
      rvo_i(s_.lthreed ? s_.nZnT : 0),
      z1e_i(s_.nZnT),
      z1o_i(s_.nZnT),
      zue_i(s_.nZnT),
      zuo_i(s_.nZnT),
      zve_i(s_.lthreed ? s_.nZnT : 0),
      zvo_i(s_.lthreed ? s_.nZnT : 0),
      lue_i(s_.nZnT),
      luo_i(s_.nZnT),
      lve_i(s_.lthreed ? s_.nZnT : 0),
      lvo_i(s_.lthreed ? s_.nZnT : 0),
      bsubu_i(s_.nZnT),
      bsubv_i(s_.nZnT),
      gvv_gsqrt_i(s_.nZnT),
      guv_bsupu_i(s_.nZnT),
      P_i(s_.nZnT),
      rup_i(s_.nZnT),
      zup_i(s_.nZnT),
      rsp_i(s_.nZnT),
      zsp_i(s_.nZnT),
      taup_i(s_.nZnT),
      gbubu_i(s_.nZnT),
      gbubv_i(s_.nZnT),
      gbvbv_i(s_.nZnT) {}
}  // namespace vmecpp
