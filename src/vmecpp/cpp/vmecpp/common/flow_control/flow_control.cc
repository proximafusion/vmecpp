// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/flow_control/flow_control.h"

#include "absl/log/check.h"
#include "absl/log/log.h"

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

namespace vmecpp {

RestartReason RestartReasonFromInt(int restart_reason) {
  switch (restart_reason) {
    case 1:
      return RestartReason::NO_RESTART;
    case 2:
      return RestartReason::BAD_JACOBIAN;
    case 3:
      return RestartReason::BAD_PROGRESS;
    case 4:
      return RestartReason::HUGE_INITIAL_FORCES;
    default:
      LOG(FATAL) << "Invalid restart_reason value: " << restart_reason;
  }
}

int get_max_threads(std::optional<int> max_threads) {
  if (max_threads == std::nullopt) {
#ifdef _OPENMP
    return omp_get_max_threads();
#endif  // _OPENMP
    // Default to 1 thread if OpenMP is not available
    return 1;
  }
  CHECK_GT(max_threads.value(), 0)
      << "The number of threads must be >=1. "
         "To automatically use all available threads, pass std::nullopt";
  return max_threads.value();
}

FlowControl::FlowControl(bool lfreeb, double delt, int num_grids,
                         std::optional<int> max_threads)
    : lfreeb(lfreeb), max_threads_(get_max_threads(max_threads)) {
  fsq = 1.0;

  // INITIALIZE PARAMETERS
  fsqr = 1.0;
  fsqz = 1.0;
  ftolv = fsqr;
  ijacob = 0;
  restart_reason = RestartReason::NO_RESTART;
  res0 = -1;
  delt0r = delt;
  multi_ns_grid = num_grids;
  neqs_old = 0;

  fResInvar.resize(3, 0.0);
  fResPrecd.resize(3, 0.0);

  ns_old = 0;
}

int FlowControl::max_threads() const { return max_threads_; }

}  // namespace vmecpp
