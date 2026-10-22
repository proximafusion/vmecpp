// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/flow_control/flow_control.h"

#include <algorithm>

#include "absl/log/check.h"
#include "vmecpp/common/util/os_compat.h"  // VMECPP_UNREACHABLE
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
      VMECPP_UNREACHABLE();
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
  res1 = -1;
  delt0r = delt;
  multi_ns_grid = num_grids;
  neqs_old = 0;

  fResInvar.setZero();
  fResPrecd.setZero();

  ns_old = 0;
}

int FlowControl::max_threads() const { return max_threads_; }

// Allocates and initializes the per-configuration state vectors that
// support the batched CUDA execution mode. The operation is idempotent
// in the sense that a call whose argument matches the currently allocated
// length returns immediately without disturbing the contents. A call with
// n_cfg <= 0 is treated as a request to leave the vectors in their
// current state.
//
// Initial values mirror the single-configuration initialization performed
// in the constructor: the residual fields are set to unity to suppress
// premature convergence, the restart disposition starts at NO_RESTART,
// every configuration starts active, and the component residual vectors
// are zeroed.
void FlowControl::ResizeForBatch(int n_cfg) {
  if (n_cfg <= 0) return;
  if (static_cast<int>(active_per_cfg.size()) == n_cfg) return;
  restart_reason_per_cfg.assign(n_cfg, RestartReason::NO_RESTART);
  active_per_cfg.assign(n_cfg, static_cast<std::uint8_t>(1));
  fsqr_per_cfg.assign(n_cfg, 1.0);
  fsqz_per_cfg.assign(n_cfg, 1.0);
  fsql_per_cfg.assign(n_cfg, 1.0);
  fsqr1_per_cfg.assign(n_cfg, 1.0);
  fsqz1_per_cfg.assign(n_cfg, 1.0);
  fsql1_per_cfg.assign(n_cfg, 1.0);
  fResInvar_per_cfg.assign(n_cfg, Eigen::Vector3d::Zero());
  fResPrecd_per_cfg.assign(n_cfg, Eigen::Vector3d::Zero());
  ijacob_per_cfg.assign(n_cfg, 0);
  iter2_per_cfg.assign(n_cfg, 0);
  converged_per_cfg.assign(n_cfg, static_cast<std::uint8_t>(0));
}

// Restores active_per_cfg to ones, zeros iter2_per_cfg, and clears
// converged_per_cfg. Called at the start of every multigrid stage.
// Configurations that converged against the coarser-stage tolerance must
// re-iterate at the finer stage's tighter tolerance, so the active mask is
// rebuilt rather than carried forward.
void FlowControl::ResetActivePerCfgForNextStage() {
  const int n_cfg = static_cast<int>(active_per_cfg.size());
  if (n_cfg <= 0) return;
  std::fill(active_per_cfg.begin(), active_per_cfg.end(),
            static_cast<std::uint8_t>(1));
  std::fill(iter2_per_cfg.begin(), iter2_per_cfg.end(), 0);
  std::fill(converged_per_cfg.begin(), converged_per_cfg.end(),
            static_cast<std::uint8_t>(0));
}

}  // namespace vmecpp
