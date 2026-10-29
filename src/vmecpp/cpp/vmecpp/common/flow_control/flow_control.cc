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

// Sizes the per-configuration vectors to n_cfg, initialized as the constructor
// initializes the scalars; n_cfg <= 0 or the current size is a no-op.
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
  force_residual_r_per_cfg.assign(n_cfg, {});
  force_residual_z_per_cfg.assign(n_cfg, {});
  force_residual_lambda_per_cfg.assign(n_cfg, {});
  mhd_energy_per_cfg.assign(n_cfg, {});
  delbsq_per_cfg.assign(n_cfg, {});
  restart_reasons_per_cfg.assign(n_cfg, {});
  delt0r_per_cfg.assign(n_cfg, 0.0);
  iter1_per_cfg.assign(n_cfg, 1);
  res0_per_cfg.assign(n_cfg, -1.0);
  res1_per_cfg.assign(n_cfg, -1.0);
  niter_per_cfg.assign(n_cfg, 0);
  failed_per_cfg.assign(n_cfg, static_cast<std::uint8_t>(0));
}

// Restarts each configuration that has not failed for the next multigrid stage:
// active, unconverged, its iteration count and restart control reset.
void FlowControl::ResetActivePerCfgForNextStage(double delt) {
  const int n_cfg = static_cast<int>(active_per_cfg.size());
  if (n_cfg <= 0) return;
  for (int c = 0; c < n_cfg; ++c) {
    active_per_cfg[c] = static_cast<std::uint8_t>(failed_per_cfg[c] == 0);
  }
  std::fill(iter2_per_cfg.begin(), iter2_per_cfg.end(), 1);
  std::fill(converged_per_cfg.begin(), converged_per_cfg.end(),
            static_cast<std::uint8_t>(0));
  std::fill(restart_reason_per_cfg.begin(), restart_reason_per_cfg.end(),
            RestartReason::NO_RESTART);
  std::fill(ijacob_per_cfg.begin(), ijacob_per_cfg.end(), 0);
  std::fill(delt0r_per_cfg.begin(), delt0r_per_cfg.end(), delt);
  std::fill(iter1_per_cfg.begin(), iter1_per_cfg.end(), 1);
  std::fill(res0_per_cfg.begin(), res0_per_cfg.end(), -1.0);
  std::fill(res1_per_cfg.begin(), res1_per_cfg.end(), -1.0);
  std::fill(niter_per_cfg.begin(), niter_per_cfg.end(), 0);
}

}  // namespace vmecpp
