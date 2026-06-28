// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/flow_control/flow_control.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "absl/log/check.h"
#include "vmecpp/common/util/os_compat.h"  // VMECPP_UNREACHABLE
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

namespace vmecpp {

// Maps the integer codes used by the Fortran reference (and by any
// serialized state that originates from that representation) onto the
// strongly-typed RestartReason enumeration. The four valid input values
// correspond one-to-one with the enumeration constants; any other input
// indicates corrupted state and triggers the compiler's unreachable hint
// so that the optimizer can elide the default branch.
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
#ifdef _OPENMP
  // Size the thread pool immediately so spin-waiting threads are not created
  // for cores that will never be used. Without this, omp_get_max_threads()
  // returns the hardware count and the runtime may spawn that many threads
  // before vmec_adjust_num_threads narrows the count at the first multigrid
  // step.
  omp_set_num_threads(max_threads.value());
#endif  // _OPENMP
  return max_threads.value();
}

// Constructs the iteration controller state for a single Vmec::run.
//
// The residual fields fsq, fsqr, and fsqz are initialized to unity, which
// is large enough to exceed any reasonable force tolerance and therefore
// prevents the convergence test from firing prematurely before the first
// force evaluation has populated them. The force-tolerance field ftolv is
// likewise initialized to unity so that the controller's stage-startup
// logic always observes ftolv > fsqr and proceeds with the iteration.
//
// The Jacobian-event counter ijacob and the reference residual res0 are
// initialized to zero and minus one respectively, the latter sentinel
// indicating that no reference residual has yet been recorded. The
// restart disposition begins at NO_RESTART, multi_ns_grid records the
// number of multigrid stages that will execute over the run, and delt0r
// preserves the input delt so that subsequent reductions remain relative
// to the original value. The component residual vectors are zeroed for
// completeness; their values are overwritten at each force evaluation.
FlowControl::FlowControl(bool lfreeb, double delt, int num_grids,
                         std::optional<int> max_threads)
    : lfreeb(lfreeb), max_threads_(get_max_threads(max_threads)) {
  // INITIALIZE PARAMETERS
  fsq = 1.0;
  fsqr = 1.0;
  fsqz = 1.0;
  ftolv = fsqr;
  ijacob = 0;
  restart_reason = RestartReason::NO_RESTART;
  res0 = -1;
  delt0r = delt;
  multi_ns_grid = num_grids;
  neqs_old = 0;

  fResInvar.setZero();
  fResPrecd.setZero();

  ns_old = 0;

  // Default to no per-cfg niter cap; the convergence gate populates this
  // from VMECPP_PER_CFG_NITER_CAP on first use when the per-cfg vectors
  // are non-empty.
  niter_max_per_cfg = std::numeric_limits<int>::max();
}

// Accessor for the effective worker count established at construction.
// Exposed for code paths that must size auxiliary thread-private buffers
// consistently with the OpenMP runtime's configuration.

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
// the active mask marks every configuration as iterating, and the
// component residual vectors are zeroed.
namespace {
// Reads VMECPP_ACTIVE_PER_CFG_OVERRIDE_BITS to produce an initial
// active-per-cfg mask. The env-var format is a string of ASCII 0/1
// characters, one per cfg. When unset, the mask is all-ones (every cfg
// active). The override is consumed by both ResizeForBatch and
// ResetActivePerCfgForNextStage so that callers (such as the per-cfg
// OutputQuantities reconstruction in pybind_vmec.cc's run_batched_gpu)
// can scope a Vmec instance to a single active configuration even when
// the CUDA state's cached n_config_max remains at the larger batched
// value.
std::vector<std::uint8_t> ResolveInitialActiveMask(int n_cfg) {
  std::vector<std::uint8_t> mask(n_cfg, static_cast<std::uint8_t>(1));
  const char* bits = std::getenv("VMECPP_ACTIVE_PER_CFG_OVERRIDE_BITS");
  if (bits == nullptr) return mask;
  const int len = static_cast<int>(std::strlen(bits));
  const int limit = std::min(n_cfg, len);
  for (int c = 0; c < limit; ++c) {
    mask[c] = (bits[c] == '0') ? static_cast<std::uint8_t>(0)
                               : static_cast<std::uint8_t>(1);
  }
  // Indices beyond the override length default to zero so callers can
  // pass a shorter prefix and have the remainder treated as inactive.
  for (int c = limit; c < n_cfg; ++c) {
    mask[c] = static_cast<std::uint8_t>(0);
  }
  return mask;
}
}  // namespace

void FlowControl::ResizeForBatch(int n_cfg) {
  if (n_cfg <= 0) return;
  if (static_cast<int>(active_per_cfg.size()) == n_cfg) return;
  restart_reason_per_cfg.assign(n_cfg, RestartReason::NO_RESTART);
  active_per_cfg = ResolveInitialActiveMask(n_cfg);
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

// Restores active_per_cfg to ones (subject to the
// VMECPP_ACTIVE_PER_CFG_OVERRIDE_BITS override), zeros iter2_per_cfg,
// and clears converged_per_cfg. Called at the start of every multigrid
// stage after the first. Configurations that converged against the
// coarser-stage tolerance must re-iterate at the finer stage's tighter
// tolerance, so the active mask is rebuilt rather than carried
// forward.
void FlowControl::ResetActivePerCfgForNextStage() {
  const int n_cfg = static_cast<int>(active_per_cfg.size());
  if (n_cfg <= 0) return;
  active_per_cfg = ResolveInitialActiveMask(n_cfg);
  std::fill(iter2_per_cfg.begin(), iter2_per_cfg.end(), 0);
  std::fill(converged_per_cfg.begin(), converged_per_cfg.end(),
            static_cast<std::uint8_t>(0));
}

}  // namespace vmecpp
