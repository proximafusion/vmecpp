// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_
#define VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <vector>

namespace vmecpp {

// enumerates values of `restart_reason`
// was `irst` = 1, 2, 3, 4 in Fortran VMEC
enum class RestartReason : std::uint8_t {
  // irst == 1, no restart required, instead make backup of current state vector
  // when calling Vmec::RestartIteration
  NO_RESTART = 1,

  // irst == 2, bad Jacobian, flux surfaces are overlapping
  BAD_JACOBIAN = 2,

  // irst == 3, bad progress, residuals not decaying as expected
  BAD_PROGRESS = 3,

  // irst == 4, huge initial forces, flux surfaces are too close to each other
  // (but not overlapping yet)
  HUGE_INITIAL_FORCES = 4
};

RestartReason RestartReasonFromInt(int restart_reason);

class FlowControl {
 public:
  // ns4: number of iterations between update of radial preconditioner matrix
  static constexpr int kPreconditionerUpdateInterval = 25;

  FlowControl(bool lfreeb, double delt, int num_grids,
              std::optional<int> max_threads = std::nullopt);

  int max_threads() const;

  const bool lfreeb;

  // was called `irst` in Fortran VMEC
  RestartReason restart_reason;

  // current ns in algorithm
  int ns;

  int neqs;
  int neqs_old;

  bool haveToFlipTheta;

  int ijacob;

  int multi_ns_grid;

  // ------ current multi-grid step settings

  // radial resolution of current multi-grid step
  int nsval;

  // radial grid spacing of flux surfaces: 1.0 / (ns - 1.0)
  double deltaS;

  // current force tolerance
  double ftolv;

  // current maximum number of iterations
  int niterv;

  // --------- end: current multi-grid step settings

  int num_surfaces_to_distribute;

  // for initialize_radial and interp
  int ns_min;
  int ns_old;

  double delt0r;

  // Cumulative force residuals (radial, vertical and lambda)
  // Populated by `evalFResInvar`
  double fsqr, fsqz, fsql;

  // Time-trace of the invariant force residuals during convergence
  // fsqt = (force_residual_r + force_residual_z + force_residual_lambda)
  std::vector<double> force_residual_r;
  std::vector<double> force_residual_z;
  std::vector<double> force_residual_lambda;

  // Preconditioned cumulative force residuals (radial, vertical and lambda)
  // Populated by `evalFResPrecd`
  double fsqr1, fsqz1, fsql1;
  double fsq;

  std::vector<double> mhd_energy;

  // Time-trace of the force at the vacuum boundary (only for free-boundary)
  std::vector<double> delbsq;
  // Time-trace of the restart reasons, for debugging purposes. Each restart is
  // a pair of <iteration, reason> (e.g. to see how many jacobian resets
  // occurred)
  std::vector<RestartReason> restart_reasons;

  // Running minimum of the preconditioned residual sum (fsq).
  double res0;
  // Running minimum of the invariant residual sum (fsqr + fsqz + fsql); used
  // only by the PARVMEC time-step control.
  double res1;

  Eigen::Vector3d fResInvar;
  Eigen::Vector3d fResPrecd;

  // ---------------------------------------------------------------------------
  // Per-configuration state vectors used by the batched CUDA execution mode.
  // ---------------------------------------------------------------------------
  // Under the batched CUDA path (active when a run carries more than one
  // configuration) the device-resident kernels write per-configuration
  // outputs into buffers sized n_config_max. The cache accessors declared in
  // fft_toroidal_cuda.h populate the vectors below from those device
  // buffers. The iteration controller in Vmec::run consults these
  // per-configuration values to drive convergence gating and per-cfg
  // kernel masking. Under single-configuration execution the per-cfg
  // vectors are populated for completeness but the equivalent scalar
  // fields above remain authoritative; the convergence gate at the
  // multigrid termination check coincides with the legacy single-cfg
  // condition by construction.
  //
  // Logical role of active_per_cfg: a nonzero entry indicates that the
  // corresponding configuration is still iterating; a zero entry indicates
  // that the configuration has converged or been terminated. The
  // convergence test for successful termination becomes the conjunction
  // over all configurations rather than the single scalar comparison
  // against fsqr.
  std::vector<RestartReason> restart_reason_per_cfg;
  // The active mask is stored as a vector of unsigned 8-bit integers
  // rather than std::vector<bool> so that the data can be copied directly
  // to a device byte buffer without the bitset packing transformation that
  // std::vector<bool> would impose. Nonzero values denote configurations
  // that should continue iterating; zero values denote configurations
  // that the per-cfg kernel skip-mask treats as inactive.
  std::vector<std::uint8_t> active_per_cfg;
  // Per-configuration scalar invariant residuals corresponding to the
  // single-configuration fsqr, fsqz, fsql values above.
  std::vector<double> fsqr_per_cfg, fsqz_per_cfg, fsql_per_cfg;
  // Per-configuration scalar preconditioned residuals corresponding to the
  // single-configuration fsqr1, fsqz1, fsql1 values above.
  std::vector<double> fsqr1_per_cfg, fsqz1_per_cfg, fsql1_per_cfg;
  // Per-configuration component-wise residual vectors. Eigen::Vector3d is
  // a fixed-size POD-like type; std::vector contiguity is preserved.
  std::vector<Eigen::Vector3d> fResInvar_per_cfg;
  std::vector<Eigen::Vector3d> fResPrecd_per_cfg;
  // Per-configuration counter of bad-Jacobian-induced restarts within the
  // current multigrid stage, used for per-configuration restart-rate
  // diagnostics and for the per-cfg recovery escalation logic.
  std::vector<int> ijacob_per_cfg;

  // Per-configuration iteration count of the current multigrid stage under
  // per_cfg_control: 1 at its start and advanced after every iteration in
  // which the configuration does not restart, as a single run's iter2.
  std::vector<int> iter2_per_cfg;

  // Per-configuration convergence outcome: 1 once the configuration met
  // ftolv in the current stage, 0 while it iterates and for a configuration
  // still iterating when the stage reached its iteration limit. Reset at
  // each multigrid stage transition.
  std::vector<std::uint8_t> converged_per_cfg;

  // Per-configuration counterparts of the time-traces above for a batch of
  // more than one configuration, logged where the traces above are, for each
  // configuration still active there; a configuration's traces end at the
  // iteration it converges, as a single run's do.
  std::vector<std::vector<double>> force_residual_r_per_cfg;
  std::vector<std::vector<double>> force_residual_z_per_cfg;
  std::vector<std::vector<double>> force_residual_lambda_per_cfg;
  std::vector<std::vector<double>> mhd_energy_per_cfg;
  std::vector<std::vector<double>> delbsq_per_cfg;
  std::vector<std::vector<RestartReason>> restart_reasons_per_cfg;

  // Restart control per configuration (per_cfg_control), set for a distinct
  // batch, whose configurations take the time step and the restarts of their
  // own single runs: the time step, the iteration of the latest restart
  // (iter1), the best preconditioned residual sum since it (res0) and, under
  // the PARVMEC iteration style, the best invariant one (res1), and the
  // iteration count reported for each configuration. restart_reason_per_cfg
  // holds the restart reason of each configuration's current iteration.
  bool per_cfg_control = false;
  std::vector<double> delt0r_per_cfg;
  std::vector<int> iter1_per_cfg;
  std::vector<double> res0_per_cfg;
  std::vector<double> res1_per_cfg;
  std::vector<int> niter_per_cfg;

  // 1 for a configuration whose iteration failed where a single run's fails:
  // a bad Jacobian in the first iteration of a stage, residuals that are not
  // finite, or 75 bad Jacobians. It sits out the later multigrid stages.
  std::vector<std::uint8_t> failed_per_cfg;

  // Allocates each of the per-configuration vectors above to a length of
  // n_cfg, default-initializing the contents. The operation is idempotent:
  // calling with a value matching the present size has no effect. Intended
  // to be invoked once at the start of Vmec::run when the CUDA path is
  // active, but safe to invoke at any multigrid stage transition.
  void ResizeForBatch(int n_cfg);

  // Re-activates every configuration that has not failed for the next
  // multigrid stage. Configurations that converged against the coarser-stage
  // tolerance must continue iterating at the finer stage's tighter
  // tolerance, so active_per_cfg is restored, converged_per_cfg is zeroed,
  // and the iteration count and the restart control of every configuration
  // start over with the time step delt. Has no effect when the per-cfg
  // vectors are empty (i.e. single-cfg or pre-ResizeForBatch).
  void ResetActivePerCfgForNextStage(double delt);

 private:
  const int max_threads_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_
