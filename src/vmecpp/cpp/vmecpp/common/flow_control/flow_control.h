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

  // Per-configuration state of a batched CUDA run, filled from the device
  // buffers; the scalar fields above govern a single-configuration run.
  std::vector<RestartReason> restart_reason_per_cfg;
  // Nonzero while the configuration iterates; bytes rather than
  // std::vector<bool> so the mask copies straight to the device.
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
  // Bad-Jacobian restarts of each configuration in the current multigrid stage.
  std::vector<int> ijacob_per_cfg;

  // Each configuration's iteration count in the stage under per_cfg_control,
  // advanced when it does not restart, as a single run's iter2.
  std::vector<int> iter2_per_cfg;

  // 1 once the configuration met ftolv in the current stage; 0 while it
  // iterates or when the stage ran out of iterations.
  std::vector<std::uint8_t> converged_per_cfg;

  // Per-configuration time traces of a batch, logged where the traces above are
  // while the configuration is active.
  std::vector<std::vector<double>> force_residual_r_per_cfg;
  std::vector<std::vector<double>> force_residual_z_per_cfg;
  std::vector<std::vector<double>> force_residual_lambda_per_cfg;
  std::vector<std::vector<double>> mhd_energy_per_cfg;
  std::vector<std::vector<double>> delbsq_per_cfg;
  std::vector<std::vector<RestartReason>> restart_reasons_per_cfg;

  // Restart control of each configuration of a distinct batch: time step,
  // iter1, residual minima res0 and res1, and the reported iteration count.
  bool per_cfg_control = false;
  std::vector<double> delt0r_per_cfg;
  std::vector<int> iter1_per_cfg;
  std::vector<double> res0_per_cfg;
  std::vector<double> res1_per_cfg;
  std::vector<int> niter_per_cfg;

  // 1 for a configuration that failed as its single run would: a bad Jacobian
  // in a stage's first iteration, non-finite residuals or 75 bad Jacobians.
  std::vector<std::uint8_t> failed_per_cfg;

  // Sizes the per-configuration vectors to n_cfg; a call with the current size
  // has no effect.
  void ResizeForBatch(int n_cfg);

  // Restarts each configuration that has not failed for the next stage, with
  // time step delt; a no-op when the per-configuration vectors are empty.
  void ResetActivePerCfgForNextStage(double delt);

 private:
  const int max_threads_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_
