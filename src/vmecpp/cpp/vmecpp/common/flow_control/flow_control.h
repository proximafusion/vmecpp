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

// Enumerated values for FlowControl::restart_reason, which controls the
// disposition of the iteration loop's recovery logic at the end of each
// time step. The integer values are preserved from the upstream Fortran
// VMEC implementation (variable `irst`) to simplify cross-reference between
// the C++ port and the Fortran reference source.
enum class RestartReason : std::uint8_t {
  // No restart is required for the current iteration. RestartIteration
  // creates a backup of the current state vector in preparation for the
  // next time step, allowing subsequent failure modes to revert cleanly
  // to a known-good configuration. Fortran equivalent: irst == 1.
  NO_RESTART = 1,

  // The Jacobian of the radial coordinate transform has changed sign,
  // indicating that adjacent flux surfaces have begun to overlap and the
  // current geometry is no longer well-posed. On the first occurrence
  // within a given multigrid stage, the iteration controller invokes
  // magnetic-axis recomputation; on subsequent occurrences it reduces the
  // time step delt and retries from the most recent backup. Fortran
  // equivalent: irst == 2.
  BAD_JACOBIAN = 2,

  // Residual norms are not decaying at the expected rate over the
  // preceding interval, signaling that the present time step is too large
  // to support stable progress. The controller treats this as a soft
  // failure, reduces delt, and retries from the most recent backup.
  // Fortran equivalent: irst == 3.
  BAD_PROGRESS = 3,

  // Initial forces evaluated at the start of an iteration exceeded a
  // safety threshold, indicating that the flux surfaces are spaced too
  // closely though not yet overlapping in the BAD_JACOBIAN sense. The
  // controller resets to the initial guess with a reduced time step,
  // treating the condition as an early-stage instability. Fortran
  // equivalent: irst == 4.
  HUGE_INITIAL_FORCES = 4
};

// Maps a raw integer restart-reason code, as produced by legacy code paths
// reading from the Fortran reference, into the corresponding RestartReason
// enumerator. Used at conversion boundaries between integer-typed state
// fields and the typed C++ representation.
RestartReason RestartReasonFromInt(int restart_reason);

// Aggregate of the iteration controller's mutable state for a single
// invocation of Vmec::run. Holds the current multigrid stage parameters,
// residual histories, restart bookkeeping, and the per-config parallel
// state vectors used by the batched CUDA execution mode. Instances are
// constructed once per Vmec object and persist across multigrid stage
// transitions; ResizeForBatch sizes the per-config vectors when the CUDA
// path is active.
class FlowControl {
 public:
  // Number of iterations between successive updates of the radial
  // preconditioner matrix. The interval balances the cost of preconditioner
  // factorization against the rate at which the preconditioner approximation
  // becomes stale relative to the evolving geometry. Corresponds to the
  // ns4 parameter in the Fortran reference.
  static constexpr int kPreconditionerUpdateInterval = 25;

  FlowControl(bool lfreeb, double delt, int num_grids,
              std::optional<int> max_threads = std::nullopt);

  int max_threads() const;

  // Whether the equilibrium is computed in free-boundary mode (the plasma
  // boundary is determined consistently with an external magnetic field) or
  // fixed-boundary mode (the boundary is supplied as input and held rigid).
  const bool lfreeb;

  // Current disposition of the iteration controller. Read at the start of
  // each iteration to determine whether a restart is in progress and which
  // recovery action to apply. Corresponds to the Fortran variable `irst`.
  RestartReason restart_reason;

  // Current number of radial flux surfaces in the discretization. Set at
  // the start of each multigrid stage and held constant within the stage.
  int ns;

  // Number of unknown coefficients in the current discretization. Equal to
  // the spectral coefficient count multiplied by the number of independent
  // poloidal and toroidal modes for the current ns value.
  int neqs;
  // The value of neqs from the previous multigrid stage, retained to drive
  // interpolation from the coarser representation to the present grid.
  int neqs_old;

  // Indicates whether the input boundary required a poloidal-angle flip
  // during setup to satisfy the sign-of-Jacobian convention. Used by
  // diagnostic and output paths to record the equivalence with the input.
  bool haveToFlipTheta;

  // Cumulative count of Jacobian-sign-induced restarts within the current
  // multigrid stage. Used by the iteration controller to escalate from
  // axis recomputation on the first occurrence to time-step reduction on
  // subsequent occurrences, and to detect pathological cases where the
  // count exceeds the runaway threshold.
  int ijacob;

  // Number of multigrid stages remaining to be executed, decremented as the
  // controller advances through the ns_array sequence supplied by the input.
  int multi_ns_grid;

  // -----------------------------------------------------------------------
  // Current multigrid stage parameters. These four fields are updated at
  // each stage transition and held constant throughout the iterations of a
  // single stage. They derive directly from the indata arrays ns_array,
  // ftol_array, and niter_array at the current multigrid index.
  // -----------------------------------------------------------------------

  // Radial resolution of the current multigrid stage. Identical to ns
  // during iteration but logically distinct: nsval is the planned stage
  // value, while ns reflects the in-progress discretization.
  int nsval;

  // Radial grid spacing of the flux surfaces, given by 1.0 / (ns - 1.0).
  // Required by every per-iteration force evaluation and metric calculation.
  double deltaS;

  // Force-residual tolerance for the current stage. The iteration is
  // considered converged when each of fsqr, fsqz, and fsql falls below
  // this threshold.
  double ftolv;

  // Maximum number of iterations permitted within the current stage.
  // Exceeding this value without convergence terminates the run.
  int niterv;

  // -----------------------------------------------------------------------
  // End of current multigrid stage parameters.
  // -----------------------------------------------------------------------

  // Number of radial surfaces distributed across the parallel worker pool
  // for the current stage. Equal to ns under single-rank execution; under
  // multi-rank execution this is the per-rank share.
  int num_surfaces_to_distribute;

  // Lower bound of the radial index range processed during InitializeRadial
  // and the multigrid interpolation step.
  int ns_min;
  // Value of ns from the previous multigrid stage, retained to drive the
  // interpolation from the coarser grid into the present finer grid.
  int ns_old;

  // Working copy of the initial time step delt at the start of an iteration
  // sequence, used by the time-step reduction logic on bad_jacobian events
  // to derive the reduced step from the original input value.
  double delt0r;

  // Cumulative invariant force residuals for the radial (R), vertical (Z),
  // and lambda equations respectively. Populated by IdealMhdModel's
  // evalFResInvar at the end of each force evaluation. The iteration is
  // considered converged when each of these falls below ftolv.
  double fsqr, fsqz, fsql;

  // Per-iteration history of the invariant residuals, retained for
  // post-run diagnostics and convergence-rate analysis. The total force
  // residual at iteration k is given by force_residual_r[k] +
  // force_residual_z[k] + force_residual_lambda[k].
  std::vector<double> force_residual_r;
  std::vector<double> force_residual_z;
  std::vector<double> force_residual_lambda;

  // Cumulative preconditioned force residuals, computed in the same units
  // and ordering as the invariant residuals above but after application of
  // the radial preconditioner. Drive the time-step controller's evolution
  // of delt and the tau-acceleration logic in performTimeStep. Populated
  // by IdealMhdModel's evalFResPrecd.
  double fsqr1, fsqz1, fsql1;
  // Sum of the three preconditioned residuals fsqr1 + fsqz1 + fsql1.
  // Cached at each force evaluation for the time-step controller's
  // convenience.
  double fsq;

  // Per-iteration history of the MHD energy functional, retained for
  // post-run diagnostics and convergence-rate analysis.
  std::vector<double> mhd_energy;

  // Per-iteration history of the integrated magnetic-field jump at the
  // plasma-vacuum interface. Populated only when lfreeb is true; otherwise
  // remains empty.
  std::vector<double> delbsq;
  // Per-iteration log of the restart-reason transitions encountered during
  // the run, retained for diagnostic purposes. Permits post-run inspection
  // of how often each recovery path was invoked and at which iterations.
  std::vector<RestartReason> restart_reasons;

  // Reference residual against which the time-step controller measures
  // relative residual decay. Updated when the iteration enters a new
  // multigrid stage or when restart_reason transitions back to NO_RESTART.
  double res0;

  // Component-wise vector form of the invariant residuals fsqr, fsqz, fsql,
  // packaged for callers that consume the three components together
  // (notably the per-cfg state vectors below for the batched CUDA path).
  Eigen::Vector3d fResInvar;
  // Component-wise vector form of the preconditioned residuals fsqr1,
  // fsqz1, fsql1, packaged for the same callers as fResInvar.
  Eigen::Vector3d fResPrecd;

  // ---------------------------------------------------------------------------
  // Per-configuration state vectors used by the batched CUDA execution mode.
  // ---------------------------------------------------------------------------
  // Under the batched CUDA path (active when VMECPP_N_CONFIG_MAX exceeds one)
  // the device-resident kernels write per-configuration outputs into buffers
  // sized n_config_max. The host-side device-to-host accessors declared in
  // fft_toroidal_cuda.h (ComputeJacobianCudaPerCfgD2H,
  // ComputeForceNormsCudaPerCfgD2H, ResidualsCudaPerCfgD2H, and the
  // associated cache accessors) populate the vectors below from those
  // device buffers. The iteration controller in Vmec::run consults these
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

  // Per-configuration iteration counter. Incremented at the convergence
  // gate each iteration the configuration is still active. Used by the
  // per-cfg niter cap to mark slow cfgs as timed out so that faster cfgs
  // in the batch can return without waiting for the slow cfg to converge
  // or for the shared niterv to be hit. Reset to zero at each multigrid
  // stage transition via ResetActivePerCfgForNextStage.
  std::vector<int> iter2_per_cfg;

  // Per-configuration convergence outcome. Set when active_per_cfg[c]
  // transitions to zero: 1 when the cfg met ftolv, 0 when it timed out
  // against the per-cfg niter cap. Used by the batch-output pipeline to
  // mark per-cfg results as converged or not converged. Reset at each
  // multigrid stage transition.
  std::vector<std::uint8_t> converged_per_cfg;

  // Per-configuration iteration ceiling. When VMECPP_PER_CFG_NITER_CAP is
  // set, the convergence gate marks any cfg whose iter2_per_cfg has
  // reached this value as timed out (active_per_cfg[c]=0,
  // converged_per_cfg[c]=0). Default value INT_MAX disables the cap and
  // preserves legacy behaviour. The cap is per-stage; iter2_per_cfg
  // resets at each multigrid stage transition.
  int niter_max_per_cfg;

  // Allocates each of the per-configuration vectors above to a length of
  // n_cfg, default-initializing the contents. The operation is idempotent:
  // calling with a value matching the present size has no effect. Intended
  // to be invoked once at the start of Vmec::run when the CUDA path is
  // active, but safe to invoke at any multigrid stage transition.
  void ResizeForBatch(int n_cfg);

  // Re-activates every configuration for the next multigrid stage and
  // resets the per-cfg iteration counter. Configurations that converged
  // against the coarser-stage tolerance must continue iterating at the
  // finer stage's tighter tolerance, so active_per_cfg is restored to
  // all-ones. iter2_per_cfg and converged_per_cfg are zeroed so the per-
  // cfg niter cap applies fresh to each stage. Has no effect when the
  // per-cfg vectors are empty (i.e. single-cfg or pre-ResizeForBatch).
  void ResetActivePerCfgForNextStage();

 private:
  const int max_threads_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_
