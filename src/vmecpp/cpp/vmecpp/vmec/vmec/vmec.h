// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_VMEC_VMEC_H_
#define VMECPP_VMEC_VMEC_VMEC_H_

#include <atomic>
#include <climits>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/free_boundary/free_boundary_base/free_boundary_base.h"
#include "vmecpp/free_boundary/nestor/nestor.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"
#include "vmecpp/vmec/boundaries/boundaries.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/fourier_velocity/fourier_velocity.h"
#include "vmecpp/vmec/geometry/geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/iteration_logger/iteration_logger.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

namespace vmecpp {

enum class MultigridInterpolationScheme : std::uint8_t {
  // 2-point linear interpolation in s (VMEC 8.52 behavior)
  kLinear,
  // 4-point Lagrange interpolation in s
  kCubic,
  // 4-point Lagrange interpolation in rho = sqrt(s), the natural radial
  // variable near the magnetic axis
  kCubicRho,
};

// The state we need to hot-restart a VMEC++ run.
struct HotRestartState {
  WOutFileContents wout;
  VmecINDATA indata;

  HotRestartState(WOutFileContents wout, VmecINDATA indata)
      : wout(std::move(wout)), indata(std::move(indata)) {}

  explicit HotRestartState(const OutputQuantities& output_quantities)
      : wout(output_quantities.wout), indata(output_quantities.indata) {}

  explicit HotRestartState(OutputQuantities&& output_quantities)
      : wout(std::move(output_quantities.wout)),
        indata(std::move(output_quantities.indata)) {}
};

// Callback that returns true if execution should be interrupted (e.g., Ctrl+C).
// Called periodically from the iteration loop.
using InterruptCallback = std::function<bool()>;

// The fields of a force evaluation on the half grid, at the angles
// theta_l = 2 pi l / ntheta_even, l < ntheta_eff, and
// zeta_k = 2 pi k / (nfp nzeta), k < nzeta. Without lasym the poloidal points
// cover [0, pi], and a field on the rest of a surface follows from
// f(theta, zeta) = f(-theta, -zeta).
struct HalfGridFields {
  // [ns - 1, nzeta * ntheta_eff], each row zeta-major: the Jacobian sqrt(g),
  // whose sign is signgs, and the contravariant and covariant components of B
  RowMatrixXd gsqrt;
  RowMatrixXd bsupu;
  RowMatrixXd bsupv;
  RowMatrixXd bsubu;
  RowMatrixXd bsubv;
  // [ntheta_eff] weight of each point in an angle average,
  // <f> = sum_{k,l} weight_l f_kl
  Eigen::VectorXd weight;
  // [ns - 1] the half-grid profiles of the wout file: buco = <B_theta>,
  // bvco = <B_zeta>, iotas, phips and vp = signgs <sqrt(g)>
  Eigen::VectorXd buco;
  Eigen::VectorXd bvco;
  Eigen::VectorXd iota;
  Eigen::VectorXd phip;
  Eigen::VectorXd vp;
  int ntheta_even = 0;
  int ntheta_eff = 0;
  int nzeta = 0;
  int nfp = 0;
  int signgs = 0;
};

// The state of the solver after the force iteration that just completed,
// handed to an IterationCallback by the master thread while the other threads
// wait.
struct SolverState {
  // iteration counter of the current multigrid stage, as printed
  int iteration;
  // index into ns_array of the current stage; -1 for the inserted ns = 3 stage
  int multigrid_step;
  int ns;
  // invariant force residuals of R, Z and lambda, the ones tested against ftol
  double fsqr;
  double fsqz;
  double fsql;
  double ftol;
  // current time step
  double delt;
  // RestartReason of this iteration; NO_RESTART unless the state was reverted
  // to the last backup
  RestartReason restart_reason;
  // Jacobian resets so far in this stage
  int jacobian_resets;
  // whether the vacuum pressure is part of the force balance yet
  bool vacuum_pressure_active;
  // MHD energy of the state
  double mhd_energy;
  // R, Z and lambda coefficients of the state, as MakeGeometry lays them out
  Geometry geometry;
  // the fields of the force evaluation of this iteration, which the time step
  // that followed it has moved geometry away from, except on the iteration
  // that converges
  HalfGridFields half_grid;
  // [ns - 1] with ncurr = 1, the enclosed toroidal current that the force
  // evaluations prescribe, in the units of buco: each evaluation solves for
  // chi' so that buco equals it. Empty with ncurr = 0.
  Eigen::VectorXd curr_h;
};

// Called once per force iteration; return false to stop the run, which then
// returns the output quantities of the state reached. The callback may change
// the values of state.curr_h, and the force evaluations prescribe the changed
// current from the next iteration to the end of the multigrid step.
using IterationCallback = std::function<bool(SolverState&)>;

// This is the preferred way to run VMEC++.
absl::StatusOr<OutputQuantities> run(
    const VmecINDATA& indata,
    std::optional<HotRestartState> initial_state = std::nullopt,
    std::optional<int> max_threads = std::nullopt,
    OutputMode verbose = OutputMode::kLegacy,
    InterruptCallback interrupt_callback = nullptr,
    bool always_fix_m1_gauge = false,
    IterationCallback iteration_callback = nullptr);

// This overload enables free-boundary runs with an in-memory mgrid file.
// The mgrid_file entry in `indata` will be ignored.
// This is useful e.g. to perform free-boundary hot-restarted runs where
// the coil geometry can be modified in-memory.
absl::StatusOr<OutputQuantities> run(
    const VmecINDATA& indata,
    const makegrid::MagneticFieldResponseTable& magnetic_response_table,
    std::optional<HotRestartState> initial_state = std::nullopt,
    std::optional<int> max_threads = std::nullopt,
    OutputMode verbose = OutputMode::kLegacy,
    InterruptCallback interrupt_callback = nullptr,
    IterationCallback iteration_callback = nullptr);

class Vmec {
 public:
  // Prefer using the FromIndata factory method, which handles both fixed-
  // and free-boundary initialization with proper error handling.
  // This constructor is public for use in external test code.
  explicit Vmec(const VmecINDATA& indata,
                std::optional<int> max_threads = std::nullopt,
                OutputMode verbose = OutputMode::kLegacy,
                InterruptCallback interrupt_callback = nullptr,
                IterationCallback iteration_callback = nullptr);

  // Vmec must not be moved or copied because members (t_, b_, h_) store
  // raw pointers to sibling members (s_, t_). Moving would leave those
  // pointers dangling.
  Vmec(const Vmec&) = delete;
  Vmec& operator=(const Vmec&) = delete;
  Vmec(Vmec&&) = delete;
  Vmec& operator=(Vmec&&) = delete;

  // sign of Jacobian between cylindrical and flux coordinates
  // This is called `signgs` in Fortran VMEC.
  static constexpr int kSignOfJacobian = -1;

  // scaling factor for blending between two different ways to compute B^zeta
  static constexpr double kPDamp = 0.05;

  // Factory method for creating a Vmec instance.
  // Handles mgrid loading for free-boundary runs with proper error handling.
  // Returns a unique_ptr because Vmec is non-movable.
  static absl::StatusOr<std::unique_ptr<Vmec>> FromIndata(
      const VmecINDATA& indata,
      const makegrid::MagneticFieldResponseTable* magnetic_response_table =
          nullptr,
      std::optional<int> max_threads = std::nullopt,
      OutputMode verbose = OutputMode::kLegacy,
      InterruptCallback interrupt_callback = nullptr,
      IterationCallback iteration_callback = nullptr);

  // checkpoint_multi_grid_step selects which entry of ns_array the checkpoints
  // taken in InitializeRadial fire on, counting from 1. Without it those
  // checkpoints always stop the run in the first multi-grid step, which leaves
  // the setup of every later step unreachable.
  absl::StatusOr<bool> run(
      const VmecCheckpoint& checkpoint = VmecCheckpoint::NONE,
      int iterations_before_checkpointing = INT_MAX,
      int maximum_multi_grid_step = 500,
      std::optional<HotRestartState> initial_state = std::nullopt,
      int checkpoint_multi_grid_step = 1);

  // -------------------

  // Size the free-boundary vacuum team (vac_num_threads_) to the threads the
  // runtime grants next to the radial team of num_threads_ threads, and build
  // the vacuum solvers (fb_vac_/tp_vac_) when that size changes. Call only
  // when lfreeb is set, after num_threads_ is set for the multigrid step and
  // the mgrid has been loaded. The solvers are ns-independent and persist
  // across multigrid steps.
  void SetupVacuumSolvers();

  // is_checkpoint_step says whether the multi-grid step being initialized is
  // the one the caller asked to checkpoint in.
  absl::StatusOr<bool> InitializeRadial(
      VmecCheckpoint checkpoint, int maximum_iterations, int nsval, int ns_old,
      double& m_delt0,
      const std::optional<HotRestartState>& initial_state = std::nullopt,
      std::optional<MultigridInterpolationScheme> interpolation_scheme =
          std::nullopt,
      bool is_checkpoint_step = true);
  absl::StatusOr<bool> SolveEquilibrium(VmecCheckpoint checkpoint,
                                        int maximum_iterations);
  // backup_evaluated_state: on a store, back up last_evaluated_x_ instead of
  // decomposed_x_ (already advanced by PerformTimeStep).
  void RestartIteration(double& m_delt0r, int thread_id,
                        bool backup_evaluated_state = false);
  absl::StatusOr<bool> Evolve(VmecCheckpoint checkpoint, int maximum_iterations,
                              double time_step, int thread_id,
                              bool& m_liter_flag);
  void Printout(double delt0r, int thread_id, int iter2);
  absl::StatusOr<bool> UpdateForwardModel(VmecCheckpoint checkpoint,
                                          int maximum_iterations,
                                          int thread_id);
  // Evaluate the model at the current state as the next iteration would,
  // without advancing the state.
  absl::Status EvaluateFinalState();
  void PerformTimeStep(double fac, double b1, double time_step, int thread_id);
  void InterpolateToNextMultigridStep(
      int ns_new, int ns_old,
      const std::vector<std::unique_ptr<RadialProfiles>>& p,
      const std::vector<std::unique_ptr<RadialPartitioning>>& r_new,
      const std::vector<std::unique_ptr<RadialPartitioning>>& r_old,
      std::vector<std::unique_ptr<FourierGeometry>>& m_x_new,
      std::vector<std::unique_ptr<FourierGeometry>>& m_x_old,
      std::optional<MultigridInterpolationScheme> interpolation_scheme =
          std::nullopt);
  // -------------------

  bool updateFwdModel(IdealMhdModel& m_m, FourierGeometry& m_decomposed_x,
                      FourierGeometry& m_physical_x, HandoverStorage& m_h,
                      FourierForces& m_decomposed_f,
                      FourierForces& m_physical_f, const RadialPartitioning& r,
                      FlowControl& m_fc, int thread_id,
                      const VmecCheckpoint& checkpoint = VmecCheckpoint::NONE,
                      int maximum_iterations = INT_MAX);

  void evolve(const RadialPartitioning& r, FourierGeometry& m_decomposed_x,
              FourierVelocity& m_decomposed_v,
              const FourierForces& decomposed_f, const FlowControl& fc);

  void performTimeStep(const Sizes& s, const FlowControl& fc,
                       const RadialPartitioning& r, double velocityScale,
                       double conjugationParameter, double time_step,
                       FourierGeometry& m_decomposed_x,
                       FourierVelocity& m_decomposed_v,
                       const FourierForces& decomposed_f,
                       HandoverStorage& m_h_) const;

  int get_ivac() const { return static_cast<int>(vacuum_pressure_state_); }
  int get_num_eqsolve_retries() const { return num_eqsolve_retries_; }
  VmecStatus get_status() const { return status_; }
  int get_iter1() const { return iter1_; }
  int get_iter2() const { return iter2_; }
  int get_last_preconditioner_update() const {
    return last_preconditioner_update_;
  }
  int get_last_full_update_nestor() const { return last_full_update_nestor_; }
  int get_jacob_off() { return jacob_off_; }
  // -------------------

  VmecINDATA indata_;
  Sizes s_;
  // Fourier cutoffs of the vacuum potential on the plasma's tangential grid
  Sizes vacuum_s_;
  FourierBasisFastPoloidal t_;
  Boundaries b_;
  VmecConstants constants_;
  HandoverStorage h_;
  FlowControl fc_;
  // Zero the m=1 gauge force (FourierForces::zeroZForceForM1) from the first
  // iteration instead of only once fsqz < 1e-6, and set the gauge from the
  // boundary in InitializeRadial. The converged gauge then equals the
  // boundary gauge scaled by sqrt(s) on every surface, independent of the
  // iteration and multigrid history, and the fixed-gauge force Jacobian is
  // the linearization of the iterated system.
  bool always_fix_m1_gauge_ = false;
  MGridProvider mgrid_;
  OutputQuantities output_quantities_;

  int num_threads_;
  // Thread count for the free-boundary solve is decoupled from
  // num_threads_ (which is capped at ns/2), since it's ns-independent.
  int vac_num_threads_ = 0;
  std::vector<std::unique_ptr<RadialPartitioning>> r_;
  std::vector<std::unique_ptr<ThreadLocalStorage>> ls_;
  std::vector<std::unique_ptr<RadialProfiles>> p_;
  std::vector<std::unique_ptr<FreeBoundaryBase>> fb_vac_;
  std::vector<std::unique_ptr<TangentialPartitioning>> tp_vac_;
  std::vector<std::unique_ptr<IdealMhdModel>> m_;
  std::vector<std::unique_ptr<FourierGeometry>> decomposed_x_;
  std::vector<std::unique_ptr<FourierGeometry>> physical_x_backup_;
  // decomposed_x_ as of the last valid force evaluation.
  std::vector<std::unique_ptr<FourierGeometry>> last_evaluated_x_;
  // decomposed_x_ at the start of the current multigrid step, which the step
  // restarts from when it is redone with tcon0 = 1.
  std::vector<std::unique_ptr<FourierGeometry>> step_initial_x_;
  std::vector<std::unique_ptr<FourierGeometry>> physical_x_;
  std::vector<std::unique_ptr<FourierForces>> decomposed_f_;
  std::vector<std::unique_ptr<FourierForces>> physical_f_;
  std::vector<std::unique_ptr<FourierVelocity>> decomposed_v_;

  std::vector<std::unique_ptr<FourierGeometry>> old_xc_scaled_;
  std::vector<std::unique_ptr<RadialPartitioning>> old_r_;

  Eigen::VectorXd matrixShare;
  // LU decomposition of matrixShare, shared across all vac_num_threads_
  // Nestor/LaplaceSolver instances (mirroring how matrixShare/bvecShare are
  // spans into shared backing storage). See LaplaceSolver's constructor for
  // why this must be a single object rather than a per-thread member.
  Eigen::PartialPivLU<Eigen::MatrixXd> lu_decomposition;
  Eigen::VectorXd bvecShare;
  // One row per vacuum thread for SumOverThreads, wide enough for the widest
  // sum of the vacuum team, which is the response matrix.
  Eigen::VectorXd vacuum_reduce_slots_;

 private:
  enum class SolveEqLoopStatus : std::uint8_t {
    NORMAL_TERMINATION,
    CHECKPOINT_REACHED,
    MUST_RETRY
  };

  // Inner multi-thread loop logic for SolveEquilibrium
  absl::StatusOr<SolveEqLoopStatus> SolveEquilibriumLoop(
      int thread_id, int maximum_iterations, VmecCheckpoint checkpoint,
      bool& m_lreset_internal, bool& m_liter_flag);

  // Returns the errors the threads reported, or sets status_ to
  // UNRECOVERABLE_ERROR and returns ok when every error is a physical
  // inconsistency and outputs were requested even if not converged.
  absl::Status RecoverFromThreadErrors(
      const absl::Status& status_of_all_threads,
      bool all_errors_are_recoverable);

  // Hand the iteration that just completed to iteration_callback_. Runs on
  // the master thread while the other threads wait on callback_running_.
  void NotifyIterationCallback(int iter2, RestartReason restart_reason,
                               bool& m_liter_flag);

  // The R, Z and lambda coefficients of the current equilibrium state as a
  // Geometry.
  Geometry EquilibriumState() const;

  // The fields of the last force evaluation on the half grid, gathered from
  // the threads.
  HalfGridFields HalfGridState() const;

  // The enclosed current each thread prescribes, on the whole half grid.
  Eigen::VectorXd EnclosedCurrent() const;

  // flag to enable or disable ALL screen output from VMEC++
  bool verbose_;

  // handles all formatted iteration output (progress bars or legacy table)
  IterationLogger logger_;

  // optional callback to check for interrupt signals (e.g., Ctrl+C)
  InterruptCallback interrupt_callback_;

  // set to true when the interrupt callback signals an interrupt
  bool interrupted_ = false;

  // set when SolveEquilibriumLoop hands a bad Jacobian that the axis guess did
  // not fix back to run(), which then retries from a three-surface mesh
  bool retry_from_three_surfaces_ = false;

  // bad-Jacobian restarts of the current multigrid step after its first
  // 2 * kPreconditionerUpdateInterval iterations, and whether the step has
  // been redone with tcon0 = 1
  int late_bad_jacobian_restarts_ = 0;
  bool redone_with_full_constraint_ = false;

  // optional callback that receives every force iteration
  IterationCallback iteration_callback_;

  // set to true when the iteration callback asks to stop the run
  bool stopped_by_callback_ = false;

  // the error of an iteration callback that left state.curr_h at another
  // length, which stops the run
  absl::Status callback_status_;

  // true while the master thread runs the iteration callback; the other
  // threads wait on it until it is false again
  std::atomic<bool> callback_running_{false};

  // index into ns_array of the multigrid stage being solved
  int multigrid_step_ = 0;

  // initialization state counter for Nestor. Called ivac in Fortran VMEC.
  VacuumPressureState vacuum_pressure_state_;

  // 0 if in regular multi-grid sequence;
  // 1 if have tried from scratch with intermediate ns=3, ftolv=1.0e-4
  // multi-grid step
  int jacob_off_ = 0;

  int num_eqsolve_retries_;

  // corresponds to PARVMEC's ier_flag
  VmecStatus status_;

  // the actual function evaluation count (the one that's printed on screen).
  // always increases.
  int iter2_;

  // value of iter2_ at which the state vector was restored the last time.
  // represents how many steps we are into the current optimization "branch".
  int iter1_;

  // history size for averaging of 1/tau
  static constexpr int kNDamp = 10;

  Eigen::VectorXd invTau_;

  // iter2 at last update of preconditioner update
  int last_preconditioner_update_;

  // iter2 at last full update (ivacskip = 0) of Nestor
  int last_full_update_nestor_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_VMEC_VMEC_H_
