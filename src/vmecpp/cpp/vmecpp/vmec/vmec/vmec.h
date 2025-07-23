// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_VMEC_VMEC_H_
#define VMECPP_VMEC_VMEC_VMEC_H_

#include <climits>
#include <memory>
#include <optional>
#include <utility>  // std::move
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
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

namespace vmecpp {

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

// This is the preferred way to run VMEC++.
absl::StatusOr<OutputQuantities> run(
    const VmecINDATA& indata,
    std::optional<HotRestartState> initial_state = std::nullopt,
    std::optional<int> max_threads = std::nullopt, bool verbose = true);

// This overload enables free-boundary runs with an in-memory mgrid file.
// The mgrid_file entry in `indata` will be ignored.
// This is useful e.g. to perform free-boundary hot-restarted runs where
// the coil geometry can be modified in-memory.
absl::StatusOr<OutputQuantities> run(
    const VmecINDATA& indata,
    const makegrid::MagneticFieldResponseTable& magnetic_response_table,
    std::optional<HotRestartState> initial_state = std::nullopt,
    std::optional<int> max_threads = std::nullopt, bool verbose = true);

class Vmec {
 public:
  // sign of Jacobian between cylindrical and flux coordinates
  // This is called `signgs` in Fortran VMEC.
  static constexpr int kSignOfJacobian = -1;

  // scaling factor for blending between two different ways to compute B^zeta
  static constexpr double kPDamp = 0.05;

  // VMEC++ constructor is only valid for fixed-boundary input files.
  // Free-boundary runs will complete initialization by loading the mgrid
  // contents when run is called, to have graceful error handling with absl.
  explicit Vmec(const VmecINDATA& indata,
                std::optional<int> max_threads = std::nullopt,
                bool verbose = true);

  // Mgrid loading for free-boundary VMEC++ from a precomputed response-table is
  // done outside of the Vmec constructor for improved exception handling
  absl::Status LoadMGrid(
      const makegrid::MagneticFieldResponseTable& magnetic_response_table);
  // Mgrid loading for free-boundary VMEC++ from the indata_.mgrid_file path is
  // done outside of the Vmec constructor for improved exception handling
  absl::Status LoadMGrid();

  absl::StatusOr<bool> run(
      const VmecCheckpoint& checkpoint = VmecCheckpoint::NONE,
      int iterations_before_checkpointing = INT_MAX,
      int maximum_multi_grid_step = 500,
      std::optional<HotRestartState> initial_state = std::nullopt);

  // -------------------

  bool InitializeRadial(
      VmecCheckpoint checkpoint, int maximum_iterations, int nsval, int ns_old,
      double& m_delt0,
      const std::optional<HotRestartState>& initial_state = std::nullopt);
  absl::StatusOr<bool> SolveEquilibrium(VmecCheckpoint checkpoint,
                                        int maximum_iterations);
  void RestartIteration(double& m_delt0r, int thread_id);
  absl::StatusOr<bool> Evolve(VmecCheckpoint checkpoint, int maximum_iterations,
                              double time_step, int thread_id,
                              bool& m_liter_flag);
  void Printout(double delt0r, int thread_id);
  absl::StatusOr<bool> UpdateForwardModel(VmecCheckpoint checkpoint,
                                          int maximum_iterations,
                                          int thread_id);
  void PerformTimeStep(double fac, double b1, double time_step, int thread_id);
  void InterpolateToNextMultigridStep(
      int ns_new, int ns_old,
      const std::vector<std::unique_ptr<RadialProfiles> >& p,
      const std::vector<std::unique_ptr<RadialPartitioning> >& r_new,
      const std::vector<std::unique_ptr<RadialPartitioning> >& r_old,
      std::vector<std::unique_ptr<FourierGeometry> >& m_x_new,
      std::vector<std::unique_ptr<FourierGeometry> >& m_x_old);
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
  FourierBasisFastPoloidal t_;
  Boundaries b_;
  VmecConstants constants_;
  HandoverStorage h_;
  FlowControl fc_;
  MGridProvider mgrid_;
  OutputQuantities output_quantities_;

  int num_threads_;
  std::vector<std::unique_ptr<RadialPartitioning> > r_;
  std::vector<std::unique_ptr<ThreadLocalStorage> > ls_;
  std::vector<std::unique_ptr<RadialProfiles> > p_;
  std::vector<std::unique_ptr<FreeBoundaryBase> > fb_;
  std::vector<std::unique_ptr<TangentialPartitioning> > tp_;
  std::vector<std::unique_ptr<IdealMhdModel> > m_;
  std::vector<std::unique_ptr<FourierGeometry> > decomposed_x_;
  std::vector<std::unique_ptr<FourierGeometry> > physical_x_backup_;
  std::vector<std::unique_ptr<FourierGeometry> > physical_x_;
  std::vector<std::unique_ptr<FourierForces> > decomposed_f_;
  std::vector<std::unique_ptr<FourierForces> > physical_f_;
  std::vector<std::unique_ptr<FourierVelocity> > decomposed_v_;

  std::vector<double> sj;
  std::vector<int> js1;
  std::vector<int> js2;
  std::vector<double> s1;
  std::vector<double> xint;
  std::vector<std::unique_ptr<FourierGeometry> > old_xc_scaled_;
  std::vector<std::unique_ptr<RadialPartitioning> > old_r_;

  std::vector<double> matrixShare;
  std::vector<int> iPiv;
  std::vector<double> bvecShare;

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

  // flag to enable or disable ALL screen output from VMEC++
  bool verbose_;

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

  std::vector<double> invTau_;

  // iter2 at last update of preconditioner update
  int last_preconditioner_update_;

  // iter2 at last full update (ivacskip = 0) of Nestor
  int last_full_update_nestor_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_VMEC_VMEC_H_
