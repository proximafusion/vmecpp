// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <pybind11/eigen.h>     // to wrap Eigen matrices
#include <pybind11/iostream.h>  // py::add_ostream_redirect
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // to wrap std::vector
#include <pybind11/stl/filesystem.h>

#include <Eigen/Dense>
#include <array>
#include <cstdio>   // FILE, fopen, fwrite, fread, fclose, remove
#include <cstdlib>  // setenv, unsetenv, getenv
#include <cstring>  // memcpy
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move
#include <vector>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/os_compat.h"  // setenv/unsetenv/getpid/OsTmpDir
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

#ifdef VMECPP_USE_CUDA
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda.h"
#ifdef VMECPP_USE_CUDA
// Defined in fft_toroidal_cuda_state.cu; forward-declared here (not in the
// widely included CUDA header) to avoid a full rebuild. Per-config normalized
// {fsqr,fsqz,fsql} snapshot used to report batched per-config convergence.
namespace vmecpp {
const std::vector<double> &GetFsqrPerCfgCache();
}  // namespace vmecpp
#endif
#endif  // VMECPP_USE_CUDA

namespace py = pybind11;
using Eigen::VectorXd;
using Eigen::VectorXi;
using pybind11::literals::operator""_a;
using vmecpp::VmecINDATA;

namespace {

// Add a property that gets/sets an Eigen data members to a Pybind11 wrapper.
// Simply using e.g. def_readwrite("mat", &WOutFileContents::mat) does
// not work because, under the hood, def_readwrite casts the data member to
// const before returning it from the getter (so the "write" part of
// "readwrite" refers to the data member itself, but not its contents). The
// getter function added here instead allows modification of the matrix or
// vector elements themselves.
//
// Use as: def_eigen_property(pywrapperclass, "rmnc", &WOutFileContents::rmnc);
template <typename PywrapperClass, typename EigenMatrix, typename Class>
void DefEigenProperty(PywrapperClass &pywrapper, const std::string &name,
                      EigenMatrix Class::*member_ptr) {
  static_assert(std::is_same_v<EigenMatrix, vmecpp::RowMatrixXd> ||
                std::is_same_v<EigenMatrix, Eigen::VectorXd> ||
                std::is_same_v<EigenMatrix, Eigen::VectorXi>);
  // similar to what pybind11's def_readwrite does, but returning a non-const
  // value from the getter
  auto getter = [member_ptr](Class &obj) -> EigenMatrix & {
    return obj.*member_ptr;
  };
  auto setter = [member_ptr](Class &obj, const EigenMatrix &val) {
    obj.*member_ptr = val;
  };
  pywrapper.def_property(name.c_str(), getter, setter);
}

template <typename T>
T &GetValueOrThrow(absl::StatusOr<T> &s) {
  if (!s.ok()) {
    // Could handle more exceptions, but only some are translated to meaningful
    // python exception types.
    // https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
    if (absl::IsInvalidArgument(s.status())) {
      throw pybind11::attribute_error(std::string(s.status().message()));
    } else {
      throw std::runtime_error(std::string(s.status().message()));
    }
  }
  return s.value();
}

vmecpp::HotRestartState MakeHotRestartState(vmecpp::WOutFileContents wout,
                                            const vmecpp::VmecINDATA &indata) {
  return vmecpp::HotRestartState(std::move(wout), indata);
}

// The active R/Z/lambda Fourier-component spans of a FourierGeometry, in a
// fixed canonical order set by (lthreed, lasym). Used to (un)flatten the VMEC++
// decision vector for Python. FourierForces shares the same underlying
// FourierCoeffs storage layout, so the same ordering applies.
std::vector<std::span<double>> ActiveSpans(vmecpp::FourierGeometry &x,
                                           const vmecpp::Sizes &s) {
  std::vector<std::span<double>> out = {x.rmncc};
  if (s.lthreed) out.push_back(x.rmnss);
  if (s.lasym) out.push_back(x.rmnsc);
  if (s.lasym && s.lthreed) out.push_back(x.rmncs);
  out.push_back(x.zmnsc);
  if (s.lthreed) out.push_back(x.zmncs);
  if (s.lasym) out.push_back(x.zmncc);
  if (s.lasym && s.lthreed) out.push_back(x.zmnss);
  out.push_back(x.lmnsc);
  if (s.lthreed) out.push_back(x.lmncs);
  if (s.lasym) out.push_back(x.lmncc);
  if (s.lasym && s.lthreed) out.push_back(x.lmnss);
  return out;
}

std::vector<std::span<double>> ActiveSpans(vmecpp::FourierForces &x,
                                           const vmecpp::Sizes &s) {
  std::vector<std::span<double>> out = {x.frcc};
  if (s.lthreed) out.push_back(x.frss);
  if (s.lasym) out.push_back(x.frsc);
  if (s.lasym && s.lthreed) out.push_back(x.frcs);
  out.push_back(x.fzsc);
  if (s.lthreed) out.push_back(x.fzcs);
  if (s.lasym) out.push_back(x.fzcc);
  if (s.lasym && s.lthreed) out.push_back(x.fzss);
  out.push_back(x.flsc);
  if (s.lthreed) out.push_back(x.flcs);
  if (s.lasym) out.push_back(x.flcc);
  if (s.lasym && s.lthreed) out.push_back(x.flss);
  return out;
}

template <typename FourierObject>
Eigen::VectorXd FlattenActive(FourierObject &x, const vmecpp::Sizes &s) {
  const std::vector<std::span<double>> spans = ActiveSpans(x, s);
  Eigen::Index total = 0;
  for (const auto &sp : spans) total += static_cast<Eigen::Index>(sp.size());
  Eigen::VectorXd out(total);
  Eigen::Index offset = 0;
  for (const auto &sp : spans) {
    const Eigen::Index n = static_cast<Eigen::Index>(sp.size());
    out.segment(offset, n) = Eigen::Map<const Eigen::VectorXd>(sp.data(), n);
    offset += n;
  }
  return out;
}

template <typename FourierObject>
void UnflattenActive(FourierObject &m_x, const vmecpp::Sizes &s,
                     const Eigen::VectorXd &flat) {
  const std::vector<std::span<double>> spans = ActiveSpans(m_x, s);
  Eigen::Index total = 0;
  for (const auto &sp : spans) total += static_cast<Eigen::Index>(sp.size());
  if (flat.size() != total) {
    throw std::runtime_error(
        "VmecModel.set_state: state vector has wrong length (got " +
        std::to_string(flat.size()) + ", expected " + std::to_string(total) +
        ")");
  }
  Eigen::Index offset = 0;
  for (auto &sp : spans) {
    const Eigen::Index n = static_cast<Eigen::Index>(sp.size());
    Eigen::Map<Eigen::VectorXd>(sp.data(), n) = flat.segment(offset, n);
    offset += n;
  }
}

// Single-resolution, single-threaded VMEC++ iteration model.
//
// Exposes the VMEC++ forward model (flux-surface geometry -> MHD forces) and
// the low-level time-step / restart primitives, so the equilibrium iteration
// ("time stepping" toward force balance) can be driven from Python. The
// expensive forward model and the per-step Fourier-coefficient arithmetic stay
// in C++; the iteration *logic* (damping, time-step control, restart decisions,
// convergence test) is owned by the Python caller. See vmecpp._iteration.
class VmecModel {
 public:
  explicit VmecModel(std::unique_ptr<vmecpp::Vmec> vmec)
      : vmec_(std::move(vmec)) {}

  // Build a single-threaded Vmec, initialized at a single radial resolution
  // (the inner solve VMEC++ performs at one multi-grid step). The Python loop
  // owns the multi-grid sequencing.
  static std::unique_ptr<VmecModel> Create(
      const VmecINDATA &indata, int ns,
      const std::optional<vmecpp::HotRestartState> &initial_state) {
    auto vmec_or = vmecpp::Vmec::FromIndata(
        indata, /*magnetic_response_table=*/nullptr, /*max_threads=*/1,
        vmecpp::OutputMode::kSilent);
    if (!vmec_or.ok()) {
      throw std::runtime_error(std::string(vmec_or.status().message()));
    }
    auto model = std::make_unique<VmecModel>(std::move(vmec_or.value()));
    vmecpp::Vmec &v = *model->vmec_;

    // Mirror the per-multi-grid-step setup that Vmec::run performs before
    // SolveEquilibrium (vmec.cc), for a single ns value.
    v.fc_.ns_old = 0;
    v.fc_.delt0r = v.indata_.delt;
    v.fc_.ns_min = 3;
    v.fc_.nsval = ns;

    // ftol/niter for this resolution: use the ns_array entry matching ns, else
    // the last entry of the arrays.
    const Eigen::VectorXi &ns_array = v.indata_.ns_array;
    int idx = static_cast<int>(ns_array.size()) - 1;
    for (int i = 0; i < ns_array.size(); ++i) {
      if (ns_array[i] == ns) {
        idx = i;
        break;
      }
    }
    if (idx >= 0 && idx < v.indata_.ftol_array.size()) {
      v.fc_.ftolv = v.indata_.ftol_array[idx];
    }
    if (idx >= 0 && idx < v.indata_.niter_array.size()) {
      v.fc_.niterv = v.indata_.niter_array[idx];
    }

    double delt0 = v.indata_.delt;
    v.InitializeRadial(vmecpp::VmecCheckpoint::NONE, INT_MAX, ns,
                       /*ns_old=*/0, delt0, initial_state);
    return model;
  }

  // Forward model: evaluate the MHD forces for the current geometry at the
  // given iteration counters. The preconditioner-update schedule keys on
  // (iter2 - iter1) (IdealMhdModel::shouldUpdateRadialPreconditioner), so the
  // Python loop must pass its own counters here, exactly as the C++
  // Vmec::SolveEquilibriumLoop advances iter1_/iter2_. After this call the
  // residual members (fsqr/fsqz/fsql, fsqr1/fsqz1/fsql1) reflect the current
  // decision vector. This is the body of Vmec::UpdateForwardModel with
  // caller-supplied counters; for free-boundary runs the caller should react to
  // `need_restart` (a vacuum-activation restart).
  // When precondition is true (default) this runs the full forward model and
  // leaves decomposed_f holding the preconditioned search direction, exactly as
  // the native solver uses it. When false, the forward model returns at the
  // INVARIANT_RESIDUALS checkpoint (vmec.cc line ~836), so decomposed_f holds
  // the raw, unpreconditioned force: the gradient of VMEC's augmented
  // Lagrangian with respect to the (decomposed) state, including the
  // lambda-constraint components. That raw gradient is what gradient-based
  // optimizers minimizing the MHD energy functional need; mhd_energy is already
  // set earlier in update(), so it is valid at the checkpoint too.
  void Evaluate(int iter1, int iter2, bool precondition = true) {
    ++force_eval_count_;
    bool need_restart = false;
    std::string error_message;
    const vmecpp::VmecCheckpoint checkpoint =
        precondition ? vmecpp::VmecCheckpoint::NONE
                     : vmecpp::VmecCheckpoint::INVARIANT_RESIDUALS;
    const int checkpoint_after = precondition ? INT_MAX : 0;
    // Clear the restart reason before evaluating the forward model, exactly as
    // Vmec::Evolve does at its start (vmec.cc): the forward model only *sets* a
    // reason (BAD_JACOBIAN when the Jacobian flips, HUGE_INITIAL_FORCES at
    // iter2 == 1), it never clears one, so without this reset a single bad
    // iteration's reason would stick to every later evaluation and poison the
    // caller's time-step / restart control.
    vmec_->fc_.restart_reason = vmecpp::RestartReason::NO_RESTART;
    // Run inside a single-thread OpenMP parallel region so the omp single /
    // barrier directives inside IdealMhdModel::update have the team context
    // they get in Vmec::SolveEquilibrium. Orphaned directives (outside any
    // parallel region) are not well-defined and give inconsistent results for
    // some configurations (e.g. ncurr=1).
#ifdef _OPENMP
#pragma omp parallel num_threads(1)
#endif
    {
      auto s = vmec_->m_[0]->update(
          *vmec_->decomposed_x_[0], *vmec_->physical_x_[0],
          *vmec_->decomposed_f_[0], *vmec_->physical_f_[0], need_restart,
          last_preconditioner_update_, last_full_update_nestor_, vmec_->fc_,
          iter1, iter2, checkpoint, checkpoint_after,
          /*verbose=*/false);
      if (!s.ok()) {
        error_message = std::string(s.status().message());
      }
    }
    if (!error_message.empty()) {
      throw std::runtime_error(error_message);
    }
    last_need_restart_ = need_restart;
  }
  bool need_restart() const { return last_need_restart_; }

  // Total forward-model (force) evaluations since construction or the last
  // reset. Counts every Evaluate, including those inside hessian_vector_product
  // and preconditioner assembly, for a fair cross-optimizer cost comparison.
  long force_eval_count() const { return force_eval_count_; }
  void reset_force_eval_count() { force_eval_count_ = 0; }

  // The Garabedian-style time step (PerformTimeStep): for each Fourier
  // coefficient, v = velocity_scale*(conjugation*v + dt*force); x += dt*v.
  void PerformTimeStep(double velocity_scale, double conjugation_parameter,
                       double time_step) const {
    vmec_->PerformTimeStep(velocity_scale, conjugation_parameter, time_step,
                           /*thread_id=*/0);
  }

  // Restart primitives (decomposed RestartIteration).
  void SaveBackup() const {
    *vmec_->physical_x_backup_[0] = *vmec_->decomposed_x_[0];
  }
  void RestoreBackup() const {
    vmec_->decomposed_v_[0]->setZero();
    *vmec_->decomposed_x_[0] = *vmec_->physical_x_backup_[0];
  }
  void ZeroVelocity() const { vmec_->decomposed_v_[0]->setZero(); }

  // Reset to the (possibly re-guessed) initial profile; used on bad Jacobian.
  void ResetToInitialGuess() const {
    vmec_->decomposed_x_[0]->setZero();
    vmec_->decomposed_x_[0]->interpFromBoundaryAndAxis(vmec_->t_, vmec_->b_,
                                                       *vmec_->p_[0]);
  }
  void RecomputeAxis() const {
    vmec_->b_.RecomputeMagneticAxisToFixJacobianSign(
        vmec_->fc_.nsval, vmecpp::Vmec::kSignOfJacobian);
  }

  // Recompute the magnetic axis to fix the Jacobian sign, then re-initialize
  // the radial state from the new axis. This is the C++ axis reguess in
  // Vmec::SolveEquilibriumLoop, but it re-runs InitializeRadial so the forward
  // model (preconditioner + scratch) starts clean; the in-place reset alone
  // leaves stale forward-model state from the failed evaluation, which the C++
  // single continuous parallel region tolerates but a step-by-step driver does
  // not.
  void Reinitialize() {
    vmec_->b_.RecomputeMagneticAxisToFixJacobianSign(
        vmec_->fc_.nsval, vmecpp::Vmec::kSignOfJacobian);
    double delt0 = vmec_->indata_.delt;
    // Vmec::run resets the accumulated constants before every
    // InitializeRadial (the rmsPhiP -> lamscale accumulation in
    // evalRadialProfiles is additive); without this reset a second
    // InitializeRadial doubles rmsPhiP and rescales the entire lambda
    // sector by sqrt(2).
    vmec_->constants_.reset();
    vmec_->InitializeRadial(vmecpp::VmecCheckpoint::NONE, INT_MAX,
                            vmec_->fc_.nsval, /*ns_old=*/0, delt0,
                            std::nullopt);
    last_preconditioner_update_ = 0;
    last_full_update_nestor_ = 0;
  }

  // Advance to the next (finer) multi-grid resolution, interpolating the
  // currently-converged geometry onto the new ns grid as the initial guess.
  //
  // This is the per-multi-grid-step setup Vmec::run performs between inner
  // solves (vmec.cc lines 680-681 then InitializeRadial with ns_old != 0):
  // record the current ns as ns_old so InitializeRadial's `linterp` path runs
  // the C++ radial interpolation (InterpolateToNextMultigridStep -- linear in
  // s, odd-m extrapolated to the axis) on the converged decomposed_x_, and pick
  // up the ftol/niter for the new resolution. The Python loop owns the
  // multi-grid sequencing; call this between solve_equilibrium calls to drive
  // the coarse->fine ramp from Python. `new_ns` must be finer than the current
  // ns (multi-grid only refines).
  void RefineTo(int new_ns) {
    vmecpp::Vmec &v = *vmec_;
    if (new_ns <= v.fc_.ns) {
      throw std::runtime_error("VmecModel.refine_to: new_ns (" +
                               std::to_string(new_ns) +
                               ") must be finer than the current ns (" +
                               std::to_string(v.fc_.ns) + ")");
    }

    // Mark the current (converged) resolution as the coarse grid to interpolate
    // from -- this is what run() does at the bottom of each multi-grid step.
    const int ns_old = v.fc_.ns;
    v.fc_.ns_old = ns_old;
    v.fc_.neqs_old = v.fc_.neqs;

    // ftol/niter for the new resolution: the ns_array entry matching new_ns,
    // else the last entry (mirrors Create()).
    const Eigen::VectorXi &ns_array = v.indata_.ns_array;
    int idx = static_cast<int>(ns_array.size()) - 1;
    for (int i = 0; i < ns_array.size(); ++i) {
      if (ns_array[i] == new_ns) {
        idx = i;
        break;
      }
    }
    if (idx >= 0 && idx < v.indata_.ftol_array.size()) {
      v.fc_.ftolv = v.indata_.ftol_array[idx];
    }
    if (idx >= 0 && idx < v.indata_.niter_array.size()) {
      v.fc_.niterv = v.indata_.niter_array[idx];
    }
    v.fc_.nsval = new_ns;

    double delt0 = v.indata_.delt;
    // Same per-multi-grid-step constants reset Vmec::run performs before
    // InitializeRadial (rmsPhiP accumulates in evalRadialProfiles).
    v.constants_.reset();
    v.InitializeRadial(vmecpp::VmecCheckpoint::NONE, INT_MAX, new_ns, ns_old,
                       delt0, std::nullopt);
    last_preconditioner_update_ = 0;
    last_full_update_nestor_ = 0;
  }

  // Reference C++ inner iteration (the loop being ported), for verification.
  void Solve() const {
    auto s = vmec_->SolveEquilibrium(vmecpp::VmecCheckpoint::NONE, INT_MAX);
    if (!s.ok()) {
      throw std::runtime_error(std::string(s.status().message()));
    }
  }

  // Flat decision vector (decomposed, i.e. preconditioner-scaled coefficients).
  Eigen::VectorXd GetState() const {
    return FlattenActive(*vmec_->decomposed_x_[0], vmec_->s_);
  }
  void SetState(const Eigen::VectorXd &flat) const {
    UnflattenActive(*vmec_->decomposed_x_[0], vmec_->s_, flat);
  }
  // Flat force vector (decomposed/preconditioned), valid after Evaluate().
  Eigen::VectorXd GetForces() const {
    return FlattenActive(*vmec_->decomposed_f_[0], vmec_->s_);
  }

  // Hessian-vector product of VMEC's augmented functional, computed inside
  // VMEC++ by a central directional derivative of the analytic force (which is
  // the gradient): H v = (F(x + eps v) - F(x - eps v)) / (2 eps), in the
  // decomposed internal basis. This is the matrix-free Hessian information an
  // internal or external Newton-Krylov solver needs; F itself is exact, so only
  // the directional step is finite-differenced. The current state is restored.
  Eigen::VectorXd HessianVectorProduct(const Eigen::VectorXd &v,
                                       double eps_rel = 1e-7) {
    const Eigen::VectorXd x =
        FlattenActive(*vmec_->decomposed_x_[0], vmec_->s_);
    const double vnorm = v.norm();
    if (vnorm == 0.0) {
      return Eigen::VectorXd::Zero(x.size());
    }
    const double eps = eps_rel * (1.0 + x.norm()) / vnorm;
    UnflattenActive(*vmec_->decomposed_x_[0], vmec_->s_, x + eps * v);
    Evaluate(2, 2, /*precondition=*/false);
    const Eigen::VectorXd fp =
        FlattenActive(*vmec_->decomposed_f_[0], vmec_->s_);
    UnflattenActive(*vmec_->decomposed_x_[0], vmec_->s_, x - eps * v);
    Evaluate(2, 2, /*precondition=*/false);
    const Eigen::VectorXd fm =
        FlattenActive(*vmec_->decomposed_f_[0], vmec_->s_);
    UnflattenActive(*vmec_->decomposed_x_[0], vmec_->s_, x);
    return (fp - fm) / (2.0 * eps);
  }

  // Apply VMEC's preconditioner M^-1 to a vector in the decomposed internal
  // basis, mirroring the native apply sequence (m=1, radial, lambda). This is
  // VMEC's hand-built approximate inverse Hessian; gradient-based solvers use
  // it as the metric (preconditioned Krylov / quasi-Newton, and as the
  // preconditioner for the Hessian solve in adjoint sensitivities).
  //
  // Requires a prior evaluate(precondition=true) at the current state: the
  // radial preconditioner is assembled inside that forward-model call.
  Eigen::VectorXd ApplyPreconditioner(const Eigen::VectorXd &v) const {
    vmecpp::FourierForces tmp(&vmec_->s_, vmec_->r_[0].get(), vmec_->fc_.ns);
    tmp.setZero();
    UnflattenActive(tmp, vmec_->s_, v);
    vmecpp::IdealMhdModel &model = *vmec_->m_[0];
    model.applyM1Preconditioner(tmp);
    const absl::Status status = model.applyRZPreconditioner(tmp);
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }
    model.applyLambdaPreconditioner(tmp);
    return FlattenActive(tmp, vmec_->s_);
  }

  // Residuals (set by Evaluate()): invariant {fsqr,fsqz,fsql} and
  // preconditioned {fsqr1,fsqz1,fsql1}.
  double fsqr() const { return vmec_->fc_.fsqr; }
  double fsqz() const { return vmec_->fc_.fsqz; }
  double fsql() const { return vmec_->fc_.fsql; }
  double fsqr1() const { return vmec_->fc_.fsqr1; }
  double fsqz1() const { return vmec_->fc_.fsqz1; }
  double fsql1() const { return vmec_->fc_.fsql1; }
  double mhd_energy() const { return vmec_->h_.mhdEnergy; }

  int restart_reason() const {
    return static_cast<int>(vmec_->fc_.restart_reason);
  }
  void set_restart_reason(int reason) const {
    vmec_->fc_.restart_reason = vmecpp::RestartReasonFromInt(reason);
  }
  int status() const { return static_cast<int>(vmec_->get_status()); }

  double ftolv() const { return vmec_->fc_.ftolv; }
  int niterv() const { return vmec_->fc_.niterv; }
  double delt() const { return vmec_->indata_.delt; }
  // Iteration style ("vmec_8_52" or "parvmec"); selects the time-step control
  // variant in vmecpp._iteration.
  std::string iteration_style() const {
    return vmecpp::ToString(vmec_->indata_.iteration_style);
  }
  int ns() const { return vmec_->fc_.ns; }
  int mpol() const { return vmec_->s_.mpol; }
  int ntor() const { return vmec_->s_.ntor; }
  bool lthreed() const { return vmec_->s_.lthreed; }
  bool lasym() const { return vmec_->s_.lasym; }

  // Invariant force-residual traces recorded during the C++ Solve().
  std::vector<double> force_residual_r() const {
    return vmec_->fc_.force_residual_r;
  }
  std::vector<double> force_residual_z() const {
    return vmec_->fc_.force_residual_z;
  }
  std::vector<double> force_residual_lambda() const {
    return vmec_->fc_.force_residual_lambda;
  }
  // Per-iteration restart-reason trace recorded alongside the residual traces
  // (one entry per recorded force iteration); NO_RESTART=1, BAD_JACOBIAN=2,
  // BAD_PROGRESS=3, HUGE_INITIAL_FORCES=4.
  std::vector<int> restart_reasons() const {
    std::vector<int> out;
    out.reserve(vmec_->fc_.restart_reasons.size());
    for (const auto &r : vmec_->fc_.restart_reasons) {
      out.push_back(static_cast<int>(r));
    }
    return out;
  }

  int ijacob() const { return vmec_->fc_.ijacob; }
  Eigen::VectorXd raxis_c() const { return vmec_->b_.raxis_c; }
  static bool openmp_enabled() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
  }

  std::unique_ptr<vmecpp::Vmec> vmec_;

  // Preconditioner / Nestor update bookkeeping, mirroring the like-named Vmec
  // members; the Python loop drives the iteration counters via Evaluate, so the
  // wrapper keeps these running values across calls.
  int last_preconditioner_update_ = 0;
  int last_full_update_nestor_ = 0;
  bool last_need_restart_ = false;
  long force_eval_count_ = 0;
};

}  // anonymous namespace

// IMPORTANT: The first argument must be the name of the module, else
// compilation will succeed but import will fail with:
//     ImportError: dynamic module does not define module export function
//     (PyInit_example)
PYBIND11_MODULE(_vmecpp, m) {
  m.doc() = "pybind11 VMEC++ plugin";

  // C++ stdout and stderr cannot easily be captured or redirected from Python.
  // This adds a Python context manager that can be used to redirect them like
  // this:
  //
  // with _vmecpp.ostream_redirect(stdout=True, stderr=True):
  //   _vmecpp.run(indata, max_thread=1) # or some other noisy function
  //
  // WARNING: Pybind11's C++ iostream redirection is thread-unsafe and does not
  // play well with OpenMP: only use it with max_thread=1 or OMP_NUM_THREADS=1!
  py::add_ostream_redirect(m, "ostream_redirect");

  auto pyindata =
      py::class_<VmecINDATA>(m, "VmecINDATA")
          .def(py::init<>())
          .def("_set_mpol_ntor", &VmecINDATA::SetMpolNtor, py::arg("new_mpol"),
               py::arg("new_ntor"))
          .def("from_file", &VmecINDATA::FromFile)
          .def("from_json", &VmecINDATA::FromJson)
          .def("to_json", &VmecINDATA::ToJsonOrException)
          .def("copy", &VmecINDATA::Copy)

          // numerical resolution, symmetry assumption
          .def_readwrite("lasym", &VmecINDATA::lasym)
          .def_readwrite("nfp", &VmecINDATA::nfp)
          .def_readonly("mpol", &VmecINDATA::mpol)  // readonly!
          .def_readonly("ntor", &VmecINDATA::ntor)  // readonly!
          .def_readwrite("ntheta", &VmecINDATA::ntheta)
          .def_readwrite("nzeta", &VmecINDATA::nzeta)
          .def_readwrite("mpol_geometry", &VmecINDATA::mpol_geometry)
          .def_readwrite("ntor_geometry", &VmecINDATA::ntor_geometry);

  // multi-grid steps
  DefEigenProperty(pyindata, "ns_array", &VmecINDATA::ns_array);
  DefEigenProperty(pyindata, "ftol_array", &VmecINDATA::ftol_array);
  DefEigenProperty(pyindata, "niter_array", &VmecINDATA::niter_array);

  // global physics parameters
  pyindata.def_readwrite("phiedge", &VmecINDATA::phiedge)
      .def_readwrite("ncurr", &VmecINDATA::ncurr)

      // mass / pressure profile
      .def_readwrite("pmass_type", &VmecINDATA::pmass_type);
  // fully read-write
  DefEigenProperty(pyindata, "am", &VmecINDATA::am);
  DefEigenProperty(pyindata, "am_aux_s", &VmecINDATA::am_aux_s);
  DefEigenProperty(pyindata, "am_aux_f", &VmecINDATA::am_aux_f);
  pyindata.def_readwrite("pres_scale", &VmecINDATA::pres_scale)
      .def_readwrite("gamma", &VmecINDATA::gamma)
      .def_readwrite("spres_ped", &VmecINDATA::spres_ped)

      // (initial guess for) iota profile
      .def_readwrite("piota_type", &VmecINDATA::piota_type);
  DefEigenProperty(pyindata, "ai", &VmecINDATA::ai);
  DefEigenProperty(pyindata, "ai_aux_s", &VmecINDATA::ai_aux_s);
  DefEigenProperty(pyindata, "ai_aux_f", &VmecINDATA::ai_aux_f);

  // enclosed toroidal current profile
  pyindata.def_readwrite("pcurr_type", &VmecINDATA::pcurr_type);
  DefEigenProperty(pyindata, "ac", &VmecINDATA::ac);
  DefEigenProperty(pyindata, "ac_aux_s", &VmecINDATA::ac_aux_s);
  DefEigenProperty(pyindata, "ac_aux_f", &VmecINDATA::ac_aux_f);
  pyindata.def_readwrite("curtor", &VmecINDATA::curtor)
      .def_readwrite("bloat", &VmecINDATA::bloat)

      // free-boundary parameters
      .def_readwrite("lfreeb", &VmecINDATA::lfreeb)
      .def_readwrite("mgrid_file", &VmecINDATA::mgrid_file);
  DefEigenProperty(pyindata, "extcur", &VmecINDATA::extcur);
  pyindata.def_readwrite("nvacskip", &VmecINDATA::nvacskip)
      .def_readwrite("free_boundary_method", &VmecINDATA::free_boundary_method)

      // tweaking parameters
      .def_readwrite("nstep", &VmecINDATA::nstep);
  DefEigenProperty(pyindata, "aphi", &VmecINDATA::aphi);
  pyindata.def_readwrite("delt", &VmecINDATA::delt)
      .def_readwrite("tcon0", &VmecINDATA::tcon0)
      .def_readwrite("lforbal", &VmecINDATA::lforbal)
      .def_readwrite("iteration_style", &VmecINDATA::iteration_style)
      .def_readwrite("return_outputs_even_if_not_converged",
                     &VmecINDATA::return_outputs_even_if_not_converged)

      // initial guess for magnetic axis
      // disallow re-assignment of the whole vector (to preserve sizes
      // consistent with mpol/ntor) but allow changing the individual elements
      .def_property_readonly(
          "raxis_c", [](VmecINDATA &w) -> VectorXd & { return w.raxis_c; })
      .def_property_readonly(
          "zaxis_s", [](VmecINDATA &w) -> VectorXd & { return w.zaxis_s; })
      .def_property_readonly(
          "raxis_s",
          [](VmecINDATA &w) -> std::optional<VectorXd> & { return w.raxis_s; })
      .def_property_readonly(
          "zaxis_c",
          [](VmecINDATA &w) -> std::optional<VectorXd> & { return w.zaxis_c; })

      // (initial guess for) boundary shape
      // disallow re-assignment of the whole matrix (to preserve shapes
      // consistent with mpol/ntor) but allow changing the individual elements
      .def_property_readonly(
          "rbc", [](VmecINDATA &w) -> vmecpp::RowMatrixXd & { return w.rbc; })
      .def_property_readonly(
          "zbs", [](VmecINDATA &w) -> vmecpp::RowMatrixXd & { return w.zbs; })
      .def_property_readonly(
          "rbs",
          [](VmecINDATA &w) -> std::optional<vmecpp::RowMatrixXd> & {
            return w.rbs;
          })
      .def_property_readonly(
          "zbc", [](VmecINDATA &w) -> std::optional<vmecpp::RowMatrixXd> & {
            return w.zbc;
          });

  py::native_enum<vmecpp::FreeBoundaryMethod>(m, "FreeBoundaryMethod",
                                              "enum.Enum")
      .value("NESTOR", vmecpp::FreeBoundaryMethod::NESTOR)
      .value("ONLY_COILS", vmecpp::FreeBoundaryMethod::ONLY_COILS)
      .value("BIEST", vmecpp::FreeBoundaryMethod::BIEST)
      .export_values()
      .finalize();

  py::native_enum<vmecpp::OutputMode>(m, "OutputMode", "enum.IntEnum")
      .value("SILENT", vmecpp::OutputMode::kSilent)
      .value("LEGACY", vmecpp::OutputMode::kLegacy)
      .value("PROGRESS", vmecpp::OutputMode::kProgress)
      .value("PROGRESS_NON_TTY", vmecpp::OutputMode::kProgressNonTTY)
      .export_values()
      .finalize();

  py::native_enum<vmecpp::IterationStyle>(m, "IterationStyle", "enum.Enum")
      .value("VMEC_8_52", vmecpp::IterationStyle::VMEC_8_52)
      .value("PARVMEC", vmecpp::IterationStyle::PARVMEC)
      .export_values()
      .finalize();

  py::class_<vmecpp::VmecCheckpoint>(m, "VmecCheckpoint");

  py::class_<vmecpp::JxBOutFileContents>(m, "JxBOutFileContents")
      .def_readonly("itheta", &vmecpp::JxBOutFileContents::itheta)
      .def_readonly("izeta", &vmecpp::JxBOutFileContents::izeta)
      .def_readonly("bdotk", &vmecpp::JxBOutFileContents::bdotk)
      //
      .def_readonly("amaxfor", &vmecpp::JxBOutFileContents::amaxfor)
      .def_readonly("aminfor", &vmecpp::JxBOutFileContents::aminfor)
      .def_readonly("avforce", &vmecpp::JxBOutFileContents::avforce)
      .def_readonly("pprim", &vmecpp::JxBOutFileContents::pprim)
      .def_readonly("jdotb", &vmecpp::JxBOutFileContents::jdotb)
      .def_readonly("bdotb", &vmecpp::JxBOutFileContents::bdotb)
      .def_readonly("bdotgradv", &vmecpp::JxBOutFileContents::bdotgradv)
      .def_readonly("jpar2", &vmecpp::JxBOutFileContents::jpar2)
      .def_readonly("jperp2", &vmecpp::JxBOutFileContents::jperp2)
      .def_readonly("phin", &vmecpp::JxBOutFileContents::phin)
      //
      .def_readonly("jsupu3", &vmecpp::JxBOutFileContents::jsupu3)
      .def_readonly("jsupv3", &vmecpp::JxBOutFileContents::jsupv3)
      .def_readonly("jsups3", &vmecpp::JxBOutFileContents::jsups3)
      .def_readonly("bsupu3", &vmecpp::JxBOutFileContents::bsupu3)
      .def_readonly("bsupv3", &vmecpp::JxBOutFileContents::bsupv3)
      .def_readonly("jcrossb", &vmecpp::JxBOutFileContents::jcrossb)
      .def_readonly("jxb_gradp", &vmecpp::JxBOutFileContents::jxb_gradp)
      .def_readonly("jdotb_sqrtg", &vmecpp::JxBOutFileContents::jdotb_sqrtg)
      .def_readonly("sqrtg3", &vmecpp::JxBOutFileContents::sqrtg3)
      .def_readonly("bsubu3", &vmecpp::JxBOutFileContents::bsubu3)
      .def_readonly("bsubv3", &vmecpp::JxBOutFileContents::bsubv3)
      .def_readonly("bsubs3", &vmecpp::JxBOutFileContents::bsubs3);

  py::class_<vmecpp::MercierFileContents>(m, "MercierFileContents")
      .def_readonly("s", &vmecpp::MercierFileContents::s)
      //
      .def_readonly("toroidal_flux",
                    &vmecpp::MercierFileContents::toroidal_flux)
      .def_readonly("iota", &vmecpp::MercierFileContents::iota)
      .def_readonly("shear", &vmecpp::MercierFileContents::shear)
      .def_readonly("d_volume_d_s", &vmecpp::MercierFileContents::d_volume_d_s)
      .def_readonly("well", &vmecpp::MercierFileContents::well)
      .def_readonly("toroidal_current",
                    &vmecpp::MercierFileContents::toroidal_current)
      .def_readonly("d_toroidal_current_d_s",
                    &vmecpp::MercierFileContents::d_toroidal_current_d_s)
      .def_readonly("pressure", &vmecpp::MercierFileContents::pressure)
      .def_readonly("d_pressure_d_s",
                    &vmecpp::MercierFileContents::d_pressure_d_s)
      //
      .def_readonly("DMerc", &vmecpp::MercierFileContents::DMerc)
      .def_readonly("Dshear", &vmecpp::MercierFileContents::Dshear)
      .def_readonly("Dwell", &vmecpp::MercierFileContents::Dwell)
      .def_readonly("Dcurr", &vmecpp::MercierFileContents::Dcurr)
      .def_readonly("Dgeod", &vmecpp::MercierFileContents::Dgeod);

  py::class_<vmecpp::Threed1FirstTable>(m, "Threed1FirstTable")
      .def_readonly("s", &vmecpp::Threed1FirstTable::s)
      .def_readonly("radial_force", &vmecpp::Threed1FirstTable::radial_force)
      .def_readonly("toroidal_flux", &vmecpp::Threed1FirstTable::toroidal_flux)
      .def_readonly("iota", &vmecpp::Threed1FirstTable::iota)
      .def_readonly("avg_jsupu", &vmecpp::Threed1FirstTable::avg_jsupu)
      .def_readonly("avg_jsupv", &vmecpp::Threed1FirstTable::avg_jsupv)
      .def_readonly("d_volume_d_phi",
                    &vmecpp::Threed1FirstTable::d_volume_d_phi)
      .def_readonly("d_pressure_d_phi",
                    &vmecpp::Threed1FirstTable::d_pressure_d_phi)
      .def_readonly("spectral_width",
                    &vmecpp::Threed1FirstTable::spectral_width)
      .def_readonly("pressure", &vmecpp::Threed1FirstTable::pressure)
      .def_readonly("buco_full", &vmecpp::Threed1FirstTable::buco_full)
      .def_readonly("bvco_full", &vmecpp::Threed1FirstTable::bvco_full)
      .def_readonly("j_dot_b", &vmecpp::Threed1FirstTable::j_dot_b)
      .def_readonly("b_dot_b", &vmecpp::Threed1FirstTable::b_dot_b);

  py::class_<vmecpp::Threed1GeometricAndMagneticQuantities>(
      m, "Threed1GeometricAndMagneticQuantities")
      .def_readonly(
          "toroidal_flux",
          &vmecpp::Threed1GeometricAndMagneticQuantities::toroidal_flux)
      //
      .def_readonly("circum_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::circum_p)
      .def_readonly("surf_area_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::surf_area_p)
      //
      .def_readonly(
          "cross_area_p",
          &vmecpp::Threed1GeometricAndMagneticQuantities::cross_area_p)
      .def_readonly("volume_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::volume_p)
      //
      .def_readonly("Rmajor_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::Rmajor_p)
      .def_readonly("Aminor_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::Aminor_p)
      .def_readonly("aspect",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::aspect)
      //
      .def_readonly("kappa_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::kappa_p)
      .def_readonly("rcen",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rcen)
      //
      .def_readonly("aminr1",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::aminr1)
      //
      .def_readonly("pavg",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::pavg)
      .def_readonly("factor",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::factor)
      //
      .def_readonly("b0", &vmecpp::Threed1GeometricAndMagneticQuantities::b0)
      //
      .def_readonly("rmax_surf",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rmax_surf)
      .def_readonly("rmin_surf",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rmin_surf)
      .def_readonly("zmax_surf",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::zmax_surf)
      //
      .def_readonly("bmin",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::bmin)
      .def_readonly("bmax",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::bmax)
      //
      .def_readonly("waist",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::waist)
      .def_readonly("height",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::height)
      //
      .def_readonly("betapol",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::betapol)
      .def_readonly("betatot",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::betatot)
      .def_readonly("betator",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::betator)
      .def_readonly("VolAvgB",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::VolAvgB)
      .def_readonly("IonLarmor",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::IonLarmor)
      //
      .def_readonly("jpar_perp",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::jpar_perp)
      .def_readonly("jparPS_perp",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::jparPS_perp)
      //
      .def_readonly(
          "toroidal_current",
          &vmecpp::Threed1GeometricAndMagneticQuantities::toroidal_current)
      //
      .def_readonly("rbtor",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rbtor)
      .def_readonly("rbtor0",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rbtor0)
      //
      .def_readonly("psi", &vmecpp::Threed1GeometricAndMagneticQuantities::psi)
      .def_readonly("ygeo",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::ygeo)
      .def_readonly("yinden",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::yinden)
      .def_readonly("yellip",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::yellip)
      .def_readonly("ytrian",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::ytrian)
      .def_readonly("yshift",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::yshift)
      //
      .def_readonly(
          "loc_jpar_perp",
          &vmecpp::Threed1GeometricAndMagneticQuantities::loc_jpar_perp)
      .def_readonly(
          "loc_jparPS_perp",
          &vmecpp::Threed1GeometricAndMagneticQuantities::loc_jparPS_perp);

  py::class_<vmecpp::Threed1Volumetrics>(m, "Threed1Volumetrics")
      .def_readonly("int_p", &vmecpp::Threed1Volumetrics::int_p)
      .def_readonly("avg_p", &vmecpp::Threed1Volumetrics::avg_p)
      //
      .def_readonly("int_bpol", &vmecpp::Threed1Volumetrics::int_bpol)
      .def_readonly("avg_bpol", &vmecpp::Threed1Volumetrics::avg_bpol)
      //
      .def_readonly("int_btor", &vmecpp::Threed1Volumetrics::int_btor)
      .def_readonly("avg_btor", &vmecpp::Threed1Volumetrics::avg_btor)
      //
      .def_readonly("int_modb", &vmecpp::Threed1Volumetrics::int_modb)
      .def_readonly("avg_modb", &vmecpp::Threed1Volumetrics::avg_modb)
      //
      .def_readonly("int_ekin", &vmecpp::Threed1Volumetrics::int_ekin)
      .def_readonly("avg_ekin", &vmecpp::Threed1Volumetrics::avg_ekin);

  py::class_<vmecpp::Threed1AxisGeometry>(m, "Threed1AxisGeometry")
      .def_readonly("raxis_symm", &vmecpp::Threed1AxisGeometry::raxis_symm)
      .def_readonly("zaxis_symm", &vmecpp::Threed1AxisGeometry::zaxis_symm)
      .def_readonly("raxis_asym", &vmecpp::Threed1AxisGeometry::raxis_asym)
      .def_readonly("zaxis_asym", &vmecpp::Threed1AxisGeometry::zaxis_asym);

  py::class_<vmecpp::Threed1Betas>(m, "Threed1Betas")
      .def_readonly("betatot", &vmecpp::Threed1Betas::betatot)
      .def_readonly("betapol", &vmecpp::Threed1Betas::betapol)
      .def_readonly("betator", &vmecpp::Threed1Betas::betator)
      .def_readonly("rbtor", &vmecpp::Threed1Betas::rbtor)
      .def_readonly("betaxis", &vmecpp::Threed1Betas::betaxis)
      .def_readonly("betstr", &vmecpp::Threed1Betas::betstr);

  py::class_<vmecpp::Threed1ShafranovIntegrals>(m, "Threed1ShafranovIntegrals")
      .def_readonly("scaling_ratio",
                    &vmecpp::Threed1ShafranovIntegrals::scaling_ratio)
      //
      .def_readonly("r_lao", &vmecpp::Threed1ShafranovIntegrals::r_lao)
      .def_readonly("f_lao", &vmecpp::Threed1ShafranovIntegrals::f_lao)
      .def_readonly("f_geo", &vmecpp::Threed1ShafranovIntegrals::f_geo)
      //
      .def_readonly("smaleli", &vmecpp::Threed1ShafranovIntegrals::smaleli)
      .def_readonly("betai", &vmecpp::Threed1ShafranovIntegrals::betai)
      .def_readonly("musubi", &vmecpp::Threed1ShafranovIntegrals::musubi)
      .def_readonly("lambda", &vmecpp::Threed1ShafranovIntegrals::lambda)
      //
      .def_readonly("s11", &vmecpp::Threed1ShafranovIntegrals::s11)
      .def_readonly("s12", &vmecpp::Threed1ShafranovIntegrals::s12)
      .def_readonly("s13", &vmecpp::Threed1ShafranovIntegrals::s13)
      .def_readonly("s2", &vmecpp::Threed1ShafranovIntegrals::s2)
      .def_readonly("s3", &vmecpp::Threed1ShafranovIntegrals::s3)
      //
      .def_readonly("delta1", &vmecpp::Threed1ShafranovIntegrals::delta1)
      .def_readonly("delta2", &vmecpp::Threed1ShafranovIntegrals::delta2)
      .def_readonly("delta3", &vmecpp::Threed1ShafranovIntegrals::delta3);

  py::class_<vmecpp::WOutFileContents>(m, "WOutFileContents")
      .def(py::init<const vmecpp::WOutFileContents &>(), py::arg("wout"))
      .def(py::init())
      .def_readwrite("version_", &vmecpp::WOutFileContents::version_)
      .def_readwrite("input_extension",
                     &vmecpp::WOutFileContents::input_extension)
      //
      .def_readwrite("signgs", &vmecpp::WOutFileContents::signgs)
      //
      .def_readwrite("gamma", &vmecpp::WOutFileContents::gamma)
      //
      .def_readwrite("pcurr_type", &vmecpp::WOutFileContents::pcurr_type)
      .def_readwrite("pmass_type", &vmecpp::WOutFileContents::pmass_type)
      .def_readwrite("piota_type", &vmecpp::WOutFileContents::piota_type)
      //
      .def_readwrite("am", &vmecpp::WOutFileContents::am)
      .def_readwrite("ac", &vmecpp::WOutFileContents::ac)
      .def_readwrite("ai", &vmecpp::WOutFileContents::ai)
      //
      .def_readwrite("am_aux_s", &vmecpp::WOutFileContents::am_aux_s)
      .def_readwrite("am_aux_f", &vmecpp::WOutFileContents::am_aux_f)
      //
      .def_readwrite("ac_aux_s", &vmecpp::WOutFileContents::ac_aux_s)
      .def_readwrite("ac_aux_f", &vmecpp::WOutFileContents::ac_aux_f)
      //
      .def_readwrite("ai_aux_s", &vmecpp::WOutFileContents::ai_aux_s)
      .def_readwrite("ai_aux_f", &vmecpp::WOutFileContents::ai_aux_f)
      //
      .def_readwrite("nfp", &vmecpp::WOutFileContents::nfp)
      .def_readwrite("mpol", &vmecpp::WOutFileContents::mpol)
      .def_readwrite("ntor", &vmecpp::WOutFileContents::ntor)
      .def_readwrite("lasym", &vmecpp::WOutFileContents::lasym)
      .def_readwrite("lrfp", &vmecpp::WOutFileContents::lrfp)
      //
      .def_readwrite("ns", &vmecpp::WOutFileContents::ns)
      .def_readwrite("ftolv", &vmecpp::WOutFileContents::ftolv)
      .def_readwrite("niter", &vmecpp::WOutFileContents::niter)
      //
      .def_readwrite("lfreeb", &vmecpp::WOutFileContents::lfreeb)
      .def_readwrite("mgrid_file", &vmecpp::WOutFileContents::mgrid_file)
      .def_readwrite("nextcur", &vmecpp::WOutFileContents::nextcur)
      .def_readwrite("extcur", &vmecpp::WOutFileContents::extcur)
      .def_readwrite("mgrid_mode", &vmecpp::WOutFileContents::mgrid_mode)
      //
      .def_readwrite("wb", &vmecpp::WOutFileContents::wb)
      .def_readwrite("wp", &vmecpp::WOutFileContents::wp)
      //
      .def_readwrite("rmax_surf", &vmecpp::WOutFileContents::rmax_surf)
      .def_readwrite("rmin_surf", &vmecpp::WOutFileContents::rmin_surf)
      .def_readwrite("zmax_surf", &vmecpp::WOutFileContents::zmax_surf)
      //
      .def_readwrite("mnmax", &vmecpp::WOutFileContents::mnmax)
      .def_readwrite("mnmax_nyq", &vmecpp::WOutFileContents::mnmax_nyq)
      //
      .def_readwrite("ier_flag", &vmecpp::WOutFileContents::ier_flag)
      //
      .def_readwrite("aspect", &vmecpp::WOutFileContents::aspect)
      //
      .def_readwrite("betatotal", &vmecpp::WOutFileContents::betatotal)
      .def_readwrite("betapol", &vmecpp::WOutFileContents::betapol)
      .def_readwrite("betator", &vmecpp::WOutFileContents::betator)
      .def_readwrite("betaxis", &vmecpp::WOutFileContents::betaxis)
      //
      .def_readwrite("b0", &vmecpp::WOutFileContents::b0)
      //
      .def_readwrite("rbtor0", &vmecpp::WOutFileContents::rbtor0)
      .def_readwrite("rbtor", &vmecpp::WOutFileContents::rbtor)
      //
      .def_readwrite("IonLarmor", &vmecpp::WOutFileContents::IonLarmor)
      .def_readwrite("volavgB", &vmecpp::WOutFileContents::volavgB)
      //
      .def_readwrite("ctor", &vmecpp::WOutFileContents::ctor)
      //
      .def_readwrite("Aminor_p", &vmecpp::WOutFileContents::Aminor_p)
      .def_readwrite("Rmajor_p", &vmecpp::WOutFileContents::Rmajor_p)
      .def_readwrite("volume", &vmecpp::WOutFileContents::volume)
      //
      .def_readwrite("fsqr", &vmecpp::WOutFileContents::fsqr)
      .def_readwrite("fsqz", &vmecpp::WOutFileContents::fsqz)
      .def_readwrite("fsql", &vmecpp::WOutFileContents::fsql)
      .def_readwrite("itfsq", &vmecpp::WOutFileContents::itfsq)
      //
      .def_readwrite("iotaf", &vmecpp::WOutFileContents::iotaf)
      .def_readwrite("q_factor", &vmecpp::WOutFileContents::q_factor)
      .def_readwrite("presf", &vmecpp::WOutFileContents::presf)
      .def_readwrite("phi", &vmecpp::WOutFileContents::phi)
      .def_readwrite("phipf", &vmecpp::WOutFileContents::phipf)
      .def_readwrite("chi", &vmecpp::WOutFileContents::chi)
      .def_readwrite("chipf", &vmecpp::WOutFileContents::chipf)
      .def_readwrite("jcuru", &vmecpp::WOutFileContents::jcuru)
      .def_readwrite("jcurv", &vmecpp::WOutFileContents::jcurv)
      //
      .def_readwrite("force_residual_r",
                     &vmecpp::WOutFileContents::force_residual_r)
      .def_readwrite("force_residual_z",
                     &vmecpp::WOutFileContents::force_residual_z)
      .def_readwrite("force_residual_lambda",
                     &vmecpp::WOutFileContents::force_residual_lambda)
      .def_readwrite("fsqt", &vmecpp::WOutFileContents::fsqt)
      .def_readwrite("delbsq", &vmecpp::WOutFileContents::delbsq)
      .def_readwrite("wdot", &vmecpp::WOutFileContents::wdot)
      .def_readwrite("restart_reason_timetrace",
                     &vmecpp::WOutFileContents::restart_reason_timetrace)
      //
      .def_readwrite("iotas", &vmecpp::WOutFileContents::iotas)
      .def_readwrite("mass", &vmecpp::WOutFileContents::mass)
      .def_readwrite("pres", &vmecpp::WOutFileContents::pres)
      .def_readwrite("beta_vol", &vmecpp::WOutFileContents::beta_vol)
      .def_readwrite("buco", &vmecpp::WOutFileContents::buco)
      .def_readwrite("bvco", &vmecpp::WOutFileContents::bvco)
      .def_readwrite("vp", &vmecpp::WOutFileContents::vp)
      .def_readwrite("specw", &vmecpp::WOutFileContents::specw)
      .def_readwrite("phips", &vmecpp::WOutFileContents::phips)
      .def_readwrite("over_r", &vmecpp::WOutFileContents::over_r)
      //
      .def_readwrite("jdotb", &vmecpp::WOutFileContents::jdotb)
      .def_readwrite("bdotb", &vmecpp::WOutFileContents::bdotb)
      .def_readwrite("bdotgradv", &vmecpp::WOutFileContents::bdotgradv)
      //
      .def_readwrite("DMerc", &vmecpp::WOutFileContents::DMerc)
      .def_readwrite("DShear", &vmecpp::WOutFileContents::DShear)
      .def_readwrite("DWell", &vmecpp::WOutFileContents::DWell)
      .def_readwrite("DCurr", &vmecpp::WOutFileContents::DCurr)
      .def_readwrite("DGeod", &vmecpp::WOutFileContents::DGeod)
      //
      .def_readwrite("equif", &vmecpp::WOutFileContents::equif)
      //
      .def_readwrite("curlabel", &vmecpp::WOutFileContents::curlabel)
      //
      .def_readwrite("potvac", &vmecpp::WOutFileContents::potvac)
      //
      .def_readwrite("xm", &vmecpp::WOutFileContents::xm)
      .def_readwrite("xn", &vmecpp::WOutFileContents::xn)
      .def_readwrite("xm_nyq", &vmecpp::WOutFileContents::xm_nyq)
      .def_readwrite("xn_nyq", &vmecpp::WOutFileContents::xn_nyq)
      //
      .def_readwrite("raxis_cc", &vmecpp::WOutFileContents::raxis_cc)
      .def_readwrite("zaxis_cs", &vmecpp::WOutFileContents::zaxis_cs)
      //
      .def_readwrite("rmnc", &vmecpp::WOutFileContents::rmnc)
      .def_readwrite("zmns", &vmecpp::WOutFileContents::zmns)
      .def_readwrite("lmns_full", &vmecpp::WOutFileContents::lmns_full)
      .def_readwrite("lmns", &vmecpp::WOutFileContents::lmns)
      .def_readwrite("gmnc", &vmecpp::WOutFileContents::gmnc)
      .def_readwrite("bmnc", &vmecpp::WOutFileContents::bmnc)
      .def_readwrite("bsubumnc", &vmecpp::WOutFileContents::bsubumnc)
      .def_readwrite("bsubvmnc", &vmecpp::WOutFileContents::bsubvmnc)
      .def_readwrite("bsubsmns", &vmecpp::WOutFileContents::bsubsmns)
      .def_readwrite("bsubsmns_full", &vmecpp::WOutFileContents::bsubsmns_full)
      .def_readwrite("bsupumnc", &vmecpp::WOutFileContents::bsupumnc)
      .def_readwrite("bsupvmnc", &vmecpp::WOutFileContents::bsupvmnc)
      //
      .def_readwrite("currumnc", &vmecpp::WOutFileContents::currumnc)
      .def_readwrite("currvmnc", &vmecpp::WOutFileContents::currvmnc)
      //
      .def_readwrite("raxis_cs", &vmecpp::WOutFileContents::raxis_cs)
      .def_readwrite("zaxis_cc", &vmecpp::WOutFileContents::zaxis_cc)
      // non-stellarator symmetric
      .def_readwrite("rmns", &vmecpp::WOutFileContents::rmns)
      .def_readwrite("zmnc", &vmecpp::WOutFileContents::zmnc)
      .def_readwrite("lmnc_full", &vmecpp::WOutFileContents::lmnc_full)
      .def_readwrite("lmnc", &vmecpp::WOutFileContents::lmnc)
      .def_readwrite("gmns", &vmecpp::WOutFileContents::gmns)
      .def_readwrite("bmns", &vmecpp::WOutFileContents::bmns)
      .def_readwrite("bsubumns", &vmecpp::WOutFileContents::bsubumns)
      .def_readwrite("bsubvmns", &vmecpp::WOutFileContents::bsubvmns)
      .def_readwrite("bsubsmnc", &vmecpp::WOutFileContents::bsubsmnc)
      .def_readwrite("bsubsmnc_full", &vmecpp::WOutFileContents::bsubsmnc_full)
      .def_readwrite("bsupumns", &vmecpp::WOutFileContents::bsupumns)
      .def_readwrite("bsupvmns", &vmecpp::WOutFileContents::bsupvmns)
      //
      .def_readwrite("currumns", &vmecpp::WOutFileContents::currumns)
      .def_readwrite("currvmns", &vmecpp::WOutFileContents::currvmns);

  py::class_<vmecpp::OutputQuantities>(m, "OutputQuantities")
      .def_readonly("jxbout", &vmecpp::OutputQuantities::jxbout)
      .def_readonly("mercier", &vmecpp::OutputQuantities::mercier)
      .def_readonly("threed1_first_table",
                    &vmecpp::OutputQuantities::threed1_first_table)
      .def_readonly("threed1_geometric_magnetic",
                    &vmecpp::OutputQuantities::threed1_geometric_magnetic)
      .def_readonly("threed1_volumetrics",
                    &vmecpp::OutputQuantities::threed1_volumetrics)
      .def_readonly("threed1_axis", &vmecpp::OutputQuantities::threed1_axis)
      .def_readonly("threed1_betas", &vmecpp::OutputQuantities::threed1_betas)
      .def_readonly("threed1_shafranov_integrals",
                    &vmecpp::OutputQuantities::threed1_shafranov_integrals)
      .def_readonly("wout", &vmecpp::OutputQuantities::wout)
      .def_readonly("indata", &vmecpp::OutputQuantities::indata)
      .def(
          "save",
          [](const vmecpp::OutputQuantities &oq,
             const std::filesystem::path &path) {
            absl::Status s = oq.Save(path);

            if (!s.ok()) {
              const std::string msg =
                  "There was an error saving OutputQuantities to file '" +
                  path.string() + "':\n" + std::string(s.message());
              throw std::runtime_error(msg);
            }
          },
          py::arg("path"))
      .def_static("load", [](const std::filesystem::path &path) {
        auto maybe_oq = vmecpp::OutputQuantities::Load(path);
        if (!maybe_oq.ok()) {
          const std::string msg =
              "There was an error loading OutputQuantities from file '" +
              path.string() + "':\n" + std::string(maybe_oq.status().message());
          throw std::runtime_error(msg);
        }
        return maybe_oq.value();
      });

  py::class_<vmecpp::HotRestartState>(m, "HotRestartState")
      .def(py::init(&MakeHotRestartState), "wout"_a, "indata"_a)
      .def_readwrite("wout", &vmecpp::HotRestartState::wout)
      .def_readwrite("indata", &vmecpp::HotRestartState::indata);

  m.def(
      "run",
      [](const VmecINDATA &indata,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads,
         vmecpp::OutputMode verbose) -> vmecpp::OutputQuantities {
        bool was_interrupted = false;
        auto interrupt_check = [&was_interrupted]() -> bool {
          if (was_interrupted) {
            return true;
          }
          py::gil_scoped_acquire acquire;
          if (PyErr_CheckSignals() != 0) {
            was_interrupted = true;
            return true;
          }
          return false;
        };
        absl::StatusOr<vmecpp::OutputQuantities> ret;
        {
          py::gil_scoped_release release;
          ret = vmecpp::run(indata, std::move(initial_state), max_threads,
                            verbose, interrupt_check);
        }
        if (was_interrupted) {
          throw py::error_already_set();
        }
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress);

  m.def(
      "recompute_outputs_from_spectra",
      [](const vmecpp::VmecINDATA &indata,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &rmncc_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &rmnss_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &zmnsc_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &zmncs_arr,
         std::optional<
             py::array_t<double, py::array::c_style | py::array::forcecast>>
             lmnsc_arr,
         std::optional<
             py::array_t<double, py::array::c_style | py::array::forcecast>>
             lmncs_arr,
         std::optional<int> max_threads, vmecpp::OutputMode verbose) {
        // Accept c-contiguous numpy arrays of shape (ns, mpol*(ntor+1))
        // or (ns, mpol, ntor+1) flattened to row-major.
        auto rmncc_info = rmncc_arr.request();
        auto rmnss_info = rmnss_arr.request();
        auto zmnsc_info = zmnsc_arr.request();
        auto zmncs_info = zmncs_arr.request();
        if (rmncc_info.ndim != 2) {
          throw std::runtime_error(
              "recompute_outputs_from_spectra: rmncc must be 2-D "
              "(ns, mpol*(ntor+1))");
        }
        // Build Eigen matrices from the numpy buffers without copy.
        auto to_matrix = [](const py::buffer_info &info,
                            const char *name) -> vmecpp::RowMatrixXd {
          if (info.ndim != 2) {
            throw std::runtime_error(
                std::string("recompute_outputs_from_spectra: ") + name +
                " must be 2-D");
          }
          const int rows = static_cast<int>(info.shape[0]);
          const int cols = static_cast<int>(info.shape[1]);
          vmecpp::RowMatrixXd m(rows, cols);
          std::memcpy(m.data(), info.ptr,
                      sizeof(double) * static_cast<size_t>(rows) * cols);
          return m;
        };
        vmecpp::RowMatrixXd rmncc_buf = to_matrix(rmncc_info, "rmncc");
        vmecpp::RowMatrixXd rmnss_buf = to_matrix(rmnss_info, "rmnss");
        vmecpp::RowMatrixXd zmnsc_buf = to_matrix(zmnsc_info, "zmnsc");
        vmecpp::RowMatrixXd zmncs_buf = to_matrix(zmncs_info, "zmncs");
        // Optional lambda spectra. When supplied (and not disabled via
        // VMECPP_RECOMPUTE_LAMBDA=0), lmns_full is reconstructed from
        // them and the hot-restart enters with converged lambda instead
        // of re-converging it from zero. VMECPP_RECOMPUTE_LAMBDA_SCALE
        // applies a uniform factor for normalization experiments.
        static int lam_use = -1;
        if (lam_use < 0) {
          const char *e = std::getenv("VMECPP_RECOMPUTE_LAMBDA");
          lam_use = (e && std::atoi(e) == 0) ? 0 : 1;
        }
        static double lam_scale = 0.0;
        if (lam_scale == 0.0) {
          const char *e = std::getenv("VMECPP_RECOMPUTE_LAMBDA_SCALE");
          lam_scale = (e && std::atof(e) != 0.0) ? std::atof(e) : 1.0;
        }
        bool have_lambda =
            lam_use && lmnsc_arr.has_value() && lmncs_arr.has_value();
        vmecpp::RowMatrixXd lmnsc_buf;
        vmecpp::RowMatrixXd lmncs_buf;
        if (have_lambda) {
          auto lsc_info = lmnsc_arr->request();
          auto lcs_info = lmncs_arr->request();
          lmnsc_buf = to_matrix(lsc_info, "lmnsc");
          lmncs_buf = to_matrix(lcs_info, "lmncs");
        }
        // Reconstructs an OutputQuantities by running a single-stage
        // vmecpp::run with a HotRestartState whose wout carries the
        // physical-form rmnc/zmns derived from the supplied converged
        // decomposed_x spectra. Intended to be invoked in a fresh
        // process: the in-process invocation following a batched
        // run_batched_gpu trips the shared global CUDA state's
        // persistent-buffer lifecycle and aborts in CheckInitialState
        // with a heap corruption, while a fresh process carries no such
        // state.
        //
        // Input arrays are flattened (ns × mpol × (ntor+1)) per the
        // d_pts_x layout written by FlushAllConfigsForOutputCuda, with
        // ns selected from the matrix row count and mpol*(ntor+1)
        // from the column count. The conversion to wout-format
        // rmnc/zmns mirrors ComputeOutputQuantities exactly: m=1
        // internal-to-physical mixing on the (rmnss, zmncs) pair via
        // m1Constraint(1.0), per-surface cc_ss_to_cos and sc_cs_to_sin
        // basis transforms, and m>0 mode zeroing at the magnetic axis
        // surface jF=0.
        const int ns_out = static_cast<int>(rmncc_buf.rows());
        const int mn_per_surface = static_cast<int>(rmncc_buf.cols());
        const int mpol_out = indata.mpol;
        const int ntor_out = indata.ntor;
        if (mn_per_surface != mpol_out * (ntor_out + 1)) {
          throw std::runtime_error(
              "recompute_outputs_from_spectra: spectra columns must equal "
              "mpol * (ntor + 1)");
        }
        if (rmnss_buf.rows() != ns_out || zmnsc_buf.rows() != ns_out ||
            zmncs_buf.rows() != ns_out) {
          throw std::runtime_error(
              "recompute_outputs_from_spectra: spectra row counts must "
              "match across rmncc/rmnss/zmnsc/zmncs");
        }
        if (rmnss_buf.cols() != mn_per_surface ||
            zmnsc_buf.cols() != mn_per_surface ||
            zmncs_buf.cols() != mn_per_surface) {
          throw std::runtime_error(
              "recompute_outputs_from_spectra: spectra column counts "
              "must match across rmncc/rmnss/zmnsc/zmncs");
        }

        // Build a single-stage indata at the converged ns. The
        // tightest tolerance from the original ftol_array is reused
        // so the hot-restart iter loop verifies convergence rather
        // than re-relaxing. Knobs: VMECPP_RECOMPUTE_FTOL (default: the
        // indata's last ftol) and VMECPP_RECOMPUTE_NITER.
        vmecpp::VmecINDATA ind_single = indata;
        ind_single.ns_array = Eigen::VectorXi::Constant(1, ns_out);
        const int last_ftol = static_cast<int>(indata.ftol_array.size()) - 1;
        const double ftolv_last =
            (last_ftol >= 0) ? indata.ftol_array[last_ftol] : 1.0e-12;
        static double recompute_ftol = -1.0;
        if (recompute_ftol < 0.0) {
          const char *e = std::getenv("VMECPP_RECOMPUTE_FTOL");
          recompute_ftol = (e && std::atof(e) > 0.0) ? std::atof(e) : 0.0;
        }
        ind_single.ftol_array = Eigen::VectorXd::Constant(
            1, recompute_ftol > 0.0 ? recompute_ftol : ftolv_last);
        static int recompute_niter = -1;
        if (recompute_niter < 0) {
          const char *e = std::getenv("VMECPP_RECOMPUTE_NITER");
          recompute_niter = (e && std::atoi(e) > 0) ? std::atoi(e) : 3000;
        }
        ind_single.niter_array = Eigen::VectorXi::Constant(1, recompute_niter);

        // Lambda normalization. The wout convention is
        //   lmns_full(mn, jF) = lmns_sin(mn) / phipF[jF] * lamscale
        // (output_quantities.cc), inverted at hot-restart consumption
        // (FourierGeometry::InitFromState). Obtain lamscale and the phipF
        // profile from a probe Vmec advanced to the RADIAL_PROFILES_EVAL
        // checkpoint of the single-stage indata; same indata and deltaS
        // as the dump's final stage, so the factors match exactly. When
        // the probe fails, the lambda input is dropped (zeroed lambda is
        // a valid, slower restart).
        double lam_norm_lamscale = 0.0;
        Eigen::VectorXd lam_norm_phipF;
        if (have_lambda) {
          auto lam_probe_or = vmecpp::Vmec::FromIndata(ind_single);
          if (lam_probe_or.ok()) {
            auto lam_probe = std::move(*lam_probe_or);
            auto lam_run =
                lam_probe->run(vmecpp::VmecCheckpoint::RADIAL_PROFILES_EVAL,
                               /*iterations_before_checkpointing=*/1,
                               /*maximum_multi_grid_step=*/1);
            if (lam_run.ok() && !lam_probe->p_.empty() && lam_probe->p_[0] &&
                lam_probe->constants_.lamscale != 0.0) {
              lam_norm_lamscale = lam_probe->constants_.lamscale;
              lam_norm_phipF = lam_probe->p_[0]->phipF;
            }
          }
          if (lam_norm_lamscale == 0.0 || lam_norm_phipF.size() < ns_out) {
            have_lambda = false;
          }
        }

        vmecpp::Sizes sizes_c(indata.lasym, indata.nfp, indata.mpol,
                              indata.ntor, indata.ntheta, indata.nzeta);
        vmecpp::FourierBasisFastPoloidal fb_c(&sizes_c);

        const int mnmax = (ntor_out + 1) + (mpol_out - 1) * (2 * ntor_out + 1);
        vmecpp::RowMatrixXd rmnc(mnmax, ns_out);
        vmecpp::RowMatrixXd zmns(mnmax, ns_out);
        vmecpp::RowMatrixXd lmns_full(mnmax, ns_out);
        rmnc.setZero();
        zmns.setZero();
        lmns_full.setZero();

        std::vector<double> rmncc_phys(mn_per_surface);
        std::vector<double> rmnss_phys(mn_per_surface);
        std::vector<double> zmnsc_phys(mn_per_surface);
        std::vector<double> zmncs_phys(mn_per_surface);
        std::vector<double> lmnsc_phys(mn_per_surface);
        std::vector<double> lmncs_phys(mn_per_surface);
        std::vector<double> rmnc_col(mnmax);
        std::vector<double> zmns_col(mnmax);
        std::vector<double> lmns_col(mnmax);
        for (int jF = 0; jF < ns_out; ++jF) {
          for (int k = 0; k < mn_per_surface; ++k) {
            rmncc_phys[k] = rmncc_buf(jF, k);
            rmnss_phys[k] = rmnss_buf(jF, k);
            zmnsc_phys[k] = zmnsc_buf(jF, k);
            zmncs_phys[k] = zmncs_buf(jF, k);
          }
          // m=1 internal-to-physical mixing on (rmnss, zmncs).
          for (int n = 0; n < ntor_out + 1; ++n) {
            const int idx_m1 = 1 * (ntor_out + 1) + n;
            const double old_rss = rmnss_phys[idx_m1];
            rmnss_phys[idx_m1] = old_rss + zmncs_phys[idx_m1];
            zmncs_phys[idx_m1] = old_rss - zmncs_phys[idx_m1];
          }
          fb_c.cc_ss_to_cos(
              std::span<const double>(rmncc_phys.data(), mn_per_surface),
              std::span<const double>(rmnss_phys.data(), mn_per_surface),
              std::span<double>(rmnc_col.data(), mnmax), ntor_out, mpol_out);
          fb_c.sc_cs_to_sin(
              std::span<const double>(zmnsc_phys.data(), mn_per_surface),
              std::span<const double>(zmncs_phys.data(), mn_per_surface),
              std::span<double>(zmns_col.data(), mnmax), ntor_out, mpol_out);
          if (have_lambda) {
            // wout convention: divide by phipF[jF], multiply by lamscale.
            // lam_scale (VMECPP_RECOMPUTE_LAMBDA_SCALE) composes on top
            // for experiments; default 1.0.
            const double lam_factor =
                lam_scale * lam_norm_lamscale / lam_norm_phipF[jF];
            for (int k = 0; k < mn_per_surface; ++k) {
              lmnsc_phys[k] = lmnsc_buf(jF, k) * lam_factor;
              lmncs_phys[k] = lmncs_buf(jF, k) * lam_factor;
            }
            fb_c.sc_cs_to_sin(
                std::span<const double>(lmnsc_phys.data(), mn_per_surface),
                std::span<const double>(lmncs_phys.data(), mn_per_surface),
                std::span<double>(lmns_col.data(), mnmax), ntor_out, mpol_out);
          }
          // Zero m>0 modes at the magnetic axis surface.
          if (jF == 0) {
            int mn = ntor_out + 1;
            for (int m = 1; m < mpol_out; ++m) {
              for (int n = -ntor_out; n <= ntor_out; ++n) {
                rmnc_col[mn] = 0.0;
                zmns_col[mn] = 0.0;
                if (have_lambda) lmns_col[mn] = 0.0;
                mn++;
              }
            }
          }
          for (int k = 0; k < mnmax; ++k) {
            rmnc(k, jF) = rmnc_col[k];
            zmns(k, jF) = zmns_col[k];
            if (have_lambda) lmns_full(k, jF) = lmns_col[k];
          }
        }

        vmecpp::WOutFileContents wout_c;
        wout_c.mpol = mpol_out;
        wout_c.ntor = ntor_out;
        wout_c.lasym = indata.lasym;
        wout_c.nfp = indata.nfp;
        wout_c.ns = ns_out;
        wout_c.rmnc = std::move(rmnc);
        wout_c.zmns = std::move(zmns);
        wout_c.lmns_full = std::move(lmns_full);

        vmecpp::HotRestartState hot_restart(std::move(wout_c), ind_single);

        bool was_interrupted = false;
        auto interrupt_check = [&was_interrupted]() -> bool {
          if (was_interrupted) return true;
          py::gil_scoped_acquire acquire;
          if (PyErr_CheckSignals() != 0) {
            was_interrupted = true;
            return true;
          }
          return false;
        };
        absl::StatusOr<vmecpp::OutputQuantities> ret;
        {
          py::gil_scoped_release release;
          ret = vmecpp::run(
              ind_single,
              std::optional<vmecpp::HotRestartState>(std::move(hot_restart)),
              max_threads, verbose, interrupt_check);
        }
        if (was_interrupted) {
          throw py::error_already_set();
        }
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("rmncc_buf"), py::arg("rmnss_buf"),
      py::arg("zmnsc_buf"), py::arg("zmncs_buf"),
      py::arg("lmnsc_buf") = std::nullopt, py::arg("lmncs_buf") = std::nullopt,
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress,
      R"pbdoc(Recompute OutputQuantities from converged decomposed_x spectra.

Drives a single-stage vmecpp.run with a HotRestartState whose wout
spectra are reconstructed from the supplied converged rmncc / rmnss /
zmnsc / zmncs arrays. Intended to be called in a fresh subprocess
following a batched run_batched_gpu so the iter loop starts from the
already-converged geometry and finishes in a small number of iters,
producing a per-configuration OutputQuantities whose wout reflects the
configuration's own physics rather than another configuration's.

The function does not honour broadcast or distinct mode env vars: it
runs a plain single-cfg vmecpp.run with the indata's ns_array
collapsed to a single stage at the row count of the supplied spectra.
Calling this in the same process as an immediately preceding
run_batched_gpu trips the shared global CUDA state's persistent-buffer
lifecycle and aborts; subprocess isolation avoids that.
)pbdoc");

  m.def(
      "reset_cuda_state_for_new_vmec_run",
      []() {
#ifdef VMECPP_USE_CUDA
        vmecpp::ResetCudaStateForNewVmecRun();
#endif
      },
      R"pbdoc(Reset the persistent CUDA state singleton.

Required between successive vmecpp.run / recompute_outputs_from_spectra
calls within one process so the next run re-stages its host state from
scratch instead of inheriting the prior run's persistent device buffers.
No-op on CPU-only builds.)pbdoc");

  m.def(
      "recompute_outputs_from_spectra_batched",
      [](std::vector<vmecpp::VmecINDATA> indatas,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &rmncc_batch,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &rmnss_batch,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &zmnsc_batch,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &zmncs_batch,
         std::optional<int> max_threads, vmecpp::OutputMode verbose) {
        // In-process batched per-configuration recompute. Each
        // configuration's wout is reconstructed from its converged
        // spectra (the same arithmetic as the singular
        // recompute_outputs_from_spectra) and fed through a single-stage
        // vmecpp::run with a HotRestartState. ResetCudaStateForNewVmecRun
        // runs between consecutive configurations so each run re-stages
        // its host decomposed position and velocity into the device
        // slots from scratch.
        //
        // This is the in-process counterpart to a multiprocessing pool
        // of single recomputes: the loop shares one warm CUDA context
        // and one warm interpreter, so the several-second per-worker
        // import and context cost disappears, and each hot restart
        // converges in a few iterations from the supplied spectra.

        auto rmncc_info = rmncc_batch.request();
        auto rmnss_info = rmnss_batch.request();
        auto zmnsc_info = zmnsc_batch.request();
        auto zmncs_info = zmncs_batch.request();
        if (rmncc_info.ndim != 3 || rmnss_info.ndim != 3 ||
            zmnsc_info.ndim != 3 || zmncs_info.ndim != 3) {
          throw std::runtime_error(
              "recompute_outputs_from_spectra_batched: spectra arrays must "
              "be 3-D (N, ns, mpol*(ntor+1))");
        }
        const int n_cfg = static_cast<int>(rmncc_info.shape[0]);
        if (static_cast<int>(indatas.size()) != n_cfg) {
          throw std::runtime_error(
              "recompute_outputs_from_spectra_batched: indatas length "
              "must match spectra batch dimension");
        }
        const int ns_out = static_cast<int>(rmncc_info.shape[1]);
        const int mn_per_surface = static_cast<int>(rmncc_info.shape[2]);

        std::vector<vmecpp::OutputQuantities> outs;
        outs.reserve(n_cfg);

        const auto *rmncc_ptr = static_cast<const double *>(rmncc_info.ptr);
        const auto *rmnss_ptr = static_cast<const double *>(rmnss_info.ptr);
        const auto *zmnsc_ptr = static_cast<const double *>(zmnsc_info.ptr);
        const auto *zmncs_ptr = static_cast<const double *>(zmncs_info.ptr);
        const size_t per_cfg = static_cast<size_t>(ns_out) * mn_per_surface;

        for (int c = 0; c < n_cfg; ++c) {
          const vmecpp::VmecINDATA &ind_c = indatas[c];
          if (ind_c.mpol * (ind_c.ntor + 1) != mn_per_surface) {
            throw std::runtime_error(
                "recompute_outputs_from_spectra_batched: indata cfg " +
                std::to_string(c) +
                " mpol*(ntor+1) does not match spectra columns");
          }

          const int mpol_out = ind_c.mpol;
          const int ntor_out = ind_c.ntor;
          vmecpp::Sizes sizes_c(ind_c.lasym, ind_c.nfp, ind_c.mpol, ind_c.ntor,
                                ind_c.ntheta, ind_c.nzeta);
          vmecpp::FourierBasisFastPoloidal fb_c(&sizes_c);

          const int mnmax =
              (ntor_out + 1) + (mpol_out - 1) * (2 * ntor_out + 1);
          vmecpp::RowMatrixXd rmnc(mnmax, ns_out);
          vmecpp::RowMatrixXd zmns(mnmax, ns_out);
          vmecpp::RowMatrixXd lmns_full(mnmax, ns_out);
          rmnc.setZero();
          zmns.setZero();
          lmns_full.setZero();

          std::vector<double> rmncc_phys(mn_per_surface);
          std::vector<double> rmnss_phys(mn_per_surface);
          std::vector<double> zmnsc_phys(mn_per_surface);
          std::vector<double> zmncs_phys(mn_per_surface);
          std::vector<double> rmnc_col(mnmax);
          std::vector<double> zmns_col(mnmax);

          const double *d_rcc = rmncc_ptr + static_cast<size_t>(c) * per_cfg;
          const double *d_rss = rmnss_ptr + static_cast<size_t>(c) * per_cfg;
          const double *d_zsc = zmnsc_ptr + static_cast<size_t>(c) * per_cfg;
          const double *d_zcs = zmncs_ptr + static_cast<size_t>(c) * per_cfg;

          for (int jF = 0; jF < ns_out; ++jF) {
            for (int k = 0; k < mn_per_surface; ++k) {
              rmncc_phys[k] =
                  d_rcc[static_cast<size_t>(jF) * mn_per_surface + k];
              rmnss_phys[k] =
                  d_rss[static_cast<size_t>(jF) * mn_per_surface + k];
              zmnsc_phys[k] =
                  d_zsc[static_cast<size_t>(jF) * mn_per_surface + k];
              zmncs_phys[k] =
                  d_zcs[static_cast<size_t>(jF) * mn_per_surface + k];
            }
            for (int n = 0; n < ntor_out + 1; ++n) {
              const int idx_m1 = 1 * (ntor_out + 1) + n;
              const double old_rss = rmnss_phys[idx_m1];
              rmnss_phys[idx_m1] = old_rss + zmncs_phys[idx_m1];
              zmncs_phys[idx_m1] = old_rss - zmncs_phys[idx_m1];
            }
            fb_c.cc_ss_to_cos(
                std::span<const double>(rmncc_phys.data(), mn_per_surface),
                std::span<const double>(rmnss_phys.data(), mn_per_surface),
                std::span<double>(rmnc_col.data(), mnmax), ntor_out, mpol_out);
            fb_c.sc_cs_to_sin(
                std::span<const double>(zmnsc_phys.data(), mn_per_surface),
                std::span<const double>(zmncs_phys.data(), mn_per_surface),
                std::span<double>(zmns_col.data(), mnmax), ntor_out, mpol_out);
            if (jF == 0) {
              int mn = ntor_out + 1;
              for (int m = 1; m < mpol_out; ++m) {
                for (int n = -ntor_out; n <= ntor_out; ++n) {
                  rmnc_col[mn] = 0.0;
                  zmns_col[mn] = 0.0;
                  mn++;
                }
              }
            }
            for (int k = 0; k < mnmax; ++k) {
              rmnc(k, jF) = rmnc_col[k];
              zmns(k, jF) = zmns_col[k];
            }
          }

          vmecpp::VmecINDATA ind_single = ind_c;
          ind_single.ns_array = Eigen::VectorXi::Constant(1, ns_out);
          const int last_ftol = static_cast<int>(ind_c.ftol_array.size()) - 1;
          const double ftolv_last =
              (last_ftol >= 0) ? ind_c.ftol_array[last_ftol] : 1.0e-12;
          // Same lambda-from-zero sizing and env knobs as the singular
          // binding.
          static double recompute_ftol_b = -1.0;
          if (recompute_ftol_b < 0.0) {
            const char *e = std::getenv("VMECPP_RECOMPUTE_FTOL");
            recompute_ftol_b = (e && std::atof(e) > 0.0) ? std::atof(e) : 0.0;
          }
          ind_single.ftol_array = Eigen::VectorXd::Constant(
              1, recompute_ftol_b > 0.0 ? recompute_ftol_b : ftolv_last);
          static int recompute_niter_b = -1;
          if (recompute_niter_b < 0) {
            const char *e = std::getenv("VMECPP_RECOMPUTE_NITER");
            recompute_niter_b = (e && std::atoi(e) > 0) ? std::atoi(e) : 3000;
          }
          ind_single.niter_array =
              Eigen::VectorXi::Constant(1, recompute_niter_b);

          vmecpp::WOutFileContents wout_c;
          wout_c.mpol = mpol_out;
          wout_c.ntor = ntor_out;
          wout_c.lasym = ind_c.lasym;
          wout_c.nfp = ind_c.nfp;
          wout_c.ns = ns_out;
          wout_c.rmnc = std::move(rmnc);
          wout_c.zmns = std::move(zmns);
          wout_c.lmns_full = std::move(lmns_full);

          vmecpp::HotRestartState hot_restart(std::move(wout_c), ind_single);

      // Reset the CUDA state singleton before each per-cfg run so the
      // next vmecpp::run re-H2Ds its host state from scratch and the
      // active-mask override doesn't carry stale persistent flags.
#ifdef VMECPP_USE_CUDA
          vmecpp::ResetCudaStateForNewVmecRun();
#endif
          unsetenv("VMECPP_BATCH_DISTINCT");
          unsetenv("VMECPP_BATCH_INPUTS_FILE");
          unsetenv("VMECPP_BATCH_OUTPUTS_FILE");
          unsetenv("VMECPP_BATCH_DEC_X_FILE");

          bool was_interrupted = false;
          auto interrupt_check = [&was_interrupted]() -> bool {
            if (was_interrupted) return true;
            py::gil_scoped_acquire acquire;
            if (PyErr_CheckSignals() != 0) {
              was_interrupted = true;
              return true;
            }
            return false;
          };
          absl::StatusOr<vmecpp::OutputQuantities> ret;
          {
            py::gil_scoped_release release;
            ret = vmecpp::run(
                ind_single,
                std::optional<vmecpp::HotRestartState>(std::move(hot_restart)),
                max_threads, verbose, interrupt_check);
          }
          if (was_interrupted) {
            throw py::error_already_set();
          }
          outs.push_back(GetValueOrThrow(ret));
        }

        return outs;
      },
      py::arg("indatas"), py::arg("rmncc_batch"), py::arg("rmnss_batch"),
      py::arg("zmnsc_batch"), py::arg("zmncs_batch"),
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress,
      R"pbdoc(In-process batched form of recompute_outputs_from_spectra.

Runs N hot-restart vmecpp.run calls sequentially in one process,
sharing a single warm CUDA context. The results match N subprocess
calls to recompute_outputs_from_spectra without the per-subprocess
interpreter and context startup cost. The persistent CUDA state resets
between configurations, so each run stages its host state to the
device from scratch.

Spectra arrays have shape (N, ns, mpol*(ntor+1)) and are c-contiguous
float64. indatas length must equal N, and each indata's
mpol*(ntor+1) must equal the column count. A configuration whose run
fails raises; the subprocess path remains available for retrying
individual configurations.
)pbdoc");

  py::class_<makegrid::MakegridParameters>(m, "MakegridParameters")
      .def(py::init<bool, bool, int, double, double, int, double, double, int,
                    int>(),
           "normalize_by_currents"_a, "assume_stellarator_symmetry"_a,
           "number_of_field_periods"_a, "r_grid_minimum"_a, "r_grid_maximum"_a,
           "number_of_r_grid_points"_a, "z_grid_minimum"_a, "z_grid_maximum"_a,
           "number_of_z_grid_points"_a, "number_of_phi_grid_points"_a)
      .def_static(
          "from_file",
          [](const std::filesystem::path &file) {
            auto maybe_params =
                makegrid::ImportMakegridParametersFromFile(file);
            return GetValueOrThrow(maybe_params);
          },
          py::arg("file"))
      .def_readonly("normalize_by_currents",
                    &makegrid::MakegridParameters::normalize_by_currents)
      .def_readonly("assume_stellarator_symmetry",
                    &makegrid::MakegridParameters::assume_stellarator_symmetry)
      .def_readonly("number_of_field_periods",
                    &makegrid::MakegridParameters::number_of_field_periods)
      .def_readonly("r_grid_minimum",
                    &makegrid::MakegridParameters::r_grid_minimum)
      .def_readonly("r_grid_maximum",
                    &makegrid::MakegridParameters::r_grid_maximum)
      .def_readonly("number_of_r_grid_points",
                    &makegrid::MakegridParameters::number_of_r_grid_points)
      .def_readonly("z_grid_minimum",
                    &makegrid::MakegridParameters::z_grid_minimum)
      .def_readonly("z_grid_maximum",
                    &makegrid::MakegridParameters::z_grid_maximum)
      .def_readonly("number_of_z_grid_points",
                    &makegrid::MakegridParameters::number_of_z_grid_points)
      .def_readonly("number_of_phi_grid_points",
                    &makegrid::MakegridParameters::number_of_phi_grid_points);

  py::class_<magnetics::MagneticConfiguration>(m, "MagneticConfiguration")
      .def_static(
          "from_file",
          [](const std::filesystem::path &file) {
            auto maybe_config =
                magnetics::ImportMagneticConfigurationFromCoilsFile(file);
            return GetValueOrThrow(maybe_config);
          },
          py::arg("file"));
  auto response_table =
      py::class_<makegrid::MagneticFieldResponseTable>(
          m, "MagneticFieldResponseTable")
          .def(py::init<const makegrid::MakegridParameters &,
                        const makegrid::RowMatrixXd &,
                        const makegrid::RowMatrixXd &,
                        const makegrid::RowMatrixXd &>(),
               py::arg("parameters"), py::arg("b_r"), py::arg("b_p"),
               py::arg("b_z"))
          .def_readonly("parameters",
                        &makegrid::MagneticFieldResponseTable::parameters);
  DefEigenProperty(response_table, "b_r",
                   &makegrid::MagneticFieldResponseTable::b_r);
  DefEigenProperty(response_table, "b_p",
                   &makegrid::MagneticFieldResponseTable::b_p);
  DefEigenProperty(response_table, "b_z",
                   &makegrid::MagneticFieldResponseTable::b_z);

  m.def(
      "compute_magnetic_field_response_table",
      [](const makegrid::MakegridParameters &mgrid_params,
         const magnetics::MagneticConfiguration &magnetic_configuration) {
        auto ret = makegrid::ComputeMagneticFieldResponseTable(
            mgrid_params, magnetic_configuration);
        return GetValueOrThrow(ret);
      },
      py::arg("makegrid_parameters"), py::arg("magnetic_configuration"));

  m.def(
      "run",
      [](const VmecINDATA &indata,
         const makegrid::MagneticFieldResponseTable &magnetic_response_table,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads, vmecpp::OutputMode verbose) {
        bool was_interrupted = false;
        auto interrupt_check = [&was_interrupted]() -> bool {
          if (was_interrupted) return true;
          py::gil_scoped_acquire acquire;
          if (PyErr_CheckSignals() != 0) {
            was_interrupted = true;
            return true;
          }
          return false;
        };
        absl::StatusOr<vmecpp::OutputQuantities> ret;
        {
          py::gil_scoped_release release;
          ret = vmecpp::run(indata, magnetic_response_table,
                            std::move(initial_state), max_threads, verbose,
                            interrupt_check);
        }
        if (was_interrupted) {
          throw py::error_already_set();
        }
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("magnetic_response_table"),
      py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress);

  // =========================================================================
  // Batched-GPU entry point: solves N fixed-boundary equilibria in a single
  // CUDA-resident iteration loop.
  // =========================================================================
  // The binding accepts a list of VmecINDATA inputs that share the shape
  // parameters mpol, ntor, nfp, lasym, and ns_array[0], and drives the
  // device-side iteration body at n_config_max = N. The shared-shape
  // constraint is required because the persistent device buffers are
  // dimensioned at first invocation of the iteration body and cannot be
  // resized across calls.
  //
  // Two execution modes are available, selected by the
  // VMECPP_BATCH_DISTINCT environment variable:
  //
  // Broadcast mode (default, environment variable unset or zero): the
  // first input's spectra are broadcast across all N configuration slots
  // in d_specs_block, the batched kernel chain executes at full N for
  // measurement and infrastructure exercise, and the returned vector
  // contains N copies of the first input's converged OutputQuantities.
  //
  // Distinct mode (VMECPP_BATCH_DISTINCT=1): a per-configuration
  // pre-initialization stage extracts each input's initial spectra into
  // a temporary file. The pre-initialization applies the proactive
  // magnetic-axis recomputation per configuration so that the iteration
  // body does not subsequently invoke the axis-recovery code path; this
  // permits the per-configuration spectra in d_specs_block to persist
  // through the first iteration without being overwritten by the host
  // triplet's broadcast of a corrected configuration-zero geometry. The
  // returned vector contains one converged OutputQuantities per input,
  // each derived from the corresponding configuration's converged state
  // after the device-resident output flush at end-of-run.
  m.def(
      "run_batched_gpu",
      [](const std::vector<VmecINDATA> &indata_list,
         std::optional<int> max_threads, vmecpp::OutputMode verbose,
         bool return_spectra, bool distinct) -> py::object {
#ifdef VMECPP_USE_CUDA
        // Derives one OutputQuantities per configuration directly from the
        // batched run's device state: the per-cfg fields flush in one
        // pass, each configuration's VmecInternalResults assembles from
        // the seed's gathered template plus the per-cfg slices, and
        // DeriveOutputQuantities runs the standard host physics chain per
        // configuration. No additional iterations run. Returns false when
        // any precondition fails; the caller falls back to the
        // hot-restart reconstruction.
        auto derive_outputs_from_batched_flush =
            [](const std::vector<vmecpp::VmecINDATA> &indata_local, int n_cfg,
               const vmecpp::OutputQuantities &seed_oq,
               std::vector<vmecpp::OutputQuantities> &outs) -> bool {
          // Capture the batched run's final per-configuration force-residual
          // state before any further CUDA call (the deriver's setup run below
          // resets the persistent device state). The wout reports the
          // normalized invariant residuals fsqr/fsqz/fsql; the iteration
          // controller builds them from the raw per-cfg residuals together
          // with the per-cfg force-norm, pressure, and volume caches exactly
          // as in ideal_mhd_model.cc:420-436, so capture all four here and
          // apply the same normalization per configuration below. Paired with
          // the lock-step batch iteration count from the seed wout, this lets
          // each per-cfg output report its own convergence instead of the
          // deriver's single-iteration placeholder fc.
          // The batched solve iterates all configurations in lock-step, so the
          // seed wout carries the per-config iteration count (the derived
          // outputs below otherwise report niter = 0 from the hard-coded
          // iter2). The per-config normalized force residuals come from the
          // snapshot cache the iteration controller fills; the live residual
          // caches are zeroed for masked (converged) configs and cannot be
          // read here directly. Captured now, before the deriver's setup run
          // resets the device state. Layout: 3*n_cfg as
          // [fsqr_c, fsqz_c, fsql_c, ...].
          const int batch_niter = seed_oq.wout.niter;
          const std::vector<double> fsqr_snapshot_pc =
              vmecpp::GetFsqrPerCfgCache();
          std::vector<double> spectra;
          int sn = 0;
          int sns = 0;
          int smpol = 0;
          int sntor = 0;
          if (!vmecpp::GetBatchOutputSpectraCuda(&spectra, &sn, &sns, &smpol,
                                                 &sntor) ||
              sn != n_cfg) {
            return false;
          }
          const vmecpp::VmecINDATA &ind0 = indata_local[0];
          vmecpp::Sizes sizes(ind0.lasym, ind0.nfp, ind0.mpol, ind0.ntor,
                              ind0.ntheta, ind0.nzeta);
          if (smpol != sizes.mpol || sntor != sizes.ntor || sizes.lasym) {
            return false;
          }
          const int ns = sns;
          const int ns_h = ns - 1;
          const int nZnT = sizes.nZnT;
          const size_t half1 = static_cast<size_t>(ns_h) * nZnT;
          const size_t full1 = static_cast<size_t>(ns) * nZnT;
          const size_t prof1 = static_cast<size_t>(ns_h);
          const size_t profF1 = static_cast<size_t>(ns);
          const size_t spec1 =
              static_cast<size_t>(ns) * sizes.mpol * (sizes.ntor + 1);
          const auto nv = [&](size_t per) {
            return std::vector<double>(static_cast<size_t>(n_cfg) * per, 0.0);
          };
          auto f_gsqrt = nv(half1), f_guu = nv(half1), f_guv = nv(half1),
               f_gvv = nv(half1), f_bsubu = nv(half1), f_bsubv = nv(half1),
               f_bsupu = nv(half1), f_bsupv = nv(half1), f_totp = nv(half1),
               f_r12 = nv(half1), f_ru12 = nv(half1), f_zu12 = nv(half1),
               f_rs = nv(half1), f_zs = nv(half1);
          auto f_r1e = nv(full1), f_r1o = nv(full1), f_z1e = nv(full1),
               f_z1o = nv(full1), f_rue = nv(full1), f_ruo = nv(full1),
               f_zue = nv(full1), f_zuo = nv(full1), f_rve = nv(full1),
               f_rvo = nv(full1), f_zve = nv(full1), f_zvo = nv(full1),
               f_ruF = nv(full1), f_zuF = nv(full1), f_blmn = nv(full1);
          auto f_presH = nv(prof1), f_dVdsH = nv(prof1), f_bvcoH = nv(prof1),
               f_bucoH = nv(prof1), f_chipH = nv(prof1), f_iotaH = nv(prof1);
          auto f_chipF = nv(profF1), f_iotaF = nv(profF1);
          // Repair dVdsH for configs masked before the batch finished (else
          // their betatot flushes as zero) before reading the per-config
          // fields.
          vmecpp::RecomputeZeroedDVdsHForOutputCuda(
              n_cfg, ns_h, nZnT, sizes.nThetaEff,
              vmecpp::Vmec::kSignOfJacobian);
          vmecpp::FlushAllConfigsForOutputCudaNs(
              ns, sizes, n_cfg, f_gsqrt.data(), f_guu.data(), f_guv.data(),
              f_gvv.data(), f_bsubu.data(), f_bsubv.data(), f_bsupu.data(),
              f_bsupv.data(), f_totp.data(), f_r12.data(), f_ru12.data(),
              f_zu12.data(), f_rs.data(), f_zs.data(), f_r1e.data(),
              f_r1o.data(), f_z1e.data(), f_z1o.data(), f_rue.data(),
              f_ruo.data(), f_zue.data(), f_zuo.data(), f_rve.data(),
              f_rvo.data(), f_zve.data(), f_zvo.data(), f_ruF.data(),
              f_zuF.data(), f_blmn.data(), f_presH.data(), f_dVdsH.data(),
              f_bvcoH.data(), f_bucoH.data(), f_chipH.data(), f_iotaH.data(),
              f_chipF.data(), f_iotaF.data(), nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr);
          const std::vector<double> pres_cache =
              vmecpp::GetPressureScalarsPerCfgCache();
          const std::vector<double> vol_cache =
              vmecpp::GetPlasmaVolumePerCfgCache();
          if (static_cast<int>(pres_cache.size()) < 3 * n_cfg ||
              static_cast<int>(vol_cache.size()) < n_cfg) {
            return false;
          }

          // Deriver instance: a setup-only run at the converged ns
          // supplies the Sizes, FlowControl, basis tables, constants,
          // profiles, and a HandoverStorage to patch per configuration.
          // Its run resets the persistent CUDA state; the flush above
          // has already consumed it.
          vmecpp::VmecINDATA ind_drv = ind0;
          ind_drv.ns_array = Eigen::VectorXi::Constant(1, ns);
          const int last_ftol = static_cast<int>(ind0.ftol_array.size()) - 1;
          ind_drv.ftol_array = Eigen::VectorXd::Constant(
              1, (last_ftol >= 0) ? ind0.ftol_array[last_ftol] : 1.0e-12);
          ind_drv.niter_array = Eigen::VectorXi::Constant(1, 1);
          auto drv_or = vmecpp::Vmec::FromIndata(ind_drv);
          if (!drv_or.ok()) {
            return false;
          }
          auto deriver = std::move(*drv_or);
          auto setup_or =
              deriver->run(vmecpp::VmecCheckpoint::SETUP_INITIAL_STATE,
                           /*iterations_before_checkpointing=*/1,
                           /*maximum_multi_grid_step=*/1);
          if (!setup_or.ok() || deriver->r_.empty() || deriver->p_.empty()) {
            return false;
          }
          const auto &res_template = seed_oq.vmec_internal_results;
          if (res_template.num_full != ns) {
            return false;
          }

          const int mpol = sizes.mpol;
          const int ntor = sizes.ntor;
          for (int c = 0; c < n_cfg; ++c) {
            const auto at = [&](const std::vector<double> &v,
                                size_t per) -> const double * {
              return v.data() + static_cast<size_t>(c) * per;
            };
            vmecpp::OutputQuantities oq;
            oq.vmec_internal_results = res_template;
            auto &res = oq.vmec_internal_results;
            const auto cp = [&](vmecpp::RowMatrixXd &dst,
                                const std::vector<double> &src, size_t per) {
              std::memcpy(dst.data(), at(src, per), per * sizeof(double));
            };
            cp(res.gsqrt, f_gsqrt, half1);
            cp(res.guu, f_guu, half1);
            if (sizes.lthreed) {
              cp(res.guv, f_guv, half1);
            }
            cp(res.gvv, f_gvv, half1);
            cp(res.bsubu, f_bsubu, half1);
            cp(res.bsubv, f_bsubv, half1);
            cp(res.bsupu, f_bsupu, half1);
            cp(res.bsupv, f_bsupv, half1);
            cp(res.total_pressure, f_totp, half1);
            cp(res.r12, f_r12, half1);
            cp(res.ru12, f_ru12, half1);
            cp(res.zu12, f_zu12, half1);
            cp(res.rs, f_rs, half1);
            cp(res.zs, f_zs, half1);
            cp(res.r_e, f_r1e, full1);
            cp(res.r_o, f_r1o, full1);
            cp(res.z_e, f_z1e, full1);
            cp(res.z_o, f_z1o, full1);
            cp(res.ru_e, f_rue, full1);
            cp(res.ru_o, f_ruo, full1);
            cp(res.zu_e, f_zue, full1);
            cp(res.zu_o, f_zuo, full1);
            if (sizes.lthreed) {
              cp(res.rv_e, f_rve, full1);
              cp(res.rv_o, f_rvo, full1);
              cp(res.zv_e, f_zve, full1);
              cp(res.zv_o, f_zvo, full1);
            }
            cp(res.ruFull, f_ruF, full1);
            cp(res.zuFull, f_zuF, full1);
            const double *blmn = at(f_blmn, full1);
            const double lamscale = deriver->constants_.lamscale;
            for (int jF = 0; jF < ns; ++jF) {
              const double unlamscale = (jF > 0) ? (-1.0 / lamscale) : 1.0;
              for (int kl = 0; kl < nZnT; ++kl) {
                res.bsubvF(jF * nZnT + kl) = blmn[jF * nZnT + kl] * unlamscale;
              }
            }
            const auto cpv = [&](Eigen::VectorXd &dst,
                                 const std::vector<double> &src, size_t per) {
              std::memcpy(dst.data(), at(src, per), per * sizeof(double));
            };
            cpv(res.presH, f_presH, prof1);
            cpv(res.dVdsH, f_dVdsH, prof1);
            cpv(res.bvcoH, f_bvcoH, prof1);
            cpv(res.chipH, f_chipH, prof1);
            cpv(res.iotaH, f_iotaH, prof1);
            cpv(res.chipF, f_chipF, profF1);
            cpv(res.iotaF, f_iotaF, profF1);

            // Converged spectra into the state vector, with the gather's
            // index transpose, plus the per-surface spectral width on the
            // raw decomposed coefficients (p = 4, q = 1, m = 1 unmixed).
            const size_t spec_base = static_cast<size_t>(c) * spec1;
            const auto sp_at = [&](int sp) -> const double * {
              return spectra.data() + static_cast<size_t>(sp) * n_cfg * spec1 +
                     spec_base;
            };
            const double *x_rcc = sp_at(0);
            const double *x_rss = sp_at(1);
            const double *x_zsc = sp_at(2);
            const double *x_zcs = sp_at(3);
            const double *x_lsc = sp_at(4);
            const double *x_lcs = sp_at(5);
            for (int jF = 0; jF < ns; ++jF) {
              for (int n = 0; n < ntor + 1; ++n) {
                for (int mm = 0; mm < mpol; ++mm) {
                  const int src = (jF * mpol + mm) * (ntor + 1) + n;
                  const int dst = (jF * (ntor + 1) + n) * mpol + mm;
                  res.rmncc(dst) = x_rcc[src];
                  res.zmnsc(dst) = x_zsc[src];
                  res.lmnsc(dst) = x_lsc[src];
                  if (sizes.lthreed) {
                    res.rmnss(dst) = x_rss[src];
                    res.zmncs(dst) = x_zcs[src];
                    res.lmncs(dst) = x_lcs[src];
                  }
                }
              }
            }
            res.spectral_width[0] = 1.0;
            for (int jF = 1; jF < ns; ++jF) {
              double num = 0.0;
              double den = 0.0;
              for (int mm = 1; mm < mpol; ++mm) {
                for (int n = 0; n < ntor + 1; ++n) {
                  const int idx = (jF * mpol + mm) * (ntor + 1) + n;
                  const double bn =
                      deriver->t_.mscale[mm] * deriver->t_.nscale[n];
                  double rcc_v = x_rcc[idx];
                  double zsc_v = x_zsc[idx];
                  double norm = rcc_v * rcc_v + zsc_v * zsc_v;
                  if (sizes.lthreed) {
                    double rss_v = x_rss[idx];
                    double zcs_v = x_zcs[idx];
                    if (mm == 1) {
                      const double r_plus = rss_v;
                      const double r_minus = zcs_v;
                      rss_v = r_plus + r_minus;
                      zcs_v = r_plus - r_minus;
                    }
                    norm += rss_v * rss_v + zcs_v * zcs_v;
                  }
                  norm *= bn * bn;
                  num += norm * std::pow(mm, 5);
                  den += norm * std::pow(mm, 4);
                }
              }
              res.spectral_width[jF] = num / den;
            }

            vmecpp::HandoverStorage h_c = deriver->h_;
            h_c.thermalEnergy = pres_cache[3 * c + 0];
            h_c.magneticEnergy = pres_cache[3 * c + 1];
            h_c.mhdEnergy = pres_cache[3 * c + 2];
            h_c.plasmaVolume = vol_cache[c];
            const double *bv = at(f_bvcoH, prof1);
            const double *bu = at(f_bucoH, prof1);
            h_c.rBtor0 = 1.5 * bv[0] - 0.5 * bv[1];
            h_c.rBtor = 1.5 * bv[ns_h - 1] - 0.5 * bv[ns_h - 2];
            h_c.cTor = (1.5 * bu[ns_h - 1] - 0.5 * bu[ns_h - 2]) *
                       vmecpp::Vmec::kSignOfJacobian * 2.0 * M_PI;
            const double *r1e = at(f_r1e, full1);
            const double *r1o = at(f_r1o, full1);
            const double *z1e = at(f_z1e, full1);
            const int outer_index = (ns - 1) * nZnT + 0;
            const int inner_index = (ns - 1) * nZnT + (sizes.nThetaReduced - 1);
            h_c.SetRadialExtent(
                {.r_outer = r1e[outer_index] + r1o[outer_index],
                 .r_inner = r1e[inner_index] + r1o[inner_index]});
            h_c.SetGeometricOffset({.r_00 = r1e[0], .z_00 = z1e[0]});

            outs.push_back(vmecpp::DeriveOutputQuantities(
                std::move(oq), indata_local[c], deriver->s_, deriver->fc_,
                deriver->constants_, deriver->t_, h_c, seed_oq.wout.mgrid_mode,
                vmecpp::VmecCheckpoint::NONE, vmecpp::VacuumPressureState::kOff,
                vmecpp::VmecStatus::SUCCESSFUL_TERMINATION,
                /*iter2=*/0));
            // DeriveOutputQuantities hard-codes iter2 = 0 and copies the
            // {fsqr,fsqz,fsql} residuals from the deriver's single-iteration
            // setup fc, so every derived configuration reports niter = 0 and
            // the same placeholder residuals. Replace them with this config's
            // batched-run values: the lock-step iteration count and the
            // normalized residual snapshot. Each output then reports its own
            // convergence (compare against wout.ftolv).
            {
              vmecpp::WOutFileContents &w_c = outs.back().wout;
              w_c.niter = batch_niter;
              const size_t s3 = static_cast<size_t>(3) * c;
              if (fsqr_snapshot_pc.size() >= s3 + 3) {
                w_c.fsqr = fsqr_snapshot_pc[s3 + 0];
                w_c.fsqz = fsqr_snapshot_pc[s3 + 1];
                w_c.fsql = fsqr_snapshot_pc[s3 + 2];
              }
            }
          }
          return true;
        };
#endif  // VMECPP_USE_CUDA
        const int n_cfg = static_cast<int>(indata_list.size());
        if (n_cfg <= 0) {
          throw std::runtime_error("run_batched_gpu: empty indata list");
        }
        const VmecINDATA &seed = indata_list[0];
        if (seed.ns_array.size() == 0) {
          throw std::runtime_error("run_batched_gpu: seed.ns_array is empty");
        }
        const int ns_first = seed.ns_array(0);
        const int mpol = seed.mpol;
        const int ntor = seed.ntor;
        // The distinct-mode block below populates indata_local with a
        // local copy of indata_list and patches each entry's
        // magnetic-axis Fourier coefficients in place. When distinct
        // mode is active, the seed-driven Vmec::run consumes
        // indata_local[0]; in broadcast mode the original indata_list[0]
        // reference suffices. The run_seed pointer is rebound below.
        std::vector<VmecINDATA> indata_local;
        const VmecINDATA *run_seed = &seed;
        // Validate shared shape across all inputs.
        for (int c = 1; c < n_cfg; ++c) {
          const VmecINDATA &v = indata_list[c];
          if (v.mpol != mpol || v.ntor != ntor || v.nfp != seed.nfp ||
              v.lasym != seed.lasym ||
              v.ns_array.size() != seed.ns_array.size() ||
              v.ns_array(0) != ns_first) {
            throw std::runtime_error(
                "run_batched_gpu: all inputs must share mpol, ntor, nfp, "
                "lasym, and ns_array[0]");
          }
        }
        // Free-boundary batches share one coil field: a single mgrid is
        // loaded for the run, so every input must carry the same mgrid
        // file and external currents. In distinct mode the boundaries and
        // magnetic axis may differ per configuration; the plasma profiles and
        // all other inputs are taken from the first configuration (the device
        // stages per-config geometry only).
        if (seed.lfreeb) {
          for (int c = 1; c < n_cfg; ++c) {
            const VmecINDATA &v = indata_list[c];
            const bool extcur_matches =
                v.extcur.size() == seed.extcur.size() &&
                (v.extcur.size() == 0 ||
                 (v.extcur.array() == seed.extcur.array()).all());
            if (!v.lfreeb || v.mgrid_file != seed.mgrid_file ||
                !extcur_matches) {
              throw std::runtime_error(
                  "run_batched_gpu: free-boundary batches must share "
                  "lfreeb, mgrid_file, and extcur across all inputs");
            }
          }
        }
        // Distinct-boundary path. Selected by setting the environment
        // variable VMECPP_BATCH_DISTINCT to a positive value. When
        // active, the per-configuration block below executes a
        // standalone Vmec::run for each input that returns at the
        // SETUP_INITIAL_STATE checkpoint with decomposed_x_[0]
        // populated; the host triplet
        // (decomposeInto, m1Constraint, extrapolateTowardsAxis) is then
        // invoked on physical_x_[0] and the resulting spectra are
        // written to a per-process batch-inputs file. The seed's first
        // invocation of the iteration body loads the per-configuration
        // spectra into d_specs_block. When the environment variable is
        // unset or zero the path is skipped and the seed's spectra are
        // broadcast across all configuration slots in the broadcast
        // branch further below.
        const char *distinct_env = std::getenv("VMECPP_BATCH_DISTINCT");
        const bool distinct_mode = distinct || (distinct_env != nullptr &&
                                                std::atoi(distinct_env) > 0);

        // run_batched_gpu stages only per-config boundary geometry and the
        // magnetic axis to the device. In broadcast mode (distinct mode off)
        // the seed -- configuration zero -- runs once and a single output is
        // returned. Any additional configurations passed in broadcast mode
        // share the seed's validated shape and are intentionally ignored; the
        // single-output broadcast contract is enforced at the return below.
        // Use distinct mode (VMECPP_BATCH_DISTINCT=1 or the run_batch distinct
        // parameter) to solve every configuration.
        // In distinct mode the device honors per-config boundary, magnetic
        // axis, and plasma profiles (each built into its own RadialProfiles and
        // staged per-config). Every other input -- the convergence controls
        // (ns_array, ftol_array, niter_array), resolution (mpol/ntor), and
        // free-boundary settings -- is still taken from the first
        // configuration. Reject batches that differ in anything else, so they
        // fail loudly instead of silently running with config 0's solver
        // settings.
        if (distinct_mode) {
          const std::string seed_json = seed.ToJsonOrException();
          for (int c = 1; c < n_cfg; ++c) {
            VmecINDATA probe = indata_list[c];
            probe.rbc = seed.rbc;
            probe.zbs = seed.zbs;
            probe.rbs = seed.rbs;
            probe.zbc = seed.zbc;
            probe.raxis_c = seed.raxis_c;
            probe.zaxis_s = seed.zaxis_s;
            probe.raxis_s = seed.raxis_s;
            probe.zaxis_c = seed.zaxis_c;
            // Per-config plasma profiles are now built from each config's
            // INDATA and staged to the device, so allow them to differ too.
            probe.phiedge = seed.phiedge;
            probe.ncurr = seed.ncurr;
            probe.pmass_type = seed.pmass_type;
            probe.am = seed.am;
            probe.am_aux_s = seed.am_aux_s;
            probe.am_aux_f = seed.am_aux_f;
            probe.pres_scale = seed.pres_scale;
            probe.gamma = seed.gamma;
            probe.spres_ped = seed.spres_ped;
            probe.piota_type = seed.piota_type;
            probe.ai = seed.ai;
            probe.ai_aux_s = seed.ai_aux_s;
            probe.ai_aux_f = seed.ai_aux_f;
            probe.pcurr_type = seed.pcurr_type;
            probe.ac = seed.ac;
            probe.ac_aux_s = seed.ac_aux_s;
            probe.ac_aux_f = seed.ac_aux_f;
            probe.curtor = seed.curtor;
            // Compare via JSON rather than the defaulted operator==, which
            // compares Eigen members coefficient-wise and is undefined for
            // arrays of differing length (e.g. a longer am) -- exactly a
            // difference we must report rather than crash on.
            if (probe.ToJsonOrException() != seed_json) {
              throw std::runtime_error(
                  "run_batched_gpu distinct mode stages per-config boundary "
                  "(rbc/zbs/rbs/zbc), magnetic axis (raxis/zaxis), and plasma "
                  "profiles (am/ai/ac/pres_scale/curtor/p*_type) to the "
                  "device; "
                  "all other inputs -- convergence controls (ns_array, "
                  "ftol_array, niter_array), resolution (mpol/ntor), and "
                  "free-boundary settings -- are taken from the first "
                  "configuration. Configuration " +
                  std::to_string(c) +
                  " differs from configuration 0 in such a field and would "
                  "otherwise run silently with configuration 0's settings. "
                  "Make "
                  "those inputs identical across configurations, or run the "
                  "configurations separately.");
            }
          }
        }

        // Restore every environment variable the binding mutates when the
        // call returns, so later runs in the process (batched or single)
        // see the caller's environment rather than this call's settings.
        static constexpr std::array<const char *, 6> kBatchEnvNames = {
            "VMECPP_N_CONFIG_MAX",
            "VMECPP_BATCH_INPUTS_FILE",
            "VMECPP_BATCH_OUTPUTS_FILE",
            "VMECPP_BATCH_DEC_X_FILE",
            "VMECPP_BATCH_MULTIGRID_UPSCALE",
            "VMECPP_BATCH_UPSCALE_KERNEL"};
        struct EnvRestore {
          std::array<std::optional<std::string>, kBatchEnvNames.size()> saved;
          EnvRestore() {
            for (size_t i = 0; i < kBatchEnvNames.size(); ++i) {
              if (const char *v = std::getenv(kBatchEnvNames[i])) {
                saved[i] = std::string(v);
              }
            }
          }
          ~EnvRestore() {
            for (size_t i = 0; i < kBatchEnvNames.size(); ++i) {
              if (saved[i]) {
                setenv(kBatchEnvNames[i], saved[i]->c_str(), 1);
              } else {
                unsetenv(kBatchEnvNames[i]);
              }
            }
          }
        } env_restore;

#ifdef VMECPP_USE_CUDA
        // Drop the per-config CUDA staging buffers when the call returns, so a
        // later run (batched or single) cannot pick up this batch's per-config
        // profiles. The per-config INDATA itself lives on the seed Vmec and is
        // released with it.
        struct BatchProfileRestore {
          ~BatchProfileRestore() { vmecpp::ClearBatchProfilesCuda(); }
        } batch_profile_restore;
#endif

        std::string batch_out_path;
        if (distinct_mode) {
          // Set N_CONFIG_MAX before init runs so GetNConfigMaxCuda caches
          // the right value.
          setenv("VMECPP_N_CONFIG_MAX", std::to_string(n_cfg).c_str(), 1);
          // The per-configuration multigrid stage transition is required
          // for distinct-mode correctness across the ns_array ramp; both
          // gates default on here, with an explicit =0 honored.
          const char *mg = std::getenv("VMECPP_BATCH_MULTIGRID_UPSCALE");
          if (!(mg != nullptr && std::atoi(mg) == 0)) {
            setenv("VMECPP_BATCH_MULTIGRID_UPSCALE", "1", 1);
          }
          const char *uk = std::getenv("VMECPP_BATCH_UPSCALE_KERNEL");
          if (!(uk != nullptr && std::atoi(uk) == 0)) {
            setenv("VMECPP_BATCH_UPSCALE_KERNEL", "1", 1);
          }
          const size_t spec_len =
              static_cast<size_t>(ns_first) * mpol * (ntor + 1);
          const int n_specs = 6;
          std::vector<double> batch_buf(static_cast<size_t>(n_specs) *
                                            static_cast<size_t>(n_cfg) *
                                            spec_len,
                                        0.0);
          // Parallel buffer for per-cfg decomposed_x_[0]. The physical_x
          // buffer above carries the post-decomposeInto + m1Constraint(1.0)
          // + extrapolateTowardsAxis state that the device-side d_specs_block
          // wants on iter 1. The decomposed_x buffer carries the pre-triplet
          // basis (the rmncc/rmnss/etc. coefficients * (1/scalxc), without
          // m=1 mixing or axis extrapolation) that the device-side
          // d_pts_x_* per-cfg slots want on iter 1. Without this, the
          // pts_x_initialized branches in PerformTimeStepCuda and
          // RecomposeToPhysicalCuda broadcast the seed's host m_decomposed_x
          // single-cfg state to all N cfg slots, overwriting the per-cfg
          // distinct initialization that the rest of the pipeline preserves.
          std::vector<double> dec_x_buf(static_cast<size_t>(n_specs) *
                                            static_cast<size_t>(n_cfg) *
                                            spec_len,
                                        0.0);

          // Per-cfg proactive axis recompute. VMEC's iter-1 path calls
          // RecomputeMagneticAxisToFixJacobianSign on bad_jacobian in
          // SolveEquilibriumLoop. At N>1 distinct, the seed's iter-1
          // recompute fires for cfg 0's boundary alone; cfg 1..N-1 keep the
          // original axis baked into their dumped spectra. Hoisting the
          // recompute into pre-init per cfg, and writing the corrected axis
          // back into the indata used by both pre-init and the seed, gives
          // each cfg its own corrected axis and leaves the iteration body's
          // axis-recovery path unexercised.
          // The pre-init Vmec drives the axis search using its own
          // setupFromIndata + flipTheta state; the corrected raxis_c /
          // zaxis_s / raxis_s / zaxis_c are then patched into the outer
          // indata_local copy, which is what the rest of the distinct-mode
          // path (and the seed run) consumes.
          indata_local.assign(indata_list.begin(), indata_list.end());
          run_seed = &indata_local[0];

          for (int c = 0; c < n_cfg; ++c) {
            // Probe Vmec: construct from the current (uncorrected) indata,
            // run the axis-recompute on its boundary, copy the corrected
            // axis arrays back into indata_local[c]. The probe Vmec is
            // discarded; pre-init below builds a fresh Vmec from the
            // patched indata.
            {
              // VMECPP_BATCH_AXIS_RECOMPUTE=0 skips the proactive axis
              // correction (the indata keeps its original axis and the
              // seed's iter-1 bad_jacobian path handles any sign flip, as
              // in single-cfg execution).
              static int axis_recompute_env = -1;
              if (axis_recompute_env < 0) {
                const char *e = std::getenv("VMECPP_BATCH_AXIS_RECOMPUTE");
                axis_recompute_env = (e && std::atoi(e) == 0) ? 0 : 1;
                if (!axis_recompute_env) {
                  std::fprintf(stderr,
                               "[pybind_vmec] per-cfg axis recompute DISABLED "
                               "(VMECPP_BATCH_AXIS_RECOMPUTE=0)\n");
                }
              }
              if (axis_recompute_env) {
                auto probe_or = vmecpp::Vmec::FromIndata(indata_local[c]);
                if (!probe_or.ok()) {
                  throw std::runtime_error(
                      "run_batched_gpu: axis-probe FromIndata failed for cfg " +
                      std::to_string(c) + ": " +
                      std::string(probe_or.status().message()));
                }
                auto probe = std::move(*probe_or);
                const int ns_first_c = indata_local[c].ns_array(0);
                probe->b_.RecomputeMagneticAxisToFixJacobianSign(
                    ns_first_c, vmecpp::Vmec::kSignOfJacobian);
                // Copy corrected axis back into the indata copy.
                const int ntor_c = indata_local[c].ntor;
                for (int n = 0; n <= ntor_c; ++n) {
                  indata_local[c].raxis_c[n] = probe->b_.raxis_c[n];
                  indata_local[c].zaxis_s[n] = probe->b_.zaxis_s[n];
                }
                if (indata_local[c].lasym) {
                  for (int n = 0; n <= ntor_c; ++n) {
                    if (indata_local[c].raxis_s.has_value()) {
                      (*indata_local[c].raxis_s)[n] = probe->b_.raxis_s[n];
                    }
                    if (indata_local[c].zaxis_c.has_value()) {
                      (*indata_local[c].zaxis_c)[n] = probe->b_.zaxis_c[n];
                    }
                  }
                }
              }
            }

            auto vmec_or = vmecpp::Vmec::FromIndata(indata_local[c]);
            if (!vmec_or.ok()) {
              throw std::runtime_error(
                  "run_batched_gpu: FromIndata failed for cfg " +
                  std::to_string(c) + ": " +
                  std::string(vmec_or.status().message()));
            }
            auto vmec = std::move(*vmec_or);

            auto run_or = vmec->run(vmecpp::VmecCheckpoint::SETUP_INITIAL_STATE,
                                    /*iterations_before_checkpointing=*/1,
                                    /*maximum_multi_grid_step=*/1);
            if (!run_or.ok()) {
              throw std::runtime_error(
                  "run_batched_gpu: init-only run failed for cfg " +
                  std::to_string(c) + ": " +
                  std::string(run_or.status().message()));
            }
            if (vmec->decomposed_x_.empty() ||
                vmec->decomposed_x_[0] == nullptr ||
                vmec->physical_x_.empty() || vmec->physical_x_[0] == nullptr ||
                vmec->p_.empty() || vmec->p_[0] == nullptr) {
              throw std::runtime_error(
                  "run_batched_gpu: null thread-vec after init for cfg " +
                  std::to_string(c));
            }
            // Extract per-cfg decomposed_x_[0] BEFORE the decomposeInto call
            // mutates physical_x_[0]; decomposed_x_[0] is the source and is
            // preserved by the call (only physical_x_[0] is written).
            auto &dx = *vmec->decomposed_x_[0];
            if (static_cast<size_t>(dx.rmncc.size()) != spec_len) {
              throw std::runtime_error(
                  "run_batched_gpu: decomposed_x rmncc.size() mismatch "
                  "for cfg " +
                  std::to_string(c));
            }
            const std::array<const double *, 6> dec_srcs = {
                dx.rmncc.data(), dx.rmnss.data(), dx.zmnsc.data(),
                dx.zmncs.data(), dx.lmnsc.data(), dx.lmncs.data()};
            const std::array<size_t, 6> dec_sizes = {
                static_cast<size_t>(dx.rmncc.size()),
                static_cast<size_t>(dx.rmnss.size()),
                static_cast<size_t>(dx.zmnsc.size()),
                static_cast<size_t>(dx.zmncs.size()),
                static_cast<size_t>(dx.lmnsc.size()),
                static_cast<size_t>(dx.lmncs.size())};
            for (int sp = 0; sp < n_specs; ++sp) {
              // The rmnss/zmncs/lmncs (odd-parity) spectra are empty for
              // axisymmetric (ntor = 0) inputs; leave their zero-initialized
              // dec_x_buf slots untouched rather than memcpy from an empty
              // span.
              if (dec_sizes[sp] < spec_len) continue;
              std::memcpy(
                  &dec_x_buf[static_cast<size_t>(sp) * n_cfg * spec_len +
                             static_cast<size_t>(c) * spec_len],
                  dec_srcs[sp], spec_len * sizeof(double));
            }
            vmec->decomposed_x_[0]->decomposeInto(*vmec->physical_x_[0],
                                                  vmec->p_[0]->scalxc);
            vmec->physical_x_[0]->m1Constraint(1.0);
            vmec->physical_x_[0]->extrapolateTowardsAxis();
            auto &px = *vmec->physical_x_[0];
            if (static_cast<size_t>(px.rmncc.size()) != spec_len) {
              throw std::runtime_error(
                  "run_batched_gpu: rmncc.size() mismatch for cfg " +
                  std::to_string(c));
            }
            const std::array<const double *, 6> srcs = {
                px.rmncc.data(), px.rmnss.data(), px.zmnsc.data(),
                px.zmncs.data(), px.lmnsc.data(), px.lmncs.data()};
            const std::array<size_t, 6> src_sizes = {
                static_cast<size_t>(px.rmncc.size()),
                static_cast<size_t>(px.rmnss.size()),
                static_cast<size_t>(px.zmnsc.size()),
                static_cast<size_t>(px.zmncs.size()),
                static_cast<size_t>(px.lmnsc.size()),
                static_cast<size_t>(px.lmncs.size())};
            for (int sp = 0; sp < n_specs; ++sp) {
              // Odd-parity spectra are empty for axisymmetric inputs; their
              // zero-initialized batch_buf slots are left untouched.
              if (src_sizes[sp] < spec_len) continue;
              std::memcpy(
                  &batch_buf[static_cast<size_t>(sp) * n_cfg * spec_len +
                             static_cast<size_t>(c) * spec_len],
                  srcs[sp], spec_len * sizeof(double));
            }
          }

      // Hand both staging blocks to the CUDA layer in memory; the
      // file pipeline (VMECPP_BATCH_INPUTS_FILE /
      // VMECPP_BATCH_DEC_X_FILE) remains for external drivers, so any
      // stale paths from the environment are cleared. The converged
      // spectra still dump to the outputs file, which the subprocess
      // recompute consumes.
#ifdef VMECPP_USE_CUDA
          vmecpp::SetBatchStagingCuda(batch_buf.data(), dec_x_buf.data(), n_cfg,
                                      ns_first, mpol, ntor);
#endif
          unsetenv("VMECPP_BATCH_INPUTS_FILE");
          unsetenv("VMECPP_BATCH_DEC_X_FILE");
          const std::string pid = std::to_string(getpid());
          batch_out_path =
              vmecpp::OsTmpDir() + "vmecpp_batch_outputs_" + pid + ".bin";
          setenv("VMECPP_BATCH_OUTPUTS_FILE", batch_out_path.c_str(), 1);
        } else {
          // Broadcast branch. The batched kernel chain executes at full
          // n_cfg with every configuration slot initialized to a copy of
          // the seed's spectra; the batch-inputs and batch-outputs file
          // environment variables are explicitly cleared so that the
          // device-side spectra-loading branch in CudaForward does not
          // attempt to consume stale paths from a prior invocation.
          (void)ns_first;
          (void)mpol;
          (void)ntor;
          setenv("VMECPP_N_CONFIG_MAX", std::to_string(n_cfg).c_str(), 1);
          unsetenv("VMECPP_BATCH_INPUTS_FILE");
          unsetenv("VMECPP_BATCH_OUTPUTS_FILE");
        }

        // Invoke the seed-driven iteration body. In distinct mode the
        // per-configuration initial spectra are loaded from the
        // batch-inputs file at the first invocation of CudaForward; in
        // broadcast mode the seed's spectra are broadcast across the
        // configuration slots. Subsequent iterations evolve each
        // configuration independently under the per-configuration
        // residual gates and the per-configuration kernel-skip mask.
        bool was_interrupted = false;
        auto interrupt_check = [&was_interrupted]() -> bool {
          if (was_interrupted) return true;
          py::gil_scoped_acquire acquire;
          if (PyErr_CheckSignals() != 0) {
            was_interrupted = true;
            return true;
          }
          return false;
        };
        absl::StatusOr<vmecpp::OutputQuantities> ret_seed;
        {
          py::gil_scoped_release release;
          // Run the seed configuration exactly as vmecpp::run would, but build
          // the Vmec here so the per-config INDATA can be handed to it directly
          // (distinct mode). Scoping it to this Vmec, instead of a global,
          // keeps it from leaking into any later run.
          auto seed_or = vmecpp::Vmec::FromIndata(
              *run_seed, /*magnetic_response_table=*/nullptr, max_threads,
              verbose, interrupt_check);
          if (!seed_or.ok()) {
            ret_seed = seed_or.status();
          } else {
            vmecpp::Vmec &seed_vmec = **seed_or;
#ifdef VMECPP_USE_CUDA
            if (distinct_mode) {
              seed_vmec.batch_indata_ = indata_local;
            }
#endif
            absl::StatusOr<bool> seed_ok = seed_vmec.run();
            if (seed_ok.ok()) {
              ret_seed = std::move(seed_vmec.output_quantities_);
            } else {
              ret_seed = seed_ok.status();
            }
          }
        }
#ifdef VMECPP_USE_CUDA
        // The staging belongs to this batched call alone; drop it before
        // any exit so a later run with a matching shape cannot consume it.
        vmecpp::ClearBatchStagingCuda();
#endif
        if (was_interrupted) {
          throw py::error_already_set();
        }
        if (!ret_seed.ok()) {
          throw std::runtime_error(
              "run_batched_gpu: batched VMEC run failed: " +
              std::string(ret_seed.status().message()));
        }

        // Construct the returned vector. Under broadcast mode every
        // configuration shares the seed's spectra and the
        // OutputQuantities is replicated. Under distinct mode each
        // configuration's converged spectra differs from cfg 0's, so
        // each cfg gets its own ComputeOutputQuantities pass via a
        // single-stage vmecpp::run with a HotRestartState whose wout
        // spectra are reconstructed from the per-cfg converged
        // decomposed_x written by FlushAllConfigsForOutputCuda to
        // /tmp/vmecpp_batch_outputs_<pid>.bin. Because the
        // HotRestartState carries the converged geometry as the
        // initial guess, each per-cfg iter loop converges in a small
        // number of iterations and the dominant cost is the
        // ComputeOutputQuantities post-processing rather than the
        // iter loop itself.
        //
        // VMECPP_PER_CFG_RECOMPUTE=0 disables per-cfg recompute and
        // restores the replication-by-ret_seed behaviour for diagnostics
        // or batched-throughput measurements where the per-cfg
        // correctness of the wout is not required.
        std::vector<vmecpp::OutputQuantities> outs;
        outs.reserve(n_cfg);

        // Per-configuration outputs are the distinct-mode default: the
        // direct derivation from the batched flush produces each
        // configuration's own OutputQuantities with no additional
        // iterations, and the hot-restart reconstruction plus the
        // subprocess recompute remain the fallbacks.
        bool per_cfg_recompute = distinct_mode && (n_cfg > 1);
        if (per_cfg_recompute) {
          const char *recompute_env = std::getenv("VMECPP_PER_CFG_RECOMPUTE");
          if (recompute_env != nullptr && std::atoi(recompute_env) == 0) {
            per_cfg_recompute = false;
          }
        }

        bool derived_direct = false;
#ifdef VMECPP_USE_CUDA
        if (per_cfg_recompute && ret_seed.ok()) {
          try {
            derived_direct = derive_outputs_from_batched_flush(
                indata_local, n_cfg, *ret_seed, outs);
          } catch (const std::exception &e) {
            std::fprintf(stderr,
                         "[pybind_vmec] direct per-cfg derivation threw "
                         "(%s); falling back to the hot-restart path\n",
                         e.what());
            derived_direct = false;
            outs.clear();
          }
          if (derived_direct) {
            std::fprintf(stderr,
                         "[pybind_vmec] per-cfg outputs derived from the "
                         "batched flush (%d cfgs)\n",
                         n_cfg);
          }
        }
#endif
        if (per_cfg_recompute && !derived_direct) {
          // Locate the batch_outputs file. The path may have been cleared
          // from the env (run_batched_gpu typically removes it at the end
          // of the call); reconstruct it from the pid in that case.
          std::string batch_out_path_eff = batch_out_path;
          if (batch_out_path_eff.empty()) {
            const std::string pid_s = std::to_string(getpid());
            batch_out_path_eff =
                vmecpp::OsTmpDir() + "vmecpp_batch_outputs_" + pid_s + ".bin";
          }
          // Read header and payload.
          FILE *f =  // NOLINT(cppcoreguidelines-owning-memory)
              std::fopen(batch_out_path_eff.c_str(), "rb");
          if (!f) {
            std::fprintf(
                stderr,
                "[pybind_vmec] per-cfg recompute: cannot open %s; falling "
                "back to ret_seed replication for all %d cfgs\n",
                batch_out_path_eff.c_str(), n_cfg);
            per_cfg_recompute = false;
          }
          std::vector<double> outs_buf;
          std::array<int32_t, 4> header = {0, 0, 0, 0};
          if (per_cfg_recompute) {
            const size_t hdr_read =
                std::fread(header.data(), sizeof(int32_t), 4, f);
            if (hdr_read != 4 || header[0] != n_cfg) {
              std::fprintf(
                  stderr,
                  "[pybind_vmec] per-cfg recompute: header mismatch "
                  "(read=%zu n=%d expected=%d); falling back to ret_seed\n",
                  hdr_read, header[0], n_cfg);
              per_cfg_recompute = false;
            }
          }
          int ns_out = 0, mpol_out = 0, ntor_out = 0;
          if (per_cfg_recompute) {
            ns_out = header[1];
            mpol_out = header[2];
            ntor_out = header[3];
            const size_t per_spec =
                static_cast<size_t>(ns_out) * mpol_out * (ntor_out + 1);
            const size_t total = static_cast<size_t>(n_cfg) * 6 * per_spec;
            outs_buf.resize(total);
            const size_t got =
                std::fread(outs_buf.data(), sizeof(double), total, f);
            if (got != total) {
              std::fprintf(
                  stderr,
                  "[pybind_vmec] per-cfg recompute: payload short read "
                  "(got=%zu expected=%zu); falling back to ret_seed\n",
                  got, total);
              per_cfg_recompute = false;
            }
          }
          if (f) std::fclose(f);  // NOLINT(cppcoreguidelines-owning-memory)

          if (per_cfg_recompute) {
            const size_t per_spec =
                static_cast<size_t>(ns_out) * mpol_out * (ntor_out + 1);
            // For each cfg, reconstruct an OutputQuantities by running
            // single-stage vmecpp::run with a HotRestartState whose
            // wout spectra are derived from the cfg's converged
            // decomposed_x via the inverse Fourier-basis transforms
            // cc_ss_to_cos and sc_cs_to_sin. The per-cfg indata's
            // ns_array, ftol_array, and niter_array are collapsed to
            // a single multigrid stage at the converged ns so the
            // hot-restart path is exercised and the multigrid ramp
            // is skipped.
            for (int c = 0; c < n_cfg; ++c) {
              try {
                // Per-spec base pointers into outs_buf at cfg c.
                // Layout: outs_buf[sp][cfg][j][m][n] flattened.
                auto spec_at = [&](int sp) -> const double * {
                  return outs_buf.data() +
                         static_cast<size_t>(sp) * n_cfg * per_spec +
                         static_cast<size_t>(c) * per_spec;
                };
                const double *d_rcc = spec_at(0);
                const double *d_rss = spec_at(1);
                const double *d_zsc = spec_at(2);
                const double *d_zcs = spec_at(3);
                const double *d_lsc = spec_at(4);
                const double *d_lcs = spec_at(5);

                const vmecpp::VmecINDATA &ind_c = indata_local[c];
                vmecpp::Sizes sizes_c(ind_c.lasym, ind_c.nfp, ind_c.mpol,
                                      ind_c.ntor, ind_c.ntheta, ind_c.nzeta);
                vmecpp::FourierBasisFastPoloidal fb_c(&sizes_c);

                const int mn_per_surface = mpol_out * (ntor_out + 1);
                const int mnmax =
                    (ntor_out + 1) + (mpol_out - 1) * (2 * ntor_out + 1);
                vmecpp::RowMatrixXd rmnc(mnmax, ns_out);
                vmecpp::RowMatrixXd zmns(mnmax, ns_out);
                vmecpp::RowMatrixXd lmns_full(mnmax, ns_out);
                rmnc.setZero();
                zmns.setZero();
                lmns_full.setZero();

                // ComputeOutputQuantities applies the m=1 internal-to-
                // physical mixing m1Constraint(1.0) on the (rmnss,
                // zmncs) pair before writing the wout, so that the
                // composition with InitFromState's m1Constraint(0.5)
                // round-trips to identity. Applying the same mixing
                // here is what makes the hot-restart land back at
                // the iter loop's converged state on iter 1; without
                // it InitFromState halves the m=1 modes and the
                // Jacobian sign flips on the very first iteration.
                std::vector<double> rmncc_phys(mn_per_surface);
                std::vector<double> rmnss_phys(mn_per_surface);
                std::vector<double> zmnsc_phys(mn_per_surface);
                std::vector<double> zmncs_phys(mn_per_surface);
                std::vector<double> rmnc_col(mnmax);
                std::vector<double> zmns_col(mnmax);
                for (int jF = 0; jF < ns_out; ++jF) {
                  const double *p_rcc =
                      d_rcc + static_cast<size_t>(jF) * mn_per_surface;
                  const double *p_rss =
                      d_rss + static_cast<size_t>(jF) * mn_per_surface;
                  const double *p_zsc =
                      d_zsc + static_cast<size_t>(jF) * mn_per_surface;
                  const double *p_zcs =
                      d_zcs + static_cast<size_t>(jF) * mn_per_surface;
                  for (int k = 0; k < mn_per_surface; ++k) {
                    rmncc_phys[k] = p_rcc[k];
                    rmnss_phys[k] = p_rss[k];
                    zmnsc_phys[k] = p_zsc[k];
                    zmncs_phys[k] = p_zcs[k];
                  }
                  // m1 internal-to-physical mixing on (rmnss, zmncs)
                  // pair: scalingFactor=1.0, mirrors the loop at
                  // output_quantities.cc:4567.
                  for (int n = 0; n < ntor_out + 1; ++n) {
                    const int idx_m1 = 1 * (ntor_out + 1) + n;
                    const double old_rss = rmnss_phys[idx_m1];
                    rmnss_phys[idx_m1] = old_rss + zmncs_phys[idx_m1];
                    zmncs_phys[idx_m1] = old_rss - zmncs_phys[idx_m1];
                  }
                  fb_c.cc_ss_to_cos(std::span<const double>(rmncc_phys.data(),
                                                            mn_per_surface),
                                    std::span<const double>(rmnss_phys.data(),
                                                            mn_per_surface),
                                    std::span<double>(rmnc_col.data(), mnmax),
                                    ntor_out, mpol_out);
                  fb_c.sc_cs_to_sin(std::span<const double>(zmnsc_phys.data(),
                                                            mn_per_surface),
                                    std::span<const double>(zmncs_phys.data(),
                                                            mn_per_surface),
                                    std::span<double>(zmns_col.data(), mnmax),
                                    ntor_out, mpol_out);
                  // ComputeOutputQuantities zeros m>0 modes at jF=0
                  // (the magnetic axis carries only m=0 components in
                  // physical-space wout). The cc_ss_to_cos basis
                  // transform is the same shape per surface, so it
                  // does not enforce this; without explicit zeroing
                  // the axis surface picks up non-zero m>0 modes that
                  // InitFromState reads back into decomposed_x at
                  // jF=0, where the iter loop's Jacobian sign check
                  // sees an ill-formed axis and flips to BAD_JACOBIAN
                  // on iter 1. The zero range mirrors the loop in
                  // output_quantities.cc that skips m>0, n!=0 at the
                  // jF == 0 surface during wout serialization.
                  if (jF == 0) {
                    int mn = 0;
                    // m=0: keep n in 0..ntor (ntor+1 entries)
                    mn = ntor_out + 1;
                    // m>=1, n in -ntor..ntor: zero (2*ntor+1 entries
                    // per m row).
                    for (int m = 1; m < mpol_out; ++m) {
                      for (int n = -ntor_out; n <= ntor_out; ++n) {
                        rmnc_col[mn] = 0.0;
                        zmns_col[mn] = 0.0;
                        mn++;
                      }
                    }
                  }
                  for (int k = 0; k < mnmax; ++k) {
                    rmnc(k, jF) = rmnc_col[k];
                    zmns(k, jF) = zmns_col[k];
                  }
                }
                // lmns_full stays zero. The wout format applies a
                // per-surface (lamscale / phipF) scaling to lambda when
                // ComputeOutputQuantities serializes it, and
                // InitFromState undoes that scaling on read. Computing
                // the scaling here without re-deriving the per-surface
                // radial profiles is awkward; zeroing lmns_full instead
                // costs a few extra iters of lambda re-convergence
                // inside the per-cfg iter loop, since R and Z are
                // already at their converged values.
                (void)d_lsc;
                (void)d_lcs;

                // Build the single-stage indata. Use the converged ns
                // value (ns_out), the tightest tolerance from the
                // original multigrid schedule for verification, and a
                // small niter cap so hot-restart converges quickly.
                vmecpp::VmecINDATA ind_single = ind_c;
                ind_single.ns_array = Eigen::VectorXi::Constant(1, ns_out);
                const int last_ftol =
                    static_cast<int>(ind_c.ftol_array.size()) - 1;
                const double ftolv_last =
                    (last_ftol >= 0) ? ind_c.ftol_array[last_ftol] : 1.0e-12;
                ind_single.ftol_array =
                    Eigen::VectorXd::Constant(1, ftolv_last);
                // R and Z hot-start at their converged values; lambda
                // re-converges from zero and needs the headroom.
                ind_single.niter_array = Eigen::VectorXi::Constant(1, 5000);

                // Construct a partial WOutFileContents carrying only
                // the spectra. InitFromState reads rmnc, zmns, and
                // lmns_full; CheckInitialState verifies lasym, mpol,
                // ntor, and ns_array.back() match the run-time
                // indata. Other wout fields stay default-initialized.
                vmecpp::WOutFileContents wout_c;
                wout_c.mpol = mpol_out;
                wout_c.ntor = ntor_out;
                wout_c.lasym = ind_c.lasym;
                wout_c.nfp = ind_c.nfp;
                wout_c.ns = ns_out;
                wout_c.rmnc = rmnc;
                wout_c.zmns = zmns;
                wout_c.lmns_full = lmns_full;

                // Construct HotRestartState. The carried indata must
                // pass CheckInitialState against ind_single, so use the
                // same single-stage shape for both the hot-restart
                // indata and the run-time indata.
                vmecpp::HotRestartState hot_restart(std::move(wout_c),
                                                    ind_single);

                // Clear the VMECPP_BATCH_* distinct-mode variables before
                // this per-configuration run so the new Vmec instance
                // does not re-enter the distinct-mode pre-initialization
                // or the batched dec_x load. The CUDA state's
                // n_config_max stays cached from the batched run, so
                // VMECPP_ACTIVE_PER_CFG_OVERRIDE_BITS pins the new Vmec
                // to configuration zero: ResizeForBatch and
                // ResetActivePerCfgForNextStage see a {1, 0, ..., 0}
                // mask and the kernels skip the stale slots left from
                // the batched run.
                unsetenv("VMECPP_BATCH_DISTINCT");
                unsetenv("VMECPP_BATCH_INPUTS_FILE");
                unsetenv("VMECPP_BATCH_OUTPUTS_FILE");
                unsetenv("VMECPP_BATCH_DEC_X_FILE");
                std::string active_mask(n_cfg, '0');
                active_mask[0] = '1';
                setenv("VMECPP_ACTIVE_PER_CFG_OVERRIDE_BITS",
                       active_mask.c_str(), 1);

            // Reset the persistent CUDA state before each
            // per-configuration run so the next vmecpp::run stages
            // its host decomposed position and velocity into the
            // configuration-zero device slots from scratch.
            // Otherwise the batched run's stale device buffers leak
            // into the hot restart and the first iteration's
            // geometry contradicts the hot-restart host state,
            // which forces BAD_JACOBIAN.
#ifdef VMECPP_USE_CUDA
                vmecpp::ResetCudaStateForNewVmecRun();
#endif

                // Single-stage hot-restart run reconstructing this cfg's
                // OutputQuantities; falls back to the seed output on
                // failure.
                absl::StatusOr<vmecpp::OutputQuantities> ret_c;
                {
                  py::gil_scoped_release release;
                  ret_c = vmecpp::run(ind_single, std::move(hot_restart),
                                      /*max_threads=*/1, verbose);
                }
                if (ret_c.ok()) {
                  outs.push_back(*ret_c);
                } else {
                  std::fprintf(
                      stderr,
                      "[pybind_vmec] per-cfg recompute: cfg %d run failed "
                      "(%s); falling back to the seed output\n",
                      c, std::string(ret_c.status().message()).c_str());
                  outs.push_back(*ret_seed);
                }
              } catch (const std::exception &e) {
                std::fprintf(
                    stderr,
                    "[pybind_vmec] per-cfg recompute: cfg %d threw (%s); "
                    "falling back to ret_seed\n",
                    c, e.what());
                outs.push_back(*ret_seed);
              }
            }
            // Clear the per-cfg active-mask override so subsequent
            // unrelated vmecpp::run() calls (in the same process) see
            // the default all-active behaviour.
            unsetenv("VMECPP_ACTIVE_PER_CFG_OVERRIDE_BITS");
            std::fprintf(
                stderr,
                "[pybind_vmec] per-cfg recompute: %d/%d cfgs reconstructed "
                "via single-stage hot-restart\n",
                static_cast<int>(outs.size()), n_cfg);
          } else {
            // Recompute disabled mid-flight (header mismatch or file
            // missing): replicate ret_seed for the remaining slots.
            for (int c = static_cast<int>(outs.size()); c < n_cfg; ++c) {
              outs.push_back(*ret_seed);
            }
          }
        }

        if (!per_cfg_recompute || static_cast<int>(outs.size()) != n_cfg) {
          outs.clear();
          if (distinct_mode) {
            // Fallback for a failed per-configuration derivation:
            // replicate the seed output so the caller still receives one
            // entry per input.
            outs.assign(n_cfg, *ret_seed);
          } else {
            // Broadcast mode solves one boundary in every slot; return
            // the single converged result instead of N copies.
            outs.push_back(*ret_seed);
          }
        }

        // Clean up the outputs file unless VMECPP_KEEP_BATCH_FILES=1 keeps
        // it for downstream inspection (the staging inputs travel in
        // memory and leave no files).
        const char *keep_env = std::getenv("VMECPP_KEEP_BATCH_FILES");
        const bool keep_files =
            (keep_env != nullptr && std::atoi(keep_env) > 0);
        if (distinct_mode && !keep_files) {
          if (!batch_out_path.empty()) std::remove(batch_out_path.c_str());
        }
        if (distinct_mode && keep_files) {
          std::fprintf(stderr,
                       "[pybind_vmec] VMECPP_KEEP_BATCH_FILES=1: preserving "
                       "%s\n",
                       batch_out_path.c_str());
        }

        if (return_spectra) {
          // Converged per-cfg spectra as a numpy array of shape
          // [6][n_cfg][spec], the same block the outputs-file dump
          // carries; None when the run held a single configuration slot.
          py::object spectra_obj = py::none();
#ifdef VMECPP_USE_CUDA
          std::vector<double> spectra;
          int sn = 0;
          int sns = 0;
          int smpol = 0;
          int sntor = 0;
          if (vmecpp::GetBatchOutputSpectraCuda(&spectra, &sn, &sns, &smpol,
                                                &sntor)) {
            const py::ssize_t spec_len =
                static_cast<py::ssize_t>(sns) * smpol * (sntor + 1);
            py::array_t<double> arr({static_cast<py::ssize_t>(6),
                                     static_cast<py::ssize_t>(sn), spec_len});
            std::memcpy(arr.mutable_data(), spectra.data(),
                        spectra.size() * sizeof(double));
            spectra_obj = std::move(arr);
          }
#endif
          return py::make_tuple(py::cast(outs), spectra_obj);
        }
        return py::cast(outs);
      },
      py::arg("indata_list"), py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress,
      py::arg("return_spectra") = false, py::arg("distinct") = false,
      R"pbdoc(Batched-GPU solver for N fixed-boundary equilibria.

Solves a list of VmecINDATA inputs in a single CUDA-resident iteration
loop. All inputs must share the shape parameters mpol, ntor, nfp, lasym,
and ns_array[0], since the persistent device buffers are dimensioned at
the first invocation of the iteration body. The behavior is selected by
the distinct argument, or the VMECPP_BATCH_DISTINCT environment variable
which is still honored as a fallback:

  Broadcast mode (default): the first input's spectra are broadcast
  across all N configuration slots, the batched kernel chain executes
  at full N for measurement and infrastructure exercise, and the
  returned list contains the single converged OutputQuantities.

  Distinct mode (distinct=True or VMECPP_BATCH_DISTINCT=1): a
  per-configuration pre-initialization stage extracts each input's
  initial spectra,
  patches the per-configuration magnetic axis with the proactive
  recomputation, and hands the concatenated spectra to the iteration
  body in memory. The iteration body loads the per-configuration
  spectra at its first invocation of CudaForward and drives each
  configuration through its own residual evolution under the
  per-configuration active-mask kernels. The returned list holds one
  OutputQuantities per input, derived in process from the batched
  flush (VMECPP_PER_CFG_RECOMPUTE=0 opts out and replicates the seed
  output instead). The per-configuration multigrid-upscale gates are
  enabled automatically in this mode.

Every environment variable the call mutates is restored on return, so
later runs in the same process see the caller's environment.

With return_spectra=True the return value is a tuple (outputs,
spectra), where spectra is a float64 array of shape [6][n_cfg][spec]
holding each configuration's converged decomposed spectra (rmncc,
rmnss, zmnsc, zmncs, lmnsc, lmncs; spec = ns * mpol * (ntor + 1) at
the final multigrid stage), or None when the run held a single
configuration slot.
)pbdoc");

  // Single-resolution iteration model: exposes the forward model and the
  // time-step / restart primitives so the equilibrium iteration can be driven
  // from Python (see vmecpp._iteration).
  py::class_<VmecModel>(m, "VmecModel")
      .def_static("create", &VmecModel::Create, py::arg("indata"),
                  py::arg("ns"), py::arg("initial_state") = std::nullopt)
      .def("evaluate", &VmecModel::Evaluate, py::arg("iter1"), py::arg("iter2"),
           py::arg("precondition") = true)
      .def_property_readonly("need_restart", &VmecModel::need_restart)
      .def("perform_time_step", &VmecModel::PerformTimeStep,
           py::arg("velocity_scale"), py::arg("conjugation_parameter"),
           py::arg("time_step"))
      .def("save_backup", &VmecModel::SaveBackup)
      .def("restore_backup", &VmecModel::RestoreBackup)
      .def("zero_velocity", &VmecModel::ZeroVelocity)
      .def("reset_to_initial_guess", &VmecModel::ResetToInitialGuess)
      .def("recompute_axis", &VmecModel::RecomputeAxis)
      .def("reinitialize", &VmecModel::Reinitialize)
      .def("refine_to", &VmecModel::RefineTo, py::arg("new_ns"))
      .def("solve", &VmecModel::Solve)
      .def("get_state", &VmecModel::GetState)
      .def("set_state", &VmecModel::SetState, py::arg("state"))
      .def("get_forces", &VmecModel::GetForces)
      .def("apply_preconditioner", &VmecModel::ApplyPreconditioner,
           py::arg("v"))
      .def("hessian_vector_product", &VmecModel::HessianVectorProduct,
           py::arg("v"), py::arg("eps_rel") = 1e-7)
      .def_property_readonly("force_eval_count", &VmecModel::force_eval_count)
      .def("reset_force_eval_count", &VmecModel::reset_force_eval_count)
      .def_property_readonly("fsqr", &VmecModel::fsqr)
      .def_property_readonly("fsqz", &VmecModel::fsqz)
      .def_property_readonly("fsql", &VmecModel::fsql)
      .def_property_readonly("fsqr1", &VmecModel::fsqr1)
      .def_property_readonly("fsqz1", &VmecModel::fsqz1)
      .def_property_readonly("fsql1", &VmecModel::fsql1)
      .def_property_readonly("mhd_energy", &VmecModel::mhd_energy)
      .def_property("restart_reason", &VmecModel::restart_reason,
                    &VmecModel::set_restart_reason)
      .def_property_readonly("status", &VmecModel::status)
      .def_property_readonly("ftolv", &VmecModel::ftolv)
      .def_property_readonly("niterv", &VmecModel::niterv)
      .def_property_readonly("delt", &VmecModel::delt)
      .def_property_readonly("iteration_style", &VmecModel::iteration_style)
      .def_property_readonly("ns", &VmecModel::ns)
      .def_property_readonly("mpol", &VmecModel::mpol)
      .def_property_readonly("ntor", &VmecModel::ntor)
      .def_property_readonly("lthreed", &VmecModel::lthreed)
      .def_property_readonly("lasym", &VmecModel::lasym)
      .def_property_readonly("force_residual_r", &VmecModel::force_residual_r)
      .def_property_readonly("force_residual_z", &VmecModel::force_residual_z)
      .def_property_readonly("force_residual_lambda",
                             &VmecModel::force_residual_lambda)
      .def_property_readonly("restart_reasons", &VmecModel::restart_reasons)
      .def_property_readonly("ijacob", &VmecModel::ijacob)
      .def_property_readonly("raxis_c", &VmecModel::raxis_c)
      .def_static("openmp_enabled", &VmecModel::openmp_enabled);
}  // NOLINT(readability/fn_size)
