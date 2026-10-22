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
#include <cstdint>
#include <cstring>  // memcpy
#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move
#include <vector>

#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/geometry/geometry.h"
#include "vmecpp/vmec/geometry/vmec_geometry.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

#ifdef VMECPP_USE_CUDA
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda.h"
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

// Raised for a mismatch between the hot restart state and the indata; a
// subclass of ValueError, which these errors used to be.
py::handle hot_restart_mismatch_error;

template <typename T>
T &GetValueOrThrow(absl::StatusOr<T> &s) {
  if (!s.ok()) {
    // Could handle more exceptions, but only some are translated to meaningful
    // python exception types.
    // https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
    if (s.status().GetPayload(vmecpp::kHotRestartMismatchPayload)) {
      PyErr_SetString(hot_restart_mismatch_error.ptr(),
                      std::string(s.status().message()).c_str());
      throw py::error_already_set();
    }
    if (absl::IsInvalidArgument(s.status())) {
      throw pybind11::value_error(std::string(s.status().message()));
    } else {
      throw std::runtime_error(std::string(s.status().message()));
    }
  }
  return s.value();
}

// Adapts a Python iteration callback to vmecpp::IterationCallback: the hook
// acquires the GIL, hands the callback a copy of the state that Python owns,
// takes curr_h back from that copy, treats a None return as "keep going", and
// stops the run on an exception, which Rethrow raises once the run has
// returned. A return value that does not convert to a bool stops the run with
// a TypeError.
struct PythonIterationCallback {
  py::object callable;
  std::optional<py::error_already_set> error;

  vmecpp::IterationCallback Hook() {
    if (callable.is_none()) {
      return nullptr;
    }
    return [this](vmecpp::SolverState &m_state) -> bool {
      py::gil_scoped_acquire acquire;
      try {
        py::object state = py::cast(m_state, py::return_value_policy::copy);
        py::object keep_going = callable(state);
        m_state.curr_h = state.cast<const vmecpp::SolverState &>().curr_h;
        if (keep_going.is_none()) {
          return true;
        }
        try {
          return py::cast<bool>(keep_going);
        } catch (const py::cast_error &) {
          PyErr_Format(PyExc_TypeError,
                       "iteration_callback must return None or a bool, but "
                       "returned an object of type %s",
                       Py_TYPE(keep_going.ptr())->tp_name);
          throw py::error_already_set();
        }
      } catch (py::error_already_set &e) {
        error.emplace(std::move(e));
        return false;
      }
    };
  }

  void Rethrow() {
    if (error.has_value()) {
      throw *error;
    }
  }
};

// Adapts a Python callable f(r, phi, z) -> (b_r, b_phi, b_z) to
// vmecpp::MagneticFieldFunction: each call acquires the GIL; an exception, or
// a result that is not three arrays of the length of r, fails the vacuum
// solve, and Rethrow raises it once the run has returned.
struct PythonMagneticField {
  py::object callable;
  std::optional<py::error_already_set> error;

  vmecpp::MagneticFieldFunction Function() {
    return [this](const Eigen::VectorXd &r, const Eigen::VectorXd &phi,
                  const Eigen::VectorXd &z, Eigen::VectorXd &m_b_r,
                  Eigen::VectorXd &m_b_phi,
                  Eigen::VectorXd &m_b_z) -> absl::Status {
      py::gil_scoped_acquire acquire;
      if (error.has_value()) {
        return absl::CancelledError("magnetic_field raised an exception");
      }
      try {
        py::object b = callable(r, phi, z);
        std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> b_rpz;
        try {
          b_rpz = b.cast<decltype(b_rpz)>();
        } catch (const py::cast_error &) {
          PyErr_SetString(PyExc_TypeError,
                          "magnetic_field must return three 1D float arrays "
                          "(b_r, b_phi, b_z)");
          throw py::error_already_set();
        }
        auto &[b_r, b_phi, b_z] = b_rpz;
        if (b_r.size() != r.size() || b_phi.size() != r.size() ||
            b_z.size() != r.size()) {
          PyErr_Format(PyExc_ValueError,
                       "magnetic_field must return arrays of length %zd, the "
                       "number of points, but returned lengths %zd, %zd, %zd",
                       r.size(), b_r.size(), b_phi.size(), b_z.size());
          throw py::error_already_set();
        }
        m_b_r = b_r;
        m_b_phi = b_phi;
        m_b_z = b_z;
        return absl::OkStatus();
      } catch (py::error_already_set &e) {
        error.emplace(std::move(e));
        return absl::CancelledError("magnetic_field raised an exception");
      }
    };
  }

  void Rethrow() {
    if (error.has_value()) {
      throw *error;
    }
  }
};

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
#ifdef VMECPP_USE_CUDA
    // The model evaluates the forward model state by state through the host
    // arrays, while the CUDA build keeps the iteration state and the
    // preconditioner on the device and sizes them once per Vmec::run.
    throw std::runtime_error(
        "VmecModel drives the host iteration and is not available in the "
        "CUDA build (VMECPP_USE_CUDA); use vmecpp.run or run_batched_gpu, or "
        "build without VMECPP_USE_CUDA for the iteration API");
#endif
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
  // always_fix_m1_gauge selects the force system: with true the m=1 gauge
  // force is zeroed (the system solve() iterates when the model's
  // always_fix_m1_gauge property is set), with false the gauge force is
  // kept until fsqz < 1e-6 as in the native iteration, so F(x) then also
  // depends on the previously evaluated residual. The exact Hessian-vector
  // products take the same flag and differentiate the same system.
  void Evaluate(int iter1, int iter2, bool precondition = true,
                bool always_fix_m1_gauge = true) {
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
          /*verbose=*/false, always_fix_m1_gauge);
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
  std::int64_t force_eval_count() const {
    return vmec_->m_[0]->forceEvaluationCount();
  }
  void reset_force_eval_count() const {
    vmec_->m_[0]->resetForceEvaluationCount();
  }
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
  // Back up the last state with a valid force evaluation.
  void SaveEvaluatedBackup() const {
    *vmec_->physical_x_backup_[0] = *vmec_->last_evaluated_x_[0];
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
    vmec_->b_.RecomputeMagneticAxisToFixJacobianSign(vmec_->fc_.nsval,
                                                     vmec_->indata_.signgs);
  }

  // Recompute the magnetic axis to fix the Jacobian sign, then re-initialize
  // the radial state from the new axis. This is the C++ axis reguess in
  // Vmec::SolveEquilibriumLoop, but it re-runs InitializeRadial so the forward
  // model (preconditioner + scratch) starts clean; the in-place reset alone
  // leaves stale forward-model state from the failed evaluation, which the C++
  // single continuous parallel region tolerates but a step-by-step driver does
  // not.
  void Reinitialize() {
    vmec_->b_.RecomputeMagneticAxisToFixJacobianSign(vmec_->fc_.nsval,
                                                     vmec_->indata_.signgs);
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
  void RefineTo(int new_ns, std::optional<vmecpp::MultigridInterpolationScheme>
                                interpolation = std::nullopt) {
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
                       delt0, std::nullopt, interpolation);
    last_preconditioner_update_ = 0;
    last_full_update_nestor_ = 0;
  }

  bool always_fix_m1_gauge() const { return vmec_->always_fix_m1_gauge_; }

  // Reference C++ inner iteration (the loop being ported), for verification.
  // Each call converges the *current* resolution ftol_array entry.
  // Exhausting the iteration budget does not raise here.
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
    exact_primal_valid_ = false;  // primal geometry cache is stale
  }
  // Flat force vector (decomposed/preconditioned), valid after Evaluate().
  Eigen::VectorXd GetForces() const {
    return FlattenActive(*vmec_->decomposed_f_[0], vmec_->s_);
  }

  // Return the output-independent geometry representation of the current
  // in-memory state. This gathers only the geometry-layer data; it does not
  // read or write a wout file.
  vmecpp::Geometry GetGeometry() const {
    if (vmec_->r_.empty()) {
      throw std::runtime_error(
          "VmecModel.get_geometry: model is not initialized");
    }
    const vmecpp::VmecInternalResults internal = vmecpp::GatherDataFromThreads(
        vmec_->indata_.signgs, vmec_->s_, vmec_->fc_, vmec_->constants_,
        vmec_->r_, vmec_->decomposed_x_, vmec_->m_, vmec_->p_);
    return vmecpp::MakeGeometry(vmec_->indata_, internal);
  }

  // Transpose of the linear state-to-geometry coefficient map used by
  // MakeGeometry. coefficient_bar contains twelve dense coefficient blocks in
  // the same order as GeometryCoefficients (r_cc, r_ss, r_sc, r_cs, z_sc,
  // z_cs, z_cc, z_ss, lambda_sc, lambda_cs, lambda_cc, lambda_ss), each with
  // surface-major (j, m, n) storage.
  //
  // poloidal_flux_bar is the separate cotangent of MakeGeometry's poloidal_flux
  // output (one entry per full surface); pass an empty vector where it is
  // zero. With ncurr=0 both flux profiles are prescribed input profiles and
  // have zero state derivative, so the toroidal_flux cotangent never enters
  // this map. With ncurr=1 the toroidal flux is still prescribed, but the
  // poloidal flux is chi_j = sum_{k<j} c_k iota_k with c_k the (state-
  // independent) half-grid flux step, and iota_k = chi'_k / phipH_k depends on
  // the state through the prescribed-current chi' (see
  // local_force_composition.h); that dependence is added to the state
  // cotangent here via chip_state_vjp.
  Eigen::VectorXd GeometryStateVjp(const Eigen::VectorXd &coefficient_bar,
                                   const Eigen::VectorXd &poloidal_flux_bar) {
    if (vmec_->indata_.lfreeb) {
      throw std::runtime_error(
          "VmecModel.geometry_state_vjp currently supports fixed-boundary "
          "models only");
    }
    if (vmec_->indata_.lasym) {
      throw std::runtime_error(
          "VmecModel.geometry_state_vjp currently supports stellarator-"
          "symmetric models only");
    }
    if (poloidal_flux_bar.size() != 0 &&
        poloidal_flux_bar.size() != vmec_->fc_.ns) {
      throw std::runtime_error(
          "VmecModel.geometry_state_vjp: poloidal_flux_bar has wrong length");
    }
    const int modes_per_surface = vmec_->s_.mpol * (vmec_->s_.ntor + 1);
    const int coefficient_size = vmec_->fc_.ns * modes_per_surface;
    if (coefficient_bar.size() != 12 * coefficient_size) {
      throw std::runtime_error(
          "VmecModel.geometry_state_vjp: coefficient cotangent has wrong "
          "length");
    }

    vmecpp::FourierGeometry state_bar(&vmec_->s_, vmec_->r_[0].get(),
                                      vmec_->fc_.ns);
    state_bar.setZero();
    const double lambda_scale = vmec_->constants_.lamscale;
    const Eigen::VectorXd &phip_f = vmec_->p_[0]->phipF;

    auto coefficient_scale = [&](int j, int m, int n, bool lambda) -> double {
      double scale = (m == 0 ? 1.0 : std::numbers::sqrt2) *
                     (n == 0 ? 1.0 : std::numbers::sqrt2);
      if (lambda) {
        if (j >= phip_f.size() || phip_f[j] == 0.0) {
          throw std::runtime_error(
              "VmecModel.geometry_state_vjp: invalid lambda scale");
        }
        scale *= lambda_scale / phip_f[j];
      }
      return scale;
    };

    auto add_block = [&](std::span<double> destination, int bar_block,
                         bool lambda, int skipped_mode = -1) {
      const int block_offset = bar_block * coefficient_size;
      for (int j = 0; j < vmec_->fc_.ns; ++j) {
        for (int m = 0; m < vmec_->s_.mpol; ++m) {
          if (m == skipped_mode) continue;
          for (int n = 0; n <= vmec_->s_.ntor; ++n) {
            const int index =
                (j * vmec_->s_.mpol + m) * (vmec_->s_.ntor + 1) + n;
            destination[index] += coefficient_scale(j, m, n, lambda) *
                                  coefficient_bar[block_offset + index];
          }
        }
      }
    };

    const double sigma = -vmec_->indata_.signgs;
    auto add_m1_pair = [&](std::span<double> first, std::span<double> second,
                           int first_bar_block, int second_bar_block) {
      const int first_offset = first_bar_block * coefficient_size;
      const int second_offset = second_bar_block * coefficient_size;
      for (int j = 0; j < vmec_->fc_.ns; ++j) {
        for (int n = 0; n <= vmec_->s_.ntor; ++n) {
          const int index = (j * vmec_->s_.mpol + 1) * (vmec_->s_.ntor + 1) + n;
          const double scale = coefficient_scale(j, 1, n, false);
          first[index] +=
              scale * (coefficient_bar[first_offset + index] +
                       sigma * coefficient_bar[second_offset + index]);
          second[index] +=
              scale * (sigma * coefficient_bar[first_offset + index] -
                       coefficient_bar[second_offset + index]);
        }
      }
    };

    add_block(state_bar.rmncc, 0, false);
    if (vmec_->s_.lthreed) {
      if (vmec_->s_.mpol > 1) {
        add_block(state_bar.rmnss, 1, false, 1);
        add_block(state_bar.zmncs, 5, false, 1);
        add_m1_pair(state_bar.rmnss, state_bar.zmncs, 1,
                    5);  // r_ss, z_cs
      } else {
        add_block(state_bar.rmnss, 1, false);
      }
    }
    add_block(state_bar.zmnsc, 4, false);
    if (vmec_->s_.lthreed) {
      if (vmec_->s_.mpol == 1) add_block(state_bar.zmncs, 5, false);
    }

    add_block(state_bar.lmnsc, 8, true);
    if (vmec_->s_.lthreed) add_block(state_bar.lmncs, 9, true);
    Eigen::VectorXd result = FlattenActive(state_bar, vmec_->s_);

    if (vmec_->indata_.ncurr != 0 && poloidal_flux_bar.size() != 0 &&
        poloidal_flux_bar.cwiseAbs().maxCoeff() != 0.0) {
#ifdef VMECPP_ENABLE_ENZYME
      // For ncurr==1 chipH_k is the state-dependent prescribed-current chi'
      // differentiated in local_force_composition.h.
      result += ChipStateVjp(ChipBarFromPoloidalFluxBar(poloidal_flux_bar));
#else
      throw std::runtime_error(
          "VmecModel.geometry_state_vjp: a nonzero poloidal_flux_bar with "
          "ncurr=1 requires an Enzyme-enabled build for chi'_state_vjp");
#endif  // VMECPP_ENABLE_ENZYME
    }
    return result;
  }

  // Cotangent of the half-grid chi' from one of MakeGeometry's poloidal_flux:
  // chi_j = sum_{k<j} signOfJacobian * 2 pi deltaS phipH_k iotaH_k, with
  // phipH_k iotaH_k = chipH_k where phipH_k != 0.
  Eigen::VectorXd ChipBarFromPoloidalFluxBar(
      const Eigen::VectorXd &poloidal_flux_bar) const {
    Eigen::VectorXd chip_bar = Eigen::VectorXd::Zero(vmec_->fc_.ns - 1);
    if (poloidal_flux_bar.size() == 0) {
      return chip_bar;
    }
    if (poloidal_flux_bar.size() != vmec_->fc_.ns) {
      throw std::runtime_error("VmecModel: poloidal_flux_bar has wrong length");
    }
    const double c = static_cast<double>(vmec_->indata_.signgs) * 2.0 *
                     std::numbers::pi * vmec_->fc_.deltaS;
    double tail = 0.0;
    for (int j = vmec_->fc_.ns - 1; j >= 1; --j) {
      tail += poloidal_flux_bar[j];
      chip_bar[j - 1] = vmec_->p_[0]->phipH[j - 1] != 0.0 ? c * tail : 0.0;
    }
    return chip_bar;
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

#ifdef VMECPP_ENABLE_ENZYME
  void RequireLforbalDisabledForExactDerivatives() const {
    if (vmec_->m_[0]->lforbal) {
      throw std::runtime_error(
          "Exact derivative operators do not support lforbal=true because "
          "the LFORBAL force replacement has no JVP/VJP; use "
          "lforbal=false or a finite-difference derivative.");
    }
  }

  // Exact Hessian-vector product H v = T^T J_g T v of the augmented
  // functional, computed with one Enzyme forward pass through the local
  // force-density composition (J_g) wrapped by the linear spectral transforms.
  // J_g covers the MHD force, the hybrid lambda force, and the spectral-
  // condensation constraint force (effective force, Fourier bandpass and
  // assembly). The geometry tangent T v is obtained exactly from the linearity
  // of geometryFromFourier: T v = geom(x+v) - geom(x), so no finite-difference
  // step enters. The constraint multiplier tcon is recomputed from the geometry
  // inside the composition, as the raw force recomputes it from the state. The
  // model state is restored to x on return. always_fix_m1_gauge has the
  // meaning it has in Evaluate: H is the Jacobian of the force that
  // Evaluate returns for the same flag.
  Eigen::VectorXd ExactHessianVectorProduct(const Eigen::VectorXd &v,
                                            bool always_fix_m1_gauge = true) {
    RequireLforbalDisabledForExactDerivatives();
    vmecpp::IdealMhdModel &model = *vmec_->m_[0];
    const int gS = static_cast<int>(model.r1_e.size());
    Eigen::VectorXd dgeom = Eigen::VectorXd::Zero(20 * gS);
    // The primal geometry depends only on the current state, not on v, so cache
    // it: a Krylov solve calls this many times at the same state. SetState
    // invalidates the cache. The geometry tangent is the same linear pre-chain
    // applied to v directly (exact, no finite difference, no full update); the
    // single nonlinear step is the Enzyme JVP inside applyExactForceJacobian.
    if (!exact_primal_valid_ ||
        exact_primal_.size() != static_cast<Eigen::Index>(20 * gS)) {
      exact_primal_.setZero(20 * gS);
      model.packGeometry(*vmec_->decomposed_x_[0], *vmec_->physical_x_[0],
                         exact_primal_.data(), gS, /*primal=*/true);
      exact_primal_valid_ = true;
    }
    vmec_->physical_x_backup_[0]->setZero();
    UnflattenActive(*vmec_->physical_x_backup_[0], vmec_->s_, v);
    model.packGeometry(*vmec_->physical_x_backup_[0], *vmec_->physical_x_[0],
                       dgeom.data(), gS, /*primal=*/false);
    model.applyExactForceJacobian(
        exact_primal_.data(), dgeom.data(), gS, *vmec_->physical_f_[0],
        *vmec_->decomposed_f_[0], always_fix_m1_gauge);
    return FlattenActive(*vmec_->decomposed_f_[0], vmec_->s_);
  }

  // Transpose of the exact Hessian-vector product: H^T w, in the decomposed
  // internal basis. The force Jacobian is non-symmetric (VMEC's force is a
  // scaled gradient), so the adjoint boundary gradient needs H^T, not H. Uses
  // the same cached primal geometry as ExactHessianVectorProduct.
  Eigen::VectorXd ExactHessianVectorProductTranspose(
      const Eigen::VectorXd &w, bool always_fix_m1_gauge = true) {
    RequireLforbalDisabledForExactDerivatives();
    vmecpp::IdealMhdModel &model = *vmec_->m_[0];
    const int gS = static_cast<int>(model.r1_e.size());
    if (!exact_primal_valid_ ||
        exact_primal_.size() != static_cast<Eigen::Index>(20 * gS)) {
      exact_primal_.setZero(20 * gS);
      model.packGeometry(*vmec_->decomposed_x_[0], *vmec_->physical_x_[0],
                         exact_primal_.data(), gS, /*primal=*/true);
      exact_primal_valid_ = true;
    }
    vmec_->decomposed_f_[0]->setZero();
    UnflattenActive(*vmec_->decomposed_f_[0], vmec_->s_, w);
    vmec_->physical_x_backup_[0]->setZero();
    model.applyExactForceJacobianTranspose(
        exact_primal_.data(), gS, *vmec_->decomposed_f_[0],
        *vmec_->physical_f_[0], *vmec_->physical_x_[0],
        *vmec_->physical_x_backup_[0], always_fix_m1_gauge);
    return FlattenActive(*vmec_->physical_x_backup_[0], vmec_->s_);
  }

  // (dchi'/dx)^T chip_bar for ncurr==1, in the decomposed internal basis:
  // the state cotangent from a chi' cotangent alone (one entry per half
  // surface), with no force-member cotangent. Uses the same cached primal
  // geometry as ExactHessianVectorProduct(Transpose). GeometryStateVjp uses
  // this to convert the poloidal-flux cotangent into a state cotangent when
  // ncurr==1.
  Eigen::VectorXd ChipStateVjp(const Eigen::VectorXd &chip_bar) {
    RequireLforbalDisabledForExactDerivatives();
    vmecpp::IdealMhdModel &model = *vmec_->m_[0];
    const int gS = static_cast<int>(model.r1_e.size());
    if (!exact_primal_valid_ ||
        exact_primal_.size() != static_cast<Eigen::Index>(20 * gS)) {
      exact_primal_.setZero(20 * gS);
      model.packGeometry(*vmec_->decomposed_x_[0], *vmec_->physical_x_[0],
                         exact_primal_.data(), gS, /*primal=*/true);
      exact_primal_valid_ = true;
    }
    vmec_->physical_x_backup_[0]->setZero();
    model.chipStateVjp(exact_primal_.data(), gS, chip_bar.data(),
                       *vmec_->physical_x_[0], *vmec_->physical_x_backup_[0]);
    return FlattenActive(*vmec_->physical_x_backup_[0], vmec_->s_);
  }

  // Cotangents of the half-grid presH, iotaH and currH from a decomposed-force
  // cotangent force_bar and a cotangent poloidal_flux_bar of MakeGeometry's
  // poloidal_flux. presH is frozen in the force composition, so gamma == 0.
  std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> ProfileVjp(
      const Eigen::VectorXd &force_bar,
      const Eigen::VectorXd &poloidal_flux_bar) {
    RequireLforbalDisabledForExactDerivatives();
    if (vmec_->indata_.gamma != 0.0 || vmec_->indata_.lasym) {
      throw std::runtime_error(
          "VmecModel.profile_vjp requires gamma == 0 and lasym == false");
    }
    vmecpp::IdealMhdModel &model = *vmec_->m_[0];
    const int gS = static_cast<int>(model.r1_e.size());
    if (!exact_primal_valid_ ||
        exact_primal_.size() != static_cast<Eigen::Index>(20 * gS)) {
      exact_primal_.setZero(20 * gS);
      model.packGeometry(*vmec_->decomposed_x_[0], *vmec_->physical_x_[0],
                         exact_primal_.data(), gS, /*primal=*/true);
      exact_primal_valid_ = true;
    }
    const Eigen::VectorXd chip_bar =
        ChipBarFromPoloidalFluxBar(poloidal_flux_bar);
    const int nHalf = vmec_->fc_.ns - 1;
    Eigen::VectorXd pres_bar = Eigen::VectorXd::Zero(nHalf);
    Eigen::VectorXd chip_profile_bar = Eigen::VectorXd::Zero(nHalf);
    Eigen::VectorXd curr_bar = Eigen::VectorXd::Zero(nHalf);
    vmec_->decomposed_f_[0]->setZero();
    UnflattenActive(*vmec_->decomposed_f_[0], vmec_->s_, force_bar);
    model.profileVjp(exact_primal_.data(), gS, *vmec_->decomposed_f_[0],
                     *vmec_->physical_f_[0], chip_bar.data(), pres_bar.data(),
                     chip_profile_bar.data(), curr_bar.data());
    // chipH = iotaH * phipH for a prescribed iota
    return {pres_bar, chip_profile_bar.cwiseProduct(vmec_->p_[0]->phipH),
            curr_bar};
  }
#endif  // VMECPP_ENABLE_ENZYME

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
  int mpol_geometry() const { return vmec_->s_.mpolGeometry; }
  int ntor_geometry() const { return vmec_->s_.ntorGeometry; }
  bool lthreed() const { return vmec_->s_.lthreed; }
  bool lasym() const { return vmec_->s_.lasym; }
  bool have_to_flip_theta() const { return vmec_->fc_.haveToFlipTheta; }
  bool has_exact_force_jacobian() const {
#ifdef VMECPP_ENABLE_ENZYME
    return !vmec_->m_[0]->lforbal;
#else
    return false;
#endif
  }

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
  // Half-grid chi' (chipH), valid after evaluate(): iota*phip when ncurr=0,
  // the current-constrained profile computeBContra solves for when ncurr=1.
  // Exposed to let callers (and tests) check chip_state_vjp and
  // geometry_state_vjp's ncurr=1 flux route against a finite difference.
  Eigen::VectorXd chip_h() const { return vmec_->p_[0]->chipH; }
  static bool openmp_enabled() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
  }

  std::unique_ptr<vmecpp::Vmec> vmec_;

  // Cached primal geometry for the exact force-Jacobian transpose: it depends
  // only on the state, so it is reused across Krylov matvecs and invalidated by
  // SetState. Mutable so SetState (const) can clear it.
  mutable Eigen::VectorXd exact_primal_;
  mutable bool exact_primal_valid_ = false;

  // Preconditioner / Nestor update bookkeeping, mirroring the like-named Vmec
  // members; the Python loop drives the iteration counters via Evaluate, so the
  // wrapper keeps these running values across calls.
  int last_preconditioner_update_ = 0;
  int last_full_update_nestor_ = 0;
  bool last_need_restart_ = false;
};

}  // anonymous namespace

// IMPORTANT: The first argument must be the name of the module, else
// compilation will succeed but import will fail with:
//     ImportError: dynamic module does not define module export function
//     (PyInit_example)
PYBIND11_MODULE(_vmecpp, m) {
  m.doc() = "pybind11 VMEC++ plugin";

  vmecpp::InitializeUserLogging();

  static py::exception<std::invalid_argument> hot_restart_mismatch(
      m, "HotRestartMismatchError", PyExc_ValueError);
  hot_restart_mismatch_error = hot_restart_mismatch;

  // Compile-time build feature: whether this wheel was built with the Enzyme
  // plugin (CMake option VMECPP_ENABLE_ENZYME). This is a static property of
  // the build, cheap to query, and does not require creating a VmecModel;
  // vmecpp.has_exact_force_jacobian() reads it to fail fast, before running
  // any solve. It does not by itself guarantee
  // VmecModel.has_exact_force_jacobian is true for a given model: that also
  // depends on the model's run-time force-balance formulation (see
  // VmecModel::has_exact_force_jacobian).
#ifdef VMECPP_ENABLE_ENZYME
  m.attr("VMECPP_ENABLE_ENZYME") = true;
#else
  m.attr("VMECPP_ENABLE_ENZYME") = false;
#endif

  // Compile-time build feature: whether the iteration body runs on the GPU
  // (CMake option VMECPP_USE_CUDA). vmecpp.has_cuda() reads it; a CUDA build
  // rejects lasym and lforbal inputs, so tests and callers can skip those
  // ahead of a solve.
#ifdef VMECPP_USE_CUDA
  m.attr("VMECPP_USE_CUDA") = true;
#else
  m.attr("VMECPP_USE_CUDA") = false;
#endif

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
          .def_static(
              "from_json",
              [](const std::string &indata_json) {
                auto maybe_indata = VmecINDATA::FromJson(indata_json);
                return GetValueOrThrow(maybe_indata);
              },
              py::arg("indata_json"))
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
          .def_readwrite("ntor_geometry", &VmecINDATA::ntor_geometry)
          .def_readwrite("vacuum_mpol", &VmecINDATA::vacuum_mpol)
          .def_readwrite("vacuum_ntor", &VmecINDATA::vacuum_ntor);

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
      .def_readwrite("signgs", &VmecINDATA::signgs)
      .def_readwrite("free_boundary_method", &VmecINDATA::free_boundary_method)

      // tweaking parameters
      .def_readwrite("nstep", &VmecINDATA::nstep);
  DefEigenProperty(pyindata, "aphi", &VmecINDATA::aphi);
  pyindata.def_readwrite("delt", &VmecINDATA::delt)
      .def_readwrite("tcon0", &VmecINDATA::tcon0)
      .def_readwrite("lgiveup", &VmecINDATA::lgiveup)
      .def_readwrite("fgiveup", &VmecINDATA::fgiveup)
      .def_readwrite("lforbal", &VmecINDATA::lforbal)
      .def_readwrite("lambda_preconditioner_scale",
                     &VmecINDATA::lambda_preconditioner_scale)
      .def_readwrite("lambda_precondition_checkerboard_terms",
                     &VmecINDATA::lambda_precondition_checkerboard_terms)
      .def_readwrite("lbsubs", &VmecINDATA::lbsubs)
      .def_readwrite("backup_evaluated_state",
                     &VmecINDATA::backup_evaluated_state)
      .def_readwrite("always_fix_m1_gauge", &VmecINDATA::always_fix_m1_gauge)
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

  py::native_enum<vmecpp::MultigridInterpolationScheme>(
      m, "MultigridInterpolationScheme", "enum.Enum")
      .value("LINEAR", vmecpp::MultigridInterpolationScheme::kLinear)
      .value("CUBIC", vmecpp::MultigridInterpolationScheme::kCubic)
      .value("CUBIC_RHO", vmecpp::MultigridInterpolationScheme::kCubicRho)
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
      .def_readonly("d_volume_d_phi",
                    &vmecpp::MercierFileContents::d_volume_d_phi)
      .def_readonly("well", &vmecpp::MercierFileContents::well)
      .def_readonly("toroidal_current",
                    &vmecpp::MercierFileContents::toroidal_current)
      .def_readonly("d_toroidal_current_d_volume",
                    &vmecpp::MercierFileContents::d_toroidal_current_d_volume)
      .def_readonly("pressure", &vmecpp::MercierFileContents::pressure)
      .def_readonly("d_pressure_d_volume",
                    &vmecpp::MercierFileContents::d_pressure_d_volume)
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
      .def_readwrite("xmpot", &vmecpp::WOutFileContents::xmpot)
      .def_readwrite("xnpot", &vmecpp::WOutFileContents::xnpot)
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

  // The output-independent geometry contract. These bindings expose the
  // product-basis representation and the C++ evaluator; the differentiable
  // evaluator lives in Python (vmecpp.geometry) and uses the C++ one as an
  // independent oracle in its tests.
  py::class_<vmecpp::GeometryDimensions>(m, "GeometryDimensions")
      .def_readonly("ns", &vmecpp::GeometryDimensions::ns)
      .def_readonly("mpol", &vmecpp::GeometryDimensions::mpol)
      .def_readonly("ntor", &vmecpp::GeometryDimensions::ntor)
      .def_readonly("nfp", &vmecpp::GeometryDimensions::nfp);
  py::class_<vmecpp::GeometryCoefficients>(m, "GeometryCoefficients")
      .def_readonly("r_cc", &vmecpp::GeometryCoefficients::r_cc)
      .def_readonly("r_ss", &vmecpp::GeometryCoefficients::r_ss)
      .def_readonly("r_sc", &vmecpp::GeometryCoefficients::r_sc)
      .def_readonly("r_cs", &vmecpp::GeometryCoefficients::r_cs)
      .def_readonly("z_sc", &vmecpp::GeometryCoefficients::z_sc)
      .def_readonly("z_cs", &vmecpp::GeometryCoefficients::z_cs)
      .def_readonly("z_cc", &vmecpp::GeometryCoefficients::z_cc)
      .def_readonly("z_ss", &vmecpp::GeometryCoefficients::z_ss)
      .def_readonly("lambda_sc", &vmecpp::GeometryCoefficients::lambda_sc)
      .def_readonly("lambda_cs", &vmecpp::GeometryCoefficients::lambda_cs)
      .def_readonly("lambda_cc", &vmecpp::GeometryCoefficients::lambda_cc)
      .def_readonly("lambda_ss", &vmecpp::GeometryCoefficients::lambda_ss);
  py::class_<vmecpp::GeometryPoint>(m, "GeometryPoint")
      .def_readonly("r", &vmecpp::GeometryPoint::r)
      .def_readonly("z", &vmecpp::GeometryPoint::z)
      .def_readonly("lambda_", &vmecpp::GeometryPoint::lambda)
      .def_readonly("toroidal_flux", &vmecpp::GeometryPoint::toroidal_flux)
      .def_readonly("poloidal_flux", &vmecpp::GeometryPoint::poloidal_flux);
  py::class_<vmecpp::Geometry>(m, "Geometry")
      .def_readonly("dimensions", &vmecpp::Geometry::dimensions)
      .def_readonly("toroidal_flux", &vmecpp::Geometry::toroidal_flux)
      .def_readonly("poloidal_flux", &vmecpp::Geometry::poloidal_flux)
      .def_readonly("coefficients", &vmecpp::Geometry::coefficients)
      .def("evaluate", &vmecpp::EvaluateGeometry, py::arg("s"),
           py::arg("theta"), py::arg("zeta"));
  py::class_<vmecpp::HalfGridFields>(m, "HalfGridFields")
      .def_readonly("gsqrt", &vmecpp::HalfGridFields::gsqrt)
      .def_readonly("bsupu", &vmecpp::HalfGridFields::bsupu)
      .def_readonly("bsupv", &vmecpp::HalfGridFields::bsupv)
      .def_readonly("bsubu", &vmecpp::HalfGridFields::bsubu)
      .def_readonly("bsubv", &vmecpp::HalfGridFields::bsubv)
      .def_readonly("weight", &vmecpp::HalfGridFields::weight)
      .def_readonly("buco", &vmecpp::HalfGridFields::buco)
      .def_readonly("bvco", &vmecpp::HalfGridFields::bvco)
      .def_readonly("iota", &vmecpp::HalfGridFields::iota)
      .def_readonly("phip", &vmecpp::HalfGridFields::phip)
      .def_readonly("vp", &vmecpp::HalfGridFields::vp)
      .def_readonly("ntheta_even", &vmecpp::HalfGridFields::ntheta_even)
      .def_readonly("ntheta_eff", &vmecpp::HalfGridFields::ntheta_eff)
      .def_readonly("nzeta", &vmecpp::HalfGridFields::nzeta)
      .def_readonly("nfp", &vmecpp::HalfGridFields::nfp)
      .def_readonly("signgs", &vmecpp::HalfGridFields::signgs);
  py::class_<vmecpp::SolverState>(m, "SolverState")
      .def_readonly("iteration", &vmecpp::SolverState::iteration)
      .def_readonly("multigrid_step", &vmecpp::SolverState::multigrid_step)
      .def_readonly("ns", &vmecpp::SolverState::ns)
      .def_readonly("fsqr", &vmecpp::SolverState::fsqr)
      .def_readonly("fsqz", &vmecpp::SolverState::fsqz)
      .def_readonly("fsql", &vmecpp::SolverState::fsql)
      .def_readonly("ftol", &vmecpp::SolverState::ftol)
      .def_readonly("delt", &vmecpp::SolverState::delt)
      .def_property_readonly("restart_reason",
                             [](const vmecpp::SolverState &state) {
                               return static_cast<int>(state.restart_reason);
                             })
      .def_readonly("jacobian_resets", &vmecpp::SolverState::jacobian_resets)
      .def_readonly("vacuum_pressure_active",
                    &vmecpp::SolverState::vacuum_pressure_active)
      .def_readonly("mhd_energy", &vmecpp::SolverState::mhd_energy)
      .def_readonly("geometry", &vmecpp::SolverState::geometry)
      .def_readonly("half_grid", &vmecpp::SolverState::half_grid)
      .def_readwrite("curr_h", &vmecpp::SolverState::curr_h);
  m.def("make_geometry", [](const vmecpp::OutputQuantities &output) {
    return vmecpp::MakeGeometry(output.indata, output.vmec_internal_results,
                                vmecpp::GeometryCoefficientState::kPhysical);
  });

  py::class_<vmecpp::HotRestartState>(m, "HotRestartState")
      .def(py::init(&MakeHotRestartState), "wout"_a, "indata"_a)
      .def_readwrite("wout", &vmecpp::HotRestartState::wout)
      .def_readwrite("indata", &vmecpp::HotRestartState::indata);

  m.def(
      "run",
      [](const VmecINDATA &indata,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads, vmecpp::OutputMode verbose,
         py::object iteration_callback) -> vmecpp::OutputQuantities {
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
        PythonIterationCallback iteration_hook{iteration_callback,
                                               std::nullopt};
        absl::StatusOr<vmecpp::OutputQuantities> ret;
        {
          py::gil_scoped_release release;
          ret = vmecpp::run(indata, std::move(initial_state), max_threads,
                            verbose, interrupt_check, iteration_hook.Hook());
        }
        iteration_hook.Rethrow();
        if (was_interrupted) {
          throw py::error_already_set();
        }
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress,
      py::arg("iteration_callback") = py::none());

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
      "write_makegrid_netcdf_file",
      [](const std::string &out_file,
         const makegrid::MakegridParameters &mgrid_params,
         const magnetics::MagneticConfiguration &magnetic_configuration,
         py::object progress) {
        // progress(stage, circuits_done, number_of_circuits) with stage "B"
        // (magnetic field) or "A" (vector potential).
        auto callback = [&progress](const char *stage) {
          return makegrid::ProgressCallback(
              [&progress, stage](int done, int total) {
                if (!progress.is_none()) progress(stage, done, total);
              });
        };
        auto maybe_currents =
            magnetics::GetCircuitCurrents(magnetic_configuration);
        Eigen::VectorXd currents = GetValueOrThrow(maybe_currents);
        auto maybe_table = makegrid::ComputeMagneticFieldResponseTable(
            mgrid_params, magnetic_configuration, callback("B"));
        auto table = GetValueOrThrow(maybe_table);
        auto maybe_vector_potential = makegrid::ComputeVectorPotentialCache(
            mgrid_params, magnetic_configuration, callback("A"));
        auto vector_potential = GetValueOrThrow(maybe_vector_potential);
        absl::Status status = makegrid::WriteMakegridNetCDFFile(
            out_file, mgrid_params, currents, table, vector_potential);
        if (!status.ok()) {
          throw std::runtime_error(std::string(status.message()));
        }
      },
      py::arg("out_file"), py::arg("makegrid_parameters"),
      py::arg("magnetic_configuration"), py::arg("progress") = py::none());

  m.def(
      "run",
      [](const VmecINDATA &indata,
         const makegrid::MagneticFieldResponseTable &magnetic_response_table,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads, vmecpp::OutputMode verbose,
         py::object iteration_callback) {
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
        PythonIterationCallback iteration_hook{iteration_callback,
                                               std::nullopt};
        absl::StatusOr<vmecpp::OutputQuantities> ret;
        {
          py::gil_scoped_release release;
          ret = vmecpp::run(indata, magnetic_response_table,
                            std::move(initial_state), max_threads, verbose,
                            interrupt_check, iteration_hook.Hook());
        }
        iteration_hook.Rethrow();
        if (was_interrupted) {
          throw py::error_already_set();
        }
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("magnetic_response_table"),
      py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress,
      py::arg("iteration_callback") = py::none());

  m.def(
      "run",
      [](const VmecINDATA &indata, py::function magnetic_field,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads, vmecpp::OutputMode verbose,
         py::object iteration_callback) {
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
        PythonIterationCallback iteration_hook{iteration_callback,
                                               std::nullopt};
        PythonMagneticField field{magnetic_field, std::nullopt};
        absl::StatusOr<vmecpp::OutputQuantities> ret;
        {
          py::gil_scoped_release release;
          ret = vmecpp::run(indata, field.Function(), std::move(initial_state),
                            max_threads, verbose, interrupt_check,
                            iteration_hook.Hook());
        }
        field.Rethrow();
        iteration_hook.Rethrow();
        if (was_interrupted) {
          throw py::error_already_set();
        }
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("magnetic_field"),
      py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress,
      py::arg("iteration_callback") = py::none());

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
  // The distinct argument selects one of two execution modes:
  //
  // Broadcast mode (distinct = false): the first input's spectra are
  // broadcast across all N configuration slots in d_specs_block, the
  // batched kernel chain executes at full N, and the returned vector
  // contains the first input's converged OutputQuantities.
  //
  // Distinct mode (distinct = true): a per-configuration
  // pre-initialization stage runs each input's first iteration alone, with
  // the magnetic-axis recovery of a single run, and extracts its initial
  // spectra into staging blocks handed to the CUDA layer; every
  // configuration then takes the time step and restart control of its own
  // single run. The returned vector contains one
  // converged OutputQuantities per input, each derived from the
  // corresponding configuration's converged state after the
  // device-resident output flush at end-of-run.
  m.def(
      "run_batched_gpu",
      [](const std::vector<VmecINDATA> &indata_list,
         std::optional<int> max_threads, vmecpp::OutputMode verbose,
         bool return_spectra, bool distinct) -> py::object {
        // Residual traces of the attempts at a configuration's first
        // iteration that the distinct-mode pre-initialization runs alone and
        // the batch does not repeat.
        struct TracePrefix {
          std::vector<double> r, z, lambda, energy, delbsq;
          std::vector<vmecpp::RestartReason> reasons;
        };
#ifdef VMECPP_USE_CUDA
        // Derives one OutputQuantities per configuration directly from the
        // batched run's device state: the per-cfg fields flush in one
        // pass, each configuration's VmecInternalResults assembles from
        // the seed's gathered template plus the per-cfg slices, and
        // DeriveOutputQuantities runs the standard host physics chain per
        // configuration. No additional iterations run. Returns false when
        // any precondition fails.
        auto derive_outputs_from_batched_flush =
            [](const std::vector<vmecpp::VmecINDATA> &indata_local, int n_cfg,
               std::optional<int> max_threads,
               const vmecpp::OutputQuantities &seed_oq,
               const vmecpp::FlowControl &batch_fc,
               const std::vector<TracePrefix> &trace_prefix,
               std::vector<vmecpp::OutputQuantities> &outs) -> bool {
          // Capture the batched run's final per-configuration force-residual
          // state before any further CUDA call (the deriver's setup run below
          // resets the persistent device state). The wout reports the
          // normalized invariant residuals fsqr/fsqz/fsql of each
          // configuration's last active iteration, from the snapshot cache
          // the iteration controller fills; the live residual caches are
          // zeroed for masked (converged) configurations. Layout: 3*n_cfg as
          // [fsqr_c, fsqz_c, fsql_c, ...].
          if (static_cast<int>(batch_fc.converged_per_cfg.size()) != n_cfg ||
              static_cast<int>(batch_fc.niter_per_cfg.size()) != n_cfg ||
              static_cast<int>(trace_prefix.size()) != n_cfg ||
              static_cast<int>(batch_fc.force_residual_r_per_cfg.size()) !=
                  n_cfg) {
            return false;
          }
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
          auto drv_or = vmecpp::Vmec::FromIndata(
              ind_drv, /*magnetic_response_table=*/nullptr, max_threads,
              vmecpp::OutputMode::kSilent);
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

            // This configuration's residuals, traces (after those of the
            // attempts at its first iteration that the batch does not
            // repeat), and iteration count in place of the deriver's
            // single-iteration setup values.
            vmecpp::FlowControl fc_c = deriver->fc_;
            const size_t s3 = static_cast<size_t>(3) * c;
            if (fsqr_snapshot_pc.size() >= s3 + 3) {
              fc_c.fsqr = fsqr_snapshot_pc[s3 + 0];
              fc_c.fsqz = fsqr_snapshot_pc[s3 + 1];
              fc_c.fsql = fsqr_snapshot_pc[s3 + 2];
            }
            const auto prefixed = [](auto prefix, const auto &trace) {
              prefix.insert(prefix.end(), trace.begin(), trace.end());
              return prefix;
            };
            const TracePrefix &prefix = trace_prefix[c];
            fc_c.force_residual_r =
                prefixed(prefix.r, batch_fc.force_residual_r_per_cfg[c]);
            fc_c.force_residual_z =
                prefixed(prefix.z, batch_fc.force_residual_z_per_cfg[c]);
            fc_c.force_residual_lambda = prefixed(
                prefix.lambda, batch_fc.force_residual_lambda_per_cfg[c]);
            fc_c.mhd_energy =
                prefixed(prefix.energy, batch_fc.mhd_energy_per_cfg[c]);
            fc_c.delbsq = prefixed(prefix.delbsq, batch_fc.delbsq_per_cfg[c]);
            fc_c.restart_reasons =
                prefixed(prefix.reasons, batch_fc.restart_reasons_per_cfg[c]);

            outs.push_back(vmecpp::DeriveOutputQuantities(
                std::move(oq), indata_local[c], deriver->s_, fc_c,
                deriver->constants_, deriver->t_, h_c, seed_oq.wout.mgrid_mode,
                seed_oq.wout.curlabel, vmecpp::VmecCheckpoint::NONE,
                vmecpp::VacuumPressureState::kOff,
                batch_fc.converged_per_cfg[c] != 0
                    ? vmecpp::VmecStatus::SUCCESSFUL_TERMINATION
                    : vmecpp::VmecStatus::NORMAL_TERMINATION,
                batch_fc.niter_per_cfg[c]));
          }
          return true;
        };
#endif  // VMECPP_USE_CUDA
#ifndef VMECPP_USE_CUDA
        throw std::runtime_error(
            "run_batched_gpu requires a build with VMECPP_USE_CUDA");
#endif
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
        // The batched kernels index radial surface and configuration together
        // in one grid dimension, which CUDA caps at 65535 blocks.
        constexpr int kMaxGridBlocks = 65535;
        const int ns_max = seed.ns_array.maxCoeff();
        if (n_cfg > kMaxGridBlocks / ns_max) {
          throw std::runtime_error(
              "run_batched_gpu: a batch at ns = " + std::to_string(ns_max) +
              " holds at most " + std::to_string(kMaxGridBlocks / ns_max) +
              " configurations, but " + std::to_string(n_cfg) +
              " were given; split the batch");
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
        // Distinct mode interpolates every configuration on the device at the
        // multigrid transitions, which does not reset the m = 1 gauge from the
        // boundary as always_fix_m1_gauge does on the host.
        if (distinct && seed.always_fix_m1_gauge && seed.ns_array.size() > 1) {
          throw std::runtime_error(
              "run_batched_gpu distinct mode supports always_fix_m1_gauge with "
              "a single-entry ns_array only; solve multigrid inputs with "
              "always_fix_m1_gauge through vmecpp.run");
        }

        // run_batched_gpu stages only per-config boundary geometry and the
        // magnetic axis to the device. In broadcast mode (distinct mode off)
        // the seed -- configuration zero -- runs once and a single output is
        // returned. Any additional configurations passed in broadcast mode
        // share the seed's validated shape and are intentionally ignored; the
        // single-output broadcast contract is enforced at the return below.
        // Use distinct mode to solve every configuration.
        // In distinct mode the device honors per-config boundary, magnetic
        // axis, and plasma profiles (each built into its own RadialProfiles and
        // staged per-config). Every other input -- the convergence controls
        // (ns_array, ftol_array, niter_array), resolution (mpol/ntor), and
        // free-boundary settings -- is still taken from the first
        // configuration. Reject batches that differ in anything else, so they
        // fail loudly instead of silently running with config 0's solver
        // settings.
        if (distinct) {
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

#ifdef VMECPP_USE_CUDA
        // Drop the per-config CUDA staging buffers when the call returns, so a
        // later run (batched or single) cannot pick up this batch's per-config
        // profiles. The per-config INDATA itself lives on the seed Vmec and is
        // released with it.
        struct BatchProfileRestore {
          ~BatchProfileRestore() { vmecpp::ClearBatchProfilesCuda(); }
        } batch_profile_restore;
#endif

        // Each configuration's first iteration, which the distinct-mode
        // pre-initialization runs alone: the ijacob it ends with, and the
        // traces of its attempts the batch does not repeat.
        std::vector<int> seed_ijacob;
        std::vector<TracePrefix> trace_prefix;
        if (distinct) {
          seed_ijacob.assign(n_cfg, 0);
          trace_prefix.assign(n_cfg, TracePrefix{});
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

          // Each configuration's first iteration runs alone below, with the
          // axis recovery of its single run. indata_local holds the inputs
          // with the recovered axes, which the pre-initialization and the
          // seed run consume.
          indata_local.assign(indata_list.begin(), indata_list.end());
          run_seed = &indata_local[0];

          for (int c = 0; c < n_cfg; ++c) {
            // As in a single run, the first iteration recomputes the magnetic
            // axis and starts over when the first force evaluation shows a
            // bad Jacobian or huge forces; the batch starts this configuration
            // from the axis that iteration ends with.
            {
              auto probe_or = vmecpp::Vmec::FromIndata(
                  indata_local[c], /*magnetic_response_table=*/nullptr,
                  max_threads, vmecpp::OutputMode::kSilent);
              if (!probe_or.ok()) {
                throw std::runtime_error(
                    "run_batched_gpu: axis-probe FromIndata failed for cfg " +
                    std::to_string(c) + ": " +
                    std::string(probe_or.status().message()));
              }
              auto probe = std::move(*probe_or);
              auto first_or = probe->run(vmecpp::VmecCheckpoint::EVOLVE,
                                         /*iterations_before_checkpointing=*/1,
                                         /*maximum_multi_grid_step=*/1);
              if (!first_or.ok()) {
                throw std::runtime_error(
                    "run_batched_gpu: the first iteration failed for cfg " +
                    std::to_string(c) + ": " +
                    std::string(first_or.status().message()));
              }
              seed_ijacob[c] = probe->fc_.ijacob;
              const vmecpp::FlowControl &probe_fc = probe->fc_;
              if (probe->get_jacob_off() == 0 &&
                  probe_fc.force_residual_r.size() > 1) {
                const auto all_but_last = [](auto trace) {
                  trace.pop_back();
                  return trace;
                };
                TracePrefix &prefix = trace_prefix[c];
                prefix.r = all_but_last(probe_fc.force_residual_r);
                prefix.z = all_but_last(probe_fc.force_residual_z);
                prefix.lambda = all_but_last(probe_fc.force_residual_lambda);
                prefix.energy = all_but_last(probe_fc.mhd_energy);
                prefix.delbsq = all_but_last(probe_fc.delbsq);
                prefix.reasons = all_but_last(probe_fc.restart_reasons);
              }
              if (probe_fc.ijacob > 0) {
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

            auto vmec_or = vmecpp::Vmec::FromIndata(
                indata_local[c], /*magnetic_response_table=*/nullptr,
                max_threads, vmecpp::OutputMode::kSilent);
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
            vmec->physical_x_[0]->m1Constraint(1.0, vmec->indata_.signgs);
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

#ifdef VMECPP_USE_CUDA
          // Hand both staging blocks to the CUDA layer.
          vmecpp::SetBatchStagingCuda(batch_buf.data(), dec_x_buf.data(), n_cfg,
                                      ns_first, mpol, ntor);
#endif
        }

        // Invoke the seed-driven iteration body. In distinct mode the
        // per-configuration initial spectra are loaded from the
        // staging blocks at the first invocation of CudaForward; in
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
        // The seed run's FlowControl, whose per-configuration fields describe
        // the configuration slots.
        std::optional<vmecpp::FlowControl> batch_fc;
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
            seed_vmec.batch_size_ = n_cfg;
            seed_vmec.batch_ijacob_ = seed_ijacob;
            if (distinct) {
              seed_vmec.batch_indata_ = indata_local;
            }
#endif
            absl::StatusOr<bool> seed_ok = seed_vmec.run();
            batch_fc.emplace(seed_vmec.fc_);
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
        // As in a single run, a configuration that misses the tolerance of the
        // last multigrid stage is an error unless the inputs ask for
        // best-effort outputs.
        if (distinct && n_cfg > 1 &&
            !run_seed->return_outputs_even_if_not_converged) {
          const std::vector<std::uint8_t> &converged_per_cfg =
              batch_fc->converged_per_cfg;
          std::string unconverged;
          for (int c = 0; c < n_cfg; ++c) {
            if (c >= static_cast<int>(converged_per_cfg.size()) ||
                converged_per_cfg[c] == 0) {
              unconverged +=
                  (unconverged.empty() ? "" : ", ") + std::to_string(c);
            }
          }
          if (!unconverged.empty()) {
            throw std::runtime_error(
                "run_batched_gpu: configuration(s) " + unconverged +
                " did not converge; set "
                "return_outputs_even_if_not_converged for their best-effort "
                "outputs");
          }
        }

        // Distinct mode returns one OutputQuantities per input, derived from
        // the batched run's device state; broadcast mode, and a distinct run
        // of one input, return the seed's.
        std::vector<vmecpp::OutputQuantities> outs;
        if (distinct && n_cfg > 1) {
#ifdef VMECPP_USE_CUDA
          outs.reserve(n_cfg);
          if (!derive_outputs_from_batched_flush(
                  indata_local, n_cfg, max_threads, *ret_seed, *batch_fc,
                  trace_prefix, outs)) {
            throw std::runtime_error(
                "run_batched_gpu: the per-configuration outputs could not be "
                "derived from the batched run");
          }
#endif
        } else {
          outs.push_back(*ret_seed);
        }

        if (return_spectra) {
          // Converged per-cfg spectra as a numpy array of shape
          // [6][n_cfg][spec]; None when the run held a single configuration
          // slot.
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
the first invocation of the iteration body. The distinct argument selects
the mode:

  Broadcast mode (distinct=False, the default): the first input's spectra
  are broadcast across all N configuration slots, the batched kernel chain
  executes at full N, and the returned list contains the single converged
  OutputQuantities.

  Distinct mode (distinct=True): a per-configuration pre-initialization
  stage runs each input's first iteration alone, with the magnetic-axis
  recovery of a single run, extracts its initial spectra, and hands the
  concatenated spectra to the iteration body in memory. The iteration body
  loads the per-configuration spectra at its first invocation of
  CudaForward and drives each configuration with the time step and the
  restart control of its own single run. The
  returned list holds one OutputQuantities per input, derived from the
  batched run's device state with its configuration's residual traces and
  iteration count; a derivation that fails raises, and so
  does a configuration that misses the tolerance of the last multigrid
  stage unless return_outputs_even_if_not_converged is set.

With return_spectra=True the return value is a tuple (outputs,
spectra), where spectra is a float64 array of shape [6][n_cfg][spec]
holding each configuration's converged decomposed spectra (rmncc,
rmnss, zmnsc, zmncs, lmnsc, lmncs; spec = ns * mpol * (ntor + 1) at
the final multigrid stage), or None when the run held a single
configuration slot.

Requires a build with VMECPP_USE_CUDA.
)pbdoc");

  // Single-resolution iteration model: exposes the forward model and the
  // time-step / restart primitives so the equilibrium iteration can be driven
  // from Python (see vmecpp._iteration).
  py::class_<VmecModel>(m, "VmecModel")
      .def_static("create", &VmecModel::Create, py::arg("indata"),
                  py::arg("ns"), py::arg("initial_state") = std::nullopt)
      .def("evaluate", &VmecModel::Evaluate, py::arg("iter1"), py::arg("iter2"),
           py::arg("precondition") = true,
           py::arg("always_fix_m1_gauge") = true)
      .def_property_readonly("need_restart", &VmecModel::need_restart)
      .def_property_readonly("have_to_flip_theta",
                             &VmecModel::have_to_flip_theta)
      .def("perform_time_step", &VmecModel::PerformTimeStep,
           py::arg("velocity_scale"), py::arg("conjugation_parameter"),
           py::arg("time_step"))
      .def("save_backup", &VmecModel::SaveBackup)
      .def("save_evaluated_backup", &VmecModel::SaveEvaluatedBackup)
      .def("restore_backup", &VmecModel::RestoreBackup)
      .def("zero_velocity", &VmecModel::ZeroVelocity)
      .def("reset_to_initial_guess", &VmecModel::ResetToInitialGuess)
      .def("recompute_axis", &VmecModel::RecomputeAxis)
      .def("reinitialize", &VmecModel::Reinitialize)
      .def("refine_to", &VmecModel::RefineTo, py::arg("new_ns"),
           py::arg("interpolation") = py::none())
      .def("solve", &VmecModel::Solve)
      .def_property_readonly(
          "always_fix_m1_gauge", &VmecModel::always_fix_m1_gauge,
          "indata.always_fix_m1_gauge: zero the m=1 gauge force from the first "
          "iteration of solve() and set the gauge from the boundary in "
          "create() and refine_to(). The converged gauge then equals the "
          "boundary gauge scaled by sqrt(s), independent of the iteration and "
          "multigrid history, and the exact Hessian-vector products with "
          "always_fix_m1_gauge=True are the Jacobian of the iterated system.")
      .def("get_state", &VmecModel::GetState)
      .def("set_state", &VmecModel::SetState, py::arg("state"))
      .def("get_forces", &VmecModel::GetForces)
      .def("get_geometry", &VmecModel::GetGeometry)
      .def("geometry_state_vjp", &VmecModel::GeometryStateVjp,
           py::arg("coefficient_bar"),
           py::arg("poloidal_flux_bar") = Eigen::VectorXd())
      .def("apply_preconditioner", &VmecModel::ApplyPreconditioner,
           py::arg("v"))
      .def("hessian_vector_product", &VmecModel::HessianVectorProduct,
           py::arg("v"), py::arg("eps_rel") = 1e-7)
#ifdef VMECPP_ENABLE_ENZYME
      // Both directions are required: the reverse adjoint needs H^T, and
      // deflating the augmented Hessian's structural null space needs both a
      // row and a column probe.
      .def("exact_hessian_vector_product",
           &VmecModel::ExactHessianVectorProduct, py::arg("v"),
           py::arg("always_fix_m1_gauge") = true)
      .def("exact_hessian_vector_product_transpose",
           &VmecModel::ExactHessianVectorProductTranspose, py::arg("w"),
           py::arg("always_fix_m1_gauge") = true)
      .def("chip_state_vjp", &VmecModel::ChipStateVjp, py::arg("chip_bar"))
      .def("profile_vjp", &VmecModel::ProfileVjp, py::arg("force_bar"),
           py::arg("poloidal_flux_bar"))
#endif  // VMECPP_ENABLE_ENZYME
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
      .def_property_readonly("mpol_geometry", &VmecModel::mpol_geometry)
      .def_property_readonly("ntor_geometry", &VmecModel::ntor_geometry)
      .def_property_readonly("lthreed", &VmecModel::lthreed)
      .def_property_readonly("lasym", &VmecModel::lasym)
      .def_property_readonly("has_exact_force_jacobian",
                             &VmecModel::has_exact_force_jacobian)
      .def_property_readonly("force_residual_r", &VmecModel::force_residual_r)
      .def_property_readonly("force_residual_z", &VmecModel::force_residual_z)
      .def_property_readonly("force_residual_lambda",
                             &VmecModel::force_residual_lambda)
      .def_property_readonly("restart_reasons", &VmecModel::restart_reasons)
      .def_property_readonly("ijacob", &VmecModel::ijacob)
      .def_property_readonly("raxis_c", &VmecModel::raxis_c)
      .def_property_readonly("chip_h", &VmecModel::chip_h)
      .def_static("openmp_enabled", &VmecModel::openmp_enabled);
}  // NOLINT(readability/fn_size)
