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
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move

#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

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
  void Evaluate(int iter1, int iter2) {
    bool need_restart = false;
    std::string error_message;
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
          iter1, iter2, vmecpp::VmecCheckpoint::NONE, INT_MAX,
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

  auto pyindata = py::class_<VmecINDATA>(m, "VmecINDATA")
                      .def(py::init<>())
                      .def("_set_mpol_ntor", &VmecINDATA::SetMpolNtor,
                           py::arg("new_mpol"), py::arg("new_ntor"))
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
                      .def_readwrite("nzeta", &VmecINDATA::nzeta);

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
                  std::string(path) + "':\n" + std::string(s.message());
              throw std::runtime_error(msg);
            }
          },
          py::arg("path"))
      .def_static("load", [](const std::filesystem::path &path) {
        auto maybe_oq = vmecpp::OutputQuantities::Load(path);
        if (!maybe_oq.ok()) {
          const std::string msg =
              "There was an error loading OutputQuantities from file '" +
              std::string(path) + "':\n" +
              std::string(maybe_oq.status().message());
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

  // Single-resolution iteration model: exposes the forward model and the
  // time-step / restart primitives so the equilibrium iteration can be driven
  // from Python (see vmecpp._iteration).
  py::class_<VmecModel>(m, "VmecModel")
      .def_static("create", &VmecModel::Create, py::arg("indata"),
                  py::arg("ns"), py::arg("initial_state") = std::nullopt)
      .def("evaluate", &VmecModel::Evaluate, py::arg("iter1"), py::arg("iter2"))
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
