// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/vmec/vmec.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vmecpp/common/flow_control/flow_control.h"

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/free_boundary/nestor/nestor.h"
#include "vmecpp/free_boundary/only_coils/only_coils.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.h"

#ifdef VMECPP_USE_CUDA
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda.h"
#endif

namespace {

void UpdateStatusForThread(absl::Status& m_status_of_all_threads, int thread_id,
                           const absl::Status& thread_status) {
  CHECK(!thread_status.ok()) << "UpdateStatusForThread expects an error status";

  auto thread_msg =
      absl::StrFormat("Thread %i:\n\t%s", thread_id, thread_status.message());

  if (m_status_of_all_threads.ok()) {
    auto msg =
        "There was an error in one or more threads during a VMEC++ run:\n" +
        thread_msg;
    m_status_of_all_threads = absl::InternalError(std::move(thread_msg));
  }

  const auto new_msg =
      std::string(m_status_of_all_threads.message()) + thread_msg;
  m_status_of_all_threads = absl::InternalError(new_msg);
}

// Check preconditions on (initial_state, indata) pair passed to Vmec::run
// in order to make sure that the state to hot-restart from can be copied over
// 1:1.
absl::Status CheckInitialState(const vmecpp::HotRestartState& initial_state,
                               const vmecpp::VmecINDATA& indata) {
  const auto msg_start = "Mismatch in variable '";
  const auto msg_end =
      "' between hot restart initial state and indata. This is not supported "
      "yet.";

  // check for match in `lasym`, since that determines whether
  // non-stellarator-symmetric terms are expected or not
  if (initial_state.indata.lasym != indata.lasym) {
    return absl::InvalidArgumentError(
        absl::StrCat(msg_start, "lasym", msg_end));
  }

  // check for `mpol` and `ntor` match, since they determine the expected array
  // size in tangential direction
  if (initial_state.indata.mpol != indata.mpol) {
    return absl::InvalidArgumentError(absl::StrCat(msg_start, "mpol", msg_end));
  }
  if (initial_state.indata.ntor != indata.ntor) {
    return absl::InvalidArgumentError(absl::StrCat(msg_start, "ntor", msg_end));
  }

  // check for matching `ns`
  if (initial_state.indata.ns_array[initial_state.indata.ns_array.size() - 1] !=
      indata.ns_array[0]) {
    return absl::InvalidArgumentError(
        absl::StrCat(msg_start, "ns_array", msg_end));
  }

  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<vmecpp::OutputQuantities> vmecpp::run(
    const VmecINDATA& indata, std::optional<HotRestartState> initial_state,
    std::optional<int> max_threads, OutputMode verbose,
    InterruptCallback interrupt_callback) {
  auto maybe_vmec = Vmec::FromIndata(indata, nullptr, max_threads, verbose,
                                     std::move(interrupt_callback));
  if (!maybe_vmec.ok()) {
    return maybe_vmec.status();
  }
  Vmec& v = **maybe_vmec;

  // the values of the first three arguments should just be VMEC's defaults
  absl::StatusOr<bool> s =
      v.run(VmecCheckpoint::NONE, INT_MAX, 500, std::move(initial_state));

  if (!s.ok()) {
    return s.status();
  }

  return std::move(v.output_quantities_);
}

absl::StatusOr<vmecpp::OutputQuantities> vmecpp::run(
    const VmecINDATA& indata,
    const makegrid::MagneticFieldResponseTable& magnetic_response_table,
    std::optional<HotRestartState> initial_state,
    std::optional<int> max_threads, OutputMode verbose,
    InterruptCallback interrupt_callback) {
  auto maybe_vmec =
      Vmec::FromIndata(indata, &magnetic_response_table, max_threads, verbose,
                       std::move(interrupt_callback));
  if (!maybe_vmec.ok()) {
    return maybe_vmec.status();
  }
  Vmec& v = **maybe_vmec;

  // the values of the first three arguments should just be VMEC's defaults
  absl::StatusOr<bool> s =
      v.run(VmecCheckpoint::NONE, INT_MAX, 500, std::move(initial_state));

  if (!s.ok()) {
    return s.status();
  }

  return std::move(v.output_quantities_);
}

namespace vmecpp {

absl::StatusOr<std::unique_ptr<Vmec>> Vmec::FromIndata(
    const VmecINDATA& indata,
    const makegrid::MagneticFieldResponseTable* magnetic_response_table,
    std::optional<int> max_threads, OutputMode verbose,
    InterruptCallback interrupt_callback) {
  auto v = std::make_unique<Vmec>(indata, max_threads, verbose,
                                  std::move(interrupt_callback));

  // This part of Vmec initialization requires Status handling, and is therefore
  // in this factory method instead of the constructor.
  if (indata.lfreeb) {
    absl::Status s{};
    if (magnetic_response_table == nullptr) {
      s = v->mgrid_.LoadFile(indata.mgrid_file, indata.extcur);
    } else {
      s = v->mgrid_.LoadFields(*magnetic_response_table, indata.extcur);
    }
    if (!s.ok()) {
      return s;
    }
  }

  return v;
}

// The asynchronous NESTOR workers (owned by the IdealMhdModel instances in m_)
// solve the vacuum field on a background thread that writes into fb_async_ and
// its response buffers, all of which are members of this Vmec. Destroy the
// models first, so each worker is joined and its final in-flight solve has
// completed, while fb_async_ is still alive. The implicit member teardown would
// otherwise free fb_async_ before m_ (m_ is declared earlier), leaving a worker
// writing into freed memory.
Vmec::~Vmec() { m_.clear(); }

// initialize based on input file contents
Vmec::Vmec(const VmecINDATA& indata, std::optional<int> max_threads,
           OutputMode verbose, InterruptCallback interrupt_callback)
    : indata_(indata),
      s_(indata_),
      t_(&s_),
      b_(&s_, &t_, kSignOfJacobian),
      h_(&s_),
      fc_(indata_.lfreeb, indata_.delt,
          static_cast<int>(indata_.ns_array.size()), max_threads),
      verbose_(verbose != OutputMode::kSilent),
      logger_(std::cout, verbose),
      interrupt_callback_(std::move(interrupt_callback)),
      vacuum_pressure_state_(VacuumPressureState::kOff),
      status_(VmecStatus::NORMAL_TERMINATION),
      iter2_(1),
      iter1_(iter2_),
      invTau_(kNDamp),
      last_preconditioner_update_(0),
      last_full_update_nestor_(0) {
  // remainder of readin()
  fc_.haveToFlipTheta = b_.setupFromIndata(indata_, verbose_);

  if (fc_.lfreeb) {
    // tangential Fourier resolution
    // 0 : ntor
    int nf = s_.ntor;
    // 0 : (mpol + 1)
    int mf = s_.mpol + 1;
    int mnpd = (2 * nf + 1) * (mf + 1);
    matrixShare.setZero(mnpd * mnpd);
    iPiv.setZero(mnpd);
    bvecShare.setZero(mnpd);

    h_.vacuum_magnetic_pressure.setZero(s_.nZnT);
    h_.vacuum_b_r.setZero(s_.nZnT);
    h_.vacuum_b_phi.setZero(s_.nZnT);
    h_.vacuum_b_z.setZero(s_.nZnT);

    // TODO(jons): move this check to better-suited place
    if (indata_.free_boundary_method == FreeBoundaryMethod::ONLY_COILS &&
        (indata_.curtor != 0.0 || indata_.pres_scale != 0.0)) {
      throw std::invalid_argument(
          absl::StrCat("curtor and pres_scale must be zero when using "
                       "'only_coils' free boundary method, but were ",
                       indata_.curtor, " and ", indata_.pres_scale));
    }  // check that cutor==0 and pres_scale==0 for only_coils
  }
}

// main worker method; equivalent of vmec.f90
// checked visually to comply with vmec.f90
absl::StatusOr<bool> Vmec::run(const VmecCheckpoint& checkpoint,
                               const int iterations_before_checkpointing,
                               const int maximum_multi_grid_step,
                               std::optional<HotRestartState> initial_state) {
#ifdef VMECPP_USE_CUDA
  // The CUDA iteration body covers stellarator-symmetric configurations,
  // both axisymmetric (ntor = 0, nZeta = 1) and three-dimensional, on a
  // single radial rank. Reject the unsupported inputs up front rather than
  // failing mid-iteration.
  if (indata_.lasym) {
    return absl::UnimplementedError(
        "non-stellarator-symmetric (lasym) runs are not supported by the "
        "CUDA build; rebuild without VMECPP_USE_CUDA for lasym inputs");
  }
  if (s_.nThetaReduced > 256) {
    return absl::UnimplementedError(
        absl::StrFormat("the CUDA build supports nThetaReduced up to 256; this "
                        "input has nThetaReduced = %d",
                        s_.nThetaReduced));
  }
  // The reset re-reads VMECPP_N_CONFIG_MAX for this run; it must precede
  // the memory pre-flight below so the budget is computed against this
  // run's configuration count rather than the previous run's. The other
  // run-scoped gates re-read with the same per-run scope.
  vmecpp::ResetCudaStateForNewVmecRun();
  sync_elide_k_ = -2;
  per_cfg_niter_cap_ = -1;
  vmecpp::SetFreeBoundaryRunCuda(indata_.lfreeb ? 1 : 0);
  if (indata_.lfreeb && vmecpp::GetNConfigMaxCuda() > 1 &&
      indata_.free_boundary_method != FreeBoundaryMethod::NESTOR) {
    return absl::UnimplementedError(
        "batched free-boundary runs support the NESTOR vacuum solver "
        "only");
  }
  {
    const int ns_max = indata_.ns_array.maxCoeff();
    const int ns_supported = vmecpp::CudaMaxRadialResolution();
    if (ns_max > ns_supported) {
      return absl::UnimplementedError(absl::StrFormat(
          "the CUDA build supports radial resolutions up to ns = %d on this "
          "device (the large-ns tridiagonal solver holds its elimination "
          "ratios in shared memory); this input requests ns = %d",
          ns_supported, ns_max));
    }
    // Pre-flight the device memory budget for the finest multigrid stage
    // at the requested configuration count, so an oversized batch is
    // rejected with a diagnosis instead of failing at an allocation deep
    // inside the iteration body.
    long long needed_bytes = 0;
    long long free_bytes = 0;
    if (!vmecpp::CudaVramBudgetCuda(vmecpp::GetNConfigMaxCuda(), ns_max,
                                    s_.mpol, s_.ntor, s_.nZeta, s_.nThetaEff,
                                    &needed_bytes, &free_bytes)) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "the CUDA device does not have enough free memory for this run: "
          "estimated need %lld MiB for %d configuration slot(s) at "
          "ns = %d, free %lld MiB; reduce VMECPP_N_CONFIG_MAX or the "
          "radial resolution",
          needed_bytes >> 20, vmecpp::GetNConfigMaxCuda(), ns_max,
          free_bytes >> 20));
    }
  }
#endif  // VMECPP_USE_CUDA
  if (indata_.lfreeb) {
    if (!mgrid_.IsLoaded()) {
      // Fallback: load mgrid from file if constructed directly via the public
      // constructor instead of the FromIndata factory method.
      absl::Status status = mgrid_.LoadFile(indata_.mgrid_file, indata_.extcur);
      if (!status.ok()) {
        return status;
      }
    }
    if (mgrid_.numPhi != indata_.nzeta) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "MGridProvider has %d phi grid points, but VmecINDATA "
          "has %d nzeta grid points. Please ensure that the two "
          "are consistent.",
          mgrid_.numPhi, indata_.nzeta));
    }
  }

  auto is_indata_consistent =
      IsConsistent(indata_, /*enable_info_messages=*/verbose_);
  if (!is_indata_consistent.ok()) {
    return is_indata_consistent;
  }

  if (initial_state.has_value()) {
    absl::Status status = CheckInitialState(*initial_state, indata_);
    if (!status.ok()) {
      return status;
    }
  }

  // !!! THIS must be the ONLY place where this gets set to zero !!!
  num_eqsolve_retries_ = 0;

  fc_.ns_old = 0;
  fc_.delt0r = indata_.delt;

  // Allocate the per-configuration state vectors on the FlowControl
  // instance for the batched CUDA execution mode. The effective
  // configuration count is queried from the CUDA-side cache populated
  // from the VMECPP_N_CONFIG_MAX environment variable. Under
  // single-configuration execution the resized vectors degenerate to
  // length one and the single-configuration scalar fields remain
  // authoritative for the convergence test; under multi-configuration
  // execution the vectors hold the per-configuration residuals and
  // restart-state values consumed by the convergence gate further below.
#ifdef VMECPP_USE_CUDA
  // The persistent device state was reset alongside the scope guards
  // above. Without that reset, a second run in the same process finds
  // pts_x still marked initialized: its first Reshape then snapshots the
  // prior run's final state as if it were a previous multigrid stage,
  // and the stage-transition upscale initializes this run from that
  // state instead of from the staged inputs.
  fc_.ResizeForBatch(vmecpp::GetNConfigMaxCuda());
#endif  // VMECPP_USE_CUDA

  // retry with ns=3 if immediately fails at lowest radial resolution
  for (jacob_off_ = 0; jacob_off_ < 2; ++jacob_off_) {
    // jacob_off=1 indicates that an initial run with ns=3 shall be inserted
    // before the user-provided ns values from ns_array are processed
    // in the multi-grid run

    // Free-boundary hot-restart attempt. Assuming that the initial_guess is
    // very good, we can immediately take the vacuum contribution into account,
    // instead of slowly activating it (as we would in a regular free-bdy
    // solve). If the initial guess completely off, jacob_off>0 will trigger a
    // retry with VacuumPressureState::kOff anyway.
    if (fc_.lfreeb && initial_state.has_value()) {
      vacuum_pressure_state_ = VacuumPressureState::kInitialized;
    }

    fc_.ns_min = 3;

    // multi-grid iterations: loop over ns_array
    // jacob_off=0,1 is required to insert one ns=3 run before
    // starting to work with the user-provided ns_array
    // if the first ns value from ns_array gave a bad jacobian

    const int max_grids = std::min(fc_.multi_ns_grid, maximum_multi_grid_step);
    for (int igrid = -jacob_off_; igrid < max_grids; igrid++) {
      constants_.reset();

      // retrieve settings for (ns, ftol, niter) for current multi-grid
      // iteration
      if (igrid < 0) {
        // igrid .lt. 1 can only happen when jacob_off == 1 (then igrid==0)

        // TRY TO GET NON-SINGULAR JACOBIAN ON A 3 PT RADIAL MESH
        // COMPUTE INITIAL SOLUTION ON COARSE GRID
        // IF PREVIOUS SEQUENCE DID NOT CONVERGE WELL
        fc_.nsval = 3;
        fc_.ftolv = 1.0e-4;
        // niterv taken from niter_array[0] in INDATA, I guess?

        // fully restart vacuum
        // TODO(jons): why then assign vacuum_pressure_state=kInitialized then
        // above?
        vacuum_pressure_state_ = VacuumPressureState::kOff;
      } else {
        // proceed regularly with ns values from ns_array
        fc_.nsval = indata_.ns_array[igrid];
        if (fc_.nsval < fc_.ns_min) {
          // skip entries that have less flux surfaces than previous iteration
          continue;
        }

        // update ns_min --> reduction in number of flux surfaces not allowed
        fc_.ns_min = fc_.nsval;

        fc_.ftolv = indata_.ftol_array[igrid];
        fc_.niterv = indata_.niter_array[igrid];
      }

      // Reserve the per-iteration convergence-history vectors for this stage so
      // the push_backs in Evolve do not reallocate inside the hot loop.
      {
        const size_t cap =
            fc_.force_residual_r.size() + static_cast<size_t>(fc_.niterv) + 1;
        fc_.force_residual_r.reserve(cap);
        fc_.force_residual_z.reserve(cap);
        fc_.force_residual_lambda.reserve(cap);
        fc_.mhd_energy.reserve(cap);
        fc_.delbsq.reserve(cap);
        fc_.restart_reasons.reserve(cap);
      }

      // notify logger of the next multigrid stage
      logger_.BeginStage(igrid, max_grids + jacob_off_, fc_.nsval, s_.mnmax,
                         fc_.ftolv, fc_.niterv, fc_.lfreeb);

      // Restore per-cfg activity for the current stage. Configurations
      // that converged against the coarser stage's looser ftolv must
      // re-iterate against the finer stage's tighter ftolv, so the
      // active mask is rebuilt and the per-cfg iteration counters are
      // zeroed before the stage's iter loop begins. Under single-cfg
      // execution the per-cfg vectors are empty and the reset is a
      // no-op.
      fc_.ResetActivePerCfgForNextStage();

      // initialize ns-dependent arrays
      // and (if previous solution is available) interpolate to current ns
      // value
      if (InitializeRadial(checkpoint, iterations_before_checkpointing,
                           fc_.nsval, fc_.ns_old, fc_.delt0r, initial_state)) {
        return true;
      }

      // *HERE* is the *ACTUAL* call to the equilibrium solver !
      const absl::StatusOr<bool> reached_checkpoint =
          SolveEquilibrium(checkpoint, iterations_before_checkpointing);
      if (!reached_checkpoint.ok() || *reached_checkpoint == true) {
        return reached_checkpoint;
      }

      // break the multi-grid sequence if current number of flux surfaces did
      // not reach convergence
      if (status_ != VmecStatus::NORMAL_TERMINATION &&
          status_ != VmecStatus::SUCCESSFUL_TERMINATION) {
        const auto msg = absl::StrFormat(
            "FATAL ERROR in SolveEquilibrium: %s\n"
            "VmecINDATA had these contents:\n%s",
            VmecStatusAsString(status_), *indata_.ToJson());

        return absl::InternalError(msg);
      }

      // TODO(jons): insert lgiveup/fgiveup logic here

      // If this point is reached, the current multi-grid step should have
      // properly converged.
    }  // igrid

    // if did not converge only because jacobian was bad
    // and the intermediate ns=3 run was not performed yet (jacob_off is still
    // == 0), retry the whole thing again
    if (status_ != VmecStatus::BAD_JACOBIAN) {
      // We can only correct a bad Jacobian (by re-trying with ns = 3);
      // all other errors are fatal.
      break;
    }

    // if ier_flag .eq. bad_jacobian_flag, repeat once again with ns=3 before
  }  // jacob_off

  if (status_ != VmecStatus::SUCCESSFUL_TERMINATION &&
      !indata_.return_outputs_even_if_not_converged) {
    const auto msg = "VMEC++ did not converge";
    return absl::InternalError(msg);
  }

  // Consolidated end-of-iteration flush of device-resident fields read
  // by the post-iteration output-derivation path. Each model invokes its
  // FlushForOutputCuda once, after the iteration loop has terminated and
  // before GatherDataFromThreads consumes the host buffers. The per-model
  // dispatch is necessary under the multi-thread CPU path where each
  // IdealMhdModel instance maintains an independent set of device shadow
  // buffers; under the single-thread CUDA path the loop iterates a single
  // model. The implementation is a no-op on builds without
  // VMECPP_USE_CUDA, so the call site remains unconditional.
  for (auto& mhd : m_) {
    mhd->FlushForOutputCuda();
  }

#ifdef VMECPP_USE_CUDA
  // Final device-to-host transfer of the decomposed-position state.
  // Mirrors the CUDA path's persistent device-resident d_pts_x buffer
  // back into the host m_decomposed_x containers so that
  // ComputeOutputQuantities and the downstream output-construction
  // routines observe the converged state.
  for (int t = 0; t < static_cast<int>(decomposed_x_.size()); ++t) {
    const int ns_local = r_[t]->nsMaxF1 - r_[t]->nsMinF1;
    vmecpp::FlushDecomposedXToHostCuda(
        0, ns_local, s_.mpol, s_.ntor, s_.lthreed,
        decomposed_x_[t]->rmncc.data(), decomposed_x_[t]->rmnss.data(),
        decomposed_x_[t]->zmnsc.data(), decomposed_x_[t]->zmncs.data(),
        decomposed_x_[t]->lmnsc.data(), decomposed_x_[t]->lmncs.data());
  }
#endif

  // compute output file quantities, but do not write them to output file yet
  // (for creating the output file, use WriteOutputFile())
  output_quantities_ = vmecpp::ComputeOutputQuantities(
      kSignOfJacobian, indata_, s_, fc_, constants_, t_, h_, mgrid_.mgrid_mode,
      r_, decomposed_x_, m_, p_, checkpoint, vacuum_pressure_state_, status_,
      iter2_);

  {
    const auto& w = output_quantities_.wout;
    RunSummary summary;
    summary.converged = (status_ == VmecStatus::SUCCESSFUL_TERMINATION);
    summary.total_iterations = w.itfsq;
    summary.num_jacobian_resets = fc_.ijacob;
    summary.fsqr = w.fsqr;
    summary.fsqz = w.fsqz;
    summary.fsql = w.fsql;
    summary.ftolv = fc_.ftolv;
    summary.betatot = w.betatotal;
    summary.betapol = w.betapol;
    summary.betator = w.betator;
    summary.w_mhd = h_.mhdEnergy * 4.0 * M_PI * M_PI;
    summary.rax = w.Rmajor_p;
    summary.aminor = w.Aminor_p;
    summary.rmajor = w.Rmajor_p;
    summary.b0 = w.b0;
    logger_.EndRun(summary);
  }

  return false;
}  // run

// initialize_radial quantities, return true if a checkpoint was reached
bool Vmec::InitializeRadial(
    VmecCheckpoint checkpoint, int iterations_before_checkpointing, int nsval,
    int ns_old, double& m_delt0,
    const std::optional<HotRestartState>& initial_state) {
  // Stage info output is now handled by logger_.BeginStage() in run().

  // Set timestep control parameters
  fc_.fsq = 1.0;

  iter2_ = 1;
  iter1_ = iter2_;

  fc_.ijacob = 0;
  fc_.restart_reason = RestartReason::NO_RESTART;
  fc_.res0 = -1;
  m_delt0 = indata_.delt;

  // INITIALIZE MESH-DEPENDENT SCALARS

  // *THIS* actually sets the global ns value for the main physics algorithm
  fc_.ns = nsval;

  fc_.deltaS = 1.0 / (fc_.ns - 1.0);
  fc_.num_surfaces_to_distribute = fc_.ns - 1;
  if (fc_.lfreeb) {
    fc_.num_surfaces_to_distribute = fc_.ns;
  }

  // number of Fourier coefficients per basis function for the whole volume
  int mns = fc_.ns * s_.mnsize;

  // number of Fourier coefficients per quantity (R, Z, lambda)
  int irzloff = s_.num_basis * mns;

  // total number of degrees-of-freedom
  fc_.neqs = 3 * irzloff;

  // check that interpolating from coarse to fine mesh
  // and that old solution is available
  bool linterp = (ns_old < fc_.ns && ns_old != 0);

  if (ns_old != fc_.ns) {
    // ALLOCATE NS-DEPENDENT ARRAYS

    // backup current xc, scalxc in xstore, scalxc
    // Note that this relies on old/previous value of num_threads_!
    if (linterp && fc_.neqs_old > 0) {
      old_xc_scaled_.resize(num_threads_);
      old_r_.resize(num_threads_);

#ifdef VMECPP_USE_CUDA
      // Flush the device-resident d_pts_x to the host m_decomposed_x
      // before old_xc_scaled_ derives from it. Under the per-iteration
      // sync deferral the host copy can lag the device by up to K
      // iterations; the host upscale and the device's own per-cfg
      // multigrid upscale must operate on bit-identical
      // configuration-zero state, or the divergence between them forces
      // distinct-mode runs into BAD_JACOBIAN at iteration 2 of the new
      // stage.
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        const int ns_old_local =
            r_[thread_id]->nsMaxF1 - r_[thread_id]->nsMinF1;
        if (ns_old_local <= 0) continue;
        vmecpp::FlushDecomposedXToHostCuda(
            0, ns_old_local, s_.mpol, s_.ntor, s_.lthreed,
            decomposed_x_[thread_id]->rmncc.data(),
            decomposed_x_[thread_id]->rmnss.data(),
            decomposed_x_[thread_id]->zmnsc.data(),
            decomposed_x_[thread_id]->zmncs.data(),
            decomposed_x_[thread_id]->lmnsc.data(),
            decomposed_x_[thread_id]->lmncs.data());
      }
#endif

      // expect that previous solution is available at this point
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        old_xc_scaled_[thread_id] = std::make_unique<FourierGeometry>(
            &s_, r_[thread_id].get(), fc_.ns_old);
        old_r_[thread_id] = std::move(r_[thread_id]);

        decomposed_x_[thread_id]->decomposeInto(*old_xc_scaled_[thread_id],
                                                p_[thread_id]->scalxc);
      }  // thread_id
    }

    // adjust parallellism for nsval at hand
#ifdef VMECPP_USE_CUDA
    // The CUDA iteration body owns the full radial domain on a single rank;
    // multi-threaded radial partitioning is not supported under the CUDA
    // build. Clamping the input keeps the OpenMP runtime thread count
    // (set inside vmec_adjust_num_threads) consistent with the per-thread
    // array sizing below.
    num_threads_ = vmec_adjust_num_threads(/*max_threads=*/1,
                                           fc_.num_surfaces_to_distribute);
#else
    num_threads_ = vmec_adjust_num_threads(fc_.max_threads(),
                                           fc_.num_surfaces_to_distribute);
#endif  // VMECPP_USE_CUDA

    r_.resize(num_threads_);
    ls_.resize(num_threads_);
    p_.resize(num_threads_);
    fb_.resize(num_threads_);
    tp_.resize(num_threads_);
    m_.resize(num_threads_);
    decomposed_x_.resize(num_threads_);
    physical_x_backup_.resize(num_threads_);
    physical_x_.resize(num_threads_);
    decomposed_f_.resize(num_threads_);
    physical_f_.resize(num_threads_);
    decomposed_v_.resize(num_threads_);

    // single-threaded creation of objects used in parallel threads
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      r_[thread_id] = std::make_unique<RadialPartitioning>();

      // Set this to `true` if you want to have the distribution
      // of radial grid points over threads to be printed out.
      // Disabled for now to reduce noise.
      const bool printout_radial_partitioning = false;
      r_[thread_id]->adjustRadialPartitioning(num_threads_, thread_id, nsval,
                                              fc_.lfreeb,
                                              printout_radial_partitioning);

      h_.allocate(*r_[thread_id], fc_.ns);

      ls_[thread_id] = std::make_unique<ThreadLocalStorage>(&s_);

      p_[thread_id] = std::make_unique<RadialProfiles>(
          r_[thread_id].get(), &h_, &indata_, &fc_, kSignOfJacobian, kPDamp);

      // update profile parameterizations based on p****_type strings
      p_[thread_id]->setupInputProfiles();

#ifdef VMECPP_USE_CUDA
      // Distinct-mode batch: build one RadialProfiles per configuration,
      // mirroring p_, so the device stages each config's pressure/current/iota/
      // flux profiles instead of broadcasting the seed's. evalRadialProfiles
      // touches no HandoverStorage, so these safely share h_; the per-level
      // eval below uses a throwaway VmecConstants to absorb the rmsPhiP sum.
      if (thread_id == 0 && !batch_indata_.empty()) {
        p_percfg_.clear();
        p_percfg_.reserve(batch_indata_.size());
        for (VmecINDATA& cfg_indata : batch_indata_) {
          auto rp = std::make_unique<RadialProfiles>(r_[thread_id].get(), &h_,
                                                     &cfg_indata, &fc_,
                                                     kSignOfJacobian, kPDamp);
          rp->setupInputProfiles();
          p_percfg_.push_back(std::move(rp));
        }
      }
#endif

      // setup free-boundary solver
      if (fc_.lfreeb) {
        // Keep the existing free-boundary solver (and its accumulated vacuum
        // response matrix and RHS, which live in ns-independent Fourier space)
        // across multi-grid steps, reproducing Fortran VMEC's persistent vacuum
        // state. On the first grid step there is no solver yet and it is built;
        // later steps reuse it.
        const bool reuse_solver = fb_[thread_id] != nullptr;
        if (!reuse_solver) {
          tp_[thread_id] = std::make_unique<TangentialPartitioning>(
              s_.nZnT, num_threads_, thread_id);

          if (indata_.free_boundary_method == FreeBoundaryMethod::NESTOR) {
            fb_[thread_id] = std::make_unique<Nestor>(
                &s_, tp_[thread_id].get(), &mgrid_,
                std::span<double>(matrixShare.data(), matrixShare.size()),
                std::span<double>(bvecShare.data(), bvecShare.size()),
                std::span<double>(h_.vacuum_magnetic_pressure.data(),
                                  h_.vacuum_magnetic_pressure.size()),
                std::span<int>(iPiv.data(), iPiv.size()),
                std::span<double>(h_.vacuum_b_r.data(), h_.vacuum_b_r.size()),
                std::span<double>(h_.vacuum_b_phi.data(),
                                  h_.vacuum_b_phi.size()),
                std::span<double>(h_.vacuum_b_z.data(), h_.vacuum_b_z.size()));
          } else if (indata_.free_boundary_method ==
                     FreeBoundaryMethod::ONLY_COILS) {
            fb_[thread_id] = std::make_unique<OnlyCoils>(
                &s_, tp_[thread_id].get(), &mgrid_,
                std::span<double>(h_.vacuum_magnetic_pressure.data(),
                                  h_.vacuum_magnetic_pressure.size()),
                std::span<double>(h_.vacuum_b_r.data(), h_.vacuum_b_r.size()),
                std::span<double>(h_.vacuum_b_phi.data(),
                                  h_.vacuum_b_phi.size()),
                std::span<double>(h_.vacuum_b_z.data(), h_.vacuum_b_z.size()));
          } else {
            LOG(FATAL) << absl::StrCat("free boundary method '",
                                       ToString(indata_.free_boundary_method),
                                       "' not implemented yet");
          }  // indata_.free_boundary_method

#ifdef VMECPP_USE_CUDA
          // Batched free-boundary: one further NESTOR instance per
          // configuration slot, each with its own persistent response
          // matrix, right-hand side, and pivots. They share the mgrid
          // provider and the HandoverStorage output spans; the vacuum
          // loop in IdealMhdModel::update consumes each configuration's
          // outputs before the next solver call overwrites them. Like
          // fb_, the instances persist across multigrid stages.
          const int n_cfg_fb = vmecpp::GetNConfigMaxCuda();
          if (n_cfg_fb > 1 &&
              indata_.free_boundary_method == FreeBoundaryMethod::NESTOR) {
            const int nf = s_.ntor;
            const int mf = s_.mpol + 1;
            const int mnpd = (2 * nf + 1) * (mf + 1);
            fb_matrix_per_cfg_.resize(n_cfg_fb - 1);
            fb_ipiv_per_cfg_.resize(n_cfg_fb - 1);
            fb_bvec_per_cfg_.resize(n_cfg_fb - 1);
            fb_extra_cfg_.resize(n_cfg_fb - 1);
            for (int c = 0; c + 1 < n_cfg_fb; ++c) {
              fb_matrix_per_cfg_[c].setZero(mnpd * mnpd);
              fb_ipiv_per_cfg_[c].setZero(mnpd);
              fb_bvec_per_cfg_[c].setZero(mnpd);
              fb_extra_cfg_[c] = std::make_unique<Nestor>(
                  &s_, tp_[thread_id].get(), &mgrid_,
                  std::span<double>(fb_matrix_per_cfg_[c].data(),
                                    fb_matrix_per_cfg_[c].size()),
                  std::span<double>(fb_bvec_per_cfg_[c].data(),
                                    fb_bvec_per_cfg_[c].size()),
                  std::span<double>(h_.vacuum_magnetic_pressure.data(),
                                    h_.vacuum_magnetic_pressure.size()),
                  std::span<int>(fb_ipiv_per_cfg_[c].data(),
                                 fb_ipiv_per_cfg_[c].size()),
                  std::span<double>(h_.vacuum_b_r.data(), h_.vacuum_b_r.size()),
                  std::span<double>(h_.vacuum_b_phi.data(),
                                    h_.vacuum_b_phi.size()),
                  std::span<double>(h_.vacuum_b_z.data(),
                                    h_.vacuum_b_z.size()));
            }
          }

          // Asynchronous NESTOR (single configuration): a thread-private
          // Nestor with its own response matrix, right-hand side, pivots, and
          // field/pressure output buffers, so the worker thread can run it
          // concurrently with the device iteration without racing the shared
          // HandoverStorage. Persists across multigrid stages like fb_.
          // The async NESTOR overlap is on by default for single-configuration
          // NESTOR free-boundary runs; set VMECPP_FB_ASYNC_NESTOR=0 to force
          // the synchronous path.
          const char* async_env = std::getenv("VMECPP_FB_ASYNC_NESTOR");
          const bool async_disabled =
              (async_env != nullptr && std::atoi(async_env) == 0);
          if (n_cfg_fb == 1 && !async_disabled &&
              indata_.free_boundary_method == FreeBoundaryMethod::NESTOR) {
            const int nf = s_.ntor;
            const int mf = s_.mpol + 1;
            const int mnpd = (2 * nf + 1) * (mf + 1);
            fb_async_matrix_.setZero(mnpd * mnpd);
            fb_async_ipiv_.setZero(mnpd);
            fb_async_bvec_.setZero(mnpd);
            fb_async_bsqvac_.setZero(h_.vacuum_magnetic_pressure.size());
            fb_async_br_.setZero(h_.vacuum_b_r.size());
            fb_async_bphi_.setZero(h_.vacuum_b_phi.size());
            fb_async_bz_.setZero(h_.vacuum_b_z.size());
            fb_async_ = std::make_unique<Nestor>(
                &s_, tp_[thread_id].get(), &mgrid_,
                std::span<double>(fb_async_matrix_.data(),
                                  fb_async_matrix_.size()),
                std::span<double>(fb_async_bvec_.data(), fb_async_bvec_.size()),
                std::span<double>(fb_async_bsqvac_.data(),
                                  fb_async_bsqvac_.size()),
                std::span<int>(fb_async_ipiv_.data(), fb_async_ipiv_.size()),
                std::span<double>(fb_async_br_.data(), fb_async_br_.size()),
                std::span<double>(fb_async_bphi_.data(), fb_async_bphi_.size()),
                std::span<double>(fb_async_bz_.data(), fb_async_bz_.size()));
          }
#endif
        }  // !reuse_solver
      }  // lfreeb

      // setup MHD model
      m_[thread_id] = std::make_unique<IdealMhdModel>(
          &fc_, &s_, &t_, p_[thread_id].get(), &constants_,
          ls_[thread_id].get(), &h_, r_[thread_id].get(), fb_[thread_id].get(),
          kSignOfJacobian, indata_.nvacskip, &vacuum_pressure_state_);
      m_[thread_id]->setFromINDATA(indata_.ncurr, indata_.gamma, indata_.tcon0);
#ifdef VMECPP_USE_CUDA
      if (!fb_extra_cfg_.empty()) {
        std::vector<FreeBoundaryBase*> fb_ptrs;
        fb_ptrs.push_back(fb_[thread_id].get());
        for (auto& fb : fb_extra_cfg_) {
          fb_ptrs.push_back(fb.get());
        }
        m_[thread_id]->SetPerCfgFreeBoundary(std::move(fb_ptrs));
      }
      if (fb_async_) {
        m_[thread_id]->SetAsyncFreeBoundary(
            fb_async_.get(), std::span<const double>(fb_async_bsqvac_.data(),
                                                     fb_async_bsqvac_.size()));
      }
#endif
    }  // thread_id

    if (checkpoint == VmecCheckpoint::SPECTRAL_CONSTRAINT &&
        iterations_before_checkpointing <= 1) {
      // break the loop over thread_id here to check spectral constraint static
      // data; need to have all "threads" initialized before being able to test
      // all at once
      return true;
    }

    // single-threaded creation of objects used in parallel threads
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      // vector of free parameters

      // physically-correct coefficients
      decomposed_x_[thread_id] =
          std::make_unique<FourierGeometry>(&s_, r_[thread_id].get(), fc_.ns);

      // even/odd-m decomposed coefficients
      physical_x_[thread_id] =
          std::make_unique<FourierGeometry>(&s_, r_[thread_id].get(), fc_.ns);

      // physically-correct coefficients
      physical_x_backup_[thread_id] =
          std::make_unique<FourierGeometry>(&s_, r_[thread_id].get(), fc_.ns);

      // even/odd-m decomposed coefficients
      physical_f_[thread_id] =
          std::make_unique<FourierForces>(&s_, r_[thread_id].get(), fc_.ns);

      // physically-correct coefficients
      decomposed_f_[thread_id] =
          std::make_unique<FourierForces>(&s_, r_[thread_id].get(), fc_.ns);

      // physically-correct coefficients
      decomposed_v_[thread_id] =
          std::make_unique<FourierVelocity>(&s_, r_[thread_id].get(), fc_.ns);
    }

    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      decomposed_v_[thread_id]->setZero();
      decomposed_x_[thread_id]->setZero();
    }

    // COMPUTE INITIAL R, Z AND MAGNETIC FLUX PROFILES
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      p_[thread_id]->evalRadialProfiles(fc_.haveToFlipTheta, constants_);
    }

#ifdef VMECPP_USE_CUDA
    // Distinct-mode batch: re-evaluate each config's profiles at this multigrid
    // level and hand the flat per-config arrays to the device staging. A
    // throwaway VmecConstants absorbs each config's rmsPhiP so the seed's
    // lamscale (computed just below) is unaffected.
    if (!p_percfg_.empty()) {
      const RadialPartitioning& rp0 = *r_[0];
      const int ns_h = rp0.nsMaxH - rp0.nsMinH;
      const int ns_f = rp0.nsMaxF1 - rp0.nsMinF1;
      const int n_cfg = static_cast<int>(p_percfg_.size());
      std::vector<double> phipF((size_t)n_cfg * ns_f),
          phipH((size_t)n_cfg * ns_h), currH((size_t)n_cfg * ns_h),
          iotaH((size_t)n_cfg * ns_h), massH((size_t)n_cfg * ns_h);
      VmecConstants throwaway = constants_;
      for (int c = 0; c < n_cfg; ++c) {
        p_percfg_[c]->evalRadialProfiles(fc_.haveToFlipTheta, throwaway);
        std::copy_n(p_percfg_[c]->phipF.data(), ns_f,
                    phipF.data() + (size_t)c * ns_f);
        std::copy_n(p_percfg_[c]->phipH.data(), ns_h,
                    phipH.data() + (size_t)c * ns_h);
        std::copy_n(p_percfg_[c]->currH.data(), ns_h,
                    currH.data() + (size_t)c * ns_h);
        std::copy_n(p_percfg_[c]->iotaH.data(), ns_h,
                    iotaH.data() + (size_t)c * ns_h);
        std::copy_n(p_percfg_[c]->massH.data(), ns_h,
                    massH.data() + (size_t)c * ns_h);
      }
      SetBatchProfilesCuda(n_cfg, ns_h, ns_f, phipF.data(), phipH.data(),
                           currH.data(), iotaH.data(), massH.data());
    }
#endif

    // Now that all contributions to lamscale have been accumulated in
    // VmecConstants::rmsPhiP, can update lamscale.
    constants_.lamscale = sqrt(constants_.rmsPhiP * fc_.deltaS);

    if (checkpoint == VmecCheckpoint::RADIAL_PROFILES_EVAL &&
        iterations_before_checkpointing <= 1) {
      return true;
    }

    // TODO(jons): lreset .and. .not.linter?
    // If xc is overwritten by interp() anyway, why bother to initialize it in
    // profil3d()?
    if (initial_state.has_value() && ns_old == 0) {
      // ns_old == 0 means we hot restart only on the very first multigrid step
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        if (indata_.lfreeb) {
          // free-boundary hot restart: use all flux surfaces from initial state
          decomposed_x_[thread_id]->InitFromState(
              t_, initial_state->wout.rmnc, initial_state->wout.zmns,
              initial_state->wout.lmns_full, *p_[thread_id], constants_);
        } else {
          // fixed-boundary hot restart: use inner flux surfaces from initial
          // state, and LCFS geometry from Boundaries (from INDATA)
          decomposed_x_[thread_id]->InitFromState(
              t_, initial_state->wout.rmnc, initial_state->wout.zmns,
              initial_state->wout.lmns_full, *p_[thread_id], constants_, &b_);
        }
      }
    } else {
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        decomposed_x_[thread_id]->interpFromBoundaryAndAxis(t_, b_,
                                                            *p_[thread_id]);
      }
    }
    if (checkpoint == VmecCheckpoint::SETUP_INITIAL_STATE &&
        iterations_before_checkpointing <= 1) {
      return true;
    }

    // restart_reason == NO_RESTART at entry of restart_iter means to store xc
    // in xstore
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      fc_.restart_reason = RestartReason::NO_RESTART;

      // TODO(jons): what exactly happens here?
      // Why do we mask potential changes on `indata_.delt` by passing a copy?
      double delt_for_restart_iter = indata_.delt;
      RestartIteration(delt_for_restart_iter, thread_id);
    }

    // INTERPOLATE FROM COARSE (ns_old) TO NEXT FINER (ns) RADIAL GRID
    if (linterp) {
      InterpolateToNextMultigridStep(fc_.ns, fc_.ns_old, p_, r_, old_r_,
                                     decomposed_x_, old_xc_scaled_);

      // TODO(jons): check for max_multigrid_steps
      // TODO(jons): maybe need `&& iter2_ >= maximum_iterations) {` ?
      if (checkpoint == VmecCheckpoint::INTERP) {
        return true;
      }
    }

    fc_.ns_old = fc_.ns;
    fc_.neqs_old = fc_.neqs;
  }

  return false;
}

// eqsolve
// This is the amalgamation of the `1000` and `20` GOTOs in `eqsolve` in Fortran
// VMEC. It is responsible for re-trying with an improved axis guess or
// resetting the time step.
absl::StatusOr<bool> Vmec::SolveEquilibrium(
    VmecCheckpoint checkpoint, int iterations_before_checkpointing) {
  // Table header output is now handled by logger_.BeginStage() in run().

  absl::Status status_of_all_threads = absl::OkStatus();
  bool any_checkpoint_reached = false;

  // Shared communication variable for all threads. Used to signal an early exit
  // of the main iteration loop.
  bool liter_flag = true;

// NOTE: *THIS* is the main parallel region for the equilibrium solver
#ifdef _OPENMP
#pragma omp parallel
#endif  // _OPENMP
  {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif

    // COMPUTE INITIAL R, Z AND MAGNETIC FLUX PROFILES

    // this needs to be persistent across loops, so we create it here
    bool m_lreset_internal = false;

    absl::StatusOr<SolveEqLoopStatus> s = SolveEqLoopStatus::MUST_RETRY;

    // n_local_eqsolve_retries is a thread-local counter
    // max iterations only to ensure this terminates eventually, should never be
    // reached.
    int n_local_eqsolve_retries = 0;
    for (n_local_eqsolve_retries = 0;
         n_local_eqsolve_retries < fc_.niterv && s.ok() &&
         *s == SolveEqLoopStatus::MUST_RETRY && liter_flag;
         n_local_eqsolve_retries++) {
// protect read of `liter_flag` from write within `SolveEquilibriumLoop` below
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

      s = SolveEquilibriumLoop(
          thread_id, iterations_before_checkpointing, checkpoint,
          /*m_lreset_internal=*/m_lreset_internal, /*m_liter_flag=*/liter_flag);
    }
// nowait because critical below has an implicit barrier
#ifdef _OPENMP
#pragma omp single nowait
#endif  // _OPENMP
    num_eqsolve_retries_ += n_local_eqsolve_retries;

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
    {
      if (s.ok()) {
        any_checkpoint_reached |= (*s == SolveEqLoopStatus::CHECKPOINT_REACHED);
      } else {
        UpdateStatusForThread(status_of_all_threads, thread_id, s.status());
      }
    }
  }  // omp parallel

  if (interrupted_) {
    return absl::CancelledError("Run interrupted by user");
  }

  if (!status_of_all_threads.ok()) {
    return status_of_all_threads;
  }

  if (!any_checkpoint_reached) {
    // write MHD energy at end of iterations for current number of surfaces
    logger_.EndStage(h_.mhdEnergy * 4.0 * M_PI * M_PI);
  }

  return any_checkpoint_reached;
}  // SolveEquilibrium

absl::StatusOr<Vmec::SolveEqLoopStatus> Vmec::SolveEquilibriumLoop(
    int thread_id, int iterations_before_checkpointing,
    VmecCheckpoint checkpoint, bool& m_lreset_internal, bool& m_liter_flag) {
  // RECOMPUTE INITIAL PROFILE, BUT WITH IMPROVED AXIS
  // OR
  // RESTART FROM INITIAL PROFILE, BUT WITH A SMALLER TIME-STEP
  if (fc_.restart_reason == RestartReason::BAD_JACOBIAN) {
    decomposed_x_[thread_id]->setZero();
    if (m_lreset_internal) {
      decomposed_x_[thread_id]->interpFromBoundaryAndAxis(t_, b_,
                                                          *p_[thread_id]);
    }

    // protect reads of restart_reason above from write below
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

    // tells restart_iter to store current xc in xstore
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
    {
      fc_.restart_reason = RestartReason::NO_RESTART;
    }

    // In the first multigrid iteration (OFF IN v8.50)
    RestartIteration(fc_.delt0r, thread_id);
  }  // restart_reason == BAD_JACOBIAN

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // start normal iterations
    m_liter_flag = true;

    // reset error flag
    status_ = VmecStatus::NORMAL_TERMINATION;
  }

  // `iter_loop`: FORCE ITERATION LOOP
  // It may hold iter2_>1 when entering this method and when prev iter
  // returned SolveEqLoopStatus::MUST_RETRY.

  // bad_resets counts the number of times we early exit with a reason OTHER
  // THAN SolveEqLoopStatus::MUST_RETRY
  for (int force_iteration = iter2_, bad_resets = 0;
       (force_iteration <= fc_.niterv) && m_liter_flag; force_iteration++) {
    const int iter2 = force_iteration - bad_resets;
    // ADVANCE FOURIER AMPLITUDES OF R, Z, AND LAMBDA
    absl::StatusOr<bool> reached_checkpoint =
        Evolve(checkpoint, iterations_before_checkpointing, fc_.delt0r,
               thread_id, /*m_liter_flag=*/m_liter_flag);
    if (!reached_checkpoint.ok()) {
      return reached_checkpoint.status();
    }
    if (*reached_checkpoint) {
      return SolveEqLoopStatus::CHECKPOINT_REACHED;
    }

    // check for bad jacobian and bad initial guess for axis
    if (fc_.ijacob == 0 &&
        (status_ == VmecStatus::BAD_JACOBIAN ||
         fc_.restart_reason == RestartReason::HUGE_INITIAL_FORCES) &&
        fc_.ns >= 3) {
// protect reads of ijacob above from write below
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
      {
        if (verbose_) {
          // Only warn about bad jacobian if that is actually the reason.
          // The other reason could be restart_reason == HUGE_INITIAL_FORCES,
          // which means that the initial forces are huge (but the Jacbian is
          // fine, i.e., flux surfaces don't overlap yet).
          if (status_ == VmecStatus::BAD_JACOBIAN) {
            std::cout << " INITIAL JACOBIAN CHANGED SIGN!\n";
          }
          std::cout << " TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS\n";
        }

        b_.RecomputeMagneticAxisToFixJacobianSign(fc_.nsval, kSignOfJacobian);
        fc_.ijacob = 1;

        // prepare parameters to functions that get called due to
        // m_lreset_internal and restart_reason == BAD_JACOBIAN
        fc_.restart_reason = RestartReason::BAD_JACOBIAN;

#ifdef VMECPP_USE_CUDA
        // The retry re-initializes the host state vector from the
        // boundary and the recomputed axis; drop the device position,
        // velocity, and backup state so the next stage preparation
        // re-stages them from that fresh host state instead of replaying
        // the failed attempt's device copy.
        vmecpp::InvalidatePtsXCuda();
#endif
      }

      // Signals to re-initialize the state vector
      // from the (updated/improved) initial guess
      // at the top of this method, in the next call.
      // NOTE: `m_lreset_internal` is thread-local, hence need to do this
      // outside the `omp single` block
      m_lreset_internal = true;

      // try again: GOTO 20
      // but need to leave m_liter_flag loop first...
      return SolveEqLoopStatus::MUST_RETRY;
    } else if (status_ != VmecStatus::NORMAL_TERMINATION &&
               status_ != VmecStatus::SUCCESSFUL_TERMINATION) {
      // if something went totally wrong even in this initial steps, do not
      // continue at all
      const auto msg = absl::StrFormat(
          "FATAL ERROR in thread=%d. The solver failed during the first "
          "iterations. This may happen if the initial boundary is poorly "
          "shaped or if it isn't spectrally condensed enough.",
          thread_id);
      return absl::UnknownError(msg);
    }

    if (checkpoint == VmecCheckpoint::EVOLVE &&
        iter2 >= iterations_before_checkpointing) {
      // need to get past re-try with guess_axis in case of bad Jacobian
      return SolveEqLoopStatus::CHECKPOINT_REACHED;
    }

    // (compute MHD energy)
    // (has been done in updateFwdModel already: `h_.mhdEnergy`)

    // ADDITIONAL STOPPING CRITERION (set m_liter_flag to FALSE)

#ifdef _OPENMP
// Protect the reads of fc_.ijacob from writes in the single region below.
#pragma omp barrier
#endif  // _OPENMP

    // the blocks for ijacob=25 or 50 are equal up to the point
    // that for 25, delt0r is reset to 0.98*delt (delt given by user)
    // and  for 50, delt0r is reset to 0.96*delt (delt given by user)
    if (fc_.ijacob == 25 || fc_.ijacob == 50) {
      // jacobian changed sign 25/50 times: hmmm? :-/

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
      {
        fc_.restart_reason = RestartReason::BAD_JACOBIAN;
      }

      RestartIteration(fc_.delt0r, thread_id);

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
      {
        // fc_.ijacob is incremented in RestartIteration
        const double scale = fc_.ijacob < 50 ? 0.98 : 0.96;

        fc_.delt0r = scale * indata_.delt;

        if (verbose_) {
          std::cout
              << absl::StrFormat(
                     "HAVING A CONVERGENCE PROBLEM: RESETTING DELT TO %8.3f. "
                     " If this does NOT resolve the problem,"
                     " try changing (decrease OR increase) the value of DELT\n",
                     fc_.delt0r)
              << std::flush;
        }

        // done by restart_iter already...
        fc_.restart_reason = RestartReason::NO_RESTART;
      }
      // try again: GOTO 20
      // but need to leave m_liter_flag loop first...
      return SolveEqLoopStatus::MUST_RETRY;
    } else if (fc_.ijacob >= 75) {
      // jacobian changed sign at least 75 times: time to give up :-(

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
      {
        // 'MORE THAN 75 JACOBIAN ITERATIONS (DECREASE DELT)'
        status_ = VmecStatus::JACOBIAN_75_TIMES_BAD;
        m_liter_flag = false;
      }
    }

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
    {
      // TIME STEP CONTROL

      if (iter2 == iter1_ || fc_.res0 == -1) {
        // if res0 has never been assigned (-1), give it the current value of
        // fsq
        fc_.res0 = fc_.fsq;
      }

      // res0 is the best force residual we got so far
      fc_.res0 = std::min(fc_.res0, fc_.fsq);
    }

#ifdef VMECPP_USE_CUDA
    if (sync_elided_iter_) {
      // Sync-elided iteration: fsq/res0 are stale; the store/restart
      // bookkeeping runs on boundary iterations only, which also sets the
      // device-state backup cadence to once per K-window.
    } else
#endif
        if (fc_.fsq <= fc_.res0 && (iter2 - iter1_) > 10) {
      // Store current state (restart_reason=NO_RESTART)
      // --> was able to reduce force consistenly over at least 10 iterations
      RestartIteration(fc_.delt0r, thread_id);
    } else if (fc_.fsq > 100.0 * fc_.res0 && iter2 > iter1_) {
      // Residuals are growing in time, reduce time step

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
      fc_.restart_reason = RestartReason::BAD_JACOBIAN;
    } else if ((iter2 - iter1_) > fc_.kPreconditionerUpdateInterval / 2 &&
               iter2 > 2 * fc_.kPreconditionerUpdateInterval &&
               fc_.fsqr + fc_.fsqz > 1.0e-2) {
      // quite some iterations and quite large forces
      // --> restart with different timestep

      // TODO(jons): maybe the threshold 0.01 is too large nowadays (at high
      // resolution)
      // --> this could help fix the cases where VMEC gets stuck immediately
      // at ~2e-3
      // --> lower threshold, e.g. 1e-4 ?

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
      fc_.restart_reason = RestartReason::BAD_PROGRESS;
    }

    const RestartReason restart_reason = fc_.restart_reason;
// protect read of restart_reason above from write (in different thread) below
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

    if (restart_reason != RestartReason::NO_RESTART) {
      // Retrieve previous good state
      RestartIteration(fc_.delt0r, thread_id);
      // This code path does not increment the iter2 counter in VMEC 8.52, so we
      // have to keep track
      bad_resets++;
#ifdef _OPENMP
#pragma omp single nowait
#endif  // _OPENMP
      iter1_ = iter2;
    } else {
      // Increment time step and printout every nstep iterations
      // status report due or
      // first iteration or
      // iterations cancelled already (last iteration)
      if (iter2 % indata_.nstep == 0 || iter2 == 1 || !m_liter_flag) {
#ifndef VMECPP_USE_CUDA
        // TODO(jons): why compute spectral width from backup and not current
        // gc (== physical xc) --> <M> includes scalxc ???
        physical_x_backup_[thread_id]->ComputeSpectralWidth(t_, *p_[thread_id]);
#endif  // !VMECPP_USE_CUDA

        // NOTE: IIRC, this still needs to be called to keep the spectral width
        // updated. Screen output will be controlled by checking the `verbose_`
        // flag inside `Printout`.
        Printout(fc_.delt0r, thread_id, iter2);

        // Check for interrupt signal (e.g., Ctrl+C from Python).
        // Only the master thread calls the callback; the subsequent barrier
        // ensures all threads see the updated m_liter_flag.
        // It MUST be the master thread, other threads cannot acquire the GIL.
#ifdef _OPENMP
#pragma omp master
#endif  // _OPENMP
        {
          if (interrupt_callback_ && interrupt_callback_()) {
            m_liter_flag = false;
            // MSVC's default OpenMP (2.0) lacks `atomic write`; the write is a
            // single-writer write-once bool here, so it stays correct without
            // it. GCC/Clang keep the atomic for memory visibility.
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp atomic write
#endif
            interrupted_ = true;
            std::cout << "Received interrupt signal from Python thread.\n";
          }
        }

        if (checkpoint == VmecCheckpoint::PRINTOUT &&
            iter2 >= iterations_before_checkpointing) {
          return SolveEqLoopStatus::CHECKPOINT_REACHED;
        }
      }
    }

// protect read of vacuum_pressure_state_ in get_delbsq called by Printout above
// from write below
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

// don't use nowait here, since need implicit barrier to protect read of iter2_
// from write below in potentially different thread
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
    {
      // vacuum_pressure_state gets set to VacuumPressureState::kInitialized in
      // vacuum() of NESTOR
      if (vacuum_pressure_state_ == VacuumPressureState::kInitialized) {
        vacuum_pressure_state_ = VacuumPressureState::kActive;

        if (verbose_) {
          std::cout << absl::StrFormat(
                           "VACUUM PRESSURE TURNED ON AT %4d ITERATIONS",
                           iter2_)
                    << "\n\n";
        }
      }
    }

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp atomic write
#endif  // _OPENMP, not MSVC (no `atomic write` in OpenMP 2.0)
    // update iter2_ for all threads, all threads have the same value of iter2,
    // it does not matter who does it.
    // bad resets didn't increment, iter2 in VMEC 8.52, so we need to compute
    // the backwards compatible iteration count
    iter2_ = (force_iteration - bad_resets) + 1;  // equivalent to iter2++
  }  // while m_liter_flag

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // Post-loop per-cfg salvage. When the iteration loop exits because
    // force_iteration exceeded the shared niterv without the per-cfg
    // convergence gate setting all_done true, the batch may still contain
    // configurations that did converge earlier. Mark any cfgs that are
    // still active as timed out, and if at least one cfg converged within
    // the stage's niterv allotment, classify the run as SUCCESSFUL.
    // Without this salvage the slowest cfg in a batch silently fails
    // every other cfg even when those reached their tolerance.
    const int n_cfg_post = static_cast<int>(fc_.active_per_cfg.size());
    if (n_cfg_post > 0 && status_ != VmecStatus::SUCCESSFUL_TERMINATION) {
      int converged_count = 0;
      int timed_out_count = 0;
      for (int c = 0; c < n_cfg_post; ++c) {
        const bool was_converged =
            (static_cast<int>(fc_.converged_per_cfg.size()) > c &&
             fc_.converged_per_cfg[c]);
        if (was_converged) ++converged_count;
        if (fc_.active_per_cfg[c]) {
          fc_.active_per_cfg[c] = 0;
          if (static_cast<int>(fc_.converged_per_cfg.size()) > c) {
            fc_.converged_per_cfg[c] = 0;
          }
          ++timed_out_count;
        }
      }
      if (converged_count > 0) {
        status_ = VmecStatus::SUCCESSFUL_TERMINATION;
        if (timed_out_count > 0) {
          std::fprintf(stderr,
                       "[vmec.cc] batch terminated at shared niterv=%d: "
                       "%d cfg(s) converged, %d cfg(s) timed out\n",
                       fc_.niterv, converged_count, timed_out_count);
        }
      }
    }
  }

  return SolveEqLoopStatus::NORMAL_TERMINATION;
}

// aligned visually with restart_iter.f90
void Vmec::RestartIteration(double& m_delt0r, int thread_id) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef VMECPP_USE_CUDA
  // A restart changes the time step baked into the captured
  // whole-iteration graph; drop it and re-capture on the next eligible
  // window.
  vmecpp::InvalidateIterationGraphCuda();
#endif

  // VMECPP_TRACE_RESTART=1: log every store/restore event with the
  // controller inputs that drove the decision.
  static int trace_restart_env = -1;
  if (trace_restart_env < 0) {
    const char* e = std::getenv("VMECPP_TRACE_RESTART");
    trace_restart_env = (e && std::atoi(e) > 0) ? 1 : 0;
  }
  if (trace_restart_env) {
    std::fprintf(
        stderr,
        "[restart] iter2=%d ns=%d reason=%d delt0r=%.6e fsq=%.6e res0=%.6e\n",
        iter2_, fc_.ns, static_cast<int>(fc_.restart_reason), m_delt0r, fc_.fsq,
        fc_.res0);
  }

  if (fc_.restart_reason == RestartReason::BAD_JACOBIAN) {
    // restore previous good state

    // zero velocity
    decomposed_v_[thread_id]->setZero();

#ifdef VMECPP_USE_CUDA
    // Device-side restore: at n_cfg=1, whole-batch RestorePtsXFromBackupCuda
    // matches the legacy behavior. At n_cfg>1, derive a per-cfg restart mask
    // from the per-cfg jacobian cache (minTau[c] * maxTau[c] < 0 or non-finite
    // → cfg c needs rollback) and route through the per-cfg variant, leaving
    // cfgs that didn't trigger BAD_JACOBIAN at their current d_pts_x state.
    {
      const auto& jac_cache = vmecpp::GetJacMinmaxPerCfgCache();
      const int n_cfg_cuda = vmecpp::GetNConfigMaxCuda();
      if (n_cfg_cuda > 1 &&
          static_cast<int>(jac_cache.size()) == 2 * n_cfg_cuda) {
        std::vector<std::uint8_t> mask(n_cfg_cuda, 0);
        bool any_bad = false;
        for (int c = 0; c < n_cfg_cuda; ++c) {
          double mn = jac_cache[2 * c + 0];
          double mx = jac_cache[2 * c + 1];
          double prod = mn * mx;
          bool bad = (prod < 0.0) || !std::isfinite(prod);
          mask[c] = bad ? 1 : 0;
          any_bad = any_bad || bad;
        }
        if (any_bad) {
          vmecpp::RestorePtsXFromBackupPerCfgCuda(mask);
        } else {
          // The restart fired without a per-configuration jacobian event
          // (the free-boundary soft start forces BAD_JACOBIAN at vacuum
          // activation): rewind every configuration, matching the
          // single-configuration restore.
          vmecpp::RestorePtsXFromBackupCuda();
        }
      } else {
        vmecpp::RestorePtsXFromBackupCuda();
      }
    }
#else
    // restore state from backup
    *decomposed_x_[thread_id] = *physical_x_backup_[thread_id];
#endif

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
    {
      // reduce time step
      m_delt0r = m_delt0r * 0.9;

      // count occurence of bad Jacobian
      fc_.ijacob = fc_.ijacob + 1;

      // update marker
      iter1_ = iter2_;

      fc_.restart_reason = RestartReason::NO_RESTART;
    }

  } else if (fc_.restart_reason == RestartReason::BAD_PROGRESS) {
    // restore previous good state

    // zero velocity
    decomposed_v_[thread_id]->setZero();

#ifdef VMECPP_USE_CUDA
    // Bad progress is detected from the shared residual trajectory, so
    // there is no per-cfg culprit to isolate; restore the still-active
    // cfgs to their backups and leave inactive cfgs frozen at their
    // converged snapshots. With the backup refreshed at every improving
    // iteration, healthy cfgs rewind at most the few iterations since
    // their last store.
    {
      const int n_cfg_cuda = vmecpp::GetNConfigMaxCuda();
      if (n_cfg_cuda > 1 &&
          static_cast<int>(fc_.active_per_cfg.size()) == n_cfg_cuda) {
        vmecpp::RestorePtsXFromBackupPerCfgCuda(fc_.active_per_cfg);
      } else {
        vmecpp::RestorePtsXFromBackupCuda();
      }
    }
#else
    // restore state from backup
    *decomposed_x_[thread_id] = *physical_x_backup_[thread_id];
#endif

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
    {
      // reduce time step
      m_delt0r = m_delt0r / 1.03;

      fc_.restart_reason = RestartReason::NO_RESTART;
    }
  } else {
    // NO_RESTART or HUGE_INITIAL_FORCES
    // save current state vector, e.g. restart_reason == NO_RESTART

#ifdef VMECPP_USE_CUDA
    // Device-side backup at the host store cadence: every improving
    // iteration past the restart window refreshes the rollback target,
    // so a BAD_JACOBIAN or BAD_PROGRESS restore rewinds the device
    // state to the same snapshot the host path would restore. A coarser
    // cadence leaves the backup pinned at an old state on restart-heavy
    // inputs; each restore then discards all progress since that state
    // and the run cannot advance past the first recurring bad-Jacobian
    // event. The copy is one fused kernel launch, asynchronous on the
    // iteration stream. Under sync elision this call participates at
    // the same K-boundary cadence as the rest of the store and restart
    // bookkeeping.
    vmecpp::BackupPtsXCuda();
#else
    // update backup
    *physical_x_backup_[thread_id] = *decomposed_x_[thread_id];
#endif
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

absl::StatusOr<bool> Vmec::Evolve(VmecCheckpoint checkpoint,
                                  int iterations_before_checkpointing,
                                  double time_step, int thread_id,
                                  bool& m_liter_flag) {
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    fc_.restart_reason = RestartReason::NO_RESTART;
  }

#ifdef VMECPP_USE_CUDA
  // K-window sync elision (VMECPP_SYNC_ELIDE=K). On elided iterations the
  // per-iteration scalar D2H + stream-sync sites in the CUDA wrappers
  // skip their transfers (host receives last boundary-synced values), the
  // device time-step controller is authoritative, and the convergence
  // gate plus the restart/store bookkeeping are skipped until the next
  // boundary. Boundaries: iterations 1-2 of a segment (HUGE_INITIAL_FORCES
  // and ring-reset semantics), every K-th iteration (K=25 aligns with the
  // preconditioner-update cadence), and any iteration entered with a
  // pending restart disposition.
  if (sync_elide_k_ == -2) {
    const char* e = std::getenv("VMECPP_SYNC_ELIDE");
    sync_elide_k_ = (e ? std::atoi(e) : 0);
    if (sync_elide_k_ < 0) sync_elide_k_ = 0;
    vmecpp::SetSyncElideRunCuda(sync_elide_k_ > 0 ? 1 : 0);
    static int last_printed = 0;
    if (sync_elide_k_ > 0 && sync_elide_k_ != last_printed) {
      last_printed = sync_elide_k_;
      std::fprintf(
          stderr,
          "[vmec.cc] sync elision ENABLED (VMECPP_SYNC_ELIDE=%d): host "
          "syncs, convergence gate, and restart bookkeeping run every %d "
          "iterations; device controller authoritative\n",
          sync_elide_k_, sync_elide_k_);
    }
  }
  const int sync_elide_k = sync_elide_k_;
  if (sync_elide_k > 0) {
    // Free-boundary runs elide only with the vacuum contribution fully
    // active: every iteration up to the kActive transition runs live, so
    // the activation check reads fresh residuals and fires on time (a
    // boundary moving without its vacuum constraint for a window of
    // K - 1 iterations can leave the mgrid extent), and the soft-start
    // restart replays the live sequence. Once active, the elision
    // covers the scalar sync sites while the vacuum block keeps its
    // per-iteration cadence: the NESTOR response must track the
    // boundary every iteration, and a window-frozen edge force leaves
    // the iteration orbiting its stale vacuum target instead of
    // converging, with or without device-side geometry tracking of the
    // staged edge pressure.
    const bool boundary =
        (iter2_ <= 2) || ((iter2_ - iter1_) <= 2) ||
        ((iter2_ % sync_elide_k) == 1) ||
        (fc_.restart_reason != RestartReason::NO_RESTART) ||
        (fc_.lfreeb && vacuum_pressure_state_ != VacuumPressureState::kActive);
    sync_elided_iter_ = !boundary;
  } else {
    sync_elided_iter_ = false;
  }
  vmecpp::SetSyncElideIterCuda(sync_elided_iter_ ? 1 : 0);

  // Whole-iteration graph (VMECPP_ITER_GRAPH=1): a captured sync-elided
  // iteration replays as one cudaGraphLaunch and the host dispatch below
  // is skipped. The residual-trace logging still runs so the output
  // trace lengths match the non-graph path. Capture brackets one normal
  // elided iteration; warmups and boundary iterations run the normal
  // path.
  bool iter_graph_capture_open = false;
  if (sync_elide_k > 0 && sync_elided_iter_ && vmecpp::IterGraphEnabledCuda()) {
    if (vmecpp::IterGraphReplayCuda()) {
      if (r_[thread_id]->nsMaxF1 == fc_.ns) {
        fc_.force_residual_r.push_back(fc_.fsqr);
        fc_.force_residual_z.push_back(fc_.fsqz);
        fc_.force_residual_lambda.push_back(fc_.fsql);
        fc_.delbsq.push_back(0.0);
        fc_.restart_reasons.push_back(fc_.restart_reason);
        fc_.mhd_energy.push_back(h_.mhdEnergy);
      }
      return false;
    }
    iter_graph_capture_open = vmecpp::IterGraphBeginCaptureCuda();
  }
#endif

  // `funct3d` - COMPUTE MHD FORCES
  absl::StatusOr<bool> reached_checkpoint = UpdateForwardModel(
      checkpoint, iterations_before_checkpointing, thread_id);
  if (!reached_checkpoint.ok() || *reached_checkpoint == true) {
    return reached_checkpoint;
  }

#ifdef VMECPP_USE_CUDA
  // Full-state dumps for cross-cfg contamination A/B runs.
  // VMECPP_STATE_DUMP_ITERS="1,2,26" writes the batched decomposed-x state
  // (and with VMECPP_STATE_DUMP_F=1 the decomposed forces, with
  // VMECPP_STATE_DUMP_PROF=1 the per-cfg chip/iota half-grid profiles) at
  // those iter2 values to files prefixed by VMECPP_STATE_DUMP_PATH. The
  // hook sits directly after UpdateForwardModel, outside any gate
  // branch, so iteration 1 of each stage is captured too; the filename
  // carries ns so per-stage dumps do not collide.
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    static std::vector<int>* state_dump_iters = nullptr;
    if (state_dump_iters == nullptr) {
      state_dump_iters = new std::vector<int>();
      if (const char* e = std::getenv("VMECPP_STATE_DUMP_ITERS")) {
        std::string spec(e);
        size_t pos = 0;
        while (pos < spec.size()) {
          size_t comma = spec.find(',', pos);
          if (comma == std::string::npos) comma = spec.size();
          int v = std::atoi(spec.substr(pos, comma - pos).c_str());
          if (v > 0) state_dump_iters->push_back(v);
          pos = comma + 1;
        }
      }
    }
    if (!state_dump_iters->empty() &&
        std::find(state_dump_iters->begin(), state_dump_iters->end(), iter2_) !=
            state_dump_iters->end()) {
      const char* prefix = std::getenv("VMECPP_STATE_DUMP_PATH");
      std::string base = std::string(prefix ? prefix : "/tmp/vmecpp_state") +
                         "_ns" + std::to_string(fc_.ns) + "_iter" +
                         std::to_string(iter2_);
      vmecpp::DumpPtsXAllCfgsCuda((base + ".bin").c_str(), iter2_);
      const char* dump_f = std::getenv("VMECPP_STATE_DUMP_F");
      if (dump_f && std::atoi(dump_f) > 0) {
        vmecpp::DumpDecomposedFAllCfgsCuda((base + "_f.bin").c_str(), iter2_);
      }
      const char* dump_prof = std::getenv("VMECPP_STATE_DUMP_PROF");
      if (dump_prof && std::atoi(dump_prof) > 0) {
        vmecpp::DumpBContraProfilesAllCfgsCuda((base + "_prof.bin").c_str(),
                                               iter2_, fc_.ns - 1);
      }
    }
  }
#endif

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // COMPUTE ABSOLUTE STOPPING CRITERION
    // A residual is either NaN or inf, guard for highly degenerate inputs
    bool all_residuals_finite = std::isfinite(fc_.fsqr) &&
                                std::isfinite(fc_.fsqz) &&
                                std::isfinite(fc_.fsql);
#ifdef VMECPP_USE_CUDA
    if (sync_elided_iter_) {
      // Sync-elided iteration: the host residual fields are stale; the
      // convergence gate, AUTH flag reads, and status update run on the
      // next boundary iteration with fresh values.
    } else
#endif
        if ((iter2_ == 1 &&
             fc_.restart_reason == RestartReason::BAD_JACOBIAN) ||
            !all_residuals_finite) {
      // first iteration and Jacobian was not computed correctly
      status_ = VmecStatus::BAD_JACOBIAN;
    } else {
      // Per-configuration successful-termination gate. The CUDA execution
      // path populates the per-configuration residual vectors fsqr_per_cfg,
      // fsqz_per_cfg, and fsql_per_cfg in addition to the equivalent
      // scalar quantities; the gate below examines the per-configuration
      // values and signals SUCCESSFUL_TERMINATION only when every
      // configuration has fallen below the multigrid tolerance and been
      // marked inactive. Under single-configuration execution the
      // per-configuration vectors degenerate to length one and the test
      // reduces exactly to the legacy scalar comparison. Under
      // multi-configuration execution with distinct boundaries each
      // configuration evolves its residuals independently and may
      // converge at different iterations; the conjunctive termination
      // ensures that the iteration loop runs until every configuration
      // has reached its own convergence threshold.
#ifdef VMECPP_USE_CUDA
      const int n_cfg = static_cast<int>(fc_.active_per_cfg.size());
#else
      const int n_cfg = 0;
#endif
      if (n_cfg > 0 && static_cast<int>(fc_.fsqr_per_cfg.size()) == n_cfg &&
          static_cast<int>(fc_.fsqz_per_cfg.size()) == n_cfg &&
          static_cast<int>(fc_.fsql_per_cfg.size()) == n_cfg) {
        // Per-cfg gate. The device k_check_convergence flag decides each
        // cfg's termination by default (same normalized-residual
        // arithmetic as evalFResInvar, from the device-resident force-norm
        // sums, energy scalars, and per-cfg plasma volumes); the host
        // comparison applies when the flag buffers are absent or
        // VMECPP_CONV_FLAG_AUTH=0.
        // VMECPP_CONV_FLAG_DEBUG=1 logs any disagreement (passive).
#ifdef VMECPP_USE_CUDA
        static int per_cfg_debug_env = -1;
        if (per_cfg_debug_env < 0) {
          const char* e = std::getenv("VMECPP_CONV_FLAG_DEBUG");
          per_cfg_debug_env = (e && std::atoi(e) > 0) ? 1 : 0;
        }
        static int per_cfg_auth_env = -1;
        if (per_cfg_auth_env < 0) {
          const char* e = std::getenv("VMECPP_CONV_FLAG_AUTH");
          per_cfg_auth_env = (e && std::atoi(e) == 0) ? 0 : 1;
          if (!per_cfg_auth_env) {
            std::fprintf(stderr,
                         "[vmec.cc] per-cfg device convergence flag disabled "
                         "(VMECPP_CONV_FLAG_AUTH=0); host gate is "
                         "authoritative\n");
          }
        }
#endif
        // Periodic per-cfg residual dump to diagnose cfg N>0 convergence in
        // distinct mode. VMECPP_PERCFG_RESIDUAL_DUMP=K logs every K iters
        // (default off). Useful for confirming whether each cfg is making
        // progress vs stuck vs diverging when the shared velocity_scale is
        // tuned for cfg 0 only.
        static int percfg_dump_period = -1;
        if (percfg_dump_period < 0) {
          const char* e = std::getenv("VMECPP_PERCFG_RESIDUAL_DUMP");
          percfg_dump_period = (e ? std::atoi(e) : 0);
          if (percfg_dump_period < 0) percfg_dump_period = 0;
        }
        if (percfg_dump_period > 0 && (iter2_ % percfg_dump_period) == 0) {
          for (int c = 0; c < n_cfg; ++c) {
            std::fprintf(
                stderr,
                "[percfg] iter2=%d cfg=%d active=%d fsqr=%.3e fsqz=%.3e "
                "fsql=%.3e ftolv=%.3e\n",
                iter2_, c, fc_.active_per_cfg[c], fc_.fsqr_per_cfg[c],
                fc_.fsqz_per_cfg[c], fc_.fsql_per_cfg[c], fc_.ftolv);
          }
        }
        // Per-cfg niter cap. Read once at first entry. When set, any active
        // cfg whose iter2_per_cfg has reached the cap is marked timed-out so
        // the batch can terminate when only the slow cfgs remain. Without
        // the cap (legacy default), the loop waits for every cfg or hits
        // the shared niterv.
        if (per_cfg_niter_cap_ < 0) {
          const char* e = std::getenv("VMECPP_PER_CFG_NITER_CAP");
          per_cfg_niter_cap_ = (e && std::atoi(e) > 0)
                                   ? std::atoi(e)
                                   : std::numeric_limits<int>::max();
          if (per_cfg_niter_cap_ != std::numeric_limits<int>::max()) {
            std::fprintf(
                stderr,
                "[vmec.cc] per-cfg niter cap = %d (VMECPP_PER_CFG_NITER_CAP)\n",
                per_cfg_niter_cap_);
          }
        }
        fc_.niter_max_per_cfg = per_cfg_niter_cap_;
        // Lazily size iter2_per_cfg and converged_per_cfg if ResizeForBatch
        // happened in a path that pre-dates these fields.
        if (static_cast<int>(fc_.iter2_per_cfg.size()) != n_cfg) {
          fc_.iter2_per_cfg.assign(n_cfg, 0);
        }
        if (static_cast<int>(fc_.converged_per_cfg.size()) != n_cfg) {
          fc_.converged_per_cfg.assign(n_cfg, static_cast<std::uint8_t>(0));
        }
        bool all_done = true;
        int any_converged_count = 0;
        int any_timed_out_count = 0;
        for (int c = 0; c < n_cfg; ++c) {
          if (fc_.active_per_cfg[c]) {
            ++fc_.iter2_per_cfg[c];
            const bool host_done = (fc_.fsqr_per_cfg[c] <= fc_.ftolv &&
                                    fc_.fsqz_per_cfg[c] <= fc_.ftolv &&
                                    fc_.fsql_per_cfg[c] <= fc_.ftolv);
#ifdef VMECPP_USE_CUDA
            bool cfg_done = host_done;
            if (per_cfg_auth_env || per_cfg_debug_env) {
              const int dev_flag = vmecpp::GetConvergenceFlag(c);
              const bool dev_done = (dev_flag == 1);
              if (per_cfg_debug_env && dev_flag >= 0 && host_done != dev_done) {
                std::fprintf(
                    stderr,
                    "[vmec.cc] WARNING cfg=%d host_norm=%d dev_norm=%d "
                    "(host_fsqr=%.3e fsqz=%.3e fsql=%.3e ftolv=%.3e)\n",
                    c, host_done ? 1 : 0, dev_done ? 1 : 0, fc_.fsqr_per_cfg[c],
                    fc_.fsqz_per_cfg[c], fc_.fsql_per_cfg[c], fc_.ftolv);
              }
              if (per_cfg_auth_env && dev_flag >= 0) {
                cfg_done = dev_done;
              }
            }
#else
            const bool cfg_done = host_done;
#endif
            const bool timed_out =
                (fc_.iter2_per_cfg[c] >= fc_.niter_max_per_cfg);
            if (cfg_done) {
              fc_.active_per_cfg[c] = 0;
              fc_.converged_per_cfg[c] = 1;
#ifdef VMECPP_USE_CUDA
              // Freeze this cfg's state at its convergence moment; the
              // live device slice keeps being modified while the rest of
              // the batch iterates.
              vmecpp::SnapshotInactiveCfgCuda(c);
#endif
            } else if (timed_out) {
              fc_.active_per_cfg[c] = 0;
              fc_.converged_per_cfg[c] = 0;
#ifdef VMECPP_USE_CUDA
              vmecpp::SnapshotInactiveCfgCuda(c);
#endif
              std::fprintf(
                  stderr,
                  "[vmec.cc] cfg %d timed out at iter2_per_cfg=%d "
                  "(cap=%d) fsqr=%.3e fsqz=%.3e fsql=%.3e ftolv=%.3e\n",
                  c, fc_.iter2_per_cfg[c], fc_.niter_max_per_cfg,
                  fc_.fsqr_per_cfg[c], fc_.fsqz_per_cfg[c], fc_.fsql_per_cfg[c],
                  fc_.ftolv);
            } else {
              all_done = false;
            }
          }
          if (fc_.converged_per_cfg[c]) ++any_converged_count;
          if (!fc_.active_per_cfg[c] && !fc_.converged_per_cfg[c]) {
            ++any_timed_out_count;
          }
        }
        if (all_done) {
          m_liter_flag = false;
          // The batch terminates successfully when at least one cfg met
          // ftolv. If every active cfg timed out without converging, the
          // batch is reported as a non-success run; the per-cfg outcome
          // is still surfaced via converged_per_cfg.
          if (any_converged_count > 0) {
            status_ = VmecStatus::SUCCESSFUL_TERMINATION;
          }
          if (any_timed_out_count > 0) {
            std::fprintf(stderr,
                         "[vmec.cc] batch terminated: %d cfg(s) converged, "
                         "%d cfg(s) timed out (per-cfg niter cap=%d)\n",
                         any_converged_count, any_timed_out_count,
                         fc_.niter_max_per_cfg);
          }
        }
      } else {
        // Fallback single-cfg gate (n_cfg vectors not yet sized, e.g. CPU
        // path or before ResizeForBatch is called).
        //
        // The device k_check_convergence flag decides termination by
        // default; the host comparison applies when the flag buffers are
        // absent or VMECPP_CONV_FLAG_AUTH=0.
        // VMECPP_CONV_FLAG_DEBUG=1 logs any disagreement (passive).
#ifdef VMECPP_USE_CUDA
        static int auth_env = -1;
        if (auth_env < 0) {
          const char* e = std::getenv("VMECPP_CONV_FLAG_AUTH");
          auth_env = (e && std::atoi(e) == 0) ? 0 : 1;
          if (!auth_env) {
            std::fprintf(stderr,
                         "[vmec.cc] device convergence flag disabled "
                         "(VMECPP_CONV_FLAG_AUTH=0); host gate is "
                         "authoritative\n");
          }
        }
        static int debug_env = -1;
        if (debug_env < 0) {
          const char* e = std::getenv("VMECPP_CONV_FLAG_DEBUG");
          debug_env = (e && std::atoi(e) > 0) ? 1 : 0;
        }
        const bool host_conv = (fc_.fsqr <= fc_.ftolv &&
                                fc_.fsqz <= fc_.ftolv && fc_.fsql <= fc_.ftolv);
        const int dev_flag = auth_env ? vmecpp::GetConvergenceFlag(0) : -1;
        const bool dev_conv =
            (auth_env && dev_flag >= 0) ? (dev_flag == 1) : host_conv;
        if (debug_env && (host_conv != dev_conv)) {
          std::fprintf(stderr,
                       "[vmec.cc] WARNING: host=%d device=%d "
                       "(fsqr=%.3e fsqz=%.3e fsql=%.3e ftolv=%.3e)\n",
                       host_conv ? 1 : 0, dev_conv ? 1 : 0, fc_.fsqr, fc_.fsqz,
                       fc_.fsql, fc_.ftolv);
        }
        if (auth_env ? dev_conv : host_conv) {
          m_liter_flag = false;
          status_ = VmecStatus::SUCCESSFUL_TERMINATION;
        }
#else
        if (fc_.fsqr <= fc_.ftolv && fc_.fsqz <= fc_.ftolv &&
            fc_.fsql <= fc_.ftolv) {
          // converged to desired tolerance
          m_liter_flag = false;
          status_ = VmecStatus::SUCCESSFUL_TERMINATION;
        }
#endif
      }
    }
  }  // #pragma omp single (there is an implicit omp barrier here)

  if (status_ != VmecStatus::NORMAL_TERMINATION || !m_liter_flag) {
#ifdef VMECPP_USE_CUDA
    // Unreachable on elided iterations (the gate above is skipped), so an
    // open capture here means a future edit changed that invariant;
    // discard the capture rather than leak it into the next iteration.
    if (iter_graph_capture_open) {
      vmecpp::AbortIterGraphCaptureCuda();
    }
#endif
    // erroneous iteration or shall not iterate further
    return false;
  }

  // ...else no error and not converged --> keep going...

  // COMPUTE DAMPING PARAMETER (DTAU) AND
  // EVOLVE R, Z, AND LAMBDA ARRAYS IN FOURIER SPACE

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // sum of preconditioned force residuals in current iteration
    const double fsq1 = fc_.fsqr1 + fc_.fsqz1 + fc_.fsql1;

    if (iter2_ == iter1_) {
      // initialize all entries in otau to 0.15/time_step --> required for
      // averaging otau: "over" tau --> 1/tau ???
      invTau_.setConstant(0.15 / time_step);
    }

    // shift elements for averaging to the left to make space at end for new
    // entry (oldest entry ends up at the end and will be overwritten later)
    {
      // Left-shift the averaging history in place (drop the oldest entry). A
      // forward copy is alias-safe since the destination precedes the source,
      // which avoids a per-iteration heap temporary.
      std::copy(invTau_.data() + 1, invTau_.data() + invTau_.size(),
                invTau_.data());
    }

    if (iter2_ > iter1_) {
      double invtau_numerator = 0.;
      if (fsq1 != 0.) {
        // fsq is 1 (first iteration) or fsq1 from previous iteration
        // fsq1/fsq is y_n assuming monotonic decrease of energy
        invtau_numerator = std::min(std::abs(std::log(fsq1 / fc_.fsq)), 0.15);
      }

      // overwrite oldest entry (at last index after rotation above) with the
      // new value of 1/tau
      invTau_[invTau_.size() - 1] = invtau_numerator / time_step;
    }

    // update backup copy of fsq1 --> here, fsq is fsq1 of previous iteration
    fc_.fsq = fsq1;
  }  // #pragma omp single (there is an implicit omp barrier here)

  // Our thread owns the last LCFS, it is our responsibility to log the
  // residuals.
  if (r_[thread_id]->nsMaxF1 == fc_.ns) {
    fc_.force_residual_r.push_back(fc_.fsqr);
    fc_.force_residual_z.push_back(fc_.fsqz);
    fc_.force_residual_lambda.push_back(fc_.fsql);
    if (fc_.lfreeb) {
      // delbsq is only available on the LCFS thread
      fc_.delbsq.push_back(m_[thread_id]->get_delbsq());
    } else {
      fc_.delbsq.push_back(0.0);
    }
    fc_.restart_reasons.push_back(fc_.restart_reason);
    fc_.mhd_energy.push_back(h_.mhdEnergy);
  }

  // averaging over ndamp entries : 1/ndamp*sum(invTau)
  const double otav = invTau_.sum() / kNDamp;

  const double dtau = time_step * otav / 2.0;
  const double b1 = 1.0 - dtau;
  const double fac = 1.0 / (1.0 + dtau);

  // THIS IS THE TIME-STEP ALGORITHM. IT IS ESSENTIALLY A CONJUGATE
  // GRADIENT METHOD, WITHOUT THE LINE SEARCHES (FLETCHER-REEVES),
  // BASED ON A METHOD GIVEN BY P. GARABEDIAN
  PerformTimeStep(fac, b1, time_step, thread_id);

#ifdef VMECPP_USE_CUDA
  if (iter_graph_capture_open) {
    vmecpp::IterGraphEndCaptureCuda();
  }
#endif

  return false;
}

void Vmec::Printout(double delt0r, int thread_id, int iter2) {
#ifdef VMECPP_USE_CUDA
  // Refresh the host decomposed spectra from the device, evaluate the
  // per-surface spectral width (defined on the raw decomposed
  // coefficients), and volume-average it over unique half-grid points
  // with the device-resident differential-volume weights. Single-rank
  // execution; the contribution is registered directly.
  {
    const int ns_local = r_[thread_id]->nsMaxF1 - r_[thread_id]->nsMinF1;
    vmecpp::FlushDecomposedXToHostCuda(0, ns_local, s_.mpol, s_.ntor,
                                       s_.lthreed,
                                       decomposed_x_[thread_id]->rmncc.data(),
                                       decomposed_x_[thread_id]->rmnss.data(),
                                       decomposed_x_[thread_id]->zmnsc.data(),
                                       decomposed_x_[thread_id]->zmncs.data(),
                                       decomposed_x_[thread_id]->lmnsc.data(),
                                       decomposed_x_[thread_id]->lmncs.data());
    decomposed_x_[thread_id]->ComputeSpectralWidth(t_, *p_[thread_id]);
    const int ns_h = r_[thread_id]->nsMaxH - r_[thread_id]->nsMinH;
    std::vector<double> dvdsh_staged(std::max(ns_h, 0), 0.0);
    vmecpp::FlushDVdsHToHostCuda(ns_h, dvdsh_staged.data());
    SpectralWidthContribution swc = {.numerator = 0.0, .denominator = 0.0};
    const auto& spectral_width = p_[thread_id]->spectral_width;
    const int nsMinH = r_[thread_id]->nsMinH;
    const int nsMaxH = r_[thread_id]->nsMaxH;
    const int nsMinF1 = r_[thread_id]->nsMinF1;
    for (int jH = nsMinH; jH < nsMaxH; ++jH) {
      if (jH < nsMaxH - 1 || jH == fc_.ns - 2) {
        const double width_on_half_grid =
            (spectral_width[jH + 1 - nsMinF1] + spectral_width[jH - nsMinF1]) /
            2.0;
        swc.numerator += width_on_half_grid * dvdsh_staged[jH - nsMinH];
        swc.denominator += dvdsh_staged[jH - nsMinH];
      }
    }
    h_.ResetSpectralWidthAccumulators();
    h_.RegisterSpectralWidthContribution(swc);
  }
#else
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    h_.ResetSpectralWidthAccumulators();
  }
  p_[thread_id]->AccumulateVolumeAveragedSpectralWidth();
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
#endif  // VMECPP_USE_CUDA

  if (r_[thread_id]->nsMaxF1 == fc_.ns) {
    // only the thread that computes the free-boundary force can compute
    // delbsq

    // radial location of magnetic axis at zeta = 0
    const GeometricOffset& geometric_offset = h_.GetGeometricOffset();
    double r00 = geometric_offset.r_00;

    // MHD energy (in SI units, i.e., Joules?)
    double energy = h_.mhdEnergy * 4.0 * M_PI * M_PI;

    // volume-averaged beta
    double betaVolAvg = h_.thermalEnergy / h_.magneticEnergy;

    // volume-averaged spectral width <M>
    double volAvgM = h_.VolumeAveragedSpectralWidth();

    // mismatch in |B|^2 at LCFS for free-boundary
    double delbsq = m_[thread_id]->get_delbsq();

    logger_.LogIteration(iter2, fc_.fsqr, fc_.fsqz, fc_.fsql, fc_.fsqr1,
                         fc_.fsqz1, fc_.fsql1, delt0r, r00, energy, betaVolAvg,
                         volAvgM, delbsq);
  }  // thread which has boundary
}

absl::StatusOr<bool> Vmec::UpdateForwardModel(
    VmecCheckpoint checkpoint, int iterations_before_checkpointing,
    int thread_id) {
  bool need_restart = false;

  absl::StatusOr<bool> reached_checkpoint = m_[thread_id]->update(
      *decomposed_x_[thread_id], *physical_x_[thread_id],
      *decomposed_f_[thread_id], *physical_f_[thread_id], need_restart,
      last_preconditioner_update_, last_full_update_nestor_, fc_, iter1_,
      iter2_, checkpoint, iterations_before_checkpointing, verbose_);
  if (!reached_checkpoint.ok()) {
    return reached_checkpoint;
  }

  // triggered at activation of vacuum forces.
  // all threads return the same value for this flag.
  if (need_restart) {
    double delt0 = indata_.delt;
    RestartIteration(delt0, thread_id);

#ifdef _OPENMP
#pragma omp single nowait
#endif  // _OPENMP
    // already done in restart_iter for restart_reason == BAD_JACOBIAN
    fc_.restart_reason = RestartReason::NO_RESTART;
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  return reached_checkpoint;
}

void Vmec::PerformTimeStep(double fac, double b1, double time_step,
                           int thread_id) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  performTimeStep(s_, fc_, *r_[thread_id], fac, b1, time_step,
                  /*m_decomposed_x=*/*decomposed_x_[thread_id],
                  /*m_decomposed_v=*/*decomposed_v_[thread_id],
                  *decomposed_f_[thread_id],
                  /*m_h_=*/h_);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

// velocity_scale == fac
// conjugation_parameter == b1
void Vmec::performTimeStep(const Sizes& s, const FlowControl& fc,
                           const RadialPartitioning& r, double velocity_scale,
                           double conjugation_parameter, double time_step,
                           FourierGeometry& m_decomposed_x,
                           FourierVelocity& m_decomposed_v,
                           const FourierForces& decomposed_f,
                           HandoverStorage& m_h_) const {
  // THIS IS THE TIME-STEP ALGORITHM. IT IS ESSENTIALLY A CONJUGATE
  // GRADIENT METHOD, WITHOUT THE LINE SEARCHES (FLETCHER-REEVES),
  // BASED ON A METHOD GIVEN BY P. GARABEDIAN
#ifdef VMECPP_USE_CUDA
  // CUDA dispatch of the time-step update. The device-resident routine
  // PerformTimeStepCuda reads the decomposed force buffers from device
  // memory (already populated by the iteration body's force-evaluation
  // path) and advances the device-resident velocity and
  // decomposed-position buffers in place; the host m_decomposed_x arrays
  // are refreshed at the controller's explicit flush sites. The host
  // decomposed_f argument is unused on this path because the device
  // routine reads its inputs directly from the device-resident shadow
  // buffers; the explicit (void) cast suppresses an unused-parameter
  // diagnostic.
  (void)decomposed_f;
  // The device time-step controller needs the host's preconditioned
  // residual normalization (fNorm1 for R/Z, fc.deltaS for L) and the ring
  // reset phase (iter2 == iter1 at stage starts and restarts).
  const int iter_phase_for_devstep = (iter2_ == iter1_) ? 0 : 1;
  vmecpp::PerformTimeStepCuda(
      r, s, fc, velocity_scale, conjugation_parameter, time_step, m_h_.fNorm1,
      iter_phase_for_devstep, m_decomposed_v.vrcc.data(),
      m_decomposed_v.vrss.data(), m_decomposed_v.vzsc.data(),
      m_decomposed_v.vzcs.data(), m_decomposed_v.vlsc.data(),
      m_decomposed_v.vlcs.data(), m_decomposed_x.rmncc.data(),
      m_decomposed_x.rmnss.data(), m_decomposed_x.zmnsc.data(),
      m_decomposed_x.zmncs.data(), m_decomposed_x.lmnsc.data(),
      m_decomposed_x.lmncs.data());
  // The multi-rank handover through m_h_ applies only when nsMinF1 > 0
  // or nsMaxF1 < fc.ns; single-rank execution, the only mode under the
  // CUDA build, satisfies neither.
  (void)m_h_;
  return;
#endif  // VMECPP_USE_CUDA

  for (int jF = r.nsMinF; jF < r.nsMaxFIncludingLcfs; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      for (int n = 0; n < s.ntor + 1; ++n) {
        const int idx_mn = ((jF - r.nsMinF) * s.mpol + m) * (s.ntor + 1) + n;
        const int idx_mn1 = ((jF - r.nsMinF1) * s.mpol + m) * (s.ntor + 1) + n;

        // update velocity
        m_decomposed_v.vrcc[idx_mn] =
            velocity_scale *
            (conjugation_parameter * m_decomposed_v.vrcc[idx_mn] +
             time_step * decomposed_f.frcc[idx_mn]);
        m_decomposed_v.vzsc[idx_mn] =
            velocity_scale *
            (conjugation_parameter * m_decomposed_v.vzsc[idx_mn] +
             time_step * decomposed_f.fzsc[idx_mn]);
        m_decomposed_v.vlsc[idx_mn] =
            velocity_scale *
            (conjugation_parameter * m_decomposed_v.vlsc[idx_mn] +
             time_step * decomposed_f.flsc[idx_mn]);
        if (s.lthreed) {
          m_decomposed_v.vrss[idx_mn] =
              velocity_scale *
              (conjugation_parameter * m_decomposed_v.vrss[idx_mn] +
               time_step * decomposed_f.frss[idx_mn]);
          m_decomposed_v.vzcs[idx_mn] =
              velocity_scale *
              (conjugation_parameter * m_decomposed_v.vzcs[idx_mn] +
               time_step * decomposed_f.fzcs[idx_mn]);
          m_decomposed_v.vlcs[idx_mn] =
              velocity_scale *
              (conjugation_parameter * m_decomposed_v.vlcs[idx_mn] +
               time_step * decomposed_f.flcs[idx_mn]);
        }
        if (s.lasym) {
          m_decomposed_v.vrsc[idx_mn] =
              velocity_scale *
              (conjugation_parameter * m_decomposed_v.vrsc[idx_mn] +
               time_step * decomposed_f.frsc[idx_mn]);
          m_decomposed_v.vzcc[idx_mn] =
              velocity_scale *
              (conjugation_parameter * m_decomposed_v.vzcc[idx_mn] +
               time_step * decomposed_f.fzcc[idx_mn]);
          m_decomposed_v.vlcc[idx_mn] =
              velocity_scale *
              (conjugation_parameter * m_decomposed_v.vlcc[idx_mn] +
               time_step * decomposed_f.flcc[idx_mn]);
          if (s.lthreed) {
            m_decomposed_v.vrcs[idx_mn] =
                velocity_scale *
                (conjugation_parameter * m_decomposed_v.vrcs[idx_mn] +
                 time_step * decomposed_f.frcs[idx_mn]);
            m_decomposed_v.vzss[idx_mn] =
                velocity_scale *
                (conjugation_parameter * m_decomposed_v.vzss[idx_mn] +
                 time_step * decomposed_f.fzss[idx_mn]);
            m_decomposed_v.vlss[idx_mn] =
                velocity_scale *
                (conjugation_parameter * m_decomposed_v.vlss[idx_mn] +
                 time_step * decomposed_f.flss[idx_mn]);
          }
        }

        // advance "position" (==Fourier coefficients of geometry) by
        // velocity*timeStep
        m_decomposed_x.rmncc[idx_mn1] +=
            time_step * m_decomposed_v.vrcc[idx_mn];
        m_decomposed_x.zmnsc[idx_mn1] +=
            time_step * m_decomposed_v.vzsc[idx_mn];
        m_decomposed_x.lmnsc[idx_mn1] +=
            time_step * m_decomposed_v.vlsc[idx_mn];
        if (s.lthreed) {
          m_decomposed_x.rmnss[idx_mn1] +=
              time_step * m_decomposed_v.vrss[idx_mn];
          m_decomposed_x.zmncs[idx_mn1] +=
              time_step * m_decomposed_v.vzcs[idx_mn];
          m_decomposed_x.lmncs[idx_mn1] +=
              time_step * m_decomposed_v.vlcs[idx_mn];
        }
        if (s.lasym) {
          m_decomposed_x.rmnsc[idx_mn1] +=
              time_step * m_decomposed_v.vrsc[idx_mn];
          m_decomposed_x.zmncc[idx_mn1] +=
              time_step * m_decomposed_v.vzcc[idx_mn];
          m_decomposed_x.lmncc[idx_mn1] +=
              time_step * m_decomposed_v.vlcc[idx_mn];
          if (s.lthreed) {
            m_decomposed_x.rmncs[idx_mn1] +=
                time_step * m_decomposed_v.vrcs[idx_mn];
            m_decomposed_x.zmnss[idx_mn1] +=
                time_step * m_decomposed_v.vzss[idx_mn];
            m_decomposed_x.lmnss[idx_mn1] +=
                time_step * m_decomposed_v.vlss[idx_mn];
          }
        }
      }  // n
    }  // m
  }  // jF

  // also evolve satellite radial locations: nsMinF1, nsMaxF1-1 in case
  // inside, outside threads exist
  bool hasInside = (r.nsMinF1 > 0);
  bool hasOutside = (r.nsMaxF1 < fc.ns);

  // get Full1-specific elements from neighboring threads
  // Uses RowMatrixXd with (thread, mn) indexing
  if (hasInside) {
    // put innermost grid point into _o storage of previous rank
    const int prev_thread = r.get_thread_id() - 1;
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
      m_h_.rmncc_o(prev_thread, mn) = m_decomposed_x.rmncc[idx_mn];
      m_h_.zmnsc_o(prev_thread, mn) = m_decomposed_x.zmnsc[idx_mn];
      m_h_.lmnsc_o(prev_thread, mn) = m_decomposed_x.lmnsc[idx_mn];
    }

    if (s_.lthreed) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
        m_h_.rmnss_o(prev_thread, mn) = m_decomposed_x.rmnss[idx_mn];
        m_h_.zmncs_o(prev_thread, mn) = m_decomposed_x.zmncs[idx_mn];
        m_h_.lmncs_o(prev_thread, mn) = m_decomposed_x.lmncs[idx_mn];
      }
    }  // lthreed

    if (s_.lasym) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
        m_h_.rmnsc_o(prev_thread, mn) = m_decomposed_x.rmnsc[idx_mn];
        m_h_.zmncc_o(prev_thread, mn) = m_decomposed_x.zmncc[idx_mn];
        m_h_.lmncc_o(prev_thread, mn) = m_decomposed_x.lmncc[idx_mn];
      }

      if (s_.lthreed) {
        for (int mn = 0; mn < s_.mnsize; ++mn) {
          int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
          m_h_.rmncs_o(prev_thread, mn) = m_decomposed_x.rmncs[idx_mn];
          m_h_.zmnss_o(prev_thread, mn) = m_decomposed_x.zmnss[idx_mn];
          m_h_.lmnss_o(prev_thread, mn) = m_decomposed_x.lmnss[idx_mn];
        }
      }
    }  // lasym
  }

  if (hasOutside) {
    // put outermost grid point into _i storage of next rank
    const int next_thread = r.get_thread_id() + 1;
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
      m_h_.rmncc_i(next_thread, mn) = m_decomposed_x.rmncc[idx_mn];
      m_h_.zmnsc_i(next_thread, mn) = m_decomposed_x.zmnsc[idx_mn];
      m_h_.lmnsc_i(next_thread, mn) = m_decomposed_x.lmnsc[idx_mn];
    }

    if (s_.lthreed) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
        m_h_.rmnss_i(next_thread, mn) = m_decomposed_x.rmnss[idx_mn];
        m_h_.zmncs_i(next_thread, mn) = m_decomposed_x.zmncs[idx_mn];
        m_h_.lmncs_i(next_thread, mn) = m_decomposed_x.lmncs[idx_mn];
      }
    }  // lthreed

    if (s_.lasym) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
        m_h_.rmnsc_i(next_thread, mn) = m_decomposed_x.rmnsc[idx_mn];
        m_h_.zmncc_i(next_thread, mn) = m_decomposed_x.zmncc[idx_mn];
        m_h_.lmncc_i(next_thread, mn) = m_decomposed_x.lmncc[idx_mn];
      }

      if (s_.lthreed) {
        for (int mn = 0; mn < s_.mnsize; ++mn) {
          int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
          m_h_.rmncs_i(next_thread, mn) = m_decomposed_x.rmncs[idx_mn];
          m_h_.zmnss_i(next_thread, mn) = m_decomposed_x.zmnss[idx_mn];
          m_h_.lmnss_i(next_thread, mn) = m_decomposed_x.lmnss[idx_mn];
        }
      }
    }  // lasym
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // Now that the crossover data is in the HandoverStorage,
  // put it locally into the correct satellite locations.

  if (hasOutside) {
    // put _o storage filled by thread_id-1 into nsMaxF1-1
    const int this_thread = r.get_thread_id();
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
      m_decomposed_x.rmncc[idx_mn] = m_h_.rmncc_o(this_thread, mn);
      m_decomposed_x.zmnsc[idx_mn] = m_h_.zmnsc_o(this_thread, mn);
      m_decomposed_x.lmnsc[idx_mn] = m_h_.lmnsc_o(this_thread, mn);
    }

    if (s_.lthreed) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
        m_decomposed_x.rmnss[idx_mn] = m_h_.rmnss_o(this_thread, mn);
        m_decomposed_x.zmncs[idx_mn] = m_h_.zmncs_o(this_thread, mn);
        m_decomposed_x.lmncs[idx_mn] = m_h_.lmncs_o(this_thread, mn);
      }
    }  // lthreed

    if (s_.lasym) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
        m_decomposed_x.rmnsc[idx_mn] = m_h_.rmnsc_o(this_thread, mn);
        m_decomposed_x.zmncc[idx_mn] = m_h_.zmncc_o(this_thread, mn);
        m_decomposed_x.lmncc[idx_mn] = m_h_.lmncc_o(this_thread, mn);
      }

      if (s_.lthreed) {
        for (int mn = 0; mn < s_.mnsize; ++mn) {
          int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
          m_decomposed_x.rmncs[idx_mn] = m_h_.rmncs_o(this_thread, mn);
          m_decomposed_x.zmnss[idx_mn] = m_h_.zmnss_o(this_thread, mn);
          m_decomposed_x.lmnss[idx_mn] = m_h_.lmnss_o(this_thread, mn);
        }
      }
    }  // lasym
  }

  if (hasInside) {
    // put _i storage filled by thread_id+1 into nsMinF1
    const int this_thread = r.get_thread_id();
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
      m_decomposed_x.rmncc[idx_mn] = m_h_.rmncc_i(this_thread, mn);
      m_decomposed_x.zmnsc[idx_mn] = m_h_.zmnsc_i(this_thread, mn);
      m_decomposed_x.lmnsc[idx_mn] = m_h_.lmnsc_i(this_thread, mn);
    }

    if (s_.lthreed) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
        m_decomposed_x.rmnss[idx_mn] = m_h_.rmnss_i(this_thread, mn);
        m_decomposed_x.zmncs[idx_mn] = m_h_.zmncs_i(this_thread, mn);
        m_decomposed_x.lmncs[idx_mn] = m_h_.lmncs_i(this_thread, mn);
      }
    }  // lthreed

    if (s_.lasym) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
        m_decomposed_x.rmnsc[idx_mn] = m_h_.rmnsc_i(this_thread, mn);
        m_decomposed_x.zmncc[idx_mn] = m_h_.zmncc_i(this_thread, mn);
        m_decomposed_x.lmncc[idx_mn] = m_h_.lmncc_i(this_thread, mn);
      }

      if (s_.lthreed) {
        for (int mn = 0; mn < s_.mnsize; ++mn) {
          int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
          m_decomposed_x.rmncs[idx_mn] = m_h_.rmncs_i(this_thread, mn);
          m_decomposed_x.zmnss[idx_mn] = m_h_.zmnss_i(this_thread, mn);
          m_decomposed_x.lmnss[idx_mn] = m_h_.lmnss_i(this_thread, mn);
        }
      }
    }  // lasym
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // performTimeStep

void Vmec::InterpolateToNextMultigridStep(
    int ns_new, int ns_old,
    const std::vector<std::unique_ptr<RadialProfiles>>& p,
    const std::vector<std::unique_ptr<RadialPartitioning>>& r_new,
    const std::vector<std::unique_ptr<RadialPartitioning>>& r_old,
    std::vector<std::unique_ptr<FourierGeometry>>& m_x_new,
    std::vector<std::unique_ptr<FourierGeometry>>& m_x_old) {
  // INTERPOLATE R,Z AND LAMBDA ON FULL GRID
  // (EXTRAPOLATE M=1 MODES,OVER SQRT(S), TO ORIGIN)
  // ON ENTRY, XOLD = X(COARSE MESH) * SCALXC(COARSE MESH)
  // ON EXIT,  XNEW = X(NEW MESH)   [ NOT SCALED BY 1/SQRTS ]

  const double hs_old = 1.0 / (ns_old - 1.0);

  const int num_threads_new = static_cast<int>(r_new.size());
  const int num_threads_old = static_cast<int>(r_old.size());

  // ------------------------
  // extrapolation to axis for odd-m modes (?)

  int thread_with_ns_0 = 0;
  int thread_with_ns_1 = 0;
  int thread_with_ns_2 = 0;
  for (int thread_id = 0; thread_id < num_threads_old; ++thread_id) {
    const int nsMinF = r_old[thread_id]->nsMinF;
    const int nsMaxFIncludingLcfs = r_old[thread_id]->nsMaxFIncludingLcfs;
    if (nsMinF <= 0 && 0 < nsMaxFIncludingLcfs) {
      thread_with_ns_0 = thread_id;
    }
    if (nsMinF <= 1 && 1 < nsMaxFIncludingLcfs) {
      thread_with_ns_1 = thread_id;
    }
    if (nsMinF <= 2 && 2 < nsMaxFIncludingLcfs) {
      thread_with_ns_2 = thread_id;
    }
  }  // thread_id

  // only odd m
  for (int m = 1; m < s_.mpol; m += 2) {
    for (int n = 0; n < s_.ntor + 1; ++n) {
      const int idx_fc_0 =
          ((0 - m_x_old[thread_with_ns_0]->nsMin()) * s_.mpol + m) *
              (s_.ntor + 1) +
          n;
      const int idx_fc_1 =
          ((1 - m_x_old[thread_with_ns_1]->nsMin()) * s_.mpol + m) *
              (s_.ntor + 1) +
          n;
      const int idx_fc_2 =
          ((2 - m_x_old[thread_with_ns_2]->nsMin()) * s_.mpol + m) *
              (s_.ntor + 1) +
          n;

      m_x_old[thread_with_ns_0]->rmncc[idx_fc_0] =
          2.0 * m_x_old[thread_with_ns_1]->rmncc[idx_fc_1] -
          m_x_old[thread_with_ns_2]->rmncc[idx_fc_2];
      m_x_old[thread_with_ns_0]->zmnsc[idx_fc_0] =
          2.0 * m_x_old[thread_with_ns_1]->zmnsc[idx_fc_1] -
          m_x_old[thread_with_ns_2]->zmnsc[idx_fc_2];
      m_x_old[thread_with_ns_0]->lmnsc[idx_fc_0] =
          2.0 * m_x_old[thread_with_ns_1]->lmnsc[idx_fc_1] -
          m_x_old[thread_with_ns_2]->lmnsc[idx_fc_2];
      if (s_.lthreed) {
        m_x_old[thread_with_ns_0]->rmnss[idx_fc_0] =
            2.0 * m_x_old[thread_with_ns_1]->rmnss[idx_fc_1] -
            m_x_old[thread_with_ns_2]->rmnss[idx_fc_2];
        m_x_old[thread_with_ns_0]->zmncs[idx_fc_0] =
            2.0 * m_x_old[thread_with_ns_1]->zmncs[idx_fc_1] -
            m_x_old[thread_with_ns_2]->zmncs[idx_fc_2];
        m_x_old[thread_with_ns_0]->lmncs[idx_fc_0] =
            2.0 * m_x_old[thread_with_ns_1]->lmncs[idx_fc_1] -
            m_x_old[thread_with_ns_2]->lmncs[idx_fc_2];
      }
      if (s_.lasym) {
        m_x_old[thread_with_ns_0]->rmnsc[idx_fc_0] =
            2.0 * m_x_old[thread_with_ns_1]->rmnsc[idx_fc_1] -
            m_x_old[thread_with_ns_2]->rmnsc[idx_fc_2];
        m_x_old[thread_with_ns_0]->zmncc[idx_fc_0] =
            2.0 * m_x_old[thread_with_ns_1]->zmncc[idx_fc_1] -
            m_x_old[thread_with_ns_2]->zmncc[idx_fc_2];
        m_x_old[thread_with_ns_0]->lmncc[idx_fc_0] =
            2.0 * m_x_old[thread_with_ns_1]->lmncc[idx_fc_1] -
            m_x_old[thread_with_ns_2]->lmncc[idx_fc_2];
        if (s_.lthreed) {
          m_x_old[thread_with_ns_0]->rmncs[idx_fc_0] =
              2.0 * m_x_old[thread_with_ns_1]->rmncs[idx_fc_1] -
              m_x_old[thread_with_ns_2]->rmncs[idx_fc_2];
          m_x_old[thread_with_ns_0]->zmnss[idx_fc_0] =
              2.0 * m_x_old[thread_with_ns_1]->zmnss[idx_fc_1] -
              m_x_old[thread_with_ns_2]->zmnss[idx_fc_2];
          m_x_old[thread_with_ns_0]->lmnss[idx_fc_0] =
              2.0 * m_x_old[thread_with_ns_1]->lmnss[idx_fc_1] -
              m_x_old[thread_with_ns_2]->lmnss[idx_fc_2];
        }
      }
    }  // n
  }  // m

  // ------------------------
  // radial interpolation from old, coarse state vector to new, finer state
  // vector

  sj.resize(ns_new);
  js1.resize(ns_new);
  js2.resize(ns_new);
  s1.resize(ns_new);
  xint.resize(ns_new);

  for (int thread_id = 0; thread_id < num_threads_new; ++thread_id) {
    for (int jNew = r_new[thread_id]->nsMinF1; jNew < r_new[thread_id]->nsMaxF1;
         ++jNew) {
      sj[jNew] = jNew / (ns_new - 1.0);

      // entries around radial position of jNew on old grid
      js1[jNew] = (jNew * (ns_old - 1)) / (ns_new - 1);
      js2[jNew] = std::min(js1[jNew] + 1, ns_old - 1);

      s1[jNew] = js1[jNew] * hs_old;

      // interpolation weight
      xint[jNew] = (sj[jNew] - s1[jNew]) / hs_old;
      xint[jNew] = std::min(1.0, xint[jNew]);
      xint[jNew] = std::max(0.0, xint[jNew]);

      // now need to figure out source threads, which have js1 and js2
      // and the target thread that has jNew
      int thread_with_js1 = 0;
      int thread_with_js2 = 0;
      for (int old_thread_id = 0; old_thread_id < num_threads_old;
           ++old_thread_id) {
        const int nsMinF1 = r_old[old_thread_id]->nsMinF1;
        const int nsMaxF1 = r_old[old_thread_id]->nsMaxF1;
        if (nsMinF1 <= js1[jNew] && js1[jNew] < nsMaxF1) {
          thread_with_js1 = old_thread_id;
        }
        if (nsMinF1 <= js2[jNew] && js2[jNew] < nsMaxF1) {
          thread_with_js2 = old_thread_id;
        }
      }  // old_thread_id

      // now can actually perform interpolation
      for (int m = 0; m < s_.mpol; ++m) {
        for (int n = 0; n < s_.ntor + 1; ++n) {
          const int m_parity = m % 2;

          const int idx_fc_js1 =
              ((js1[jNew] - m_x_old[thread_with_js1]->nsMin()) * s_.mpol + m) *
                  (s_.ntor + 1) +
              n;
          const int idx_fc_js2 =
              ((js2[jNew] - m_x_old[thread_with_js2]->nsMin()) * s_.mpol + m) *
                  (s_.ntor + 1) +
              n;
          const int idx_fc_jNew =
              ((jNew - m_x_new[thread_id]->nsMin()) * s_.mpol + m) *
                  (s_.ntor + 1) +
              n;

          const double scalxc =
              p[thread_id]
                  ->scalxc[(jNew - r_new[thread_id]->nsMinF1) * 2 + m_parity];

          m_x_new[thread_id]->rmncc[idx_fc_jNew] =
              ((1.0 - xint[jNew]) *
                   m_x_old[thread_with_js1]->rmncc[idx_fc_js1] +
               xint[jNew] * m_x_old[thread_with_js2]->rmncc[idx_fc_js2]) /
              scalxc;
          m_x_new[thread_id]->zmnsc[idx_fc_jNew] =
              ((1.0 - xint[jNew]) *
                   m_x_old[thread_with_js1]->zmnsc[idx_fc_js1] +
               xint[jNew] * m_x_old[thread_with_js2]->zmnsc[idx_fc_js2]) /
              scalxc;
          m_x_new[thread_id]->lmnsc[idx_fc_jNew] =
              ((1.0 - xint[jNew]) *
                   m_x_old[thread_with_js1]->lmnsc[idx_fc_js1] +
               xint[jNew] * m_x_old[thread_with_js2]->lmnsc[idx_fc_js2]) /
              scalxc;
          if (s_.lthreed) {
            m_x_new[thread_id]->rmnss[idx_fc_jNew] =
                ((1.0 - xint[jNew]) *
                     m_x_old[thread_with_js1]->rmnss[idx_fc_js1] +
                 xint[jNew] * m_x_old[thread_with_js2]->rmnss[idx_fc_js2]) /
                scalxc;
            m_x_new[thread_id]->zmncs[idx_fc_jNew] =
                ((1.0 - xint[jNew]) *
                     m_x_old[thread_with_js1]->zmncs[idx_fc_js1] +
                 xint[jNew] * m_x_old[thread_with_js2]->zmncs[idx_fc_js2]) /
                scalxc;
            m_x_new[thread_id]->lmncs[idx_fc_jNew] =
                ((1.0 - xint[jNew]) *
                     m_x_old[thread_with_js1]->lmncs[idx_fc_js1] +
                 xint[jNew] * m_x_old[thread_with_js2]->lmncs[idx_fc_js2]) /
                scalxc;
          }
          if (s_.lasym) {
            m_x_new[thread_id]->rmnsc[idx_fc_jNew] =
                ((1.0 - xint[jNew]) *
                     m_x_old[thread_with_js1]->rmnsc[idx_fc_js1] +
                 xint[jNew] * m_x_old[thread_with_js2]->rmnsc[idx_fc_js2]) /
                scalxc;
            m_x_new[thread_id]->zmncc[idx_fc_jNew] =
                ((1.0 - xint[jNew]) *
                     m_x_old[thread_with_js1]->zmncc[idx_fc_js1] +
                 xint[jNew] * m_x_old[thread_with_js2]->zmncc[idx_fc_js2]) /
                scalxc;
            m_x_new[thread_id]->lmncc[idx_fc_jNew] =
                ((1.0 - xint[jNew]) *
                     m_x_old[thread_with_js1]->lmncc[idx_fc_js1] +
                 xint[jNew] * m_x_old[thread_with_js2]->lmncc[idx_fc_js2]) /
                scalxc;
            if (s_.lthreed) {
              m_x_new[thread_id]->rmncs[idx_fc_jNew] =
                  ((1.0 - xint[jNew]) *
                       m_x_old[thread_with_js1]->rmncs[idx_fc_js1] +
                   xint[jNew] * m_x_old[thread_with_js2]->rmncs[idx_fc_js2]) /
                  scalxc;
              m_x_new[thread_id]->zmnss[idx_fc_jNew] =
                  ((1.0 - xint[jNew]) *
                       m_x_old[thread_with_js1]->zmnss[idx_fc_js1] +
                   xint[jNew] * m_x_old[thread_with_js2]->zmnss[idx_fc_js2]) /
                  scalxc;
              m_x_new[thread_id]->lmnss[idx_fc_jNew] =
                  ((1.0 - xint[jNew]) *
                       m_x_old[thread_with_js1]->lmnss[idx_fc_js1] +
                   xint[jNew] * m_x_old[thread_with_js2]->lmnss[idx_fc_js2]) /
                  scalxc;
            }
          }
        }  // n
      }  // m
    }  // jNew
  }  // thread_id

  // ------------------------
  // Zero M=1 modes at origin

  // Actually, all odd-m modes are zeroed!
  // odd m only
  for (int m = 1; m < s_.mpol; m += 2) {
    for (int n = 0; n < s_.ntor + 1; ++n) {
      const int idx_fc_0 =
          ((0 - m_x_old[thread_with_ns_0]->nsMin()) * s_.mpol + m) *
              (s_.ntor + 1) +
          n;

      m_x_new[thread_with_ns_0]->rmncc[idx_fc_0] = 0.0;
      m_x_new[thread_with_ns_0]->zmnsc[idx_fc_0] = 0.0;
      m_x_new[thread_with_ns_0]->lmnsc[idx_fc_0] = 0.0;
      if (s_.lthreed) {
        m_x_new[thread_with_ns_0]->rmnss[idx_fc_0] = 0.0;
        m_x_new[thread_with_ns_0]->zmncs[idx_fc_0] = 0.0;
        m_x_new[thread_with_ns_0]->lmncs[idx_fc_0] = 0.0;
      }
      if (s_.lasym) {
        m_x_new[thread_with_ns_0]->rmnsc[idx_fc_0] = 0.0;
        m_x_new[thread_with_ns_0]->zmncc[idx_fc_0] = 0.0;
        m_x_new[thread_with_ns_0]->lmncc[idx_fc_0] = 0.0;
        if (s_.lthreed) {
          m_x_new[thread_with_ns_0]->rmncs[idx_fc_0] = 0.0;
          m_x_new[thread_with_ns_0]->zmnss[idx_fc_0] = 0.0;
          m_x_new[thread_with_ns_0]->lmnss[idx_fc_0] = 0.0;
        }
      }
    }  // n
  }  // m
}  // InterpolateToNextMultigridStep

}  // namespace vmecpp
