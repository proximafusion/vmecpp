// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/vmec/vmec.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
#include "vmecpp/vmec/output_quantities/output_quantities.h"

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

  // check for having only a single element in `ns_array`, `ftol_array`, and
  // `niter_array`, since we don't support hot-restarting with multiple
  // multi-grid steps
  if (indata.ns_array.size() != 1ull) {
    return absl::InvalidArgumentError(
        "Only ns array with a single element is supported when hot-restarting");
  }
  if (indata.ftol_array.size() != 1ull) {
    return absl::InvalidArgumentError(
        "Only ftol array with a single element is supported when "
        "hot-restarting");
  }
  if (indata.niter_array.size() != 1ull) {
    return absl::InvalidArgumentError(
        "Only niter array with a single element is supported when "
        "hot-restarting");
  }

  // check for matching `ns`
  if (initial_state.indata.ns_array.back() != indata.ns_array[0]) {
    return absl::InvalidArgumentError(
        absl::StrCat(msg_start, "ns_array", msg_end));
  }

  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<vmecpp::OutputQuantities> vmecpp::run(
    const VmecINDATA& indata, std::optional<HotRestartState> initial_state,
    std::optional<int> max_threads, bool verbose) {
  Vmec v = Vmec(indata, max_threads, verbose);

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
    std::optional<int> max_threads, bool verbose) {
  Vmec v(indata, max_threads, verbose);
  absl::Status mgrid_status = v.LoadMGrid(magnetic_response_table);
  if (!mgrid_status.ok()) {
    return mgrid_status;
  }

  // the values of the first three arguments should just be VMEC's defaults
  absl::StatusOr<bool> s =
      v.run(VmecCheckpoint::NONE, INT_MAX, 500, std::move(initial_state));

  if (!s.ok()) {
    return s.status();
  }

  return std::move(v.output_quantities_);
}

namespace vmecpp {

Vmec::Vmec(const VmecINDATA& indata, std::optional<int> max_threads,
           bool verbose)
    : indata_(indata),
      s_(indata_),
      t_(&s_),
      b_(&s_, &t_, kSignOfJacobian),
      h_(&s_),
      fc_(indata_.lfreeb, indata_.delt,
          static_cast<int>(indata_.ns_array.size()), max_threads),
      verbose_(verbose),
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
    matrixShare.resize(mnpd * mnpd, 0.0);
    iPiv.resize(mnpd, 0);
    bvecShare.resize(mnpd, 0.0);

    // not so nice to do this here, but meh...
    h_.vacuum_magnetic_pressure.resize(s_.nZnT, 0.0);

    h_.vacuum_b_r.resize(s_.nZnT);
    h_.vacuum_b_phi.resize(s_.nZnT);
    h_.vacuum_b_z.resize(s_.nZnT);
  }
}

absl::Status Vmec::LoadMGrid(
    const makegrid::MagneticFieldResponseTable& magnetic_response_table) {
  if (!fc_.lfreeb) {
    return absl::InvalidArgumentError(
        "The MGridProvider is only required for free-boundary VMEC++ runs.");
  }

  return mgrid_.LoadFields(magnetic_response_table.parameters,
                           magnetic_response_table, indata_.extcur);
}

absl::Status Vmec::LoadMGrid() {
  if (!fc_.lfreeb) {
    return absl::InvalidArgumentError(
        "The MGridProvider is only required for free-boundary VMEC++ runs.");
  }

  return mgrid_.LoadFile(indata_.mgrid_file, indata_.extcur);
}

// main worker method; equivalent of vmec.f90
// checked visually to comply with vmec.f90
absl::StatusOr<bool> Vmec::run(const VmecCheckpoint& checkpoint,
                               const int iterations_before_checkpointing,
                               const int maximum_multi_grid_step,
                               std::optional<HotRestartState> initial_state) {
  if (indata_.lfreeb) {
    if (!mgrid_.IsLoaded()) {
      // if we have a free-boundary VMEC run, we may need to load the
      // magnetic field response table
      absl::Status status = LoadMGrid();
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
            "FATAL ERROR in SolveEquilibrium.\nVmec status "
            "code: %s\nVmecINDATA had these contents:\n%s",
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

  // compute output file quantities, but do not write them to output file yet
  // (for creating the output file, use WriteOutputFile())
  output_quantities_ = vmecpp::ComputeOutputQuantities(
      kSignOfJacobian, indata_, s_, fc_, constants_, t_, h_, mgrid_.mgrid_mode,
      r_, decomposed_x_, m_, p_, checkpoint, vacuum_pressure_state_, status_,
      iter2_);

  if (verbose_) {
    std::cout << "\nNUMBER OF JACOBIAN RESETS = " << fc_.ijacob << '\n';
  }

  return false;
}  // run

// initialize_radial quantities, return true if a checkpoint was reached
bool Vmec::InitializeRadial(
    VmecCheckpoint checkpoint, int iterations_before_checkpointing, int nsval,
    int ns_old, double& m_delt0,
    const std::optional<HotRestartState>& initial_state) {
  if (verbose_) {
    std::cout << absl::StrFormat(
        "\n NS = %d   NO. FOURIER MODES = %d   FTOLV = %9.3e   NITER = %d\n",
        nsval, s_.mnmax, fc_.ftolv, fc_.niterv);
  }

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
    num_threads_ = vmec_adjust_num_threads(fc_.max_threads(),
                                           fc_.num_surfaces_to_distribute);

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

      // setup free-boundary solver
      if (fc_.lfreeb) {
        tp_[thread_id] = std::make_unique<TangentialPartitioning>(
            s_.nZnT, num_threads_, thread_id);

        if (indata_.free_boundary_method == FreeBoundaryMethod::NESTOR) {
          fb_[thread_id] = std::make_unique<Nestor>(
              &s_, tp_[thread_id].get(), &mgrid_, matrixShare, bvecShare,
              h_.vacuum_magnetic_pressure, iPiv, h_.vacuum_b_r, h_.vacuum_b_phi,
              h_.vacuum_b_z);
        } else {
          LOG(FATAL) << absl::StrCat("free boundary method '",
                                     ToString(indata_.free_boundary_method),
                                     "' not implemented yet");
        }  // indata_.free_boundary_method
      }  // lfreeb

      // setup MHD model
      m_[thread_id] = std::make_unique<IdealMhdModel>(
          &fc_, &s_, &t_, p_[thread_id].get(), &constants_,
          ls_[thread_id].get(), &h_, r_[thread_id].get(), fb_[thread_id].get(),
          kSignOfJacobian, indata_.nvacskip, &vacuum_pressure_state_);
      m_[thread_id]->setFromINDATA(indata_.ncurr, indata_.gamma, indata_.tcon0);
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
    if (initial_state.has_value()) {
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
  if (verbose_) {
    std::cout << '\n';
    if (fc_.lfreeb) {
      std::cout
          << " ITER |    FSQR     FSQZ     FSQL    |    fsqr     fsqz      "
             "fsql   |   DELT   |  RAX(v=0) |    W_MHD   |   <BETA>   |  "
             "<M>  |  DELBSQ  \n";
      std::cout
          << "------+------------------------------+-----------------------"
             "-------+----------+-----------+------------+------------+----"
             "---+----------\n";
    } else {
      std::cout
          << " ITER |    FSQR     FSQZ     FSQL    |    fsqr     fsqz    "
             "  fsql  "
             " |   DELT   |  RAX(v=0) |    W_MHD   |   <BETA>   |  <M>  \n";
      std::cout
          << "------+------------------------------+---------------------"
             "--------"
             "-+----------+-----------+------------+------------+-------\n";
    }
  }

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
         *s == SolveEqLoopStatus::MUST_RETRY;
         n_local_eqsolve_retries++) {
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

  if (!status_of_all_threads.ok()) {
    return status_of_all_threads;
  }

  if (!any_checkpoint_reached && verbose_) {
    // write MHD energy at end of iterations for current number of surfaces
    std::cout << absl::StrFormat("MHD Energy = %12.6e\n",
                                 h_.mhdEnergy * 4.0 * M_PI * M_PI);
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
        const double scale = fc_.ijacob == 25 ? 0.98 : 0.96;
        fc_.delt0r = scale * indata_.delt;

        if (verbose_) {
          std::cout << absl::StrFormat(
              "HAVING A CONVERGENCE PROBLEM: RESETTING DELT TO %8.3f. "
              " If this does NOT resolve the problem,"
              " try changing (decrease OR increase) the value of DELT\n",
              fc_.delt0r);
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

    if (fc_.restart_reason != RestartReason::NO_RESTART) {
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
        // TODO(jons): why compute spectral width from backup and not current
        // gc (== physical xc) --> <M> includes scalxc ???
        physical_x_backup_[thread_id]->ComputeSpectralWidth(t_, *p_[thread_id]);

        // NOTE: IIRC, this still needs to be called to keep the spectral width
        // updated. Screen output will be controlled by checking the `verbose_`
        // flag inside `Printout`.
        Printout(fc_.delt0r, thread_id);

        if (checkpoint == VmecCheckpoint::PRINTOUT &&
            iter2 >= iterations_before_checkpointing) {
          return SolveEqLoopStatus::CHECKPOINT_REACHED;
        }
      }
    }

#ifdef _OPENMP
#pragma omp single nowait
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

#ifdef _OPENMP
#pragma omp atomic write
#endif  // _OPENMP
    // update iter2_ for all threads, all threads have the same value of iter2,
    // it does not matter who does it.
    // bad resets didn't increment, iter2 in VMEC 8.52, so we need to compute
    // the backwards compatible iteration count
    iter2_ = force_iteration + 1 - bad_resets;
  }  // while m_liter_flag

  return SolveEqLoopStatus::NORMAL_TERMINATION;
}

// aligned visually with restart_iter.f90
void Vmec::RestartIteration(double& m_delt0r, int thread_id) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  if (fc_.restart_reason == RestartReason::BAD_JACOBIAN) {
    // restore previous good state

    // zero velocity
    decomposed_v_[thread_id]->setZero();

    // restore state from backup
    decomposed_x_[thread_id]->copyFrom(*physical_x_backup_[thread_id]);

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

    // restore state from backup
    decomposed_x_[thread_id]->copyFrom(*physical_x_backup_[thread_id]);

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

    // update backup
    physical_x_backup_[thread_id]->copyFrom(*decomposed_x_[thread_id]);
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

  // `funct3d` - COMPUTE MHD FORCES
  absl::StatusOr<bool> reached_checkpoint = UpdateForwardModel(
      checkpoint, iterations_before_checkpointing, thread_id);
  if (!reached_checkpoint.ok() || *reached_checkpoint == true) {
    return reached_checkpoint;
  }

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // COMPUTE ABSOLUTE STOPPING CRITERION
    if (iter2_ == 1 && fc_.restart_reason == RestartReason::BAD_JACOBIAN) {
      // first iteration and Jacobian was not computed correctly
      status_ = VmecStatus::BAD_JACOBIAN;
    } else if (fc_.fsqr <= fc_.ftolv && fc_.fsqz <= fc_.ftolv &&
               fc_.fsql <= fc_.ftolv) {
      // converged to desired tolerance

      m_liter_flag = false;
      status_ = VmecStatus::SUCCESSFUL_TERMINATION;
    }
  }  // #pragma omp single (there is an implicit omp barrier here)

  if (status_ != VmecStatus::NORMAL_TERMINATION || !m_liter_flag) {
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
      absl::c_fill(invTau_, 0.15 / time_step);
    }

    // shift elements for averaging to the left to make space at end for new
    // entry (oldest entry ends up at the end and will be overwritten later)
    absl::c_rotate(invTau_, invTau_.begin() + 1);

    if (iter2_ > iter1_) {
      double invtau_numerator = 0.;
      if (fsq1 != 0.) {
        // fsq is 1 (first iteration) or fsq1 from previous iteration
        // fsq1/fsq is y_n assuming monotonic decrease of energy
        invtau_numerator = std::min(std::abs(std::log(fsq1 / fc_.fsq)), 0.15);
      }

      // overwrite oldest entry (at last index after rotation above) with the
      // new value of 1/tau
      invTau_.back() = invtau_numerator / time_step;
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
  const double otav = absl::c_accumulate(invTau_, 0.) / kNDamp;

  const double dtau = time_step * otav / 2.0;
  const double b1 = 1.0 - dtau;
  const double fac = 1.0 / (1.0 + dtau);

  // THIS IS THE TIME-STEP ALGORITHM. IT IS ESSENTIALLY A CONJUGATE
  // GRADIENT METHOD, WITHOUT THE LINE SEARCHES (FLETCHER-REEVES),
  // BASED ON A METHOD GIVEN BY P. GARABEDIAN
  PerformTimeStep(fac, b1, time_step, thread_id);

  return false;
}

void Vmec::Printout(double delt0r, int thread_id) {
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

  if (verbose_ && r_[thread_id]->nsMaxF1 == fc_.ns) {
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

    if (fc_.lfreeb) {
      std::cout << absl::StrFormat(
          "%5d | %.2e  %.2e  %.2e | %.2e  %.2e  %.2e | %.2e | "
          "%.3e | %.4e | %.4e | %5.3f | %.3e\n",
          iter2_, fc_.fsqr, fc_.fsqz, fc_.fsql, fc_.fsqr1, fc_.fsqz1, fc_.fsql1,
          delt0r, r00, energy, betaVolAvg, volAvgM, delbsq);
    } else {
      // omit DELBSQ column in fixed-boundary case
      std::cout << absl::StrFormat(
          "%5d | %.2e  %.2e  %.2e | %.2e  %.2e  %.2e | %.2e | "
          "%.3e | %.4e | %.4e | %5.3f\n",
          iter2_, fc_.fsqr, fc_.fsqz, fc_.fsql, fc_.fsqr1, fc_.fsqz1, fc_.fsql1,
          delt0r, r00, energy, betaVolAvg, volAvgM);
    }
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
  if (hasInside) {
    // put innermost grid point into _o storage of previous rank
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
      m_h_.rmncc_o[r.get_thread_id() - 1][mn] = m_decomposed_x.rmncc[idx_mn];
      m_h_.zmnsc_o[r.get_thread_id() - 1][mn] = m_decomposed_x.zmnsc[idx_mn];
      m_h_.lmnsc_o[r.get_thread_id() - 1][mn] = m_decomposed_x.lmnsc[idx_mn];
    }

    if (s_.lthreed) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMinF - r.nsMinF1) * s_.mnsize + mn;
        m_h_.rmnss_o[r.get_thread_id() - 1][mn] = m_decomposed_x.rmnss[idx_mn];
        m_h_.zmncs_o[r.get_thread_id() - 1][mn] = m_decomposed_x.zmncs[idx_mn];
        m_h_.lmncs_o[r.get_thread_id() - 1][mn] = m_decomposed_x.lmncs[idx_mn];
      }
    }  // lthreed

    // TODO(jons) : non-stellarator-symmetric terms!
  }

  if (hasOutside) {
    // put outermost grid point into _i storage of next rank
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
      m_h_.rmncc_i[r.get_thread_id() + 1][mn] = m_decomposed_x.rmncc[idx_mn];
      m_h_.zmnsc_i[r.get_thread_id() + 1][mn] = m_decomposed_x.zmnsc[idx_mn];
      m_h_.lmnsc_i[r.get_thread_id() + 1][mn] = m_decomposed_x.lmnsc[idx_mn];
    }

    if (s_.lthreed) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMaxF - 1 - r.nsMinF1) * s_.mnsize + mn;
        m_h_.rmnss_i[r.get_thread_id() + 1][mn] = m_decomposed_x.rmnss[idx_mn];
        m_h_.zmncs_i[r.get_thread_id() + 1][mn] = m_decomposed_x.zmncs[idx_mn];
        m_h_.lmncs_i[r.get_thread_id() + 1][mn] = m_decomposed_x.lmncs[idx_mn];
      }
    }  // lthreed

    // TODO(jons) : non-stellarator-symmetric terms!
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // Now that the crossover data is in the HandoverStorge,
  // put it locally into the correct satellite locations.

  if (hasOutside) {
    // put _o storage filled by thread_id-1 into nsMaxF1-1
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
      m_decomposed_x.rmncc[idx_mn] = m_h_.rmncc_o[r.get_thread_id()][mn];
      m_decomposed_x.zmnsc[idx_mn] = m_h_.zmnsc_o[r.get_thread_id()][mn];
      m_decomposed_x.lmnsc[idx_mn] = m_h_.lmnsc_o[r.get_thread_id()][mn];
    }

    if (s_.lthreed) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMaxF1 - 1 - r.nsMinF1) * s_.mnsize + mn;
        m_decomposed_x.rmnss[idx_mn] = m_h_.rmnss_o[r.get_thread_id()][mn];
        m_decomposed_x.zmncs[idx_mn] = m_h_.zmncs_o[r.get_thread_id()][mn];
        m_decomposed_x.lmncs[idx_mn] = m_h_.lmncs_o[r.get_thread_id()][mn];
      }
    }  // lthreed
  }

  if (hasInside) {
    // put _i storage filled by thread_id-1 into nsMinF1
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
      m_decomposed_x.rmncc[idx_mn] = m_h_.rmncc_i[r.get_thread_id()][mn];
      m_decomposed_x.zmnsc[idx_mn] = m_h_.zmnsc_i[r.get_thread_id()][mn];
      m_decomposed_x.lmnsc[idx_mn] = m_h_.lmnsc_i[r.get_thread_id()][mn];
    }

    if (s_.lthreed) {
      for (int mn = 0; mn < s_.mnsize; ++mn) {
        int idx_mn = (r.nsMinF1 - r.nsMinF1) * s_.mnsize + mn;
        m_decomposed_x.rmnss[idx_mn] = m_h_.rmnss_i[r.get_thread_id()][mn];
        m_decomposed_x.zmncs[idx_mn] = m_h_.zmncs_i[r.get_thread_id()][mn];
        m_decomposed_x.lmncs[idx_mn] = m_h_.lmncs_i[r.get_thread_id()][mn];
      }
    }  // lthreed
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // performTimeStep

void Vmec::InterpolateToNextMultigridStep(
    int ns_new, int ns_old,
    const std::vector<std::unique_ptr<RadialProfiles> >& p,
    const std::vector<std::unique_ptr<RadialPartitioning> >& r_new,
    const std::vector<std::unique_ptr<RadialPartitioning> >& r_old,
    std::vector<std::unique_ptr<FourierGeometry> >& m_x_new,
    std::vector<std::unique_ptr<FourierGeometry> >& m_x_old) {
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
