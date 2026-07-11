// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/vac2/vac2.h"

#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace vmecpp {

Vac2::Vac2(const Sizes* s, const TangentialPartitioning* tp,
           const MGridProvider* mgrid, std::span<double> coil_b_r_share,
           std::span<double> coil_b_p_share, std::span<double> coil_b_z_share,
           std::span<double> rub_share, std::span<double> rvb_share,
           std::span<double> zub_share, std::span<double> zvb_share,
           std::span<double> bsq_out_share, std::span<double> pot_u_share,
           std::span<double> pot_v_share, std::span<double> bSqVacShare,
           std::span<double> vacuum_b_r_share,
           std::span<double> vacuum_b_phi_share,
           std::span<double> vacuum_b_z_share)
    : FreeBoundaryBase(s, tp, mgrid, bSqVacShare, vacuum_b_r_share,
                       vacuum_b_phi_share, vacuum_b_z_share),
      coil_b_r_share_(coil_b_r_share),
      coil_b_p_share_(coil_b_p_share),
      coil_b_z_share_(coil_b_z_share),
      rub_share_(rub_share),
      rvb_share_(rvb_share),
      zub_share_(zub_share),
      zvb_share_(zvb_share),
      bsq_out_share_(bsq_out_share),
      pot_u_share_(pot_u_share),
      pot_v_share_(pot_v_share),
      // potential mode cutoffs matching NESTOR: poloidal modes up to
      // (mpol + 1) inclusive -> Vac2Solver cutoff mpol + 2 (exclusive);
      // toroidal modes -ntor..ntor
      solver_(s_.mpol + 2, s_.ntor, s_.nThetaEven, s_.nZeta, s_.nfp) {
  CHECK(!s_.lasym) << "Vac2 free-boundary solver does not support "
                      "lasym = true yet";
  CHECK_EQ(static_cast<int>(coil_b_r_share_.size()), s_.nZnT);
  CHECK_EQ(static_cast<int>(rub_share_.size()), s_.nZnT);
  const int nuv = s_.nThetaEven * s_.nZeta;
  CHECK_EQ(static_cast<int>(bsq_out_share_.size()), nuv);
  CHECK_EQ(static_cast<int>(pot_u_share_.size()), nuv);
  CHECK_EQ(static_cast<int>(pot_v_share_.size()), nuv);
}

bool Vac2::update(
    const std::span<const double> rCC, const std::span<const double> rSS,
    const std::span<const double> rSC, const std::span<const double> rCS,
    const std::span<const double> zSC, const std::span<const double> zCS,
    const std::span<const double> zCC, const std::span<const double> zSS,
    int signOfJacobian, const std::span<const double> rAxis,
    const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
    double netToroidalCurrent, int ivacskip,
    const VmecCheckpoint& vmec_checkpoint, bool at_checkpoint_iteration) {
  if (vmec_checkpoint == VmecCheckpoint::VAC1_VACUUM &&
      at_checkpoint_iteration) {
    return true;
  }

  sg_.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian,
             /*fullUpdate=*/true);

  // Coil field only: the net toroidal plasma current enters through the
  // secular potential term (curtor) below, so the axis-filament
  // contribution in ExternalMagneticField is disabled.
  ef_.update(rAxis, zAxis, /*netToroidalCurrent=*/0.0);

  const int nZeta = s_.nZeta;
  const int nThetaEven = s_.nThetaEven;
  const int nuv = nThetaEven * nZeta;

  // Deposit this thread's tangential partition into the shared full-surface
  // buffers (VMEC layout: kl = l * nZeta + k).
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    coil_b_r_share_[kl] = ef_.interpBr[kl - tp_.ztMin];
    coil_b_p_share_[kl] = ef_.interpBp[kl - tp_.ztMin];
    coil_b_z_share_[kl] = ef_.interpBz[kl - tp_.ztMin];
    rub_share_[kl] = sg_.rub[kl - tp_.ztMin];
    rvb_share_[kl] = sg_.rvb[kl - tp_.ztMin];
    zub_share_[kl] = sg_.zub[kl - tp_.ztMin];
    zvb_share_[kl] = sg_.zvb[kl - tp_.ztMin];
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // The thread owning the start of the tangential partition runs the Vac2
  // solve (same thread every update; the solver instance holds trig
  // tables).
  if (tp_.ztMin == 0) {
    // Expand from the reduced poloidal range [0, pi] to the full range
    // using stellarator symmetry; parities under
    // (theta, phi) -> (-theta, -phi):
    //   B_R odd, B_phi even, B_Z even,
    //   dR/dtheta odd, dR/dphi odd, dZ/dtheta even, dZ/dphi even.
    std::vector<double> br(nuv), bp(nuv), bz(nuv);
    std::vector<double> rub(nuv), rvb(nuv), zub(nuv), zvb(nuv);
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      br[kl] = coil_b_r_share_[kl];
      bp[kl] = coil_b_p_share_[kl];
      bz[kl] = coil_b_z_share_[kl];
      rub[kl] = rub_share_[kl];
      rvb[kl] = rvb_share_[kl];
      zub[kl] = zub_share_[kl];
      zvb[kl] = zvb_share_[kl];
    }
    for (int l = 1; l < s_.nThetaReduced - 1; ++l) {
      const int lRev = nThetaEven - l;
      for (int k = 0; k < nZeta; ++k) {
        const int kRev = (nZeta - k) % nZeta;
        const int src = l * nZeta + k;
        const int dst = lRev * nZeta + kRev;
        br[dst] = -br[src];
        bp[dst] = bp[src];
        bz[dst] = bz[src];
        rub[dst] = -rub[src];
        rvb[dst] = -rvb[src];
        zub[dst] = zub[src];
        zvb[dst] = zvb[src];
      }  // k
    }  // l

    // Assemble the Vac2 input on the full grid. VMEC's [l * nZeta + k]
    // (theta slow, zeta fast) matches Vac2's [ku * nv + kv] directly.
    // Vac2 uses u, v in [0, 1) with v covering one field period:
    //   d/du = 2 pi d/dtheta,  d/dv = (2 pi / nfp) d/dphi.
    Vac2Solver::Input in;
    in.x.resize(nuv);
    in.y.resize(nuv);
    in.z.resize(nuv);
    in.xu.resize(nuv);
    in.yu.resize(nuv);
    in.zu.resize(nuv);
    in.xv.resize(nuv);
    in.yv.resize(nuv);
    in.zv.resize(nuv);
    in.guu.resize(nuv);
    in.guv.resize(nuv);
    in.gvv.resize(nuv);
    in.snx.resize(nuv);
    in.sny.resize(nuv);
    in.snz.resize(nuv);
    in.bexn.resize(nuv);

    const double two_pi = 2.0 * M_PI;
    const double dv_scale = two_pi / s_.nfp;
    double curpol = 0.0;
    for (int l = 0; l < nThetaEven; ++l) {
      for (int k = 0; k < nZeta; ++k) {
        const int idx = l * nZeta + k;
        const double cos_phi = sg_.cos_phi[k];
        const double sin_phi = sg_.sin_phi[k];

        in.x[idx] = sg_.rcosuv[idx];
        in.y[idx] = sg_.rsinuv[idx];
        in.z[idx] = sg_.z1b[idx];

        in.xu[idx] = two_pi * rub[idx] * cos_phi;
        in.yu[idx] = two_pi * rub[idx] * sin_phi;
        in.zu[idx] = two_pi * zub[idx];

        in.xv[idx] = dv_scale * (rvb[idx] * cos_phi - sg_.r1b[idx] * sin_phi);
        in.yv[idx] = dv_scale * (rvb[idx] * sin_phi + sg_.r1b[idx] * cos_phi);
        in.zv[idx] = dv_scale * zvb[idx];

        in.guu[idx] = in.xu[idx] * in.xu[idx] + in.yu[idx] * in.yu[idx] +
                      in.zu[idx] * in.zu[idx];
        in.guv[idx] = in.xu[idx] * in.xv[idx] + in.yu[idx] * in.yv[idx] +
                      in.zu[idx] * in.zv[idx];
        in.gvv[idx] = in.xv[idx] * in.xv[idx] + in.yv[idx] * in.yv[idx] +
                      in.zv[idx] * in.zv[idx];

        // surface-area-weighted normal signgs * (xu x xv), pointing outward
        // (NOT normalized -- Vac2 expects the area weighting, matching the
        // Fortran reference where sn* = Xu x Xv without normalization)
        in.snx[idx] = signOfJacobian *
                      (in.yu[idx] * in.zv[idx] - in.zu[idx] * in.yv[idx]);
        in.sny[idx] = signOfJacobian *
                      (in.zu[idx] * in.xv[idx] - in.xu[idx] * in.zv[idx]);
        in.snz[idx] = signOfJacobian *
                      (in.xu[idx] * in.yv[idx] - in.yu[idx] * in.xv[idx]);

        // Cartesian coil field
        const double bx = br[idx] * cos_phi - bp[idx] * sin_phi;
        const double by = br[idx] * sin_phi + bp[idx] * cos_phi;

        // bexn = +(B_coil . n), Vac2's sign convention
        in.bexn[idx] =
            bx * in.snx[idx] + by * in.sny[idx] + bz[idx] * in.snz[idx];

        // curpol = <xv . B> = (2 pi / nfp) <R B_phi>
        curpol += in.xv[idx] * bx + in.yv[idx] * by + in.zv[idx] * bz[idx];
      }  // k
    }  // l
    curpol /= static_cast<double>(nuv);

    in.curpol = curpol;
    in.curtor = MU_0 * netToroidalCurrent;
    in.lasym = false;

    // NESTOR-style partial updates (frozen operator + fresh RHS) are
    // implemented but opt-in: they preserve the converged answer yet slow
    // the descent (~2.6x iterations on the CTH-like case), so the default
    // is a full solve every update. Enable for experiments with
    // VMECPP_VAC2_PARTIAL_UPDATES=1.
    static const bool partial_updates_enabled = []() {
      const char* env = std::getenv("VMECPP_VAC2_PARTIAL_UPDATES");
      return env != nullptr && env[0] == '1';
    }();
    const bool reuse_operator =
        partial_updates_enabled && (ivacskip != 0) && has_full_solution_;

    // The solve runs on this single thread inside VMEC's parallel region,
    // where nested parallelism is disabled by default -- which would leave
    // the kernel assembly serial while the other threads wait at the
    // barrier below. Temporarily enable one extra level of nesting so the
    // solver's internal OpenMP regions can use the idle cores.
#ifdef _OPENMP
    const int prev_max_levels = omp_get_max_active_levels();
    const int prev_num_threads = omp_get_max_threads();
    omp_set_max_active_levels(2);
    // Bounded inner team: spawning a full-width team for a millisecond-
    // scale kernel costs more in thread management and barrier-spinner
    // contention than it gains. Override via VMECPP_VAC2_SOLVE_THREADS.
    static const int solve_threads = []() {
      const char* env = std::getenv("VMECPP_VAC2_SOLVE_THREADS");
      if (env != nullptr) return std::atoi(env);
      return std::min(8, omp_get_num_procs());
    }();
    omp_set_num_threads(solve_threads);
#endif  // _OPENMP
    auto maybe_out = solver_.Solve(in, reuse_operator);
#ifdef _OPENMP
    omp_set_num_threads(prev_num_threads);
    omp_set_max_active_levels(prev_max_levels);
#endif  // _OPENMP
    CHECK_OK(maybe_out.status()) << "Vac2 solve failed";
    const auto& out = *maybe_out.value();

    for (int i = 0; i < nuv; ++i) {
      bsq_out_share_[i] = out.bsqvac[i];
      pot_u_share_[i] = out.potU[i];
      pot_v_share_[i] = out.potV[i];
    }
  }  // solver thread (tp_.ztMin == 0)
  has_full_solution_ = true;
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // Each thread fills its tangential partition of the outputs. Convert
  // potU = dPhi/du, potV = dPhi/dv back to VMEC's covariant components
  // (per-radian in theta and phi).
  double local_bSubUVac = 0.0;
  double local_bSubVVac = 0.0;
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    const int l = kl / nZeta;

    const double bSubU = pot_u_share_[kl] / (2.0 * M_PI);
    const double bSubV = pot_v_share_[kl] * s_.nfp / (2.0 * M_PI);

    local_bSubUVac += bSubU * s_.wInt[l];
    local_bSubVVac += bSubV * s_.wInt[l];

    bSqVacShare[kl] = bsq_out_share_[kl];

    // cylindrical components of the vacuum magnetic field, from the
    // covariant components and the surface metric (as in Nestor)
    const double guu = sg_.guu[kl - tp_.ztMin];
    const double guv = sg_.guv[kl - tp_.ztMin] * s_.nfp * 0.5;
    const double gvv = sg_.gvv[kl - tp_.ztMin] * s_.nfp * s_.nfp;
    const double det = guu * gvv - guv * guv;
    const double bSupU = (gvv * bSubU - guv * bSubV) / det;
    const double bSupV = (-guv * bSubU + guu * bSubV) / det;
    vacuum_b_r_share_[kl] =
        sg_.rub[kl - tp_.ztMin] * bSupU + sg_.rvb[kl - tp_.ztMin] * bSupV;
    vacuum_b_phi_share_[kl] = sg_.r1b[kl] * bSupV;
    vacuum_b_z_share_[kl] =
        sg_.zub[kl - tp_.ztMin] * bSupU + sg_.zvb[kl - tp_.ztMin] * bSupV;
  }  // kl
  local_bSubUVac *= signOfJacobian * 2.0 * M_PI;

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    *bSubUVac = 0.0;
    *bSubVVac = 0.0;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    *bSubUVac += local_bSubUVac;
    *bSubVVac += local_bSubVVac;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  if (vmec_checkpoint == VmecCheckpoint::VAC1_BSQVAC &&
      at_checkpoint_iteration) {
    return true;
  }

  return false;
}  // update

}  // namespace vmecpp
