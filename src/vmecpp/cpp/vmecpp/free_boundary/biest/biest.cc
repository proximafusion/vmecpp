// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/biest/biest.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "absl/log/check.h"

namespace vmecpp {

Biest::Biest(const Sizes* s, const TangentialPartitioning* tp,
             const MGridProvider* mgrid, int accuracy_digits,
             std::span<double> coil_b_r_share, std::span<double> coil_b_p_share,
             std::span<double> coil_b_z_share, std::span<double> b_plasma_share,
             std::span<double> bSqVacShare, std::span<double> vacuum_b_r_share,
             std::span<double> vacuum_b_phi_share,
             std::span<double> vacuum_b_z_share)
    : FreeBoundaryBase(s, tp, mgrid, bSqVacShare, vacuum_b_r_share,
                       vacuum_b_phi_share, vacuum_b_z_share),
      accuracy_digits_(accuracy_digits),
      coil_b_r_share_(coil_b_r_share),
      coil_b_p_share_(coil_b_p_share),
      coil_b_z_share_(coil_b_z_share),
      b_plasma_share_(b_plasma_share) {
  CHECK_GE(accuracy_digits_, 1) << "BIEST accuracy_digits must be >= 1";
  CHECK_LE(accuracy_digits_, 14) << "BIEST accuracy_digits must be <= 14";
  // BIEST needs an actual 3D toroidal discretization; axisymmetric runs
  // (nzeta == 1) must use NESTOR.
  CHECK_GE(s_.nZeta, 4)
      << "BIEST requires nzeta >= 4; use free_boundary_method = NESTOR "
         "for axisymmetric configurations";
  CHECK_EQ(static_cast<int>(coil_b_r_share_.size()), s_.nZnT);
  CHECK_EQ(static_cast<int>(coil_b_p_share_.size()), s_.nZnT);
  CHECK_EQ(static_cast<int>(coil_b_z_share_.size()), s_.nZnT);
  CHECK_EQ(static_cast<int>(b_plasma_share_.size()),
           3 * s_.nZeta * s_.nThetaEven);
}

bool Biest::update(
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

  // Partial updates (ivacskip != 0) mirror NESTOR's strategy of freezing the
  // expensive geometry-bound operator while tracking the moving boundary:
  // the BIEST singular-quadrature setup (and the harmonic net-current field
  // J0) are kept from the last full update, while the coil field is
  // re-evaluated on the fresh boundary and the BIE is re-solved with a
  // GMRES warm start from the previous sigma. Without this per-iteration
  // tracking, the boundary force acts on a stale vacuum pressure and the
  // VMEC descent stagnates.
  //
  // A partial update is promoted to a full one when the boundary has moved
  // too far since the last full update: the frozen quadrature and normals
  // are only a good approximation for small boundary displacements.
  bool full_update = (ivacskip == 0) || !has_full_solution_;
  if (!full_update) {
    // relative boundary drift since the last full update, scaled by the
    // major-radius coefficient
    double max_drift = 0.0;
    for (size_t mn = 0; mn < r_cc_at_last_full_update_.size(); ++mn) {
      max_drift = std::max(max_drift,
                           std::abs(rCC[mn] - r_cc_at_last_full_update_[mn]));
    }
    // Default: no partial updates (threshold 0 -> every update solves on
    // the fresh boundary). Frozen-setup partial updates were observed to
    // slow VMEC's convergence even at tight thresholds (1e-4: restarts and
    // ~2x iterations; 1e-5: ~1.7x iterations on the CTH-like case), while
    // the GMRES warm start already makes fresh solves cheap near
    // convergence. The experimental threshold can be set via the
    // VMECPP_BIEST_PARTIAL_DRIFT_TOL environment variable (relative to the
    // major-radius coefficient).
    static const double kMaxRelativeDriftForPartialUpdate = []() {
      const char* env = std::getenv("VMECPP_BIEST_PARTIAL_DRIFT_TOL");
      return env != nullptr ? std::atof(env) : 0.0;
    }();
    if (max_drift >= kMaxRelativeDriftForPartialUpdate * std::abs(rCC[0])) {
      full_update = true;
    }
  }

  sg_.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian,
             /*fullUpdate=*/true);

  // Coil field only: the net toroidal plasma current is handed to BIEST as
  // the poloidal circulation of B_plasma below, so the axis-filament
  // contribution in ExternalMagneticField is disabled.
  ef_.update(rAxis, zAxis, /*netToroidalCurrent=*/0.0);

  const int nZeta = s_.nZeta;
  const int nThetaEven = s_.nThetaEven;

  // Deposit this thread's tangential partition of the cylindrical coil field
  // into the shared full-surface buffers (VMEC layout: kl = l * nZeta + k).
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    coil_b_r_share_[kl] = ef_.interpBr[kl - tp_.ztMin];
    coil_b_p_share_[kl] = ef_.interpBp[kl - tp_.ztMin];
    coil_b_z_share_[kl] = ef_.interpBz[kl - tp_.ztMin];
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // The thread owning the start of the tangential partition performs the
  // BIEST solve; results are shared through b_plasma_share_. This must be
  // the same thread on every update (NOT an omp single, which any thread
  // may execute) because the frozen quadrature setup and the GMRES warm
  // start live in this instance's vacuum_field_ across updates.
  if (tp_.ztMin == 0) {
    const int num_surf = nZeta * nThetaEven;

    // Expand the coil field from the reduced poloidal range [0, pi] to the
    // full range using stellarator symmetry:
    //   (B_R, B_phi, B_Z)(R, -phi, -Z) = (-B_R, B_phi, B_Z)(R, phi, Z).
    std::vector<double> full_br(num_surf);
    std::vector<double> full_bp(num_surf);
    std::vector<double> full_bz(num_surf);
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      const int l = kl / nZeta;
      const int k = kl % nZeta;
      const int idx_full = l * nZeta + k;
      full_br[idx_full] = coil_b_r_share_[kl];
      full_bp[idx_full] = coil_b_p_share_[kl];
      full_bz[idx_full] = coil_b_z_share_[kl];
    }
    if (!s_.lasym) {
      for (int l = 1; l < s_.nThetaReduced - 1; ++l) {
        const int lRev = nThetaEven - l;
        for (int k = 0; k < nZeta; ++k) {
          const int kRev = (nZeta - k) % nZeta;
          const int idx_src = l * nZeta + k;
          const int idx_dst = lRev * nZeta + kRev;
          full_br[idx_dst] = -coil_b_r_share_[idx_src];
          full_bp[idx_dst] = coil_b_p_share_[idx_src];
          full_bz[idx_dst] = coil_b_z_share_[idx_src];
        }  // k
      }  // l
    }

    // Surface coordinates and Cartesian coil field in BIEST layout:
    // index (d * Nt + t) * Np + p with t = zeta index, p = theta index,
    // covering one field period.
    const int Nt = nZeta;
    const int Np = nThetaEven;
    std::vector<double> X(3 * num_surf);
    std::vector<double> B_coil(3 * num_surf);
    for (int k = 0; k < nZeta; ++k) {
      const double cos_phi = sg_.cos_phi[k];
      const double sin_phi = sg_.sin_phi[k];
      for (int l = 0; l < nThetaEven; ++l) {
        const int idx_vmec = l * nZeta + k;
        const int idx_biest = k * Np + l;

        X[0 * num_surf + idx_biest] = sg_.rcosuv[idx_vmec];
        X[1 * num_surf + idx_biest] = sg_.rsinuv[idx_vmec];
        X[2 * num_surf + idx_biest] = sg_.z1b[idx_vmec];

        const double br = full_br[idx_vmec];
        const double bp = full_bp[idx_vmec];
        B_coil[0 * num_surf + idx_biest] = br * cos_phi - bp * sin_phi;
        B_coil[1 * num_surf + idx_biest] = br * sin_phi + bp * cos_phi;
        B_coil[2 * num_surf + idx_biest] = full_bz[idx_vmec];
      }  // l
    }  // k

    const auto t0 = std::chrono::steady_clock::now();
    if (full_update) {
      // rebuilds the singular quadrature and the harmonic net-current field
      // for the new boundary (expensive)
      vacuum_field_.Setup(accuracy_digits_, s_.nfp, Nt, Np, X, Nt, Np);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // B_coil . n with BIEST's surface normal (from the last full update on
    // partial updates, mirroring NESTOR's frozen kernel matrix)
    const std::vector<double> b_coil_dot_n = vacuum_field_.ComputeBdotN(B_coil);

    // Net toroidal plasma current as poloidal circulation of B_plasma;
    // the sign convention relative to VMEC's theta orientation carries the
    // sign of the Jacobian (the poloidal circulation direction of BIEST's
    // harmonic surface current follows the handedness of the surface
    // parameterization, which for VMEC's theta carries signgs).
    const double j_plasma = signOfJacobian * MU_0 * netToroidalCurrent;

    std::vector<double> b_plasma, sigma, j_surf;
    std::tie(b_plasma, sigma, j_surf) =
        vacuum_field_.ComputeBplasma(b_coil_dot_n, j_plasma);
    const auto t2 = std::chrono::steady_clock::now();

    static const bool print_timing =
        std::getenv("VMECPP_BIEST_TIMING") != nullptr;
    if (print_timing) {
      const auto ms = [](auto dt) {
        return std::chrono::duration<double, std::milli>(dt).count();
      };
      std::cout << "[biest] " << (full_update ? "full" : "partial")
                << " update: setup " << ms(t1 - t0) << " ms, solve "
                << ms(t2 - t1) << " ms\n";
    }

    for (int i = 0; i < 3 * num_surf; ++i) {
      b_plasma_share_[i] = b_plasma[i];
    }
  }  // solver thread (tp_.ztMin == 0)
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // Each thread fills its tangential partition of the outputs.
  const int num_surf = nZeta * nThetaEven;
  double local_bSubUVac = 0.0;
  double local_bSubVVac = 0.0;
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    const int l = kl / nZeta;
    const int k = kl % nZeta;
    const int idx_biest = k * nThetaEven + l;

    const double plasma_bx = b_plasma_share_[0 * num_surf + idx_biest];
    const double plasma_by = b_plasma_share_[1 * num_surf + idx_biest];
    const double plasma_bz = b_plasma_share_[2 * num_surf + idx_biest];

    const double cos_phi = sg_.cos_phi[k];
    const double sin_phi = sg_.sin_phi[k];
    const double plasma_br = cos_phi * plasma_bx + sin_phi * plasma_by;
    const double plasma_bp = cos_phi * plasma_by - sin_phi * plasma_bx;

    const double full_br = coil_b_r_share_[kl] + plasma_br;
    const double full_bp = coil_b_p_share_[kl] + plasma_bp;
    const double full_bz = coil_b_z_share_[kl] + plasma_bz;

    // covariant components of the total vacuum field
    const double bSubU =
        full_br * sg_.rub[kl - tp_.ztMin] + full_bz * sg_.zub[kl - tp_.ztMin];
    const double bSubV = full_br * sg_.rvb[kl - tp_.ztMin] +
                         full_bz * sg_.zvb[kl - tp_.ztMin] +
                         full_bp * sg_.r1b[kl];

    local_bSubUVac += bSubU * s_.wInt[l];
    local_bSubVVac += bSubV * s_.wInt[l];

    // magnetic pressure from vacuum: |B|^2 / 2
    bSqVacShare[kl] =
        0.5 * (full_br * full_br + full_bp * full_bp + full_bz * full_bz);

    // cylindrical components of the vacuum magnetic field
    vacuum_b_r_share_[kl] = full_br;
    vacuum_b_phi_share_[kl] = full_bp;
    vacuum_b_z_share_[kl] = full_bz;
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

  if (full_update) {
    r_cc_at_last_full_update_.assign(rCC.begin(), rCC.end());
  }
  has_full_solution_ = true;

  if (vmec_checkpoint == VmecCheckpoint::VAC1_BSQVAC &&
      at_checkpoint_iteration) {
    return true;
  }

  return false;
}  // update

}  // namespace vmecpp
