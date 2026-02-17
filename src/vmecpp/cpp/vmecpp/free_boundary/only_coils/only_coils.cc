// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/only_coils/only_coils.h"

namespace vmecpp {

OnlyCoils::OnlyCoils(const Sizes* s, const TangentialPartitioning* tp,
                     const MGridProvider* mgrid, std::span<double> bSqVacShare,
                     std::span<double> vacuum_b_r_share,
                     std::span<double> vacuum_b_phi_share,
                     std::span<double> vacuum_b_z_share)
    : FreeBoundaryBase(s, tp, mgrid, bSqVacShare, vacuum_b_r_share,
                       vacuum_b_phi_share, vacuum_b_z_share) {}  // OnlyCoils

bool OnlyCoils::update(
    const std::span<const double> rCC, const std::span<const double> rSS,
    const std::span<const double> rSC, const std::span<const double> rCS,
    const std::span<const double> zSC, const std::span<const double> zCS,
    const std::span<const double> zCC, const std::span<const double> zSS,
    int signOfJacobian, const std::span<const double> rAxis,
    const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
    double netToroidalCurrent, int ivacskip,
    const VmecCheckpoint& vmec_checkpoint, bool at_checkpoint_iteration) {
  // only need surface geometry, not all derived quantities
  bool full_update = false;

  sg_.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian,
             full_update);

  // blindly assume netToroidalCurrent == 0.0,
  // since checked for that during initialization
  ef_.update(rAxis, zAxis, 0.0);

  // compute net covariant magnetic field components on surface
  double local_bsubuvac = 0.0;
  double local_bsubvvac = 0.0;
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    int l = kl / s_.nZeta;
    local_bsubuvac += ef_.bSubU[kl - tp_.ztMin] * s_.wInt[l];
    local_bsubvvac += ef_.bSubV[kl - tp_.ztMin] * s_.wInt[l];
  }
  local_bsubuvac *= signOfJacobian * 2.0 * M_PI;

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
    *bSubUVac += local_bsubuvac;
    *bSubVVac += local_bsubvvac;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // compute magnetic pressure from only coils: |B|^2/2
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    // cylindrical components of vacuum magnetic field
    vacuum_b_r_share_[kl] = ef_.interpBr[kl - tp_.ztMin];
    vacuum_b_phi_share_[kl] = ef_.interpBp[kl - tp_.ztMin];
    vacuum_b_z_share_[kl] = ef_.interpBz[kl - tp_.ztMin];

    // magnetic pressure from vacuum: |B|^2/2
    bSqVacShare[kl] = 0.5 * (vacuum_b_r_share_[kl] * vacuum_b_r_share_[kl] +
                             vacuum_b_phi_share_[kl] * vacuum_b_phi_share_[kl] +
                             vacuum_b_z_share_[kl] * vacuum_b_z_share_[kl]);
  }  // kl

  // ... done ...

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // TODO(jons): could move bSubUVac, bSubVVac collection here to spare on
  // barrier

  return false;
}  // update

}  // namespace vmecpp
