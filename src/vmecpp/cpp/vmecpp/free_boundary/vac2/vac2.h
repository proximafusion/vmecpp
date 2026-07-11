// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_VAC2_VAC2_H_
#define VMECPP_FREE_BOUNDARY_VAC2_VAC2_H_

#include <memory>
#include <span>
#include <vector>

#include "vmecpp/free_boundary/free_boundary_base/free_boundary_base.h"
#include "vmecpp/free_boundary/vac2/vac2_solver.h"

namespace vmecpp {

// Free-boundary solver based on the Vac2 (Strumberger/Tichmann) formulation,
// a C++ port of the Strumberger/Tichmann Fortran vac2 solver.
//
// Vac2 solves the same exterior Neumann problem as NESTOR (scalar magnetic
// potential with B . n = 0 on the LCFS), but with a reformulated kernel
// that avoids the factorial growth of NESTOR's analytic c_mn coefficients:
// its accuracy improves monotonically with resolution (O(h^3) observed
// against BIEST) instead of degrading beyond mpol ~ 12.
//
// Conventions at the interface (following the Fortran vac2 reference):
// - geometry and fields on the full poloidal grid, toroidal-fastest
//   [ku * nv + kv], with u, v in [0, 1) (one field period toroidally);
// - bexn = +(B_coil . n), with the coil field only: the net toroidal
//   plasma current enters through the secular potential term curtor =
//   mu0 * I_tor (the axis-filament contribution in ExternalMagneticField
//   is disabled accordingly);
// - curpol = (2 pi / nfp) <R B_phi> from the toroidal surface tangent.
//
// Threading mirrors Biest: threads deposit their tangential partition of
// the coil field and surface derivatives into shared buffers, the thread
// owning the partition start runs the (internally OpenMP-parallel, here
// effectively serial) Vac2 solve, and all threads read back their
// partition of the outputs.
//
// Partial updates (ivacskip != 0) reuse the cached kernel mode matrices
// and Cholesky factor from the last full solve (like the Fortran
// original's ivac_skip and NESTOR's frozen-LU scheme) and only refresh
// the right-hand side (coil field on the fresh boundary) and the
// reconstruction with the fresh metric. The vacuum pressure keeps
// tracking the boundary every iteration, which is required for
// convergence (see docs/free_boundary_solvers.md).
class Vac2 : public FreeBoundaryBase {
 public:
  // coil_b_{r,p,z}_share, {rub,rvb,zub,zvb}_share: shared scratch, size
  // Sizes::nZnT. bsq_out_share, pot_u_share, pot_v_share: shared scratch,
  // size nThetaEven * nZeta.
  Vac2(const Sizes* s, const TangentialPartitioning* tp,
       const MGridProvider* mgrid, std::span<double> coil_b_r_share,
       std::span<double> coil_b_p_share, std::span<double> coil_b_z_share,
       std::span<double> rub_share, std::span<double> rvb_share,
       std::span<double> zub_share, std::span<double> zvb_share,
       std::span<double> bsq_out_share, std::span<double> pot_u_share,
       std::span<double> pot_v_share, std::span<double> bSqVacShare,
       std::span<double> vacuum_b_r_share, std::span<double> vacuum_b_phi_share,
       std::span<double> vacuum_b_z_share);

  bool update(
      const std::span<const double> rCC, const std::span<const double> rSS,
      const std::span<const double> rSC, const std::span<const double> rCS,
      const std::span<const double> zSC, const std::span<const double> zCS,
      const std::span<const double> zCC, const std::span<const double> zSS,
      int signOfJacobian, const std::span<const double> rAxis,
      const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
      double netToroidalCurrent, int ivacskip,
      const VmecCheckpoint& vmec_checkpoint = VmecCheckpoint::NONE,
      bool at_checkpoint_iteration = false) override;

 private:
  std::span<double> coil_b_r_share_;
  std::span<double> coil_b_p_share_;
  std::span<double> coil_b_z_share_;
  std::span<double> rub_share_;
  std::span<double> rvb_share_;
  std::span<double> zub_share_;
  std::span<double> zvb_share_;
  std::span<double> bsq_out_share_;
  std::span<double> pot_u_share_;
  std::span<double> pot_v_share_;

  // true once a full solve has populated the solver's cached operator
  bool has_full_solution_ = false;

  Vac2Solver solver_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_VAC2_VAC2_H_
