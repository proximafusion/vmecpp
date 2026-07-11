// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_BIEST_BIEST_H_
#define VMECPP_FREE_BOUNDARY_BIEST_BIEST_H_

#include <biest.hpp>
#include <span>
#include <vector>

#include "vmecpp/free_boundary/free_boundary_base/free_boundary_base.h"

namespace vmecpp {

// Free-boundary solver based on BIEST (Malhotra et al.,
// https://github.com/dmalhotra/BIEST).
//
// Solves the same exterior Neumann problem as NESTOR: find the vacuum
// magnetic field B_vac = B_coil + B_plasma such that B_vac . n = 0 on the
// LCFS and the poloidal circulation of B_plasma equals the net toroidal
// plasma current. B_plasma is represented by layer potentials,
// grad(S[sigma]) + curl(S[J]), and sigma is obtained from a boundary
// integral equation solved with GMRES using high-order partition-of-unity
// singular quadrature. This replaces NESTOR's low-order singularity
// subtraction and dense LU solve.
//
// The normal component of the coil field is evaluated with BIEST's own
// surface normals (ComputeBdotN), which keeps the normal-orientation
// convention internal to BIEST.
//
// Threading: update() is called by all OpenMP threads, each holding its own
// Biest instance (mirroring Nestor). The threads cooperate through the
// shared buffers passed to the constructor: each thread deposits the coil
// field on its tangential partition, a single thread runs the BIEST solve,
// and all threads then fill their partition of the vacuum-pressure outputs.
//
// Partial updates (ivacskip != 0) mirror NESTOR's strategy (which reuses
// its LU decomposition and only refreshes the right-hand side): the
// singular-quadrature setup and the harmonic net-current field are frozen
// from the last full update, the coil field is re-evaluated on the fresh
// boundary, and the boundary integral equation is re-solved with a GMRES
// warm start from the previous solution. This keeps the vacuum pressure
// tracking the moving boundary every iteration, which is essential for the
// convergence of VMEC's descent (a stale vacuum pressure between full
// updates was observed to stall the iteration).
class Biest : public FreeBoundaryBase {
 public:
  // accuracy_digits: requested number of decimal digits of accuracy of the
  // BIEST singular quadrature and GMRES solve.
  // coil_b_{r,p,z}_share: shared scratch, size Sizes::nZnT.
  // b_plasma_share: shared scratch, size 3 * nZeta * nThetaEven.
  Biest(const Sizes* s, const TangentialPartitioning* tp,
        const MGridProvider* mgrid, int accuracy_digits,
        std::span<double> coil_b_r_share, std::span<double> coil_b_p_share,
        std::span<double> coil_b_z_share, std::span<double> b_plasma_share,
        std::span<double> bSqVacShare, std::span<double> vacuum_b_r_share,
        std::span<double> vacuum_b_phi_share,
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
  int accuracy_digits_;

  // shared across threads; see constructor docs for sizes
  std::span<double> coil_b_r_share_;
  std::span<double> coil_b_p_share_;
  std::span<double> coil_b_z_share_;
  std::span<double> b_plasma_share_;

  // true once a full BIEST solve has populated the shared output buffers
  bool has_full_solution_ = false;

  // boundary Fourier coefficients (rCC) at the last full update, used to
  // promote partial updates to full ones when the boundary has moved too
  // far for the frozen quadrature setup to remain a good approximation
  std::vector<double> r_cc_at_last_full_update_;

  biest::ExtVacuumField<double> vacuum_field_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_BIEST_BIEST_H_
