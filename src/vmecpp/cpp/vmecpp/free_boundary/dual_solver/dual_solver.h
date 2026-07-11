// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_DUAL_SOLVER_DUAL_SOLVER_H_
#define VMECPP_FREE_BOUNDARY_DUAL_SOLVER_DUAL_SOLVER_H_

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vmecpp/free_boundary/free_boundary_base/free_boundary_base.h"

namespace vmecpp {

// Diagnostic decorator that runs two free-boundary solvers side by side.
//
// The primary solver drives the VMEC iteration: its outputs land in the
// shared buffers that IdealMhdModel consumes. The shadow solver writes into
// separate buffers that only appear in the dump. After every vacuum update,
// one JSON line per update is appended to the dump file with both |B|^2/2
// fields, both net-current integrals, and the boundary Fourier
// coefficients, so that the evolution of the two solvers can be compared
// across the iteration (see tools/plot_fb_dual_run.py).
//
// This class is constructed by Vmec when the environment variables
// VMECPP_FB_SHADOW (nestor|biest) and VMECPP_FB_DUAL_DUMP (output path,
// JSON lines) are set. It is a pure diagnostic: the shadow result never
// feeds back into the iteration.
class DualSolver : public FreeBoundaryBase {
 public:
  // primary/shadow: the two solvers, already wired to their respective
  // output buffers. primary must be wired to the real shared buffers used
  // by IdealMhdModel; shadow to separate scratch buffers.
  // {primary,shadow}_b_sq_vac: the |B|^2/2 output buffers of the two
  // solvers, for dumping.
  // dump_path: file to append JSON-lines records to (one per vacuum
  // update); opened by the thread that owns the dump (thread 0).
  // shadow_b_sub_{u,v}_vac: shared (across threads) locations receiving the
  // shadow solver's net-current integrals; only used for the dump.
  DualSolver(const Sizes* s, const TangentialPartitioning* tp,
             const MGridProvider* mgrid,
             std::unique_ptr<FreeBoundaryBase> primary,
             std::unique_ptr<FreeBoundaryBase> shadow,
             std::span<const double> primary_b_sq_vac,
             std::span<const double> shadow_b_sq_vac,
             double* shadow_b_sub_u_vac, double* shadow_b_sub_v_vac,
             const std::string& dump_path, std::span<double> bSqVacShare,
             std::span<double> vacuum_b_r_share,
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
  std::unique_ptr<FreeBoundaryBase> primary_;
  std::unique_ptr<FreeBoundaryBase> shadow_;

  std::span<const double> primary_b_sq_vac_;
  std::span<const double> shadow_b_sq_vac_;

  std::string dump_path_;

  // number of update() calls so far (same on every thread)
  int update_counter_ = 0;

  // shared (across threads) targets for the shadow solver's net-current
  // integrals; only used for the dump
  double* shadow_b_sub_u_vac_;
  double* shadow_b_sub_v_vac_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_DUAL_SOLVER_DUAL_SOLVER_H_
