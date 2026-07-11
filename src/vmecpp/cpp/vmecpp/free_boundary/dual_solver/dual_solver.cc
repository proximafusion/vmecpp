// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/dual_solver/dual_solver.h"

#include <fstream>
#include <utility>

#include "absl/log/check.h"
#include "nlohmann/json.hpp"

namespace vmecpp {

DualSolver::DualSolver(
    const Sizes* s, const TangentialPartitioning* tp,
    const MGridProvider* mgrid, std::unique_ptr<FreeBoundaryBase> primary,
    std::unique_ptr<FreeBoundaryBase> shadow,
    std::span<const double> primary_b_sq_vac,
    std::span<const double> shadow_b_sq_vac, double* shadow_b_sub_u_vac,
    double* shadow_b_sub_v_vac, const std::string& dump_path,
    std::span<double> bSqVacShare, std::span<double> vacuum_b_r_share,
    std::span<double> vacuum_b_phi_share, std::span<double> vacuum_b_z_share)
    : FreeBoundaryBase(s, tp, mgrid, bSqVacShare, vacuum_b_r_share,
                       vacuum_b_phi_share, vacuum_b_z_share),
      primary_(std::move(primary)),
      shadow_(std::move(shadow)),
      primary_b_sq_vac_(primary_b_sq_vac),
      shadow_b_sq_vac_(shadow_b_sq_vac),
      dump_path_(dump_path),
      shadow_b_sub_u_vac_(shadow_b_sub_u_vac),
      shadow_b_sub_v_vac_(shadow_b_sub_v_vac) {
  CHECK(primary_ != nullptr);
  CHECK(shadow_ != nullptr);
}

bool DualSolver::update(
    const std::span<const double> rCC, const std::span<const double> rSS,
    const std::span<const double> rSC, const std::span<const double> rCS,
    const std::span<const double> zSC, const std::span<const double> zCS,
    const std::span<const double> zCC, const std::span<const double> zSS,
    int signOfJacobian, const std::span<const double> rAxis,
    const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
    double netToroidalCurrent, int ivacskip,
    const VmecCheckpoint& vmec_checkpoint, bool at_checkpoint_iteration) {
  // The primary result drives the iteration; run it first so a checkpoint
  // stop behaves exactly like a run without the shadow.
  const bool reached_checkpoint =
      primary_->update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian,
                       rAxis, zAxis, bSubUVac, bSubVVac, netToroidalCurrent,
                       ivacskip, vmec_checkpoint, at_checkpoint_iteration);
  if (reached_checkpoint) {
    return true;
  }

  // The shadow solver sees exactly the same inputs; its outputs go to
  // separate buffers and are only dumped.
  shadow_->update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian, rAxis,
                  zAxis, shadow_b_sub_u_vac_, shadow_b_sub_v_vac_,
                  netToroidalCurrent, ivacskip, VmecCheckpoint::NONE,
                  /*at_checkpoint_iteration=*/false);

  update_counter_++;

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // one JSON line per vacuum update
    nlohmann::json record;
    record["update_index"] = update_counter_;
    record["ivacskip"] = ivacskip;
    record["n_theta_eff"] = s_.nThetaEff;
    record["n_zeta"] = s_.nZeta;
    record["net_toroidal_current"] = netToroidalCurrent;
    record["primary_b_sub_u_vac"] = *bSubUVac;
    record["primary_b_sub_v_vac"] = *bSubVVac;
    record["shadow_b_sub_u_vac"] = *shadow_b_sub_u_vac_;
    record["shadow_b_sub_v_vac"] = *shadow_b_sub_v_vac_;
    record["primary_b_sq_vac"] =
        std::vector<double>(primary_b_sq_vac_.begin(), primary_b_sq_vac_.end());
    record["shadow_b_sq_vac"] =
        std::vector<double>(shadow_b_sq_vac_.begin(), shadow_b_sq_vac_.end());
    record["rCC"] = std::vector<double>(rCC.begin(), rCC.end());
    record["rSS"] = std::vector<double>(rSS.begin(), rSS.end());
    record["zSC"] = std::vector<double>(zSC.begin(), zSC.end());
    record["zCS"] = std::vector<double>(zCS.begin(), zCS.end());

    std::ofstream dump_file(dump_path_, std::ios::app);
    dump_file << record << "\n";
  }

  return false;
}  // update

}  // namespace vmecpp
