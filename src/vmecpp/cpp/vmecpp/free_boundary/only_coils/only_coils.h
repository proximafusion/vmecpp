// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_ONLY_COILS_ONLY_COILS_H_
#define VMECPP_FREE_BOUNDARY_ONLY_COILS_ONLY_COILS_H_

#include <span>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/free_boundary_base/free_boundary_base.h"
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class OnlyCoils : public FreeBoundaryBase {
 public:
  OnlyCoils(const Sizes* s, const TangentialPartitioning* tp,
            const MGridProvider* mgrid, std::span<double> bSqVacShare,
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
      bool at_checkpoint_iteration = false) final;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_ONLY_COILS_ONLY_COILS_H_
