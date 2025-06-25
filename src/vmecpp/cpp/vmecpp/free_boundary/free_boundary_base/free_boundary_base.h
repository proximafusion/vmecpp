// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_FREE_BOUNDARY_BASE_FREE_BOUNDARY_BASE_H_
#define VMECPP_FREE_BOUNDARY_FREE_BOUNDARY_BASE_FREE_BOUNDARY_BASE_H_

#include <span>
#include <vector>

#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.h"
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class FreeBoundaryBase {
 public:
  virtual ~FreeBoundaryBase() = default;

  FreeBoundaryBase(const Sizes* s, const TangentialPartitioning* tp,
                   const MGridProvider* mgrid, std::span<double> bSqVacShare,
                   std::span<double> vacuum_b_r_share,
                   std::span<double> vacuum_b_phi_share,
                   std::span<double> vacuum_b_z_share)
      : s_(*s),
        fb_(&s_),
        tp_(*tp),
        sg_(s, &fb_, tp),
        ef_(s, tp, &sg_, mgrid),
        bSqVacShare(bSqVacShare),
        vacuum_b_r_share_(vacuum_b_r_share),
        vacuum_b_phi_share_(vacuum_b_phi_share),
        vacuum_b_z_share_(vacuum_b_z_share) {}

  virtual bool update(
      const std::span<const double> rCC, const std::span<const double> rSS,
      const std::span<const double> rSC, const std::span<const double> rCS,
      const std::span<const double> zSC, const std::span<const double> zCS,
      const std::span<const double> zCC, const std::span<const double> zSS,
      int signOfJacobian, const std::span<const double> rAxis,
      const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
      double netToroidalCurrent, int m_ivacskip,
      const VmecCheckpoint& vmec_checkpoint = VmecCheckpoint::NONE,
      bool at_checkpoint_iteration = false) = 0;

  const SurfaceGeometry& GetSurfaceGeometry() const { return sg_; }
  const ExternalMagneticField& GetExternalMagneticField() const { return ef_; }

 protected:
  const Sizes& s_;
  const FourierBasisFastToroidal fb_;
  const TangentialPartitioning& tp_;

  SurfaceGeometry sg_;
  ExternalMagneticField ef_;

  // [nZnT] vacuum magnetic pressure |B_vac^2|/2 at the plasma boundary
  // Points to vacuum_magnetic_pressure in HandoverStorage
  std::span<double> bSqVacShare;

  // [nZnT] cylindrical B^R of Nestor's vacuum magnetic field
  // Points to vacuum_b_r in HandoverStorage
  std::span<double> vacuum_b_r_share_;
  // [nZnT] cylindrical B^phi of Nestor's vacuum magnetic field
  // Points to vacuum_b_phi in HandoverStorage
  std::span<double> vacuum_b_phi_share_;
  // [nZnT] cylindrical B^Z of Nestor's vacuum magnetic field
  // Points to vacuum_b_z in HandoverStorage
  std::span<double> vacuum_b_z_share_;
};  // FreeBoundaryBase

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_FREE_BOUNDARY_BASE_FREE_BOUNDARY_BASE_H_
