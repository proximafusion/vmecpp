// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_NESTOR_NESTOR_H_
#define VMECPP_FREE_BOUNDARY_NESTOR_NESTOR_H_

#include <Eigen/Dense>
#include <span>

#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.h"
#include "vmecpp/free_boundary/free_boundary_base/free_boundary_base.h"
#include "vmecpp/free_boundary/laplace_solver/laplace_solver.h"
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"
#include "vmecpp/free_boundary/regularized_integrals/regularized_integrals.h"
#include "vmecpp/free_boundary/singular_integrals/singular_integrals.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class Nestor : public FreeBoundaryBase {
 public:
  Nestor(const Sizes* s, const TangentialPartitioning* tp,
         const MGridProvider* mgrid, std::span<real_t> matrixShare,
         std::span<real_t> bvecShare, std::span<real_t> bSqVacShare,
         std::span<int> iPiv, std::span<real_t> vacuum_b_r_share,
         std::span<real_t> vacuum_b_phi_share,
         std::span<real_t> vacuum_b_z_share);

  bool update(
      const std::span<const real_t> rCC, const std::span<const real_t> rSS,
      const std::span<const real_t> rSC, const std::span<const real_t> rCS,
      const std::span<const real_t> zSC, const std::span<const real_t> zCS,
      const std::span<const real_t> zCC, const std::span<const real_t> zSS,
      int signOfJacobian, const std::span<const real_t> rAxis,
      const std::span<const real_t> zAxis, real_t* bSubUVac, real_t* bSubVVac,
      real_t netToroidalCurrent, int ivacskip,
      const VmecCheckpoint& vmec_checkpoint = VmecCheckpoint::NONE,
      bool at_checkpoint_iteration = false) final;

  const SingularIntegrals& GetSingularIntegrals() const;
  const RegularizedIntegrals& GetRegularizedIntegrals() const;
  const LaplaceSolver& GetLaplaceSolver() const;

  // tangential derivatives of scalar magnetic potential
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> potU;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> potV;

  // covariant magnetic field components on surface
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bSubU;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bSubV;

 private:
  // tangential Fourier resolution
  // 0 : ntor
  const int nf;
  // 0 : (mpol + 1)
  const int mf;

  SingularIntegrals si_;
  RegularizedIntegrals ri_;
  LaplaceSolver ls_;

  std::span<real_t> bvecShare;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_NESTOR_NESTOR_H_
