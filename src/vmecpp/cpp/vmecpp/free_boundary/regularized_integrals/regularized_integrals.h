// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_
#define VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_

#include <Eigen/Dense>

#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class RegularizedIntegrals {
 public:
  RegularizedIntegrals(const Sizes* s, const TangentialPartitioning* tp,
                       const SurfaceGeometry* sg);

  void update(const Eigen::VectorXd& bDotN);

  Eigen::VectorXd gsave;
  Eigen::VectorXd dsave;

  Eigen::VectorXd tanu;
  Eigen::VectorXd tanv;

  Eigen::VectorXd greenp;
  Eigen::VectorXd gstore;

 private:
  // educational_VMEC resolves the toroidal direction of an axisymmetric
  // (nZeta == 1) plasma with this many toroidal images (nvper for the tokamak).
  static constexpr int kAxisymmetricToroidalImages = 64;

  const Sizes& s_;
  const TangentialPartitioning& tp_;
  const SurfaceGeometry& sg_;

  // Number of toroidal images used to perform the toroidal integral of the
  // Green's function: kAxisymmetricToroidalImages for an axisymmetric plasma
  // (nZeta == 1), the number of field periods otherwise.
  int nvper_;
  // 2 tan(pi p / nvper_): the toroidal-angle factor of the analytic
  // approximation at toroidal image p (axisymmetric case only).
  std::vector<double> tanv_per_;

  void computeConstants();

  // Axisymmetric (nZeta == 1) specialization of update(): performs the toroidal
  // integral by summing over nvper_ toroidal images of the evaluation point,
  // since the single-plane surface grid does not resolve the toroidal angle.
  void updateAxisymmetric(const std::vector<double>& bDotN);
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_
