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
  // The fewest toroidal images over which the Green's function of an
  // axisymmetric (nZeta == 1) vacuum is integrated.
  static constexpr int kAxisymmetricToroidalImages = 64;

  const Sizes& s_;
  const TangentialPartitioning& tp_;
  const SurfaceGeometry& sg_;

  // Toroidal images of the Green's function integral: for nZeta == 1, as many
  // as the boundary needs; the number of field periods otherwise.
  int nvper_;
  // 2 tan(pi p / nvper_): the toroidal-angle factor of the analytic
  // approximation at toroidal image p (axisymmetric case only).
  std::vector<double> tanv_per_;

  void computeConstants();

  // Sets nvper_ to count and the analytic approximation's toroidal-angle factor
  // at each of the count images.
  void SetToroidalImages(int count);

  // Axisymmetric (nZeta == 1) specialization of update(): performs the toroidal
  // integral by summing over nvper_ toroidal images of the evaluation point,
  // since the single-plane surface grid does not resolve the toroidal angle.
  void updateAxisymmetric(const Eigen::VectorXd& bDotN);
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_
