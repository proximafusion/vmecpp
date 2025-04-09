// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_
#define VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_

#include <vector>

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

  void update(const std::vector<double>& bDotN);

  std::vector<double> gsave;
  std::vector<double> dsave;

  std::vector<double> tanu;
  std::vector<double> tanv;

  std::vector<double> greenp;
  std::vector<double> gstore;

 private:
  const Sizes& s_;
  const TangentialPartitioning& tp_;
  const SurfaceGeometry& sg_;

  void computeConstants();
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_REGULARIZED_INTEGRALS_REGULARIZED_INTEGRALS_H_
