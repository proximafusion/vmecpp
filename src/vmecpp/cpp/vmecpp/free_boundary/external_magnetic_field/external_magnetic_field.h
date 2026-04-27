// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_EXTERNAL_MAGNETIC_FIELD_EXTERNAL_MAGNETIC_FIELD_H_
#define VMECPP_FREE_BOUNDARY_EXTERNAL_MAGNETIC_FIELD_EXTERNAL_MAGNETIC_FIELD_H_

#include <span>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class ExternalMagneticField {
 public:
  ExternalMagneticField(const Sizes* s, const TangentialPartitioning* tp,
                        const SurfaceGeometry* sg, const MGridProvider* mgrid);

  void update(const std::span<const real_t> rAxis,
              const std::span<const real_t> zAxis, real_t netToroidalCurrent);

  // axis geometry around whole machine
  std::vector<real_t> axisXYZ;
  std::vector<real_t> surfaceXYZ;
  std::vector<real_t> bCoilsXYZ;

  // interpolated magnetic field from mgrid
  std::vector<real_t> interpBr;
  std::vector<real_t> interpBp;
  std::vector<real_t> interpBz;

  // interpolated magnetic field from mgrid
  real_t axis_current;
  std::vector<real_t> curtorBr;
  std::vector<real_t> curtorBp;
  std::vector<real_t> curtorBz;

  // outputs to Nestor
  std::vector<real_t> bSubU;
  std::vector<real_t> bSubV;
  std::vector<real_t> bDotN;

 private:
  // We /can/ use ABSCAB to compute the magnetic field due to the axis current,
  // but since ABSCAB was not optimized for performance (yet),
  // this acutally slows down the overall computation quite a bit.
  // Hence, we offer an option here to use ABSCAB if you really need to,
  // but use the simpler method by default,
  // which is a straightforward implementation of the Hanson-Hirshman 2002
  // paper. Regarding VMEC convergence, this should be not a (relevant) problem,
  // as the axis is usually far away enough from the LCFS.
  // This is evident from a call graph analysis of VMEC++ for a few test cases,
  // where the most part of the ABSCAB calls are spent anyway in the far-field
  // (== standard) cases. NOTE: When changing this, also need to change
  // educational_VMEC branch and re-generate test data; use `master` for
  // ABSCAB-version and `no_abscab_in_belicu` for simple version.
  static constexpr bool kUseAbscabForAxisCurrent = false;

  const Sizes& s_;
  const TangentialPartitioning& tp_;

  const SurfaceGeometry& sg_;
  const MGridProvider& mgrid_;

  void updateInterpolation();

  // Compute the contribution to the external magnetic field from net toroidal
  // current along magnetic axis. Here, the contribution is computed using
  // ABSCAB.
  void AddAxisCurrentFieldAbscab(const std::span<const real_t> rAxis,
                                 const std::span<const real_t> zAxis,
                                 real_t netToroidalCurrent);

  // Compute the contribution to the external magnetic field from net toroidal
  // current along magnetic axis. Here, this contribution is computed using a
  // straightforward implementation of the Hanson-Hirshman 2002 paper.
  void AddAxisCurrentFieldSimple(const std::span<const real_t> rAxis,
                                 const std::span<const real_t> zAxis,
                                 real_t netToroidalCurrent);

  void covariantAndNormalComponents();
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_EXTERNAL_MAGNETIC_FIELD_EXTERNAL_MAGNETIC_FIELD_H_
