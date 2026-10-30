// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_EXTERNAL_MAGNETIC_FIELD_EXTERNAL_MAGNETIC_FIELD_H_
#define VMECPP_FREE_BOUNDARY_EXTERNAL_MAGNETIC_FIELD_EXTERNAL_MAGNETIC_FIELD_H_

#include <Eigen/Dense>
#include <functional>
#include <span>

#include "absl/status/status.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

// Evaluates the cylindrical components (B^R, B^phi, B^Z) of the external
// magnetic field at the points (r[i], phi[i], z[i]), with phi the geometric
// toroidal angle. The outputs come sized to the number of points. Called
// concurrently by the vacuum threads, each on its own slice of the surface.
using MagneticFieldFunction = std::function<absl::Status(
    const Eigen::VectorXd& r, const Eigen::VectorXd& phi,
    const Eigen::VectorXd& z, Eigen::VectorXd& m_b_r, Eigen::VectorXd& m_b_phi,
    Eigen::VectorXd& m_b_z)>;

class ExternalMagneticField {
 public:
  ExternalMagneticField(const Sizes* s, const TangentialPartitioning* tp,
                        const SurfaceGeometry* sg,
                        const MagneticFieldFunction* magnetic_field);

  // Returns the status of the magnetic field evaluation on this thread's
  // slice.
  [[nodiscard]] absl::Status update(const std::span<const double> rAxis,
                                    const std::span<const double> zAxis,
                                    double netToroidalCurrent);

  // axis geometry around whole machine
  Eigen::VectorXd axisXYZ;
  Eigen::VectorXd surfaceXYZ;
  Eigen::VectorXd bCoilsXYZ;

  // thread-local surface points in cylindrical coordinates
  Eigen::VectorXd surfaceR;
  Eigen::VectorXd surfacePhi;
  Eigen::VectorXd surfaceZ;

  // external magnetic field at the surface points
  Eigen::VectorXd interpBr;
  Eigen::VectorXd interpBp;
  Eigen::VectorXd interpBz;

  // interpolated magnetic field from mgrid
  double axis_current;
  Eigen::VectorXd curtorBr;
  Eigen::VectorXd curtorBp;
  Eigen::VectorXd curtorBz;

  // outputs to Nestor
  Eigen::VectorXd bSubU;
  Eigen::VectorXd bSubV;
  Eigen::VectorXd bDotN;

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

  // For an axisymmetric (nZeta == 1) plasma the toroidal direction is not
  // resolved by the surface grid; the net-toroidal-current axis filament is
  // then replicated over this many equally-spaced toroidal angles to resolve
  // it (educational_VMEC uses nvper = 64 for the tokamak case). For nZeta > 1,
  // nvper is the number of field periods.
  static constexpr int kAxisymmetricToroidalReplication = 64;
  int nvper_;

  const Sizes& s_;
  const TangentialPartitioning& tp_;

  const SurfaceGeometry& sg_;
  const MagneticFieldFunction* magnetic_field_;

  // Compute the contribution to the external magnetic field from net toroidal
  // current along magnetic axis. Here, the contribution is computed using
  // ABSCAB.
  void AddAxisCurrentFieldAbscab(const std::span<const double> rAxis,
                                 const std::span<const double> zAxis,
                                 double netToroidalCurrent);

  // Compute the contribution to the external magnetic field from net toroidal
  // current along magnetic axis. Here, this contribution is computed using a
  // straightforward implementation of the Hanson-Hirshman 2002 paper.
  void AddAxisCurrentFieldSimple(const std::span<const double> rAxis,
                                 const std::span<const double> zAxis,
                                 double netToroidalCurrent);

  void covariantAndNormalComponents();
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_EXTERNAL_MAGNETIC_FIELD_EXTERNAL_MAGNETIC_FIELD_H_
