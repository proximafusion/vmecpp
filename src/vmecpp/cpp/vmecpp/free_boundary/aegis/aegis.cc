// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/aegis/aegis.h"

#include <cmath>
#include <numbers>
#include <utility>

namespace vmecpp {

VirtualCasing::VirtualCasing(std::vector<Eigen::Vector3d> surface_points,
                             std::vector<Eigen::Vector3d> surface_current,
                             std::vector<double> normal_field,
                             std::vector<double> area)
    : x_(std::move(surface_points)),
      k_(std::move(surface_current)),
      sigma_(std::move(normal_field)),
      dA_(std::move(area)) {}

Eigen::Vector3d VirtualCasing::Raw(const Eigen::Vector3d& r) const {
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < x_.size(); ++i) {
    const Eigen::Vector3d d = r - x_[i];
    const double inv = 1.0 / (d.norm() * d.squaredNorm());
    acc += (k_[i].cross(d) + sigma_[i] * d) * (inv * dA_[i]);
  }
  return acc / (4.0 * std::numbers::pi);
}

Eigen::Vector3d VirtualCasing::OnSurface(const Eigen::Vector3d& r0,
                                         const Eigen::Vector3d& outward_normal,
                                         double expansion_scale) const {
  // Expansion centers at 3..8 source-spacings out; a cubic fit in the offset,
  // extrapolated to the surface (the constant term). This ordering keeps the
  // centers far enough that the Biot-Savart quadrature is well resolved and
  // close enough that the local expansion converges.
  constexpr int kNumCenters = 6;
  constexpr int kDegree = 3;
  static const double kOrders[kNumCenters] = {3, 4, 5, 6, 7, 8};

  Eigen::VectorXd offsets(kNumCenters);
  Eigen::MatrixXd samples(kNumCenters, 3);
  for (int j = 0; j < kNumCenters; ++j) {
    offsets(j) = kOrders[j] * expansion_scale;
    samples.row(j) = Raw(r0 + offsets(j) * outward_normal).transpose();
  }

  // Vandermonde in the offset, degrees 0..kDegree; column-pivoted QR least
  // squares. The degree-0 coefficient is the value extrapolated to offset 0.
  Eigen::MatrixXd vandermonde(kNumCenters, kDegree + 1);
  for (int j = 0; j < kNumCenters; ++j) {
    double p = 1.0;
    for (int k = 0; k <= kDegree; ++k) {
      vandermonde(j, k) = p;
      p *= offsets(j);
    }
  }
  const auto qr = vandermonde.colPivHouseholderQr();
  Eigen::Vector3d on_surface;
  for (int c = 0; c < 3; ++c) {
    on_surface(c) = qr.solve(samples.col(c))(0);
  }
  return on_surface;
}

}  // namespace vmecpp
