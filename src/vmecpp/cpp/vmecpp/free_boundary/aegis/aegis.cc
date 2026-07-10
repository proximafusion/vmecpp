// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/aegis/aegis.h"

#include <array>
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
  static constexpr std::array<double, kNumCenters> kOrders = {3, 4, 5, 6, 7, 8};

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

std::vector<double> VacuumPressure(
    const std::vector<Eigen::Vector3d>& surface_points,
    const std::vector<Eigen::Vector3d>& outward_normals,
    const std::vector<double>& area,
    const std::vector<Eigen::Vector3d>& interior_field,
    const std::vector<Eigen::Vector3d>& coil_field, double expansion_scale) {
  const std::size_t n = surface_points.size();
  std::vector<Eigen::Vector3d> current(n);
  std::vector<double> normal_field(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Eigen::Vector3d b_plasma = interior_field[i] - coil_field[i];
    current[i] = outward_normals[i].cross(b_plasma);
    normal_field[i] = outward_normals[i].dot(b_plasma);
  }
  // The constructor copies surface_points and area (passed by value); the
  // caller's vectors remain valid for the on-surface evaluations below.
  const VirtualCasing vc(surface_points, std::move(current),
                         std::move(normal_field), area);
  std::vector<double> pressure(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Eigen::Vector3d b_exterior =
        vc.OnSurface(surface_points[i], outward_normals[i], expansion_scale) +
        coil_field[i];
    pressure[i] = 0.5 * b_exterior.squaredNorm();
  }
  return pressure;
}

}  // namespace vmecpp
