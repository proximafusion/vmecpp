// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/aegis/aegis.h"

#include <Eigen/Dense>
#include <cmath>
#include <numbers>
#include <vector>

#include "gtest/gtest.h"

namespace vmecpp {
namespace {

using ::Eigen::Vector3d;

// On any closed surface carrying a surface current K, the virtual-casing field
// satisfies the jump condition n x (B_out - B_in) = K (Ampere's law for the
// equivalent surface current). Build a sphere with a smooth tangential current
// K = n x c and verify that the QBX exterior and interior on-surface limits
// reproduce K. This exercises the Biot-Savart integrator and the QBX
// extrapolation with no external inputs.
TEST(AegisVirtualCasingTest, MagnetizedSphereJumpCondition) {
  constexpr int kNTheta = 128;
  constexpr int kNPhi = 128;
  constexpr double kRadius = 1.0;
  const Vector3d c(0.0, 0.0, 1.0);  // K = n x c is tangential and smooth

  std::vector<Vector3d> points;
  std::vector<Vector3d> current;
  std::vector<Vector3d> normals;
  std::vector<double> sigma;
  std::vector<double> area;
  double total_area = 0.0;
  for (int i = 0; i < kNTheta; ++i) {
    const double theta = (i + 0.5) * std::numbers::pi / kNTheta;
    for (int j = 0; j < kNPhi; ++j) {
      const double phi = (j + 0.5) * 2.0 * std::numbers::pi / kNPhi;
      const Vector3d n(std::sin(theta) * std::cos(phi),
                       std::sin(theta) * std::sin(phi), std::cos(theta));
      points.push_back(kRadius * n);
      normals.push_back(n);
      current.push_back(n.cross(c));
      sigma.push_back(0.0);
      const double dA = kRadius * kRadius * std::sin(theta) *
                        (std::numbers::pi / kNTheta) *
                        (2.0 * std::numbers::pi / kNPhi);
      area.push_back(dA);
      total_area += dA;
    }
  }

  const VirtualCasing vc(points, current, sigma, area);
  const double h = std::sqrt(total_area / (kNTheta * kNPhi));

  for (int idx : {17 * kNPhi + 40, 64 * kNPhi + 10, 100 * kNPhi + 90}) {
    const Vector3d r0 = points[idx];
    const Vector3d n0 = normals[idx];
    const Vector3d jump =
        n0.cross(vc.OnSurface(r0, n0, h) - vc.OnSurface(r0, -n0, h));
    const Vector3d k_ref = current[idx];
    const double rel_err = (jump - k_ref).norm() / k_ref.norm();
    EXPECT_LT(rel_err, 0.02) << "jump condition at point " << idx;
  }
}

// VacuumPressure forms the surface current from interior_field - coil_field,
// evaluates the exterior field, and returns |B_exterior|^2 / 2. Smoke test on a
// sphere: finite, positive, and of order |coil|^2/2 when the plasma current is
// weak.
TEST(AegisVirtualCasingTest, VacuumPressureIsFinitePositive) {
  constexpr int kN = 48;
  const Vector3d coil_field_value(0.1, 0.0, 0.2);
  std::vector<Vector3d> points;
  std::vector<Vector3d> normals;
  std::vector<Vector3d> interior;
  std::vector<Vector3d> coil;
  std::vector<double> area;
  double total_area = 0.0;
  for (int i = 0; i < kN; ++i) {
    const double theta = (i + 0.5) * std::numbers::pi / kN;
    for (int j = 0; j < kN; ++j) {
      const double phi = (j + 0.5) * 2.0 * std::numbers::pi / kN;
      const Vector3d n(std::sin(theta) * std::cos(phi),
                       std::sin(theta) * std::sin(phi), std::cos(theta));
      points.push_back(n);
      normals.push_back(n);
      coil.push_back(coil_field_value);
      // interior = coil field + a weak tangential plasma field.
      interior.push_back(coil_field_value + n.cross(Vector3d(0, 0, 0.05)));
      const double dA = std::sin(theta) * (std::numbers::pi / kN) *
                        (2.0 * std::numbers::pi / kN);
      area.push_back(dA);
      total_area += dA;
    }
  }
  const double h = std::sqrt(total_area / (kN * kN));
  const std::vector<double> pressure =
      VacuumPressure(points, normals, area, interior, coil, h);

  ASSERT_EQ(static_cast<int>(pressure.size()), kN * kN);
  const double coil_pressure = 0.5 * coil_field_value.squaredNorm();
  for (double p : pressure) {
    EXPECT_TRUE(std::isfinite(p));
    EXPECT_GT(p, 0.0);
    EXPECT_LT(p, 10.0 * coil_pressure);  // weak plasma -> near coil pressure
  }
}

}  // namespace
}  // namespace vmecpp
