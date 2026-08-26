// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/geometry/geometry.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace vmecpp {
namespace {

Geometry ExampleGeometry() {
  Geometry geometry{
      .dimensions = {.ns = 2, .mpol = 2, .ntor = 1, .nfp = 3},
      .toroidal_flux = {0.0, 2.0},
      .poloidal_flux = {0.0, 0.8},
  };
  const int size = 8;
  geometry.coefficients.r_cc.assign(size, 0.0);
  geometry.coefficients.z_sc.assign(size, 0.0);
  geometry.coefficients.lambda_sc.assign(size, 0.0);
  // R = 10 + s + (2 + 2s) cos(theta) cos(3 zeta)
  geometry.coefficients.r_cc[0] = 10.0;
  geometry.coefficients.r_cc[4] = 11.0;
  geometry.coefficients.r_cc[3] = 2.0;
  geometry.coefficients.r_cc[7] = 4.0;
  // Z = (1 + 2s) sin(theta)
  geometry.coefficients.z_sc[2] = 1.0;
  geometry.coefficients.z_sc[6] = 3.0;
  // lambda = (0.1 + 0.2s) sin(theta)
  geometry.coefficients.lambda_sc[2] = 0.1;
  geometry.coefficients.lambda_sc[6] = 0.3;
  return geometry;
}

double Contract(const GeometryPoint& left, const GeometryPoint& right) {
  double result = 0.0;
  for (const auto member :
       {&GeometryPoint::r, &GeometryPoint::z, &GeometryPoint::lambda,
        &GeometryPoint::toroidal_flux, &GeometryPoint::poloidal_flux}) {
    for (int i = 0; i < 4; ++i)
      result += (left.*member)[i] * (right.*member)[i];
  }
  return result;
}

TEST(GeometryTest, EvaluatesKnownFourierSeriesAndDerivatives) {
  const Geometry geometry = ExampleGeometry();
  const double s = 0.25;
  const double theta = 0.7;
  const double zeta = -0.2;
  const GeometryPoint point = EvaluateGeometry(geometry, s, theta, zeta);
  const double ct = std::cos(theta);
  const double st = std::sin(theta);
  const double cp = std::cos(3.0 * zeta);
  const double sp = std::sin(3.0 * zeta);

  EXPECT_NEAR(point.r[0], 10.0 + s + (2.0 + 2.0 * s) * ct * cp, 1e-14);
  EXPECT_NEAR(point.r[1], 1.0 + 2.0 * ct * cp, 1e-14);
  EXPECT_NEAR(point.r[2], -(2.0 + 2.0 * s) * st * cp, 1e-14);
  EXPECT_NEAR(point.r[3], -3.0 * (2.0 + 2.0 * s) * ct * sp, 1e-14);
  EXPECT_NEAR(point.z[0], (1.0 + 2.0 * s) * st, 1e-14);
  EXPECT_NEAR(point.z[1], 2.0 * st, 1e-14);
  EXPECT_NEAR(point.lambda[0], (0.1 + 0.2 * s) * st, 1e-14);
  EXPECT_DOUBLE_EQ(point.toroidal_flux[0], 2.0 * s);
  EXPECT_DOUBLE_EQ(point.toroidal_flux[1], 2.0);
  EXPECT_DOUBLE_EQ(point.poloidal_flux[0], 0.8 * s);
}

TEST(GeometryTest, CubicRadialProfileHasExactValueAndDerivative) {
  Geometry geometry{
      .dimensions = {.ns = 4, .mpol = 1, .ntor = 0, .nfp = 1},
      .toroidal_flux = {0.0, 1.0 / 27.0, 8.0 / 27.0, 1.0},
      .poloidal_flux = {0.0, 0.0, 0.0, 0.0},
  };
  geometry.coefficients.r_cc.assign(4, 1.0);
  geometry.coefficients.z_sc.assign(4, 0.0);
  geometry.coefficients.lambda_sc.assign(4, 0.0);

  const double s = 0.43;
  const GeometryPoint point = EvaluateGeometry(geometry, s, 0.0, 0.0);
  EXPECT_NEAR(point.toroidal_flux[0], s * s * s, 1e-14);
  EXPECT_NEAR(point.toroidal_flux[1], 3.0 * s * s, 1e-14);
}

TEST(GeometryTest, VjpMatchesIndependentDirectionalDerivative) {
  const Geometry geometry = ExampleGeometry();
  const double s = 0.37;
  const double theta = 0.4;
  const double zeta = 0.23;
  GeometryPoint seed{};
  seed.r = {0.2, -0.3, 0.7, 0.1};
  seed.z = {-0.4, 0.5, 0.2, -0.1};
  seed.lambda = {0.3, -0.2, 0.1, 0.4};
  seed.toroidal_flux = {0.6, -0.7, 0.0, 0.0};
  seed.poloidal_flux = {-0.1, 0.8, 0.0, 0.0};
  const GeometryVjp vjp = EvaluateGeometryVjp(geometry, s, theta, zeta, seed);

  Geometry direction = geometry;
  for (double& value : direction.toroidal_flux) value = 0.11;
  for (double& value : direction.poloidal_flux) value = -0.07;
  for (double& value : direction.coefficients.r_cc) value = 0.03;
  for (double& value : direction.coefficients.z_sc) value = -0.02;
  for (double& value : direction.coefficients.lambda_sc) value = 0.04;
  const std::array<double, 3> coordinate_direction = {0.05, -0.08, 0.09};

  double adjoint = coordinate_direction[0] * vjp.coordinates[0] +
                   coordinate_direction[1] * vjp.coordinates[1] +
                   coordinate_direction[2] * vjp.coordinates[2];
  for (int i = 0; i < 2; ++i) {
    adjoint += direction.toroidal_flux[i] * vjp.geometry.toroidal_flux[i];
    adjoint += direction.poloidal_flux[i] * vjp.geometry.poloidal_flux[i];
  }
  for (int i = 0; i < 8; ++i) {
    adjoint +=
        direction.coefficients.r_cc[i] * vjp.geometry.coefficients.r_cc[i];
    adjoint +=
        direction.coefficients.z_sc[i] * vjp.geometry.coefficients.z_sc[i];
    adjoint += direction.coefficients.lambda_sc[i] *
               vjp.geometry.coefficients.lambda_sc[i];
  }

  const double epsilon = 1e-6;
  auto perturb = [&](double sign) {
    Geometry perturbed = geometry;
    for (int i = 0; i < 2; ++i) {
      perturbed.toroidal_flux[i] += sign * epsilon * direction.toroidal_flux[i];
      perturbed.poloidal_flux[i] += sign * epsilon * direction.poloidal_flux[i];
    }
    for (int i = 0; i < 8; ++i) {
      perturbed.coefficients.r_cc[i] +=
          sign * epsilon * direction.coefficients.r_cc[i];
      perturbed.coefficients.z_sc[i] +=
          sign * epsilon * direction.coefficients.z_sc[i];
      perturbed.coefficients.lambda_sc[i] +=
          sign * epsilon * direction.coefficients.lambda_sc[i];
    }
    return EvaluateGeometry(perturbed,
                            s + sign * epsilon * coordinate_direction[0],
                            theta + sign * epsilon * coordinate_direction[1],
                            zeta + sign * epsilon * coordinate_direction[2]);
  };
  const double finite_difference =
      (Contract(seed, perturb(1.0)) - Contract(seed, perturb(-1.0))) /
      (2.0 * epsilon);
  EXPECT_NEAR(adjoint, finite_difference, 2e-8);
}

}  // namespace
}  // namespace vmecpp
