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
  EXPECT_NEAR(point.r[4], 0.0, 1e-14);
  EXPECT_NEAR(point.r[7], -(2.0 + 2.0 * s) * ct * cp, 1e-14);
  EXPECT_NEAR(point.r[9], -9.0 * (2.0 + 2.0 * s) * ct * cp, 1e-14);
  EXPECT_NEAR(point.z[0], (1.0 + 2.0 * s) * st, 1e-14);
  EXPECT_NEAR(point.z[1], 2.0 * st, 1e-14);
  EXPECT_NEAR(point.z[5], 2.0 * ct, 1e-14);
  EXPECT_NEAR(point.z[7], -(1.0 + 2.0 * s) * st, 1e-14);
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

}  // namespace
}  // namespace vmecpp
