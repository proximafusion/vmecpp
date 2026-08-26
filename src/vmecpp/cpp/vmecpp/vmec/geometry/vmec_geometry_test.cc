// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/geometry/vmec_geometry.h"

#include <cmath>

#include "gtest/gtest.h"

namespace vmecpp {
namespace {

TEST(VmecGeometryTest, ConvertsInternalScalingWithoutWout) {
  VmecINDATA indata;
  indata.nfp = 2;
  indata.mpol = 2;
  indata.ntor = 0;
  VmecInternalResults internal;
  internal.sign_of_jacobian = -1;
  internal.lamscale = 3.0;
  internal.num_full = 2;
  internal.phiF = Eigen::Vector2d(0.0, -4.0);
  internal.phipF = Eigen::Vector2d(2.0, 2.0);
  internal.phipH = Eigen::VectorXd::Constant(1, 2.0);
  internal.iotaH = Eigen::VectorXd::Constant(1, 0.4);
  internal.rmncc = RowMatrixXd::Zero(2, 2);
  internal.zmnsc = RowMatrixXd::Zero(2, 2);
  internal.lmnsc = RowMatrixXd::Zero(2, 2);
  internal.rmncc << 10.0, 1.0, 12.0, 2.0;
  internal.zmnsc << 0.0, 0.5, 0.0, 1.0;
  internal.lmnsc << 0.0, 0.2, 0.0, 0.4;

  const Geometry geometry = MakeGeometry(indata, internal);
  const GeometryPoint point = EvaluateGeometry(geometry, 0.25, 0.6, 0.7);
  const double root_two = std::sqrt(2.0);
  const double expected_r = 10.5 + root_two * 1.25 * std::cos(0.6);
  const double expected_z = root_two * 0.625 * std::sin(0.6);
  const double expected_lambda = 1.5 * root_two * 0.25 * std::sin(0.6);

  EXPECT_NEAR(point.r[0], expected_r, 1e-14);
  EXPECT_NEAR(point.z[0], expected_z, 1e-14);
  EXPECT_NEAR(point.lambda[0], expected_lambda, 1e-14);
  EXPECT_DOUBLE_EQ(point.toroidal_flux[0], -1.0);
  EXPECT_NEAR(point.poloidal_flux[1], -1.6 * M_PI, 1e-14);
}

}  // namespace
}  // namespace vmecpp
