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

TEST(VmecGeometryTest, SolverAndPhysicalStatesAgree) {
  VmecINDATA indata;
  indata.mpol = 2;
  indata.ntor = 1;
  VmecInternalResults solver;
  solver.sign_of_jacobian = 1;
  solver.lamscale = 1.0;
  solver.num_full = 2;
  solver.phiF = Eigen::Vector2d(0.0, 1.0);
  solver.phipF = Eigen::Vector2d(1.0, 1.0);
  solver.phipH = Eigen::VectorXd::Ones(1);
  solver.iotaH = Eigen::VectorXd::Constant(1, 0.4);
  solver.rmncc = RowMatrixXd::Zero(2, 4);
  solver.zmnsc = RowMatrixXd::Zero(2, 4);
  solver.lmnsc = RowMatrixXd::Zero(2, 4);
  solver.rmnss = RowMatrixXd::Zero(2, 4);
  solver.zmncs = RowMatrixXd::Zero(2, 4);
  solver.lmncs = RowMatrixXd::Zero(2, 4);
  solver.rmnss << 0.0, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0, 0.5;
  solver.zmncs << 0.0, -0.1, 0.0, 0.6, 0.0, -0.2, 0.0, 0.7;

  VmecInternalResults physical = solver;
  for (int j = 0; j < solver.num_full; ++j) {
    for (int n = 0; n <= indata.ntor; ++n) {
      const double old_r = solver.rmnss(j, n * indata.mpol + 1);
      const double old_z = solver.zmncs(j, n * indata.mpol + 1);
      physical.rmnss(j, n * indata.mpol + 1) = old_r + old_z;
      physical.zmncs(j, n * indata.mpol + 1) = old_r - old_z;
    }
  }

  const Geometry from_solver = MakeGeometry(indata, solver);
  const Geometry from_physical =
      MakeGeometry(indata, physical, GeometryCoefficientState::kPhysical);
  ASSERT_EQ(from_solver.coefficients.r_ss.size(),
            from_physical.coefficients.r_ss.size());
  for (std::size_t i = 0; i < from_solver.coefficients.r_ss.size(); ++i) {
    EXPECT_DOUBLE_EQ(from_solver.coefficients.r_ss[i],
                     from_physical.coefficients.r_ss[i]);
    EXPECT_DOUBLE_EQ(from_solver.coefficients.z_cs[i],
                     from_physical.coefficients.z_cs[i]);
  }
}

}  // namespace
}  // namespace vmecpp
