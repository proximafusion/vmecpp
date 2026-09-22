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

// With no phiF to copy, the adapter integrates the enclosed toroidal flux from
// the half-grid dphi/ds between the two surfaces. That is exact for a linear
// dphi/ds, where the value at the inner endpoint of each interval is not.
TEST(VmecGeometryTest, DerivesToroidalFluxFromTheHalfGridDerivative) {
  VmecINDATA indata;
  indata.nfp = 1;
  indata.mpol = 2;
  indata.ntor = 0;

  const int num_full = 4;
  const double delta_s = 1.0 / (num_full - 1);
  VmecInternalResults internal;
  internal.sign_of_jacobian = 1;
  internal.lamscale = 1.0;
  internal.num_full = num_full;
  // dphi/ds = 1 + 3 s, so the enclosed flux is 2 pi (s + 3 s^2 / 2)
  internal.phiF = Eigen::VectorXd::Zero(num_full);
  internal.phipF = Eigen::VectorXd(num_full);
  for (int jF = 0; jF < num_full; ++jF) {
    internal.phipF[jF] = 1.0 + 3.0 * jF * delta_s;
  }
  internal.phipH = Eigen::VectorXd(num_full - 1);
  for (int jH = 0; jH < num_full - 1; ++jH) {
    internal.phipH[jH] = 1.0 + 3.0 * (jH + 0.5) * delta_s;
  }
  internal.iotaH = Eigen::VectorXd::Zero(num_full - 1);
  internal.rmncc = RowMatrixXd::Zero(num_full, 2);
  internal.zmnsc = RowMatrixXd::Zero(num_full, 2);
  internal.lmnsc = RowMatrixXd::Zero(num_full, 2);

  const Geometry geometry = MakeGeometry(indata, internal);
  ASSERT_EQ(geometry.toroidal_flux.size(), static_cast<std::size_t>(num_full));
  for (int jF = 0; jF < num_full; ++jF) {
    const double s = jF * delta_s;
    EXPECT_NEAR(geometry.toroidal_flux[jF], 2.0 * M_PI * (s + 1.5 * s * s),
                1e-13)
        << "surface " << jF;
  }
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
