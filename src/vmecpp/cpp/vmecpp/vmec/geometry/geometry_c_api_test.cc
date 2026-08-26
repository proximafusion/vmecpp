// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/geometry/geometry_c_api.h"

#include <cmath>

#include "gtest/gtest.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace {

TEST(GeometryCApiTest, BoundaryMatchesInputWithoutWout) {
  constexpr char kInput[] = "vmecpp/test_data/solovev.json";
  const vmecpp::VmecINDATA indata = vmecpp::VmecINDATA::FromFile(kInput);
  vmecpp_geometry_handle* handle = nullptr;
  ASSERT_EQ(vmecpp_geometry_create(kInput, &handle), 0)
      << vmecpp_geometry_error();

  vmecpp_geometry_metadata metadata{};
  ASSERT_EQ(vmecpp_geometry_get_metadata(handle, &metadata), 0)
      << vmecpp_geometry_error();
  EXPECT_EQ(metadata.nfp, indata.nfp);
  EXPECT_NEAR(metadata.major_radius, indata.rbc(0, indata.ntor), 2e-12);

  const double theta = 0.37;
  vmecpp_geometry_point point{};
  ASSERT_EQ(vmecpp_geometry_evaluate(handle, 1.0, theta, 0.0, &point), 0)
      << vmecpp_geometry_error();
  double expected_r = 0.0;
  double expected_z = 0.0;
  double expected_r_theta2 = 0.0;
  double expected_z_theta2 = 0.0;
  for (int m = 0; m < indata.mpol; ++m) {
    expected_r += indata.rbc(m, indata.ntor) * std::cos(m * theta);
    expected_z += indata.zbs(m, indata.ntor) * std::sin(m * theta);
    expected_r_theta2 -=
        m * m * indata.rbc(m, indata.ntor) * std::cos(m * theta);
    expected_z_theta2 -=
        m * m * indata.zbs(m, indata.ntor) * std::sin(m * theta);
  }
  EXPECT_NEAR(point.r[0], expected_r, 2e-12);
  EXPECT_NEAR(point.z[0], expected_z, 2e-12);
  EXPECT_NEAR(point.r[7], expected_r_theta2, 2e-12);
  EXPECT_NEAR(point.z[7], expected_z_theta2, 2e-12);
  vmecpp_geometry_destroy(handle);
}

TEST(GeometryCApiTest, ReportsInvalidArguments) {
  EXPECT_NE(vmecpp_geometry_create(nullptr, nullptr), 0);
  EXPECT_NE(vmecpp_geometry_error()[0], '\0');
}

}  // namespace
