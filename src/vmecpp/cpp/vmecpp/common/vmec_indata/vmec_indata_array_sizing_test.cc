// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/vmec_indata/vmec_indata.h"

#include <string>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

namespace vmecpp {
namespace {

TEST(VmecINDATAArraySizingTest, SymmetricArraySizes) {
  // Test symmetric case with mpol=6, ntor=0
  const std::string json_str = R"({
    "lasym": false,
    "nfp": 1,
    "mpol": 6,
    "ntor": 0,
    "ns_array": [5],
    "ftol_array": [1e-12],
    "niter_array": [100],
    "raxis_c": [4.0],
    "zaxis_s": [0.0],
    "rbc": [
      {"m": 0, "n": 0, "value": 4.0},
      {"m": 1, "n": 0, "value": 1.0},
      {"m": 2, "n": 0, "value": -0.1}
    ],
    "zbs": [
      {"m": 1, "n": 0, "value": 1.5},
      {"m": 2, "n": 0, "value": 0.01}
    ]
  })";
  
  auto result = VmecINDATA::FromJson(json_str);
  ASSERT_TRUE(result.ok()) << result.status();
  
  const VmecINDATA& vmec_indata = result.value();
  
  // Check array sizes
  const size_t expected_size = (vmec_indata.mpol + 1) * (2 * vmec_indata.ntor + 1);
  EXPECT_EQ(expected_size, 7);  // (6+1) * (2*0+1) = 7
  EXPECT_EQ(vmec_indata.rbc.size(), expected_size);
  EXPECT_EQ(vmec_indata.zbs.size(), expected_size);
  
  // Asymmetric arrays should be empty for lasym=false
  EXPECT_EQ(vmec_indata.rbs.size(), 0);
  EXPECT_EQ(vmec_indata.zbc.size(), 0);
}

TEST(VmecINDATAArraySizingTest, AsymmetricArraySizes) {
  // Test asymmetric case with mpol=5, ntor=0
  const std::string json_str = R"({
    "lasym": true,
    "nfp": 1,
    "mpol": 5,
    "ntor": 0,
    "ns_array": [5],
    "ftol_array": [1e-12],
    "niter_array": [100],
    "raxis_c": [6.0],
    "zaxis_s": [0.0],
    "raxis_s": [0.0],
    "zaxis_c": [0.0],
    "rbc": [
      {"m": 0, "n": 0, "value": 6.0},
      {"m": 2, "n": 0, "value": 0.6},
      {"m": 4, "n": 0, "value": 0.12}
    ],
    "zbs": [],
    "rbs": [
      {"m": 2, "n": 0, "value": 0.189737}
    ],
    "zbc": [
      {"m": 2, "n": 0, "value": 0.189737}
    ]
  })";
  
  auto result = VmecINDATA::FromJson(json_str);
  ASSERT_TRUE(result.ok()) << result.status();
  
  const VmecINDATA& vmec_indata = result.value();
  
  // Check array sizes
  const size_t expected_size = (vmec_indata.mpol + 1) * (2 * vmec_indata.ntor + 1);
  EXPECT_EQ(expected_size, 6);  // (5+1) * (2*0+1) = 6
  EXPECT_EQ(vmec_indata.rbc.size(), expected_size);
  EXPECT_EQ(vmec_indata.zbs.size(), expected_size);
  EXPECT_EQ(vmec_indata.rbs.size(), expected_size);
  EXPECT_EQ(vmec_indata.zbc.size(), expected_size);
}

TEST(VmecINDATAArraySizingTest, MaxModeNumberAllowed) {
  // Test that m=mpol is allowed (not rejected)
  const std::string json_str = R"({
    "lasym": false,
    "nfp": 1,
    "mpol": 3,
    "ntor": 0,
    "ns_array": [5],
    "ftol_array": [1e-12],
    "niter_array": [100],
    "raxis_c": [4.0],
    "zaxis_s": [0.0],
    "rbc": [
      {"m": 0, "n": 0, "value": 4.0},
      {"m": 3, "n": 0, "value": 0.1}
    ],
    "zbs": []
  })";
  
  auto result = VmecINDATA::FromJson(json_str);
  ASSERT_TRUE(result.ok()) << result.status();
  
  const VmecINDATA& vmec_indata = result.value();
  
  // Check that m=3 coefficient is present (at index 3)
  EXPECT_EQ(vmec_indata.rbc[3], 0.1);
}

TEST(VmecINDATAArraySizingTest, LargeModeNumberRejected) {
  // Test that m > mpol is rejected with a log message
  // This test just verifies the input is accepted but the coefficient is ignored
  const std::string json_str = R"({
    "lasym": false,
    "nfp": 1,
    "mpol": 3,
    "ntor": 0,
    "ns_array": [5],
    "ftol_array": [1e-12],
    "niter_array": [100],
    "raxis_c": [4.0],
    "zaxis_s": [0.0],
    "rbc": [
      {"m": 0, "n": 0, "value": 4.0},
      {"m": 4, "n": 0, "value": 0.1}
    ],
    "zbs": []
  })";
  
  auto result = VmecINDATA::FromJson(json_str);
  ASSERT_TRUE(result.ok()) << result.status();
  
  const VmecINDATA& vmec_indata = result.value();
  
  // Array should still be size 4 (mpol+1)
  EXPECT_EQ(vmec_indata.rbc.size(), 4);
  // The m=4 coefficient should not be stored (all zeros)
  for (size_t i = 0; i < vmec_indata.rbc.size(); ++i) {
    if (i == 0) {
      EXPECT_EQ(vmec_indata.rbc[i], 4.0);
    } else {
      EXPECT_EQ(vmec_indata.rbc[i], 0.0);
    }
  }
}

}  // namespace
}  // namespace vmecpp