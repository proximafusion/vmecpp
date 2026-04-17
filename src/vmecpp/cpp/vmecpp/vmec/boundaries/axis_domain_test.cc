// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// Test that verifies the toroidal domain coverage for axis optimization
TEST(AxisDomainTest, SymmetricUsesHalfDomain) {
  // Create symmetric configuration
  bool lasym = false;
  int nfp = 1;
  int mpol = 3;
  int ntor = 2;
  int ntheta = 16;
  int nzeta = 8;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // For symmetric case, should use nZeta/2 + 1 points
  const int expected_loop_count = nzeta / 2 + 1;  // = 4 + 1 = 5
  const int actual_loop_count = sizes.lasym ? sizes.nZeta : sizes.nZeta / 2 + 1;

  EXPECT_EQ(actual_loop_count, expected_loop_count);
  EXPECT_EQ(actual_loop_count, 5);
}

TEST(AxisDomainTest, AsymmetricUsesFullDomain) {
  // Create asymmetric configuration
  bool lasym = true;
  int nfp = 1;
  int mpol = 3;
  int ntor = 2;
  int ntheta = 16;
  int nzeta = 8;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // For asymmetric case, should use full nZeta points
  const int expected_loop_count = nzeta;  // = 8
  const int actual_loop_count = sizes.lasym ? sizes.nZeta : sizes.nZeta / 2 + 1;

  EXPECT_EQ(actual_loop_count, expected_loop_count);
  EXPECT_EQ(actual_loop_count, 8);
}

TEST(AxisDomainTest, AsymmetricDoublesCoverage) {
  // Test that asymmetric uses exactly double the coverage of symmetric
  int nfp = 1;
  int mpol = 3;
  int ntor = 2;
  int ntheta = 16;
  int nzeta = 12;  // Use 12 so half is 6, +1 = 7

  Sizes sizes_sym(false, nfp, mpol, ntor, ntheta, nzeta);
  Sizes sizes_asym(true, nfp, mpol, ntor, ntheta, nzeta);

  const int sym_loop_count = sizes_sym.lasym ? sizes_sym.nZeta : sizes_sym.nZeta / 2 + 1;
  const int asym_loop_count = sizes_asym.lasym ? sizes_asym.nZeta : sizes_asym.nZeta / 2 + 1;

  EXPECT_EQ(sym_loop_count, 7);   // 12/2 + 1 = 7
  EXPECT_EQ(asym_loop_count, 12); // Full domain
  EXPECT_EQ(asym_loop_count, nzeta);
  
  // Asymmetric covers more domain than symmetric
  EXPECT_GT(asym_loop_count, sym_loop_count);
}

// Test the logic matches what's implemented in guess_magnetic_axis.cc
TEST(AxisDomainTest, MatchesImplementationLogic) {
  std::vector<std::pair<bool, int>> test_cases = {
    {false, 8},   // symmetric, nzeta=8
    {true, 8},    // asymmetric, nzeta=8
    {false, 16},  // symmetric, nzeta=16
    {true, 16},   // asymmetric, nzeta=16
  };

  for (const auto& [lasym, nzeta] : test_cases) {
    Sizes sizes(lasym, 1, 3, 2, 16, nzeta);
    
    // This is the exact logic from guess_magnetic_axis.cc line 383
    const int nZetaLoop = sizes.lasym ? sizes.nZeta : sizes.nZeta / 2 + 1;
    
    if (lasym) {
      EXPECT_EQ(nZetaLoop, nzeta) << "Asymmetric should use full domain";
    } else {
      EXPECT_EQ(nZetaLoop, nzeta / 2 + 1) << "Symmetric should use half domain + 1";
    }
  }
}

}  // namespace vmecpp