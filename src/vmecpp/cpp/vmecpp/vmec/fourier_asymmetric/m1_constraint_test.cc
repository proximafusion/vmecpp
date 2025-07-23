// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

TEST(M1ConstraintTest, EnsureM1ConstrainedSymmetric) {
  // Test M=1 constraint coupling for symmetric boundary coefficients
  bool lasym = false;
  int nfp = 1;
  int mpol = 2;  // Need m=1 mode
  int ntor = 1;  // Need n=0,1 modes
  int ntheta = 8;
  int nzeta = 4;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // Create test arrays for boundary coefficients
  std::vector<double> rbss((ntor + 1) * (mpol + 1), 0.0);
  std::vector<double> zbcs((ntor + 1) * (mpol + 1), 0.0);
  std::vector<double> rbsc((ntor + 1) * (mpol + 1), 0.0);  // Not used for lasym=false
  std::vector<double> zbcc((ntor + 1) * (mpol + 1), 0.0);  // Not used for lasym=false

  // Set initial values for m=1 modes
  // Index: n * (mpol + 1) + m = n * 3 + 1
  rbss[0 * 3 + 1] = 1.0;  // rbss[n=0,m=1] = 1.0
  zbcs[0 * 3 + 1] = 0.5;  // zbcs[n=0,m=1] = 0.5
  rbss[1 * 3 + 1] = 2.0;  // rbss[n=1,m=1] = 2.0
  zbcs[1 * 3 + 1] = 1.5;  // zbcs[n=1,m=1] = 1.5

  // Call M=1 constraint function
  EnsureM1Constrained(sizes, absl::MakeSpan(rbss), absl::MakeSpan(zbcs),
                      absl::MakeSpan(rbsc), absl::MakeSpan(zbcc));

  // Verify constraint coupling: 
  // rbss[n,1] = (original_rbss + zbcs) / 2
  // zbcs[n,1] = (original_rbss - zbcs) / 2
  EXPECT_NEAR(rbss[0 * 3 + 1], (1.0 + 0.5) / 2.0, 1e-10);  // 0.75
  EXPECT_NEAR(zbcs[0 * 3 + 1], (1.0 - 0.5) / 2.0, 1e-10);  // 0.25
  EXPECT_NEAR(rbss[1 * 3 + 1], (2.0 + 1.5) / 2.0, 1e-10);  // 1.75
  EXPECT_NEAR(zbcs[1 * 3 + 1], (2.0 - 1.5) / 2.0, 1e-10);  // 0.25
}

TEST(M1ConstraintTest, EnsureM1ConstrainedAsymmetric) {
  // Test M=1 constraint coupling for asymmetric boundary coefficients
  bool lasym = true;
  int nfp = 1;
  int mpol = 2;  // Need m=1 mode
  int ntor = 1;  // Need n=0,1 modes
  int ntheta = 8;
  int nzeta = 4;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // Create test arrays for boundary coefficients
  std::vector<double> rbss((ntor + 1) * (mpol + 1), 0.0);  // Not used for asymmetric test
  std::vector<double> zbcs((ntor + 1) * (mpol + 1), 0.0);  // Not used for asymmetric test
  std::vector<double> rbsc((ntor + 1) * (mpol + 1), 0.0);
  std::vector<double> zbcc((ntor + 1) * (mpol + 1), 0.0);

  // Set initial values for asymmetric m=1 modes
  rbsc[0 * 3 + 1] = 3.0;  // rbsc[n=0,m=1] = 3.0
  zbcc[0 * 3 + 1] = 1.0;  // zbcc[n=0,m=1] = 1.0
  rbsc[1 * 3 + 1] = 4.0;  // rbsc[n=1,m=1] = 4.0
  zbcc[1 * 3 + 1] = 2.0;  // zbcc[n=1,m=1] = 2.0

  // Call M=1 constraint function
  EnsureM1Constrained(sizes, absl::MakeSpan(rbss), absl::MakeSpan(zbcs),
                      absl::MakeSpan(rbsc), absl::MakeSpan(zbcc));

  // Verify asymmetric constraint coupling:
  // rbsc[n,1] = (original_rbsc + zbcc) / 2
  // zbcc[n,1] = (original_rbsc - zbcc) / 2
  EXPECT_NEAR(rbsc[0 * 3 + 1], (3.0 + 1.0) / 2.0, 1e-10);  // 2.0
  EXPECT_NEAR(zbcc[0 * 3 + 1], (3.0 - 1.0) / 2.0, 1e-10);  // 1.0
  EXPECT_NEAR(rbsc[1 * 3 + 1], (4.0 + 2.0) / 2.0, 1e-10);  // 3.0
  EXPECT_NEAR(zbcc[1 * 3 + 1], (4.0 - 2.0) / 2.0, 1e-10);  // 1.0
}

TEST(M1ConstraintTest, ConvertToM1Constrained) {
  // Test M=1 constraint coupling for force coefficients
  bool lasym = true;
  int nfp = 1;
  int mpol = 2;  // Need m=1 mode
  int ntor = 1;  // Need n=0,1 modes
  int ntheta = 8;
  int nzeta = 4;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);
  
  const int num_surfaces = 10;  // Test with 10 surfaces
  const int total_size = num_surfaces * (ntor + 1) * (mpol + 1);

  // Create test arrays for force coefficients
  std::vector<double> rss_rsc(total_size, 0.0);
  std::vector<double> zcs_zcc(total_size, 0.0);

  // Set test values for m=1 modes on different surfaces
  for (int j = 0; j < num_surfaces; ++j) {
    for (int n = 0; n <= ntor; ++n) {
      const int idx = j * (ntor + 1) * (mpol + 1) + n * (mpol + 1) + 1;
      rss_rsc[idx] = 10.0 + j + n;  // Different values per surface/mode
      zcs_zcc[idx] = 5.0 + j + n;
    }
  }

  const double scaling_factor = 1.0 / sqrt(2.0);  // jVMEC uses this scaling

  // Call M=1 constraint function
  ConvertToM1Constrained(sizes, num_surfaces, absl::MakeSpan(rss_rsc), 
                         absl::MakeSpan(zcs_zcc), scaling_factor);

  // Verify constraint coupling with scaling:
  // rss_rsc[j,n,1] = scaling * (original_rss_rsc + zcs_zcc)
  // zcs_zcc[j,n,1] = scaling * (original_rss_rsc - zcs_zcc)
  for (int j = 0; j < num_surfaces; ++j) {
    for (int n = 0; n <= ntor; ++n) {
      const int idx = j * (ntor + 1) * (mpol + 1) + n * (mpol + 1) + 1;
      const double original_rss_rsc = 10.0 + j + n;
      const double original_zcs_zcc = 5.0 + j + n;
      
      const double expected_rss_rsc = scaling_factor * (original_rss_rsc + original_zcs_zcc);
      const double expected_zcs_zcc = scaling_factor * (original_rss_rsc - original_zcs_zcc);
      
      EXPECT_NEAR(rss_rsc[idx], expected_rss_rsc, 1e-10);
      EXPECT_NEAR(zcs_zcc[idx], expected_zcs_zcc, 1e-10);
    }
  }
}

}  // namespace vmecpp