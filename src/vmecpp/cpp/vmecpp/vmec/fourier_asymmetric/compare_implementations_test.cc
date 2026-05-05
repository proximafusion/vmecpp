// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

class CompareImplementationsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::cout << "===== COMPARING ORIGINAL VS CORRECTED IMPLEMENTATIONS ====="
              << std::endl;
    std::cout << "Test parameters:" << std::endl;
    std::cout << "  mpol=" << sizes.mpol << ", ntor=" << sizes.ntor
              << ", nfp=" << sizes.nfp << std::endl;
    std::cout << "  mnmax=" << sizes.mnmax << ", nZnT=" << sizes.nZnT
              << std::endl;
  }

  Sizes sizes{true, 1, 3, 0,
              8,    1};  // lasym=true, nfp=1, mpol=3, ntor=0, ntheta=8, nzeta=1
};

TEST_F(CompareImplementationsTest, CompareOriginalVsCorrected2D) {
  std::cout
      << "\n===== SETTING UP IDENTICAL INPUT FOR BOTH IMPLEMENTATIONS ====="
      << std::endl;

  // Create identical coefficient arrays for both tests
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);  // Asymmetric R
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);  // Asymmetric Z
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set test coefficients
  rmncc[0] = 1.0;   // (m=0,n=0) symmetric baseline R
  rmnsc[1] = 0.1;   // (m=1,n=0) asymmetric R perturbation
  zmncc[1] = 0.05;  // (m=1,n=0) asymmetric Z perturbation

  std::cout << "Input coefficients:" << std::endl;
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    std::cout << "  mn=" << mn << ": rmncc=" << rmncc[mn]
              << ", rmnsc=" << rmnsc[mn] << ", zmncc=" << zmncc[mn]
              << std::endl;
  }

  // Test ORIGINAL implementation
  std::cout << "\n===== RUNNING ORIGINAL IMPLEMENTATION =====" << std::endl;
  std::vector<double> r_real_orig(sizes.nZnT, 0.0);
  std::vector<double> z_real_orig(sizes.nZnT, 0.0);
  std::vector<double> lambda_real_orig(sizes.nZnT, 0.0);

  FourierToReal2DAsymmFastPoloidal(
      sizes, absl::Span<const double>(rmncc), absl::Span<const double>(rmnss),
      absl::Span<const double>(rmnsc), absl::Span<const double>(rmncs),
      absl::Span<const double>(zmnsc), absl::Span<const double>(zmncs),
      absl::Span<const double>(zmncc), absl::Span<const double>(zmnss),
      absl::Span<double>(r_real_orig), absl::Span<double>(z_real_orig),
      absl::Span<double>(lambda_real_orig));

  // Test CORRECTED implementation (with proper symmetric baseline)
  std::cout << "\n===== RUNNING CORRECTED IMPLEMENTATION (with symmetric "
               "baseline) ====="
            << std::endl;
  std::vector<double> r_real_corr(sizes.nZnT, 0.0);
  std::vector<double> z_real_corr(sizes.nZnT, 0.0);
  std::vector<double> lambda_real_corr(sizes.nZnT, 0.0);

  // First: Initialize with symmetric baseline (rmncc[0] = 1.0)
  // This simulates what the symmetric transform would do
  for (int i = 0; i < sizes.nZnT; ++i) {
    r_real_corr[i] = 1.0;  // Symmetric R baseline from rmncc[0] = 1.0
    z_real_corr[i] = 0.0;  // No symmetric Z contribution in this test
  }

  std::cout << "Initialized with symmetric baseline: R=1.0, Z=0.0" << std::endl;

  // Second: Add asymmetric contribution
  FourierToReal2DAsymmFastPoloidal_Corrected(
      sizes, absl::Span<const double>(rmncc), absl::Span<const double>(rmnss),
      absl::Span<const double>(rmnsc), absl::Span<const double>(rmncs),
      absl::Span<const double>(zmnsc), absl::Span<const double>(zmncs),
      absl::Span<const double>(zmncc), absl::Span<const double>(zmnss),
      absl::Span<double>(r_real_corr), absl::Span<double>(z_real_corr),
      absl::Span<double>(lambda_real_corr));

  // DETAILED COMPARISON
  std::cout << "\n===== DETAILED COMPARISON OF RESULTS =====" << std::endl;
  std::cout << "idx | R_orig      | R_corr      | Delta_R     | Z_orig      | "
               "Z_corr      | Delta_Z"
            << std::endl;
  std::cout << "----+-------------+-------------+-------------+-------------+--"
               "-----------+------------"
            << std::endl;

  double max_r_diff = 0.0, max_z_diff = 0.0;
  bool has_nan_orig = false, has_nan_corr = false;

  for (int i = 0; i < sizes.nZnT; ++i) {
    double delta_r = r_real_corr[i] - r_real_orig[i];
    double delta_z = z_real_corr[i] - z_real_orig[i];

    if (!std::isfinite(r_real_orig[i]) || !std::isfinite(z_real_orig[i])) {
      has_nan_orig = true;
    }
    if (!std::isfinite(r_real_corr[i]) || !std::isfinite(z_real_corr[i])) {
      has_nan_corr = true;
    }

    max_r_diff = std::max(max_r_diff, std::abs(delta_r));
    max_z_diff = std::max(max_z_diff, std::abs(delta_z));

    printf("%3d | %11.6f | %11.6f | %11.6f | %11.6f | %11.6f | %11.6f\n", i,
           r_real_orig[i], r_real_corr[i], delta_r, z_real_orig[i],
           z_real_corr[i], delta_z);
  }

  std::cout << "\n===== SUMMARY COMPARISON =====" << std::endl;
  std::cout << "Original implementation has NaN/Inf: "
            << (has_nan_orig ? "YES" : "NO") << std::endl;
  std::cout << "Corrected implementation has NaN/Inf: "
            << (has_nan_corr ? "YES" : "NO") << std::endl;
  std::cout << "Maximum R difference: " << max_r_diff << std::endl;
  std::cout << "Maximum Z difference: " << max_z_diff << std::endl;

  // Verify corrected implementation produces finite results
  EXPECT_FALSE(has_nan_corr)
      << "Corrected implementation should not produce NaN/Inf";

  for (int i = 0; i < sizes.nZnT; ++i) {
    EXPECT_TRUE(std::isfinite(r_real_corr[i]))
        << "r_real_corr[" << i << "] is not finite";
    EXPECT_TRUE(std::isfinite(z_real_corr[i]))
        << "z_real_corr[" << i << "] is not finite";
  }
}

}  // namespace vmecpp
