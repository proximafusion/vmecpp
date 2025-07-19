// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

class DebugCorrectedTransformTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create simple 2D asymmetric test case that matches jVMEC exactly
    // Use minimal parameters for detailed debugging

    std::cout << "===== SETTING UP DEBUG TEST =====" << std::endl;
    std::cout << "Test parameters:" << std::endl;
    std::cout << "  mpol=" << sizes.mpol << ", ntor=" << sizes.ntor
              << ", nfp=" << sizes.nfp << std::endl;
    std::cout << "  mnmax=" << sizes.mnmax << ", nZnT=" << sizes.nZnT
              << std::endl;
    std::cout << "  nThetaEff=" << sizes.nThetaEff
              << ", nThetaReduced=" << sizes.nThetaReduced << std::endl;
    std::cout << "  nZeta=" << sizes.nZeta << std::endl;
  }

  // Initialize sizes with simple 2D parameters
  Sizes sizes{true, 1, 3, 0,
              8,    1};  // lasym=true, nfp=1, mpol=3, ntor=0, ntheta=8, nzeta=1
};

TEST_F(DebugCorrectedTransformTest, TestMinimalAsymmetricTransform) {
  std::cout << "\n===== TESTING MINIMAL ASYMMETRIC TRANSFORM ====="
            << std::endl;

  // Create coefficient arrays
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);  // Asymmetric R
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);  // Asymmetric Z
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set up simple test coefficients
  // Mode (m=0,n=0): rmncc[0] = 1.0 (symmetric baseline)
  // Mode (m=1,n=0): rmnsc[1] = 0.1 (asymmetric R perturbation)
  //                 zmncc[1] = 0.05 (asymmetric Z perturbation)

  rmncc[0] = 1.0;   // (m=0,n=0) symmetric baseline R
  rmnsc[1] = 0.1;   // (m=1,n=0) asymmetric R perturbation
  zmncc[1] = 0.05;  // (m=1,n=0) asymmetric Z perturbation

  std::cout << "Input asymmetric coefficients:" << std::endl;
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    std::cout << "  mn=" << mn << ": rmnsc=" << rmnsc[mn]
              << ", zmncc=" << zmncc[mn] << std::endl;
  }

  // Create output arrays
  std::vector<double> r_real(sizes.nZnT, 0.0);
  std::vector<double> z_real(sizes.nZnT, 0.0);
  std::vector<double> lambda_real(sizes.nZnT, 0.0);

  // Call corrected implementation
  std::cout << "\n===== CALLING CORRECTED 2D TRANSFORM =====" << std::endl;
  FourierToReal2DAsymmFastPoloidal_Corrected(
      sizes, absl::Span<const double>(rmncc), absl::Span<const double>(rmnss),
      absl::Span<const double>(rmnsc), absl::Span<const double>(rmncs),
      absl::Span<const double>(zmnsc), absl::Span<const double>(zmncs),
      absl::Span<const double>(zmncc), absl::Span<const double>(zmnss),
      absl::Span<double>(r_real), absl::Span<double>(z_real),
      absl::Span<double>(lambda_real));

  std::cout << "\n===== TRANSFORM RESULTS =====" << std::endl;
  std::cout << "Output real space values:" << std::endl;
  for (int i = 0; i < sizes.nZnT; ++i) {
    std::cout << "  i=" << i << ": R=" << r_real[i] << ", Z=" << z_real[i]
              << std::endl;
  }

  // Verify results are finite and reasonable
  for (int i = 0; i < sizes.nZnT; ++i) {
    EXPECT_TRUE(std::isfinite(r_real[i]))
        << "r_real[" << i << "] is not finite";
    EXPECT_TRUE(std::isfinite(z_real[i]))
        << "z_real[" << i << "] is not finite";
  }

  std::cout << "===== TEST COMPLETED =====" << std::endl;
}

}  // namespace vmecpp
