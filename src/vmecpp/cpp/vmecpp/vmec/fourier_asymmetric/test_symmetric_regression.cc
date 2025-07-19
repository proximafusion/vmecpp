// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

TEST(SymmetricRegressionTest, VerifyAsymmTransformConstantMode) {
  // Test that asymmetric transform handles constant mode correctly
  // NOTE: This tests the asymmetric transform function in isolation
  // The actual VMEC routing logic should use symmetric transforms for
  // lasym=false

  bool lasym = true;  // Use asymmetric case for this test
  int nfp = 1;
  int mpol = 2;
  int ntor = 1;
  int ntheta = 8;
  int nzeta = 4;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // Initialize arrays
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  // Set constant mode coefficient (m=0, n=0)
  rmncc[0] = 1.0;

  // Test asymmetric transform function
  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // For constant mode (m=0,n=0), no sqrt(2) normalization is applied
  // This is correct behavior - constant mode should give the coefficient value
  const double expected = 1.0;
  const double tolerance = 1e-10;

  for (int i = 0; i < sizes.nZnT; ++i) {
    EXPECT_NEAR(r_real[i], expected, tolerance)
        << "Constant mode failed at index " << i;
  }

  std::cout << "Asymmetric transform constant mode working correctly"
            << std::endl;
}

}  // namespace vmecpp
