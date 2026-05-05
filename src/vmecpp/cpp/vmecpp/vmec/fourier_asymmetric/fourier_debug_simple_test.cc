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

namespace {
const double PI = 3.14159265358979323846;
}

TEST(FourierDebugSimpleTest, TestFirstHalfThetaOnly) {
  // Test cosine mode but only check the first half of theta range [0,pi]
  // This bypasses the problematic reflection logic

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 0;
  int ntheta = 8;
  int nzeta = 1;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "nThetaEff=" << sizes.nThetaEff
            << ", nThetaReduced=" << sizes.nThetaReduced << std::endl;

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
  std::vector<double> ru_real(sizes.nZnT);
  std::vector<double> zu_real(sizes.nZnT);

  // Set cosine mode (m=1, n=0)
  rmncc[1] = 1.0;  // Assuming mode index 1 is (m=1,n=0)

  // Transform
  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Check only the first half: theta=[0,pi]
  std::cout << "Checking first half theta=[0,pi]:" << std::endl;
  for (int i = 0; i < sizes.nThetaReduced; ++i) {
    double u = 2.0 * PI * i / sizes.nThetaEff;
    int idx = i * sizes.nZeta;

    double expected = cos(u) * sqrt(2.0);  // With normalization

    std::cout << "i=" << i << ", u=" << u << ", cos(u)=" << cos(u)
              << ", expected=" << expected << ", actual=" << r_real[idx]
              << ", diff=" << (r_real[idx] - expected) << std::endl;

    EXPECT_NEAR(r_real[idx], expected, 1e-10);
  }

  std::cout << "\nChecking second half theta=[pi,2pi]:" << std::endl;
  for (int i = sizes.nThetaReduced; i < sizes.nThetaEff; ++i) {
    double u = 2.0 * PI * i / sizes.nThetaEff;
    int idx = i * sizes.nZeta;

    double expected = cos(u) * sqrt(2.0);  // With normalization

    std::cout << "i=" << i << ", u=" << u << ", cos(u)=" << cos(u)
              << ", expected=" << expected << ", actual=" << r_real[idx]
              << ", diff=" << (r_real[idx] - expected) << std::endl;

    // Don't assert here, just observe what happens
  }
}

TEST(FourierDebugSimpleTest, TestBasisFunctionNormalization) {
  // Test what the FourierBasisFastPoloidal produces for cos(u)

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 0;
  int ntheta = 8;
  int nzeta = 1;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastPoloidal fourier_basis(&sizes);

  std::cout << "Basis function values for m=1:" << std::endl;
  for (int l = 0; l < sizes.nThetaReduced; ++l) {
    double cosmu = fourier_basis.cosmu[1 * sizes.nThetaReduced + l];
    double sinmu = fourier_basis.sinmu[1 * sizes.nThetaReduced + l];

    double u = 2.0 * PI * l / sizes.nThetaEff;
    double expected_cos = cos(u);
    double expected_sin = sin(u);

    std::cout << "l=" << l << ", u=" << u << ", basis_cos=" << cosmu
              << ", expected_cos=" << expected_cos << ", basis_sin=" << sinmu
              << ", expected_sin=" << expected_sin << std::endl;
  }
}

}  // namespace vmecpp
