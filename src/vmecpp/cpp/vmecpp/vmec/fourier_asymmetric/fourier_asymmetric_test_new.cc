// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Additional TDD tests for asymmetric Fourier transforms
// Following strict Test Driven Development principles

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
}  // namespace

// Test specific to jVMEC algorithm comparison
TEST(FourierAsymmetricTestNew, jVMECAlgorithmComparison) {
  // Test that exactly replicates jVMEC's asymmetric transform behavior
  // Based on the jVMEC analysis findings

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 1;
  int ntheta = 8;
  int nzeta = 4;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // Test case: R ~ sin(mu)cos(nv), Z ~ cos(mu)cos(nv) (asymmetric basis)
  // This should exactly match jVMEC's asymmetric coefficient usage

  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);  // R asymmetric: sin(mu)cos(nv)
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);  // Z asymmetric: cos(mu)cos(nv)
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set asymmetric coefficient for m=1, n=1 mode
  FourierBasisFastPoloidal fourier_basis(&sizes);
  int mn_target = -1;
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;
    if (m == 1 && n == 1) {
      mn_target = mn;
      break;
    }
  }

  ASSERT_NE(mn_target, -1) << "Could not find m=1, n=1 mode";

  // Set asymmetric coefficients matching jVMEC pattern
  rmnsc[mn_target] = 1.0;  // R ~ sin(mu)cos(nv)
  zmncc[mn_target] = 1.0;  // Z ~ cos(mu)cos(nv)

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Verify specific jVMEC patterns are followed
  // For asymmetric case, R should follow sin(mu)cos(nv) pattern
  // Z should follow cos(mu)cos(nv) pattern

  bool has_correct_pattern = false;
  for (int i = 0; i < sizes.nThetaEff; ++i) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      double u = 2.0 * PI * i / sizes.nThetaEff;
      double v = 2.0 * PI * k / sizes.nZeta;

      // Check if we have non-zero values that match expected pattern
      if (std::abs(r_real[idx]) > 1e-12 || std::abs(z_real[idx]) > 1e-12) {
        has_correct_pattern = true;

        // Expected pattern with sqrt(2) normalization
        double expected_r = sin(u) * cos(v) * sqrt(2.0);  // rmnsc basis
        double expected_z = cos(u) * cos(v) * sqrt(2.0);  // zmncc basis

        std::cout << "jVMEC pattern at u=" << u << ", v=" << v
                  << ": R=" << r_real[idx] << " (exp: " << expected_r << ")"
                  << ", Z=" << z_real[idx] << " (exp: " << expected_z << ")"
                  << std::endl;

        // This test will FAIL until the implementation exactly matches jVMEC
        EXPECT_NEAR(r_real[idx], expected_r, 1e-10);
        EXPECT_NEAR(z_real[idx], expected_z, 1e-10);
      }
    }
  }

  // Ensure we actually tested something
  EXPECT_TRUE(has_correct_pattern)
      << "No non-zero values found in transform output";
}

// Test for correct normalization factors based on jVMEC behavior
TEST(FourierAsymmetricTestNew, NormalizationFactorsJVMEC) {
  // Based on jVMEC analysis: verify normalization matches exactly
  // jVMEC uses only n ∈ [0, ntor], no negative n modes

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 1;
  int ntheta = 8;
  int nzeta = 8;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Test with m=1, n=1 mode to verify exact jVMEC normalization
  FourierBasisFastPoloidal fourier_basis(&sizes);
  int mn_test = -1;
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;
    if (m == 1 && n == 1) {
      mn_test = mn;
      break;
    }
  }

  ASSERT_NE(mn_test, -1) << "Could not find m=1, n=1 mode";

  // Set only asymmetric R coefficient: sin(mu)cos(nv)
  rmnsc[mn_test] = 1.0;

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Check normalization at specific points
  // For jVMEC compatibility, rmnsc coefficient should produce
  // sqrt(2)*sin(mu)cos(nv)
  int idx_test = 4;  // Some test index
  double u = 2.0 * PI * (idx_test / sizes.nZeta) / sizes.nThetaEff;
  double v = 2.0 * PI * (idx_test % sizes.nZeta) / sizes.nZeta;
  double expected = sqrt(2.0) * sin(u) * cos(v);

  std::cout << "Normalization test: u=" << u << ", v=" << v
            << ", R=" << r_real[idx_test] << ", expected=" << expected
            << std::endl;

  // This test verifies exact jVMEC normalization
  EXPECT_NEAR(r_real[idx_test], expected, 1e-10);
}

// Test for correct array combination as found in educational_VMEC
TEST(FourierAsymmetricTestNew, AsymmetricArrayCombination) {
  // Test that asymmetric and symmetric contributions are properly combined
  // Based on educational_VMEC pattern: r1s = r1s + r1a

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 0;  // 2D case for simplicity
  int ntheta = 8;
  int nzeta = 1;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set both symmetric and asymmetric coefficients for m=1
  rmncc[1] = 1.0;  // R symmetric: cos(mu)
  rmnsc[1] = 0.5;  // R asymmetric: sin(mu)
  zmnsc[1] = 1.0;  // Z symmetric: sin(mu)
  zmncc[1] = 0.5;  // Z asymmetric: cos(mu)

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Verify that both symmetric and asymmetric contributions appear
  // in the final result (proper array combination)
  for (int i = 0; i < sizes.nThetaEff; ++i) {
    double u = 2.0 * PI * i / sizes.nThetaEff;
    // Expected: R = cos(mu)*sqrt(2) + sin(mu)*sqrt(2)
    // Expected: Z = sin(mu)*sqrt(2) + cos(mu)*sqrt(2)
    double expected_r = (cos(u) + sin(u)) * sqrt(2.0);
    double expected_z = (sin(u) + cos(u)) * sqrt(2.0);

    std::cout << "Combination test u=" << u << ": R=" << r_real[i]
              << " (exp: " << expected_r << "), Z=" << z_real[i]
              << " (exp: " << expected_z << ")" << std::endl;

    // This test verifies proper combination - will FAIL if not implemented
    // correctly
    EXPECT_NEAR(r_real[i], expected_r, 1e-10);
    EXPECT_NEAR(z_real[i], expected_z, 1e-10);
  }
}

// Test for correct theta range handling [0, 2π]
TEST(FourierAsymmetricTestNew, FullThetaRangeHandling) {
  // Test that asymmetric transform properly handles full [0, 2π] range
  // Not just [0, π] with reflection

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 0;
  int ntheta = 8;  // Will give nThetaEff = 10
  int nzeta = 1;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set simple asymmetric mode
  rmnsc[1] = 1.0;  // R ~ sin(mu)

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Check values across full theta range [0, 2π]
  // Should be continuous sin(u) function, not reflected
  for (int i = 0; i < sizes.nThetaEff; ++i) {
    double u = 2.0 * PI * i / sizes.nThetaEff;
    double expected_r = sin(u) * sqrt(2.0);

    std::cout << "Full range u=" << u << " (" << i << "/" << sizes.nThetaEff
              << "): R=" << r_real[i] << ", expected=" << expected_r
              << std::endl;

    // This tests proper full-range computation - will FAIL if reflection is
    // wrong
    EXPECT_NEAR(r_real[i], expected_r, 1e-10);
  }

  // Specifically test that theta > π values are not just reflections
  int idx_3pi2 = 3 * sizes.nThetaEff / 4;  // u = 3π/2
  double u_3pi2 = 2.0 * PI * idx_3pi2 / sizes.nThetaEff;
  double expected_3pi2 = sin(u_3pi2) * sqrt(2.0);  // Should be negative

  EXPECT_LT(expected_3pi2, -0.5) << "Expected sin(3π/2) should be negative";
  EXPECT_NEAR(r_real[idx_3pi2], expected_3pi2, 1e-10);
}

// Test for coefficient indexing matching jVMEC exactly
TEST(FourierAsymmetricTestNew, CoefficientIndexingJVMECMatch) {
  // Test that coefficient arrays are used exactly as in jVMEC
  // This verifies the mapping: symmetric vs asymmetric coefficients

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 1;
  int ntheta = 4;
  int nzeta = 4;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // Test coefficient usage pattern matching jVMEC exactly
  std::vector<double> rmncc(sizes.mnmax, 0.0);  // R symmetric cos(mu)cos(nv)
  std::vector<double> rmnss(sizes.mnmax, 0.0);  // R symmetric sin(mu)sin(nv)
  std::vector<double> rmnsc(sizes.mnmax, 0.0);  // R asymmetric sin(mu)cos(nv)
  std::vector<double> rmncs(sizes.mnmax, 0.0);  // R asymmetric cos(mu)sin(nv)
  std::vector<double> zmnsc(sizes.mnmax, 0.0);  // Z symmetric sin(mu)cos(nv)
  std::vector<double> zmncs(sizes.mnmax, 0.0);  // Z symmetric cos(mu)sin(nv)
  std::vector<double> zmncc(sizes.mnmax, 0.0);  // Z asymmetric cos(mu)cos(nv)
  std::vector<double> zmnss(sizes.mnmax, 0.0);  // Z asymmetric sin(mu)sin(nv)

  // Set exactly one coefficient type to verify usage
  FourierBasisFastPoloidal fourier_basis(&sizes);
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;

    if (m == 1 && n == 1) {
      // Test R asymmetric sin(mu)cos(nv) coefficient usage
      rmnsc[mn] = 1.0;
      break;
    }
  }

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Verify that rmnsc[mn] coefficient produces sin(mu)cos(nv) pattern
  bool found_correct_pattern = false;
  for (int i = 0; i < sizes.nThetaEff; ++i) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      double u = 2.0 * PI * i / sizes.nThetaEff;
      double v = 2.0 * PI * k / sizes.nZeta;
      double expected = sin(u) * cos(v) * sqrt(2.0);

      if (std::abs(expected) > 1e-12) {
        found_correct_pattern = true;
        std::cout << "Coefficient mapping test: u=" << u << ", v=" << v
                  << ", R=" << r_real[idx]
                  << ", expected sin(u)cos(v)=" << expected << std::endl;

        // This verifies exact coefficient usage - will FAIL if mapping is wrong
        EXPECT_NEAR(r_real[idx], expected, 1e-10);
      }
    }
  }

  EXPECT_TRUE(found_correct_pattern)
      << "Did not find expected sin(mu)cos(nv) pattern";
}

}  // namespace vmecpp
