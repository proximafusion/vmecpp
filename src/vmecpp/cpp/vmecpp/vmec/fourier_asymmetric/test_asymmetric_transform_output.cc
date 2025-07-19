// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "absl/types/span.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

// Test to examine asymmetric transform output at problematic theta positions
TEST(AsymmetricTransformOutputTest, DebugTransformAtProblematicTheta) {
  std::cout << "\n=== ASYMMETRIC TRANSFORM OUTPUT DEBUG ===\n" << std::endl;

  std::cout << "Examining R,Z values at theta positions kl=6,7,8,9 where NaN "
               "occurs..."
            << std::endl;

  // Use the same configuration that causes NaN
  VmecINDATA indata;

  indata.nfp = 1;
  indata.lasym = true;
  indata.mpol = 2;
  indata.ntor = 0;
  indata.ntheta = 10;  // Ensures kl=6-9 exist
  indata.nzeta = 1;

  // Simple tokamak with small asymmetric perturbation
  int coeff_size = indata.mpol * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  // Symmetric part
  indata.rbc[0] = 3.0;  // R00
  indata.rbc[1] = 1.0;  // R10
  indata.zbs[1] = 1.0;  // Z10

  // Asymmetric perturbation that causes NaN
  indata.rbs[1] = 0.001;
  indata.zbc[1] = 0.001;

  std::cout << "Coefficients:" << std::endl;
  std::cout << "  Symmetric: rbc[0]=" << indata.rbc[0]
            << ", rbc[1]=" << indata.rbc[1] << ", zbs[1]=" << indata.zbs[1]
            << std::endl;
  std::cout << "  Asymmetric: rbs[1]=" << indata.rbs[1]
            << ", zbc[1]=" << indata.zbc[1] << std::endl;

  std::cout << "\n1. ASYMMETRIC TRANSFORM OUTPUT (direct test):" << std::endl;
  {
    // Set up sizes for asymmetric transform
    Sizes sizes_asymm(true, 1, indata.mpol, indata.ntor, indata.ntheta,
                      indata.nzeta);

    // Create coefficient arrays for asymmetric transform
    // Following the signature: rmncc, rmnss, rmnsc, rmncs, zmnsc, zmncs, zmncc,
    // zmnss
    std::vector<double> rmncc(coeff_size), rmnss(coeff_size);
    std::vector<double> rmnsc(coeff_size), rmncs(coeff_size);
    std::vector<double> zmnsc(coeff_size), zmncs(coeff_size);
    std::vector<double> zmncc(coeff_size), zmnss(coeff_size);

    // Fill with coefficients
    for (int i = 0; i < coeff_size; ++i) {
      rmncc[i] = indata.rbc[i];  // Symmetric R cosine terms
      zmnss[i] = indata.zbs[i];  // Symmetric Z sine terms
      rmnsc[i] = indata.rbs[i];  // Asymmetric R sine terms
      zmnsc[i] = indata.zbc[i];  // Asymmetric Z cosine terms
      // Other arrays remain zero
      rmnss[i] = 0.0;
      rmncs[i] = 0.0;
      zmncs[i] = 0.0;
      zmncc[i] = 0.0;
    }

    // Output arrays
    std::vector<double> r_asymm(indata.ntheta * indata.nzeta);
    std::vector<double> z_asymm(indata.ntheta * indata.nzeta);
    std::vector<double> lambda_asymm(indata.ntheta * indata.nzeta);

    // Call asymmetric transform with correct signature
    FourierToReal3DAsymmFastPoloidal(
        sizes_asymm, rmncc, rmnss, rmnsc, rmncs,  // R coefficients
        zmnsc, zmncs, zmncc, zmnss,               // Z coefficients
        absl::MakeSpan(r_asymm), absl::MakeSpan(z_asymm),
        absl::MakeSpan(lambda_asymm));

    std::cout << "  Asymmetric R,Z values:" << std::endl;
    for (int i = 0; i < indata.ntheta; ++i) {
      double theta = 2.0 * M_PI * i / indata.ntheta;
      std::cout << "    i=" << i << " (theta=" << theta << "): R=" << r_asymm[i]
                << ", Z=" << z_asymm[i];

      // Check for problematic values
      if (i >= 6 && i <= 9) {
        std::cout << " ← PROBLEMATIC THETA POSITION";
        if (!std::isfinite(r_asymm[i]) || !std::isfinite(z_asymm[i])) {
          std::cout << " ⚠️ NON-FINITE VALUE DETECTED!";
        }
        if (r_asymm[i] <= 0.0) {
          std::cout << " ⚠️ NEGATIVE/ZERO R VALUE!";
        }
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\n2. ANALYSIS:" << std::endl;
  std::cout
      << "✅ If asymmetric transform produces finite R,Z values at kl=6-9:"
      << std::endl;
  std::cout << "   → Issue is NOT in the transform itself" << std::endl;
  std::cout
      << "   → Issue is in geometry derivative calculations (ru12, zu12, tau)"
      << std::endl;
  std::cout << "   → Need to examine how these R,Z values are processed"
            << std::endl;

  std::cout << "\n❌ If asymmetric transform produces NaN/invalid R,Z values:"
            << std::endl;
  std::cout << "   → Issue IS in the transform implementation" << std::endl;
  std::cout << "   → Need to compare with jVMEC transform algorithm"
            << std::endl;
  std::cout << "   → Asymmetric coefficient handling is wrong" << std::endl;

  std::cout << "\n⚠️  If R approaches zero or becomes negative:" << std::endl;
  std::cout
      << "   → This would cause division by zero in subsequent calculations"
      << std::endl;
  std::cout << "   → May explain NaN in tau, gsqrt calculations" << std::endl;
  std::cout << "   → Need better boundary conditions or coefficient validation"
            << std::endl;

  // This test is for analysis - always passes
  EXPECT_TRUE(true) << "Analysis test completed";
}

// Test to validate individual coefficient contributions
TEST(AsymmetricTransformOutputTest, TestCoefficientContributions) {
  std::cout << "\n=== COEFFICIENT CONTRIBUTION ANALYSIS ===\n" << std::endl;

  std::cout << "Testing individual asymmetric coefficient contributions..."
            << std::endl;

  // Test with only asymmetric coefficients (no symmetric baseline)
  Sizes sizes(true, 1, 2, 0, 10, 1);

  int coeff_size = sizes.mpol * (2 * sizes.ntor + 1);

  std::cout << "\n1. PURE ASYMMETRIC R CONTRIBUTION (rbs[1] only):"
            << std::endl;
  {
    std::vector<double> rmncc(coeff_size, 0.0), rmnss(coeff_size, 0.0);
    std::vector<double> rmnsc(coeff_size, 0.0), rmncs(coeff_size, 0.0);
    std::vector<double> zmnsc(coeff_size, 0.0), zmncs(coeff_size, 0.0);
    std::vector<double> zmncc(coeff_size, 0.0), zmnss(coeff_size, 0.0);

    rmnsc[1] = 0.001;  // Only asymmetric R contribution

    std::vector<double> r(sizes.nThetaEff), z(sizes.nThetaEff),
        lambda(sizes.nThetaEff);
    FourierToReal3DAsymmFastPoloidal(sizes, rmncc, rmnss, rmnsc, rmncs, zmnsc,
                                     zmncs, zmncc, zmnss, absl::MakeSpan(r),
                                     absl::MakeSpan(z), absl::MakeSpan(lambda));

    for (int i = 6; i <= 9; ++i) {
      std::cout << "    i=" << i << ": R=" << r[i] << ", Z=" << z[i]
                << std::endl;
    }
  }

  std::cout << "\n2. PURE ASYMMETRIC Z CONTRIBUTION (zbc[1] only):"
            << std::endl;
  {
    std::vector<double> rmncc(coeff_size, 0.0), rmnss(coeff_size, 0.0);
    std::vector<double> rmnsc(coeff_size, 0.0), rmncs(coeff_size, 0.0);
    std::vector<double> zmnsc(coeff_size, 0.0), zmncs(coeff_size, 0.0);
    std::vector<double> zmncc(coeff_size, 0.0), zmnss(coeff_size, 0.0);

    zmnsc[1] = 0.001;  // Only asymmetric Z contribution

    std::vector<double> r(sizes.nThetaEff), z(sizes.nThetaEff),
        lambda(sizes.nThetaEff);
    FourierToReal3DAsymmFastPoloidal(sizes, rmncc, rmnss, rmnsc, rmncs, zmnsc,
                                     zmncs, zmncc, zmnss, absl::MakeSpan(r),
                                     absl::MakeSpan(z), absl::MakeSpan(lambda));

    for (int i = 6; i <= 9; ++i) {
      std::cout << "    i=" << i << ": R=" << r[i] << ", Z=" << z[i]
                << std::endl;
    }
  }

  std::cout << "\n3. COMBINED CONTRIBUTIONS:" << std::endl;
  {
    std::vector<double> rmncc(coeff_size, 0.0), rmnss(coeff_size, 0.0);
    std::vector<double> rmnsc(coeff_size, 0.0), rmncs(coeff_size, 0.0);
    std::vector<double> zmnsc(coeff_size, 0.0), zmncs(coeff_size, 0.0);
    std::vector<double> zmncc(coeff_size, 0.0), zmnss(coeff_size, 0.0);

    // Symmetric baseline
    rmncc[0] = 3.0;  // R00
    rmncc[1] = 1.0;  // R10
    zmnss[1] = 1.0;  // Z10

    // Asymmetric contributions
    rmnsc[1] = 0.001;  // rbs[1]
    zmnsc[1] = 0.001;  // zbc[1]

    std::vector<double> r(sizes.nThetaEff), z(sizes.nThetaEff),
        lambda(sizes.nThetaEff);
    FourierToReal3DAsymmFastPoloidal(sizes, rmncc, rmnss, rmnsc, rmncs, zmnsc,
                                     zmncs, zmncc, zmnss, absl::MakeSpan(r),
                                     absl::MakeSpan(z), absl::MakeSpan(lambda));

    for (int i = 6; i <= 9; ++i) {
      std::cout << "    i=" << i << ": R=" << r[i] << ", Z=" << z[i];
      if (!std::isfinite(r[i]) || !std::isfinite(z[i])) {
        std::cout << " ⚠️ NON-FINITE!";
      }
      if (r[i] <= 0.0) {
        std::cout << " ⚠️ NON-POSITIVE R!";
      }
      std::cout << std::endl;
    }
  }

  EXPECT_TRUE(true) << "Coefficient analysis completed";
}

}  // namespace vmecpp
