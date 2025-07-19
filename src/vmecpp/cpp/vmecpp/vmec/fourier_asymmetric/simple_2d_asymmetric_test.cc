// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

TEST(Simple2DAsymmetricTest, TestBasic2DTransform) {
  // Test the 2D asymmetric transform with minimal setup
  std::cout << "Testing basic 2D asymmetric transform..." << std::endl;

  // Create simple 2D asymmetric configuration (tokamak: ntor=0)
  bool lasym = true;
  int nfp = 1;
  int mpol = 5;
  int ntor = 0;  // Axisymmetric
  int ntheta = 16;
  int nzeta = 1;  // Axisymmetric -> single zeta point

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "Sizes: mnmax=" << sizes.mnmax
            << ", nThetaEff=" << sizes.nThetaEff << ", nZnT=" << sizes.nZnT
            << std::endl;

  // Create coefficient arrays
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);  // Asymmetric
  std::vector<double> rmncs(sizes.mnmax, 0.0);

  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);  // Asymmetric
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  std::vector<double> lmnsc(sizes.mnmax, 0.0);
  std::vector<double> lmncs(sizes.mnmax, 0.0);
  std::vector<double> lmncc(sizes.mnmax, 0.0);
  std::vector<double> lmnss(sizes.mnmax, 0.0);

  // Set up simple up-down asymmetric tokamak
  // R = R0 + R1*cos(θ) + R1_asym*sin(θ)
  // Z = Z1*sin(θ) + Z1_asym*cos(θ)
  rmncc[0] = 6.0;   // R(0,0) = R0 (major radius)
  rmncc[1] = 0.6;   // R(1,0) = R1 (minor radius, symmetric)
  rmnsc[1] = 0.6;   // R(1,0) asymmetric component
  rmnsc[2] = 0.12;  // R(2,0) asymmetric component

  zmnsc[1] = 0.6;  // Z(1,0) symmetric component
  zmncc[1] = 0.1;  // Z(1,0) asymmetric component (small)

  // Create real space arrays
  std::vector<double> r_real(sizes.nZnT, 0.0);
  std::vector<double> z_real(sizes.nZnT, 0.0);
  std::vector<double> lambda_real(sizes.nZnT, 0.0);

  std::cout << "Input coefficients:" << std::endl;
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    std::cout << "  mn=" << mn << ": rmncc=" << rmncc[mn]
              << ", rmnsc=" << rmnsc[mn] << ", zmnsc=" << zmnsc[mn]
              << ", zmncc=" << zmncc[mn] << std::endl;
  }

  // Test the forward transform
  std::cout << "Calling FourierToReal2DAsymmFastPoloidal..." << std::endl;

  try {
    FourierToReal2DAsymmFastPoloidal(
        sizes, absl::Span<const double>(rmncc.data(), rmncc.size()),
        absl::Span<const double>(rmnss.data(), rmnss.size()),
        absl::Span<const double>(rmnsc.data(), rmnsc.size()),
        absl::Span<const double>(rmncs.data(), rmncs.size()),
        absl::Span<const double>(zmnsc.data(), zmnsc.size()),
        absl::Span<const double>(zmncs.data(), zmncs.size()),
        absl::Span<const double>(zmncc.data(), zmncc.size()),
        absl::Span<const double>(zmnss.data(), zmnss.size()),
        absl::Span<double>(r_real.data(), r_real.size()),
        absl::Span<double>(z_real.data(), z_real.size()),
        absl::Span<double>(lambda_real.data(), lambda_real.size()));

    std::cout << "Forward transform completed successfully!" << std::endl;

    // Check for NaN values
    bool has_nan = false;
    for (int i = 0; i < sizes.nZnT; ++i) {
      if (std::isnan(r_real[i]) || std::isnan(z_real[i]) ||
          std::isnan(lambda_real[i])) {
        has_nan = true;
        std::cout << "NaN detected at i=" << i << ": R=" << r_real[i]
                  << ", Z=" << z_real[i] << ", L=" << lambda_real[i]
                  << std::endl;
      }
    }

    EXPECT_FALSE(has_nan) << "Forward transform produced NaN values";

    // Print some results
    std::cout << "Real space results:" << std::endl;
    for (int i = 0; i < std::min(8, sizes.nZnT); ++i) {
      double theta = 2.0 * M_PI * i / sizes.nThetaEff;
      std::cout << "  i=" << i << ", theta=" << theta << ": R=" << r_real[i]
                << ", Z=" << z_real[i] << std::endl;
    }

    // Test the inverse transform
    std::cout << "Testing inverse transform..." << std::endl;

    std::vector<double> rmncc_recov(sizes.mnmax, 0.0);
    std::vector<double> rmnss_recov(sizes.mnmax, 0.0);
    std::vector<double> rmnsc_recov(sizes.mnmax, 0.0);
    std::vector<double> rmncs_recov(sizes.mnmax, 0.0);
    std::vector<double> zmnsc_recov(sizes.mnmax, 0.0);
    std::vector<double> zmncs_recov(sizes.mnmax, 0.0);
    std::vector<double> zmncc_recov(sizes.mnmax, 0.0);
    std::vector<double> zmnss_recov(sizes.mnmax, 0.0);

    RealToFourier2DAsymmFastPoloidal(
        sizes, absl::Span<const double>(r_real.data(), r_real.size()),
        absl::Span<const double>(z_real.data(), z_real.size()),
        absl::Span<const double>(lambda_real.data(), lambda_real.size()),
        absl::Span<double>(rmncc_recov.data(), rmncc_recov.size()),
        absl::Span<double>(rmnss_recov.data(), rmnss_recov.size()),
        absl::Span<double>(rmnsc_recov.data(), rmnsc_recov.size()),
        absl::Span<double>(rmncs_recov.data(), rmncs_recov.size()),
        absl::Span<double>(zmnsc_recov.data(), zmnsc_recov.size()),
        absl::Span<double>(zmncs_recov.data(), zmncs_recov.size()),
        absl::Span<double>(zmncc_recov.data(), zmncc_recov.size()),
        absl::Span<double>(zmnss_recov.data(), zmnss_recov.size()),
        absl::Span<double>(lmnsc.data(), lmnsc.size()),
        absl::Span<double>(lmncs.data(), lmncs.size()),
        absl::Span<double>(lmncc.data(), lmncc.size()),
        absl::Span<double>(lmnss.data(), lmnss.size()));

    std::cout << "Inverse transform completed!" << std::endl;

    // Check round-trip accuracy
    std::cout << "Round-trip comparison:" << std::endl;
    for (int mn = 0; mn < sizes.mnmax; ++mn) {
      std::cout << "  mn=" << mn << ": rmncc " << rmncc[mn] << " -> "
                << rmncc_recov[mn] << ", rmnsc " << rmnsc[mn] << " -> "
                << rmnsc_recov[mn] << std::endl;
    }

  } catch (const std::exception& e) {
    std::cout << "Transform failed with exception: " << e.what() << std::endl;
    FAIL() << "Transform should not throw exception";
  }

  std::cout << "2D asymmetric transform test completed" << std::endl;
}

TEST(Simple2DAsymmetricTest, TestSymmetrizationOnly) {
  // Test just the symmetrization function
  std::cout << "Testing symmetrization function..." << std::endl;

  bool lasym = true;
  int nfp = 1;
  int mpol = 5;
  int ntor = 0;
  int ntheta = 16;
  int nzeta = 1;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // Create simple real space arrays
  std::vector<double> r_real(sizes.nZnT, 0.0);
  std::vector<double> z_real(sizes.nZnT, 0.0);
  std::vector<double> lambda_real(sizes.nZnT, 0.0);

  // Fill with a simple pattern
  for (int i = 0; i < sizes.nZnT; ++i) {
    double theta = 2.0 * M_PI * i / sizes.nThetaEff;
    r_real[i] = 6.0 + 0.6 * cos(theta);  // Simple tokamak
    z_real[i] = 0.6 * sin(theta);
    lambda_real[i] = 0.0;
  }

  std::cout << "Before symmetrization:" << std::endl;
  for (int i = 0; i < std::min(8, sizes.nZnT); ++i) {
    std::cout << "  i=" << i << ": R=" << r_real[i] << ", Z=" << z_real[i]
              << std::endl;
  }

  // Test symmetrization
  try {
    SymmetrizeRealSpaceGeometry(
        sizes, absl::Span<double>(r_real.data(), r_real.size()),
        absl::Span<double>(z_real.data(), z_real.size()),
        absl::Span<double>(lambda_real.data(), lambda_real.size()));

    std::cout << "Symmetrization completed!" << std::endl;

    std::cout << "After symmetrization:" << std::endl;
    for (int i = 0; i < std::min(8, sizes.nZnT); ++i) {
      std::cout << "  i=" << i << ": R=" << r_real[i] << ", Z=" << z_real[i]
                << std::endl;
    }

    // Check for NaN values
    bool has_nan = false;
    for (int i = 0; i < sizes.nZnT; ++i) {
      if (std::isnan(r_real[i]) || std::isnan(z_real[i]) ||
          std::isnan(lambda_real[i])) {
        has_nan = true;
        std::cout << "NaN detected at i=" << i << ": R=" << r_real[i]
                  << ", Z=" << z_real[i] << ", L=" << lambda_real[i]
                  << std::endl;
      }
    }

    EXPECT_FALSE(has_nan) << "Symmetrization produced NaN values";

  } catch (const std::exception& e) {
    std::cout << "Symmetrization failed with exception: " << e.what()
              << std::endl;
    FAIL() << "Symmetrization should not throw exception";
  }

  std::cout << "Symmetrization test completed" << std::endl;
}

}  // namespace vmecpp
