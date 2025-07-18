// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iostream>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

TEST(MinimalDebugTest, TestAsymmetricTransformDirect) {
  // Create a minimal asymmetric configuration
  Sizes sizes_asym(true, 1, 3, 2, 16, 16);  // Simple tokamak-like, lasym=true

  std::cout << "Asymmetric sizes: mnmax=" << sizes_asym.mnmax
            << ", nThetaEff=" << sizes_asym.nThetaEff
            << ", nZnT=" << sizes_asym.nZnT << std::endl;

  // Create minimal coefficient arrays
  std::vector<double> rmncc(sizes_asym.mnmax, 0.0);
  std::vector<double> rmnss(sizes_asym.mnmax, 0.0);
  std::vector<double> rmnsc(sizes_asym.mnmax, 0.0);
  std::vector<double> rmncs(sizes_asym.mnmax, 0.0);

  std::vector<double> zmnsc(sizes_asym.mnmax, 0.0);
  std::vector<double> zmncs(sizes_asym.mnmax, 0.0);
  std::vector<double> zmncc(sizes_asym.mnmax, 0.0);
  std::vector<double> zmnss(sizes_asym.mnmax, 0.0);

  std::vector<double> lmnsc(sizes_asym.mnmax, 0.0);
  std::vector<double> lmncs(sizes_asym.mnmax, 0.0);
  std::vector<double> lmncc(sizes_asym.mnmax, 0.0);
  std::vector<double> lmnss(sizes_asym.mnmax, 0.0);

  // Set simple test values
  rmncc[0] = 1.0;  // R00
  if (sizes_asym.mnmax > 1) {
    zmnsc[1] = 0.3;  // Z10
  }

  // Create real space arrays
  std::vector<double> r_real(sizes_asym.nZnT, 0.0);
  std::vector<double> z_real(sizes_asym.nZnT, 0.0);
  std::vector<double> lambda_real(sizes_asym.nZnT, 0.0);

  std::cout << "Calling FourierToReal3DAsymmFastPoloidal..." << std::endl;

  // This should not crash
  try {
    FourierToReal3DAsymmFastPoloidal(
        sizes_asym, absl::Span<const double>(rmncc.data(), rmncc.size()),
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

    std::cout << "Transform completed successfully!" << std::endl;
    std::cout << "First few R values: " << r_real[0] << ", " << r_real[1]
              << ", " << r_real[2] << std::endl;

  } catch (const std::exception& e) {
    std::cout << "Transform failed with exception: " << e.what() << std::endl;
    FAIL() << "Transform should not throw exception";
  }
}

}  // namespace vmecpp
