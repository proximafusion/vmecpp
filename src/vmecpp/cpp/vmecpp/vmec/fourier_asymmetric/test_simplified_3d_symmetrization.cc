// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
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

// Test SymmetrizeRealSpaceGeometry with 3D arrays matching ideal_mhd_model.cc
// pattern
class Simplified3DSymmetrizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test configuration matching ideal_mhd_model.cc usage
  }

  Sizes CreateTestSizes() {
    // Use minimal realistic parameters
    return Sizes(true, 1, 4, 2, 9,
                 4);  // lasym=true, nfp=1, mpol=4, ntor=2, ntheta=9, nzeta=4
  }
};

TEST_F(Simplified3DSymmetrizationTest, TestBasic3DArraySymmetrization) {
  std::cout << "\n=== SIMPLIFIED 3D ARRAY SYMMETRIZATION TEST ===\n";

  Sizes sizes = CreateTestSizes();

  std::cout << "Test parameters:\n";
  std::cout << "  mpol = " << sizes.mpol << ", ntor = " << sizes.ntor << "\n";
  std::cout << "  nThetaReduced = " << sizes.nThetaReduced
            << " ([0, π] range)\n";
  std::cout << "  nThetaEff = " << sizes.nThetaEff << " ([0, 2π] range)\n";
  std::cout << "  nZeta = " << sizes.nZeta << "\n\n";

  // Use minimal surface range matching ideal_mhd_model.cc pattern
  const int nsurfaces = 2;  // Two surfaces: j=0 (boundary) and j=1 (interior)
  const int reduced_size = sizes.nThetaReduced * sizes.nZeta * nsurfaces;
  const int full_size = sizes.nThetaEff * sizes.nZeta * nsurfaces;

  std::cout << "Array sizes:\n";
  std::cout << "  reduced_size = " << reduced_size
            << " (nThetaReduced * nZeta * nsurfaces)\n";
  std::cout << "  full_size = " << full_size
            << " (nThetaEff * nZeta * nsurfaces)\n\n";

  // Create separate arrays as SymmetrizeRealSpaceGeometry expects
  std::vector<double> r_sym(reduced_size);
  std::vector<double> r_asym(reduced_size);
  std::vector<double> z_sym(reduced_size);
  std::vector<double> z_asym(reduced_size);
  std::vector<double> lambda_sym(reduced_size);
  std::vector<double> lambda_asym(reduced_size);

  // Initialize with simple test pattern matching realistic physics
  for (int surface = 0; surface < nsurfaces; ++surface) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      for (int j = 0; j < sizes.nThetaReduced; ++j) {
        const int idx = j + k * sizes.nThetaReduced +
                        surface * (sizes.nThetaReduced * sizes.nZeta);

        // Realistic tokamak geometry
        const double major_radius = 3.0 - 0.5 * surface;  // R decreases inward
        const double minor_radius = 0.2;
        const double theta = M_PI * j / (sizes.nThetaReduced - 1);
        const double zeta = 2.0 * M_PI * k / sizes.nZeta;

        // Symmetric components (standard tokamak)
        r_sym[idx] = major_radius + minor_radius * cos(theta);
        z_sym[idx] = minor_radius * sin(theta);
        lambda_sym[idx] = 0.1 * sin(theta);

        // Small asymmetric perturbations
        r_asym[idx] = 0.02 * sin(theta) * cos(zeta);  // Up-down asymmetry
        z_asym[idx] = 0.01 * cos(theta) * sin(zeta);  // Left-right asymmetry
        lambda_asym[idx] = 0.005 * cos(theta);
      }
    }
  }

  // Output arrays matching ideal_mhd_model.cc pattern
  std::vector<double> r_full(full_size, 0.0);
  std::vector<double> z_full(full_size, 0.0);
  std::vector<double> lambda_full(full_size, 0.0);

  std::cout << "Calling SymmetrizeRealSpaceGeometry...\n";
  SymmetrizeRealSpaceGeometry(
      absl::MakeConstSpan(r_sym), absl::MakeConstSpan(r_asym),
      absl::MakeConstSpan(z_sym), absl::MakeConstSpan(z_asym),
      absl::MakeConstSpan(lambda_sym), absl::MakeConstSpan(lambda_asym),
      absl::MakeSpan(r_full), absl::MakeSpan(z_full),
      absl::MakeSpan(lambda_full), sizes);

  // Verify arrays are populated
  bool r_populated = false, z_populated = false;
  for (int i = 0; i < full_size; ++i) {
    if (std::abs(r_full[i]) > 1e-10) r_populated = true;
    if (std::abs(z_full[i]) > 1e-10) z_populated = true;
  }

  std::cout << "Result verification:\n";
  std::cout << "  r_full populated: " << (r_populated ? "YES" : "NO") << "\n";
  std::cout << "  z_full populated: " << (z_populated ? "YES" : "NO") << "\n\n";

  EXPECT_TRUE(r_populated) << "R full array should be populated";
  EXPECT_TRUE(z_populated) << "Z full array should be populated";

  // Verify symmetrization properties with 3D indexing
  std::cout << "Verifying symmetrization properties...\n";

  for (int surface = 0; surface < nsurfaces; ++surface) {
    for (int k = 0; k < std::min(2, sizes.nZeta); ++k) {
      for (int j = 0; j < std::min(2, sizes.nThetaReduced); ++j) {
        // Indices in separate arrays (3D: theta x zeta x surface)
        const int idx_sep = j + k * sizes.nThetaReduced +
                            surface * (sizes.nThetaReduced * sizes.nZeta);

        // Indices in full arrays (3D: theta x zeta x surface)
        const int idx_first =
            j + k * sizes.nThetaEff +
            surface * (sizes.nThetaEff * sizes.nZeta);  // [0, π] range
        const int idx_second =
            (j + sizes.nThetaReduced) + k * sizes.nThetaEff +
            surface * (sizes.nThetaEff * sizes.nZeta);  // [π, 2π] range

        if (idx_sep < reduced_size && idx_first < full_size &&
            idx_second < full_size) {
          // Expected values based on educational_VMEC pattern
          const double expected_first = r_sym[idx_sep] + r_asym[idx_sep];

          // For second half, use reflection
          const int j_reflected = sizes.nThetaReduced - 1 - j;
          const int idx_reflected =
              j_reflected + k * sizes.nThetaReduced +
              surface * (sizes.nThetaReduced * sizes.nZeta);

          double expected_second = 0.0;
          if (idx_reflected < reduced_size) {
            expected_second = r_sym[idx_reflected] - r_asym[idx_reflected];
          }

          std::cout << "  surface=" << surface << ", k=" << k << ", j=" << j
                    << ":\n";
          std::cout << "    r_sym[" << idx_sep << "] = " << r_sym[idx_sep]
                    << "\n";
          std::cout << "    r_asym[" << idx_sep << "] = " << r_asym[idx_sep]
                    << "\n";
          std::cout << "    r_full[" << idx_first << "] = " << r_full[idx_first]
                    << " (expected " << expected_first << ")\n";
          std::cout << "    r_full[" << idx_second
                    << "] = " << r_full[idx_second] << " (expected "
                    << expected_second << ")\n";

          // Verify symmetrization is correct
          EXPECT_NEAR(r_full[idx_first], expected_first, 1e-12)
              << "First half symmetrization at surface=" << surface
              << ", k=" << k << ", j=" << j;

          if (idx_reflected < reduced_size) {
            EXPECT_NEAR(r_full[idx_second], expected_second, 1e-12)
                << "Second half symmetrization at surface=" << surface
                << ", k=" << k << ", j=" << j;
          }
        }
      }
    }
  }

  std::cout << "\n✅ 3D SYMMETRIZATION TEST COMPLETE\n";
  std::cout
      << "SymmetrizeRealSpaceGeometry working correctly with 3D arrays\n\n";
}

}  // namespace vmecpp

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
