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

// Test complete pipeline integration: separated transform + symmetrization
class PipelineIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test configuration matching realistic VMEC setup
  }

  Sizes CreateTestSizes() {
    // Use realistic parameters for asymmetric tokamak
    return Sizes(true, 1, 6, 3, 17,
                 8);  // lasym=true, nfp=1, mpol=6, ntor=3, ntheta=17, nzeta=8
  }
};

TEST_F(PipelineIntegrationTest, TestCompleteAsymmetricPipeline) {
  std::cout << "\n=== COMPLETE ASYMMETRIC PIPELINE INTEGRATION TEST ===\n";

  Sizes sizes = CreateTestSizes();

  std::cout << "Pipeline parameters:\n";
  std::cout << "  mpol = " << sizes.mpol << ", ntor = " << sizes.ntor << "\n";
  std::cout << "  nThetaReduced = " << sizes.nThetaReduced
            << " ([0, π] range)\n";
  std::cout << "  nThetaEff = " << sizes.nThetaEff << " ([0, 2π] range)\n";
  std::cout << "  nZeta = " << sizes.nZeta << "\n\n";

  // Create realistic test Fourier coefficients
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set realistic tokamak coefficients (major radius + asymmetric perturbation)
  // R symmetric: major radius (m=0,n=0) + ellipticity (m=2,n=0)
  // R antisymmetric: up-down asymmetry (m=1,n=0)
  // Z symmetric: sine component (m=1,n=0)
  // Z antisymmetric: up-down tilt (m=0,n=0), (m=2,n=0)

  // Find mode indices (educational_VMEC/jVMEC pattern)
  auto find_mode = [&](int m, int n) -> int {
    for (int mn = 0; mn < sizes.mnmax; ++mn) {
      // Mode index calculation: m * (2*ntor + 1) + (n + ntor)
      int expected_m = mn / (2 * sizes.ntor + 1);
      int expected_n = (mn % (2 * sizes.ntor + 1)) - sizes.ntor;
      if (expected_m == m && expected_n == n) {
        return mn;
      }
    }
    return -1;
  };

  // Set up tokamak-like coefficients
  int idx_m0n0 = find_mode(0, 0);
  int idx_m1n0 = find_mode(1, 0);
  int idx_m2n0 = find_mode(2, 0);

  if (idx_m0n0 >= 0) {
    rmncc[idx_m0n0] = 3.0;  // Major radius R0
    zmncc[idx_m0n0] = 0.1;  // Up-down asymmetric shift (antisymmetric)
  }
  if (idx_m1n0 >= 0) {
    zmnsc[idx_m1n0] = 1.0;  // Standard tokamak Z sine (symmetric)
    rmnsc[idx_m1n0] =
        0.05;  // Up-down asymmetric R perturbation (antisymmetric)
  }
  if (idx_m2n0 >= 0) {
    rmncc[idx_m2n0] = 0.3;   // Ellipticity (symmetric)
    zmncc[idx_m2n0] = 0.02;  // Up-down asymmetric Z component (antisymmetric)
  }

  std::cout << "Input coefficients set:\n";
  std::cout << "  Symmetric: rmncc[" << idx_m0n0 << "] = " << rmncc[idx_m0n0]
            << " (R major radius)\n";
  std::cout << "  Symmetric: zmnsc[" << idx_m1n0 << "] = " << zmnsc[idx_m1n0]
            << " (Z sine)\n";
  std::cout << "  Antisymmetric: rmnsc[" << idx_m1n0
            << "] = " << rmnsc[idx_m1n0] << " (R up-down asym)\n";
  std::cout << "  Antisymmetric: zmncc[" << idx_m0n0
            << "] = " << zmncc[idx_m0n0] << " (Z up-down shift)\n\n";

  // STEP 1: Apply separated asymmetric transform
  const int reduced_size = sizes.nThetaReduced * sizes.nZeta;
  const int full_size = sizes.nThetaEff * sizes.nZeta;

  std::vector<double> r_sym(reduced_size, 0.0);
  std::vector<double> r_asym(reduced_size, 0.0);
  std::vector<double> z_sym(reduced_size, 0.0);
  std::vector<double> z_asym(reduced_size, 0.0);
  std::vector<double> lambda_sym(reduced_size, 0.0);
  std::vector<double> lambda_asym(reduced_size, 0.0);

  std::cout
      << "STEP 1: Applying FourierToReal3DAsymmFastPoloidalSeparated...\n";
  FourierToReal3DAsymmFastPoloidalSeparated(
      sizes, absl::MakeConstSpan(rmncc), absl::MakeConstSpan(rmnss),
      absl::MakeConstSpan(rmnsc), absl::MakeConstSpan(rmncs),
      absl::MakeConstSpan(zmnsc), absl::MakeConstSpan(zmncs),
      absl::MakeConstSpan(zmncc), absl::MakeConstSpan(zmnss),
      absl::MakeSpan(r_sym), absl::MakeSpan(r_asym), absl::MakeSpan(z_sym),
      absl::MakeSpan(z_asym), absl::MakeSpan(lambda_sym),
      absl::MakeSpan(lambda_asym));

  // Verify separated arrays are populated
  bool r_sym_populated = false, r_asym_populated = false;
  bool z_sym_populated = false, z_asym_populated = false;

  for (int i = 0; i < reduced_size; ++i) {
    if (std::abs(r_sym[i]) > 1e-10) r_sym_populated = true;
    if (std::abs(r_asym[i]) > 1e-10) r_asym_populated = true;
    if (std::abs(z_sym[i]) > 1e-10) z_sym_populated = true;
    if (std::abs(z_asym[i]) > 1e-10) z_asym_populated = true;
  }

  std::cout << "Separated array population:\n";
  std::cout << "  r_sym populated: " << (r_sym_populated ? "YES" : "NO")
            << "\n";
  std::cout << "  r_asym populated: " << (r_asym_populated ? "YES" : "NO")
            << "\n";
  std::cout << "  z_sym populated: " << (z_sym_populated ? "YES" : "NO")
            << "\n";
  std::cout << "  z_asym populated: " << (z_asym_populated ? "YES" : "NO")
            << "\n\n";

  EXPECT_TRUE(r_sym_populated) << "R symmetric array should be populated";
  EXPECT_TRUE(r_asym_populated) << "R antisymmetric array should be populated";
  EXPECT_TRUE(z_sym_populated) << "Z symmetric array should be populated";
  EXPECT_TRUE(z_asym_populated) << "Z antisymmetric array should be populated";

  // STEP 2: Apply symmetrization with separated arrays
  std::vector<double> r_full(full_size, 0.0);
  std::vector<double> z_full(full_size, 0.0);
  std::vector<double> lambda_full(full_size, 0.0);

  std::cout << "STEP 2: Applying SymmetrizeRealSpaceGeometry...\n";
  SymmetrizeRealSpaceGeometry(
      absl::MakeConstSpan(r_sym), absl::MakeConstSpan(r_asym),
      absl::MakeConstSpan(z_sym), absl::MakeConstSpan(z_asym),
      absl::MakeConstSpan(lambda_sym), absl::MakeConstSpan(lambda_asym),
      absl::MakeSpan(r_full), absl::MakeSpan(z_full),
      absl::MakeSpan(lambda_full), sizes);

  // Verify combined arrays are populated
  bool r_full_populated = false, z_full_populated = false;
  for (int i = 0; i < full_size; ++i) {
    if (std::abs(r_full[i]) > 1e-10) r_full_populated = true;
    if (std::abs(z_full[i]) > 1e-10) z_full_populated = true;
  }

  std::cout << "Combined array population:\n";
  std::cout << "  r_full populated: " << (r_full_populated ? "YES" : "NO")
            << "\n";
  std::cout << "  z_full populated: " << (z_full_populated ? "YES" : "NO")
            << "\n\n";

  EXPECT_TRUE(r_full_populated) << "R full array should be populated";
  EXPECT_TRUE(z_full_populated) << "Z full array should be populated";

  // STEP 3: Verify educational_VMEC symmetrization properties
  std::cout
      << "STEP 3: Verifying educational_VMEC symmetrization properties...\n";

  const int ntheta = sizes.nThetaReduced;
  const int nzeta = sizes.nZeta;

  // Check first few points for symmetrization correctness
  for (int k = 0; k < std::min(2, nzeta); ++k) {
    for (int j = 0; j < std::min(3, ntheta); ++j) {
      // Indices in separate arrays (2D: theta x zeta)
      const int idx_sep = j + k * ntheta;

      // Indices in full arrays - match SymmetrizeRealSpaceGeometry indexing
      // exactly
      const int idx_first = j + k * sizes.nThetaEff;  // [0, π] range
      const int idx_second =
          (j + ntheta) + k * sizes.nThetaEff;  // [π, 2π] range

      if (idx_sep < reduced_size && idx_first < full_size &&
          idx_second < full_size) {
        // Expected values based on educational_VMEC pattern
        const double expected_first = r_sym[idx_sep] + r_asym[idx_sep];

        // For second half, we need the reflected index in the separate arrays
        // Following exact jVMEC/educational_VMEC zeta reflection pattern
        const int j_reflected = ntheta - 1 - j;
        const int k_reflected =
            (nzeta - k) % nzeta;  // Critical zeta reflection
        const int idx_reflected = j_reflected + k_reflected * ntheta;

        double expected_second = 0.0;
        if (idx_reflected < reduced_size) {
          expected_second = r_sym[idx_reflected] - r_asym[idx_reflected];
        }

        std::cout << "  k=" << k << ", j=" << j << " -> j_refl=" << j_reflected
                  << ", k_refl=" << k_reflected << ":\n";
        std::cout << "    r_sym[" << idx_sep << "] = " << r_sym[idx_sep]
                  << "\n";
        std::cout << "    r_asym[" << idx_sep << "] = " << r_asym[idx_sep]
                  << "\n";
        std::cout << "    r_full[" << idx_first << "] = " << r_full[idx_first]
                  << " (expected " << expected_first << ")\n";
        std::cout << "    r_sym[" << idx_reflected << "] = "
                  << (idx_reflected < reduced_size ? r_sym[idx_reflected] : 0.0)
                  << "\n";
        std::cout << "    r_asym[" << idx_reflected << "] = "
                  << (idx_reflected < reduced_size ? r_asym[idx_reflected]
                                                   : 0.0)
                  << "\n";
        std::cout << "    r_full[" << idx_second << "] = " << r_full[idx_second]
                  << " (expected " << expected_second << ")\n";

        // Allow some numerical tolerance
        EXPECT_NEAR(r_full[idx_first], expected_first, 1e-12)
            << "First half symmetrization mismatch at k=" << k << ", j=" << j;

        if (idx_reflected < reduced_size) {
          EXPECT_NEAR(r_full[idx_second], expected_second, 1e-12)
              << "Second half symmetrization mismatch at k=" << k
              << ", j=" << j;
        }
      }
    }
  }

  std::cout << "\n✅ PIPELINE INTEGRATION TEST COMPLETE\n";
  std::cout
      << "All steps executed successfully with proper array population\n\n";
}

}  // namespace vmecpp

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
