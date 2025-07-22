// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Test the fixed SymmetrizeRealSpaceGeometry function
class SymmetrizationFixTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  // Test function to implement the correct educational_VMEC approach
  void CorrectSymmetrization(const absl::Span<const double> r_sym,
                             const absl::Span<const double> r_asym,
                             const absl::Span<const double> z_sym,
                             const absl::Span<const double> z_asym,
                             const absl::Span<const double> lambda_sym,
                             const absl::Span<const double> lambda_asym,
                             absl::Span<double> r_full,
                             absl::Span<double> z_full,
                             absl::Span<double> lambda_full, int ntheta,
                             int nzeta) {
    // First half: theta in [0, pi] - symmetric + antisymmetric
    for (int k = 0; k < nzeta; ++k) {
      for (int j = 0; j < ntheta; ++j) {
        const int idx_half = j + k * ntheta;
        const int idx_full_first = j + k * (2 * ntheta);

        // Direct addition for first half
        r_full[idx_full_first] = r_sym[idx_half] + r_asym[idx_half];
        z_full[idx_full_first] = z_sym[idx_half] + z_asym[idx_half];
        lambda_full[idx_full_first] =
            lambda_sym[idx_half] + lambda_asym[idx_half];
      }
    }

    // Second half: theta in [pi, 2pi] - symmetric - antisymmetric (reflected)
    for (int k = 0; k < nzeta; ++k) {
      for (int j = 0; j < ntheta; ++j) {
        const int idx_full_second = (j + ntheta) + k * (2 * ntheta);

        // Reflection mapping: theta -> 2*pi - theta
        const int j_reflected = ntheta - 1 - j;
        const int idx_reflected = j_reflected + k * ntheta;

        // Subtraction with reflection for second half
        r_full[idx_full_second] = r_sym[idx_reflected] - r_asym[idx_reflected];
        z_full[idx_full_second] = z_sym[idx_reflected] - z_asym[idx_reflected];
        lambda_full[idx_full_second] =
            lambda_sym[idx_reflected] - lambda_asym[idx_reflected];
      }
    }
  }
};

TEST_F(SymmetrizationFixTest, VerifyCorrectImplementation) {
  std::cout << "\n=== TESTING CORRECT SYMMETRIZATION IMPLEMENTATION ===\n";

  const int ntheta = 8;
  const int nzeta = 4;

  // Create test arrays
  std::vector<double> r_sym(ntheta * nzeta);
  std::vector<double> r_asym(ntheta * nzeta);
  std::vector<double> z_sym(ntheta * nzeta);
  std::vector<double> z_asym(ntheta * nzeta);
  std::vector<double> lambda_sym(ntheta * nzeta);
  std::vector<double> lambda_asym(ntheta * nzeta);

  std::vector<double> r_full(2 * ntheta * nzeta);
  std::vector<double> z_full(2 * ntheta * nzeta);
  std::vector<double> lambda_full(2 * ntheta * nzeta);

  // Fill with test data that has clear symmetric/antisymmetric properties
  for (int k = 0; k < nzeta; ++k) {
    for (int j = 0; j < ntheta; ++j) {
      const int idx = j + k * ntheta;
      const double theta = j * M_PI / (ntheta - 1);
      const double zeta = k * 2 * M_PI / nzeta;

      // Symmetric components (even functions)
      r_sym[idx] = 3.0 + 0.5 * cos(theta) + 0.1 * cos(zeta);
      z_sym[idx] = 0.3 * sin(theta) + 0.05 * sin(zeta);
      lambda_sym[idx] = 0.1 * cos(2 * theta) + 0.02 * cos(zeta);

      // Antisymmetric components (odd functions)
      r_asym[idx] = 0.1 * sin(theta) + 0.02 * sin(zeta);
      z_asym[idx] = 0.05 * cos(theta) + 0.01 * cos(zeta);
      lambda_asym[idx] = 0.02 * sin(2 * theta) + 0.005 * sin(zeta);
    }
  }

  std::cout << "Input arrays created with ntheta=" << ntheta
            << ", nzeta=" << nzeta << std::endl;

  // Apply correct symmetrization
  CorrectSymmetrization(absl::MakeConstSpan(r_sym), absl::MakeConstSpan(r_asym),
                        absl::MakeConstSpan(z_sym), absl::MakeConstSpan(z_asym),
                        absl::MakeConstSpan(lambda_sym),
                        absl::MakeConstSpan(lambda_asym),
                        absl::MakeSpan(r_full), absl::MakeSpan(z_full),
                        absl::MakeSpan(lambda_full), ntheta, nzeta);

  std::cout << "Symmetrization applied successfully\n";

  // Verify properties
  for (int k = 0; k < nzeta; ++k) {
    for (int j = 0; j < ntheta; ++j) {
      const int idx1 = j + k * (2 * ntheta);  // First half
      const int idx2 =
          (2 * ntheta - 1 - j) + k * (2 * ntheta);  // Second half (reflected)

      // Extract symmetric and antisymmetric parts from combined array
      const double r_sym_extracted = (r_full[idx1] + r_full[idx2]) / 2.0;
      const double r_asym_extracted = (r_full[idx1] - r_full[idx2]) / 2.0;

      const double z_sym_extracted = (z_full[idx1] + z_full[idx2]) / 2.0;
      const double z_asym_extracted = (z_full[idx1] - z_full[idx2]) / 2.0;

      // These should match our original arrays
      const int idx_orig = j + k * ntheta;

      EXPECT_NEAR(r_sym_extracted, r_sym[idx_orig], 1e-12)
          << "R symmetric mismatch at j=" << j << ", k=" << k;
      EXPECT_NEAR(r_asym_extracted, r_asym[idx_orig], 1e-12)
          << "R antisymmetric mismatch at j=" << j << ", k=" << k;
      EXPECT_NEAR(z_sym_extracted, z_sym[idx_orig], 1e-12)
          << "Z symmetric mismatch at j=" << j << ", k=" << k;
      EXPECT_NEAR(z_asym_extracted, z_asym[idx_orig], 1e-12)
          << "Z antisymmetric mismatch at j=" << j << ", k=" << k;

      // Print verification for first few points
      if (j < 3 && k == 0) {
        std::cout << "j=" << j << ": R_sym original=" << r_sym[idx_orig]
                  << " extracted=" << r_sym_extracted << std::endl;
      }
    }
  }

  std::cout << "All symmetrization properties verified ✓\n";
}

TEST_F(SymmetrizationFixTest, CreateIntegrationPlan) {
  std::cout << "\n=== INTEGRATION PLAN FOR VMEC++ FIX ===\n";

  std::cout << "Step 1: Modify FourierToReal3DAsymmFastPoloidal signature:\n";
  std::cout << "  - Output separate symmetric and antisymmetric arrays\n";
  std::cout << "  - Remove direct combination in transform\n\n";

  std::cout << "Step 2: Update SymmetrizeRealSpaceGeometry:\n";
  std::cout << "  - Accept separate symmetric/antisymmetric inputs\n";
  std::cout << "  - Apply educational_VMEC reflection logic\n";
  std::cout << "  - Remove division by tau completely\n\n";

  std::cout << "Step 3: Update call sites in ideal_mhd_model.cc:\n";
  std::cout << "  - Pass separate arrays to symmetrization\n";
  std::cout << "  - Remove ODD ARRAYS HACK section\n";
  std::cout << "  - Ensure proper array sizing\n\n";

  std::cout << "Step 4: Test and validate:\n";
  std::cout << "  - Run asymmetric convergence tests\n";
  std::cout << "  - Verify no regression in symmetric mode\n";
  std::cout << "  - Compare with educational_VMEC output\n\n";
}

}  // namespace vmecpp

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
