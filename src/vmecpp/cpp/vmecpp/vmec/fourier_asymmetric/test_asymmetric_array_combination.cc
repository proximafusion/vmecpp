// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Test to understand the array combination logic for asymmetric equilibria
// This documents the exact issue seen with the "ODD ARRAYS HACK" in the debug
// output

class AsymmetricArrayCombinationTest : public ::testing::Test {
 protected:
  // Helper function to create test data
  void SetupTestArrays(int ntheta, int nzeta) {
    ntheta_ = ntheta;
    nzeta_ = nzeta;

    // Initialize symmetric and antisymmetric arrays
    symmetric_r_.resize(ntheta_ * nzeta_);
    antisymmetric_r_.resize(ntheta_ * nzeta_);
    combined_r_.resize(2 * ntheta_ * nzeta_);

    symmetric_z_.resize(ntheta_ * nzeta_);
    antisymmetric_z_.resize(ntheta_ * nzeta_);
    combined_z_.resize(2 * ntheta_ * nzeta_);
  }

  // Reproduce the educational_VMEC symrzl logic
  void CombineArraysEducationalVMEC() {
    // First half: theta in [0, pi]
    for (int k = 0; k < nzeta_; ++k) {
      for (int j = 0; j < ntheta_; ++j) {
        int idx_half = j + k * ntheta_;
        int idx_full = j + k * (2 * ntheta_);

        // R: symmetric + antisymmetric
        combined_r_[idx_full] =
            symmetric_r_[idx_half] + antisymmetric_r_[idx_half];

        // Z: symmetric + antisymmetric
        combined_z_[idx_full] =
            symmetric_z_[idx_half] + antisymmetric_z_[idx_half];
      }
    }

    // Second half: theta in [pi, 2pi]
    // Note: In educational_VMEC, this is done with index arithmetic
    for (int k = 0; k < nzeta_; ++k) {
      for (int j = 0; j < ntheta_; ++j) {
        int idx_half = j + k * ntheta_;
        int idx_full = (j + ntheta_) + k * (2 * ntheta_);

        // Map theta -> 2pi - theta for antisymmetric part
        int j_reflected = ntheta_ - 1 - j;
        int idx_reflected = j_reflected + k * ntheta_;

        // R: symmetric - antisymmetric (with reflection)
        combined_r_[idx_full] =
            symmetric_r_[idx_reflected] - antisymmetric_r_[idx_reflected];

        // Z: symmetric - antisymmetric (with reflection)
        combined_z_[idx_full] =
            symmetric_z_[idx_reflected] - antisymmetric_z_[idx_reflected];
      }
    }
  }

  // The problematic version that uses division by odd array values
  void CombineArraysWithOddHack() {
    // First half: theta in [0, pi]
    for (int k = 0; k < nzeta_; ++k) {
      for (int j = 0; j < ntheta_; ++j) {
        int idx_half = j + k * ntheta_;
        int idx_full = j + k * (2 * ntheta_);

        // Direct copy for first half
        combined_r_[idx_full] =
            symmetric_r_[idx_half] + antisymmetric_r_[idx_half];
        combined_z_[idx_full] =
            symmetric_z_[idx_half] + antisymmetric_z_[idx_half];
      }
    }

    // Second half with "ODD ARRAYS HACK"
    for (int k = 0; k < nzeta_; ++k) {
      for (int j = 0; j < ntheta_; ++j) {
        int idx_full_first = j + k * (2 * ntheta_);
        int idx_full_second = (2 * ntheta_ - 1 - j) + k * (2 * ntheta_);

        // The problematic division approach
        double r_first = combined_r_[idx_full_first];
        double z_first = combined_z_[idx_full_first];

        // This assumes:
        // r_first = r_sym + r_anti
        // r_second = r_sym - r_anti
        // Therefore: r_sym = (r_first + r_second) / 2
        //           r_anti = (r_first - r_second) / 2

        // But if we're trying to compute r_second from r_first...
        // We need the individual symmetric and antisymmetric components

        // The hack attempts to reconstruct these, but it's circular logic
        // if we don't have the original symmetric/antisymmetric arrays
      }
    }
  }

  int ntheta_;
  int nzeta_;
  std::vector<double> symmetric_r_, antisymmetric_r_, combined_r_;
  std::vector<double> symmetric_z_, antisymmetric_z_, combined_z_;
};

TEST_F(AsymmetricArrayCombinationTest, TestSymmetricPlusAntisymmetric) {
  // Test the basic combination for theta in [0, pi]
  SetupTestArrays(4, 2);  // Small test case

  // Fill with test data
  for (int k = 0; k < nzeta_; ++k) {
    for (int j = 0; j < ntheta_; ++j) {
      int idx = j + k * ntheta_;
      double theta = j * M_PI / (ntheta_ - 1);
      double zeta = k * 2 * M_PI / nzeta_;

      // Simple test functions
      symmetric_r_[idx] = cos(theta);
      antisymmetric_r_[idx] = sin(theta);

      symmetric_z_[idx] = sin(theta);
      antisymmetric_z_[idx] = cos(theta);
    }
  }

  CombineArraysEducationalVMEC();

  // Verify first half
  std::cout << "\nFirst half (theta in [0, pi]):\n";
  for (int k = 0; k < nzeta_; ++k) {
    for (int j = 0; j < ntheta_; ++j) {
      int idx_half = j + k * ntheta_;
      int idx_full = j + k * (2 * ntheta_);

      double expected_r = symmetric_r_[idx_half] + antisymmetric_r_[idx_half];
      double expected_z = symmetric_z_[idx_half] + antisymmetric_z_[idx_half];

      EXPECT_NEAR(combined_r_[idx_full], expected_r, 1e-12);
      EXPECT_NEAR(combined_z_[idx_full], expected_z, 1e-12);

      std::cout << "theta[" << j << "] R: " << combined_r_[idx_full]
                << " Z: " << combined_z_[idx_full] << std::endl;
    }
  }
}

TEST_F(AsymmetricArrayCombinationTest, TestReflectionAndSignChange) {
  // Test the reflection for theta in [pi, 2pi]
  SetupTestArrays(5, 1);  // Odd number of theta points for clarity

  // Fill with test data
  for (int j = 0; j < ntheta_; ++j) {
    double theta = j * M_PI / (ntheta_ - 1);

    // Use functions that make the reflection obvious
    symmetric_r_[j] = cos(theta);      // Even in theta
    antisymmetric_r_[j] = sin(theta);  // Odd in theta

    symmetric_z_[j] = sin(theta);  // Odd in theta (but symmetric for Z)
    antisymmetric_z_[j] =
        cos(theta);  // Even in theta (but antisymmetric for Z)
  }

  CombineArraysEducationalVMEC();

  // Verify second half with reflection
  std::cout << "\nSecond half (theta in [pi, 2pi]) with reflection:\n";
  for (int j = 0; j < ntheta_; ++j) {
    int idx_full_second = (j + ntheta_);
    int j_reflected = ntheta_ - 1 - j;

    double expected_r =
        symmetric_r_[j_reflected] - antisymmetric_r_[j_reflected];
    double expected_z =
        symmetric_z_[j_reflected] - antisymmetric_z_[j_reflected];

    std::cout << "theta[" << (j + ntheta_) << "] (from theta[" << j_reflected
              << "])"
              << " R: " << combined_r_[idx_full_second]
              << " (expected: " << expected_r << ")"
              << " Z: " << combined_z_[idx_full_second]
              << " (expected: " << expected_z << ")" << std::endl;

    EXPECT_NEAR(combined_r_[idx_full_second], expected_r, 1e-12);
    EXPECT_NEAR(combined_z_[idx_full_second], expected_z, 1e-12);
  }
}

TEST_F(AsymmetricArrayCombinationTest, DocumentOddArraysHackIssue) {
  // Document why the "ODD ARRAYS HACK" with division is problematic
  std::cout << "\n=== ODD ARRAYS HACK Issue Documentation ===\n";
  std::cout
      << "The issue occurs when trying to compute the second half of arrays\n";
  std::cout << "by dividing by values that should come from 'odd' "
               "(antisymmetric) arrays.\n\n";

  std::cout << "Problem:\n";
  std::cout << "1. We have combined arrays where:\n";
  std::cout << "   - First half: combined = symmetric + antisymmetric\n";
  std::cout << "   - Second half: combined = symmetric - antisymmetric "
               "(reflected)\n\n";

  std::cout
      << "2. The hack attempts to use tau/2 (from odd arrays) for division,\n";
  std::cout << "   but this creates several issues:\n";
  std::cout << "   - Division by values that might be zero or near-zero\n";
  std::cout
      << "   - Incorrect assumption about the relationship between arrays\n";
  std::cout << "   - Missing the proper reflection operation\n\n";

  std::cout << "3. The correct approach (from educational_VMEC):\n";
  std::cout << "   - Store symmetric and antisymmetric components separately\n";
  std::cout << "   - Apply proper reflection: theta -> 2*pi - theta\n";
  std::cout << "   - Combine with correct signs\n";
}

TEST_F(AsymmetricArrayCombinationTest, ProposedFix) {
  // Document the proposed fix based on educational_VMEC pattern
  std::cout << "\n=== Proposed Fix ===\n";
  std::cout << "Replace the division-based approach with the educational_VMEC "
               "method:\n\n";

  std::cout << "1. In FourierToReal3DAsymmFastPoloidal:\n";
  std::cout << "   - Keep symmetric and antisymmetric transforms separate\n";
  std::cout << "   - Don't combine them immediately\n\n";

  std::cout << "2. In SymmetrizeRealSpaceGeometry:\n";
  std::cout << "   - Take symmetric and antisymmetric arrays as input\n";
  std::cout << "   - For j in [0, ntheta):\n";
  std::cout << "     * combined[j] = symmetric[j] + antisymmetric[j]\n";
  std::cout
      << "     * combined[2*ntheta-1-j] = symmetric[j] - antisymmetric[j]\n\n";

  std::cout << "3. Key insight:\n";
  std::cout << "   - The 'tau/2' division is trying to extract antisymmetric "
               "components\n";
  std::cout
      << "   - But we should keep these components separate from the start\n";
  std::cout << "   - The reflection index is: j_reflected = ntheta - 1 - j\n\n";

  std::cout << "4. Specific code changes needed:\n";
  std::cout << "   a) Modify SymmetrizeRealSpaceGeometry signature:\n";
  std::cout << "      void SymmetrizeRealSpaceGeometry(\n";
  std::cout
      << "          absl::Span<double> r_sym, absl::Span<double> r_asym,\n";
  std::cout
      << "          absl::Span<double> z_sym, absl::Span<double> z_asym,\n";
  std::cout << "          absl::Span<double> lambda_sym, absl::Span<double> "
               "lambda_asym,\n";
  std::cout
      << "          absl::Span<double> r_full, absl::Span<double> z_full,\n";
  std::cout << "          absl::Span<double> lambda_full,\n";
  std::cout << "          const Sizes& sizes);\n\n";

  std::cout << "   b) Remove the division by tau/2 completely\n";
  std::cout << "   c) Use the educational_VMEC index mapping:\n";
  std::cout << "      - First half: direct addition\n";
  std::cout << "      - Second half: reflection with sign change\n";
}

TEST_F(AsymmetricArrayCombinationTest, TestCorrectImplementation) {
  // Demonstrate the correct implementation
  SetupTestArrays(8, 4);

  // Fill with realistic test data
  for (int k = 0; k < nzeta_; ++k) {
    for (int j = 0; j < ntheta_; ++j) {
      int idx = j + k * ntheta_;
      double theta = j * M_PI / (ntheta_ - 1);
      double zeta = k * 2 * M_PI / nzeta_;

      // Typical Fourier components
      symmetric_r_[idx] = 1.5 + 0.3 * cos(theta) + 0.1 * cos(zeta);
      antisymmetric_r_[idx] = 0.2 * sin(theta) + 0.05 * sin(zeta);

      symmetric_z_[idx] = 0.3 * sin(theta) + 0.1 * sin(zeta);
      antisymmetric_z_[idx] = 0.1 * cos(theta) + 0.02 * cos(zeta);
    }
  }

  CombineArraysEducationalVMEC();

  // Verify symmetry properties
  std::cout << "\nVerifying symmetry properties:\n";
  for (int k = 0; k < nzeta_; ++k) {
    for (int j = 0; j < ntheta_; ++j) {
      int idx1 = j + k * (2 * ntheta_);
      int idx2 = (2 * ntheta_ - 1 - j) + k * (2 * ntheta_);

      // Extract symmetric and antisymmetric parts from combined array
      double r_sym = (combined_r_[idx1] + combined_r_[idx2]) / 2.0;
      double r_anti = (combined_r_[idx1] - combined_r_[idx2]) / 2.0;

      double z_sym = (combined_z_[idx1] + combined_z_[idx2]) / 2.0;
      double z_anti = (combined_z_[idx1] - combined_z_[idx2]) / 2.0;

      // These should match our original symmetric arrays at j
      int idx_orig = j + k * ntheta_;

      if (j == 0) {  // Print first point for each toroidal angle
        std::cout << "k=" << k << ", j=" << j << ":\n";
        std::cout << "  R symmetric: extracted=" << r_sym
                  << " original=" << symmetric_r_[idx_orig] << "\n";
        std::cout << "  R antisymmetric: extracted=" << r_anti
                  << " original=" << antisymmetric_r_[idx_orig] << "\n";
      }

      EXPECT_NEAR(r_sym, symmetric_r_[idx_orig], 1e-12);
      EXPECT_NEAR(r_anti, antisymmetric_r_[idx_orig], 1e-12);
      EXPECT_NEAR(z_sym, symmetric_z_[idx_orig], 1e-12);
      EXPECT_NEAR(z_anti, antisymmetric_z_[idx_orig], 1e-12);
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
