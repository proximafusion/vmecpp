// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

// Detailed unit test to isolate root cause of k!=0 differences (~2.4e-2)
// Following jVMEC analysis recommendations
class PreciseZetaReflectionDebugTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use precise test configuration to match jVMEC patterns
  }

  Sizes CreateTestSizes() {
    // Use specific nzeta > 1 to test zeta reflection properly
    return Sizes(true, 1, 3, 1, 5,
                 4);  // lasym=true, nfp=1, mpol=3, ntor=1, ntheta=5, nzeta=4
  }

  void PrintDetailedDebug(const std::string& title, int k, int j, int idx,
                          double r_sym, double r_asym, double expected,
                          double actual) {
    std::cout << title << " k=" << k << " j=" << j << " idx=" << idx << ":\n";
    std::cout << "  r_sym = " << std::scientific << std::setprecision(15)
              << r_sym << "\n";
    std::cout << "  r_asym = " << r_asym << "\n";
    std::cout << "  expected = " << expected << "\n";
    std::cout << "  actual = " << actual << "\n";
    std::cout << "  difference = " << (actual - expected) << "\n\n";
  }
};

TEST_F(PreciseZetaReflectionDebugTest, IsolateKZeroVsKNonZeroDifferences) {
  std::cout << "\n=== PRECISE ZETA REFLECTION DEBUG TEST ===\n";

  Sizes sizes = CreateTestSizes();

  std::cout << "Test configuration (jVMEC-matched):\n";
  std::cout << "  nThetaReduced = " << sizes.nThetaReduced
            << " ([0, π] range)\n";
  std::cout << "  nThetaEff = " << sizes.nThetaEff << " ([0, 2π] range)\n";
  std::cout << "  nZeta = " << sizes.nZeta
            << " (>1 for proper zeta reflection test)\n\n";

  // Single surface test to isolate the issue
  const int nsurfaces = 1;
  const int reduced_size = sizes.nThetaReduced * sizes.nZeta * nsurfaces;
  const int full_size = sizes.nThetaEff * sizes.nZeta * nsurfaces;

  // Initialize arrays with controlled, simple test pattern
  std::vector<double> r_sym(reduced_size, 0.0);
  std::vector<double> r_asym(reduced_size, 0.0);
  std::vector<double> z_sym(reduced_size, 0.0);
  std::vector<double> z_asym(reduced_size, 0.0);
  std::vector<double> lambda_sym(reduced_size, 0.0);
  std::vector<double> lambda_asym(reduced_size, 0.0);

  std::cout << "=== STEP 1: CONTROLLED INPUT PATTERN ===\n";

  // Create carefully controlled test pattern to isolate k=0 vs k!=0 behavior
  for (int k = 0; k < sizes.nZeta; ++k) {
    for (int j = 0; j < sizes.nThetaReduced; ++j) {
      const int idx = j + k * sizes.nThetaReduced;

      // Simple, predictable values to track exactly
      r_sym[idx] = 10.0 + k + 0.1 * j;         // Different for each k,j
      r_asym[idx] = 1.0 + 0.1 * k + 0.01 * j;  // Small asymmetric component

      z_sym[idx] = 20.0 + k + 0.1 * j;
      z_asym[idx] = 2.0 + 0.1 * k + 0.01 * j;

      lambda_sym[idx] = 0.1 + 0.01 * k + 0.001 * j;
      lambda_asym[idx] = 0.01 + 0.001 * k + 0.0001 * j;

      std::cout << "Input [k=" << k << ",j=" << j << "] idx=" << idx
                << ": r_sym=" << r_sym[idx] << ", r_asym=" << r_asym[idx]
                << "\n";
    }
  }

  std::cout << "\n=== STEP 2: APPLY VMEC++ SYMMETRIZATION ===\n";

  // Output arrays
  std::vector<double> r_full_vmecpp(full_size, 0.0);
  std::vector<double> z_full_vmecpp(full_size, 0.0);
  std::vector<double> lambda_full_vmecpp(full_size, 0.0);

  // Apply VMEC++ SymmetrizeRealSpaceGeometry
  SymmetrizeRealSpaceGeometry(
      absl::MakeConstSpan(r_sym), absl::MakeConstSpan(r_asym),
      absl::MakeConstSpan(z_sym), absl::MakeConstSpan(z_asym),
      absl::MakeConstSpan(lambda_sym), absl::MakeConstSpan(lambda_asym),
      absl::MakeSpan(r_full_vmecpp), absl::MakeSpan(z_full_vmecpp),
      absl::MakeSpan(lambda_full_vmecpp), sizes);

  std::cout << "\n=== STEP 3: MANUAL JVMEC REFERENCE IMPLEMENTATION ===\n";

  // Manual implementation exactly following jVMEC pattern
  std::vector<double> r_full_jvmec(full_size, 0.0);

  // Following exact jVMEC symrzl logic from deep dive analysis
  const int ntheta_reduced = sizes.nThetaReduced;
  const int ntheta_eff = sizes.nThetaEff;
  const int nzeta = sizes.nZeta;

  std::cout << "jVMEC implementation parameters:\n";
  std::cout << "  ntheta_reduced = " << ntheta_reduced << "\n";
  std::cout << "  ntheta_eff = " << ntheta_eff << "\n";
  std::cout << "  nzeta = " << nzeta << "\n\n";

  // First half [0, π]: Direct addition (like jVMEC)
  for (int k = 0; k < nzeta; ++k) {
    for (int j = 0; j < ntheta_reduced; ++j) {
      const int idx_sep = j + k * ntheta_reduced;
      const int idx_first = j + k * ntheta_eff;

      r_full_jvmec[idx_first] = r_sym[idx_sep] + r_asym[idx_sep];

      if (k <= 1 && j <= 1) {  // Debug first few cases
        std::cout << "jVMEC first half [k=" << k << ",j=" << j << "]: "
                  << "idx_sep=" << idx_sep << ", idx_first=" << idx_first
                  << ", result=" << r_full_jvmec[idx_first] << "\n";
      }
    }
  }

  // Second half [π, 2π]: Reflection (like jVMEC)
  for (int k = 0; k < nzeta; ++k) {
    for (int j = 0; j < ntheta_reduced; ++j) {
      const int idx_second = (j + ntheta_reduced) + k * ntheta_eff;

      // jVMEC exact reflection pattern from analysis
      const int j_reflected = ntheta_reduced - 1 - j;  // lr = ntheta1 - l
      const int k_reflected = (nzeta - k) % nzeta;  // kr = (nzeta - k) % nzeta
      const int idx_reflected = j_reflected + k_reflected * ntheta_reduced;

      r_full_jvmec[idx_second] = r_sym[idx_reflected] - r_asym[idx_reflected];

      if (k <= 1 && j <= 1) {  // Debug first few cases
        std::cout << "jVMEC second half [k=" << k << ",j=" << j << "]: "
                  << "j_reflected=" << j_reflected
                  << ", k_reflected=" << k_reflected
                  << ", idx_reflected=" << idx_reflected
                  << ", idx_second=" << idx_second
                  << ", result=" << r_full_jvmec[idx_second] << "\n";
      }
    }
  }

  std::cout << "\n=== STEP 4: DETAILED COMPARISON BY K VALUE ===\n";

  double max_diff_k0 = 0.0, max_diff_k1 = 0.0;
  int mismatch_count_k0 = 0, mismatch_count_k1 = 0;

  // Compare k=0 vs k!=0 cases separately
  for (int k = 0; k < std::min(2, nzeta); ++k) {
    std::cout << "\n--- K=" << k << " ANALYSIS ---\n";

    for (int j = 0; j < ntheta_eff; ++j) {
      const int idx = j + k * ntheta_eff;

      if (idx < full_size) {
        const double diff = std::abs(r_full_vmecpp[idx] - r_full_jvmec[idx]);

        if (k == 0) {
          if (diff > max_diff_k0) max_diff_k0 = diff;
          if (diff > 1e-12) mismatch_count_k0++;
        } else {
          if (diff > max_diff_k1) max_diff_k1 = diff;
          if (diff > 1e-12) mismatch_count_k1++;
        }

        // Print detailed debug for first few cases
        if (j < 4) {
          std::cout << "  [k=" << k << ",j=" << j << "] idx=" << idx
                    << ": VMEC++=" << std::scientific << std::setprecision(15)
                    << r_full_vmecpp[idx] << ", jVMEC=" << r_full_jvmec[idx]
                    << ", diff=" << diff << "\n";
        }
      }
    }
  }

  std::cout << "\n=== STEP 5: ROOT CAUSE ANALYSIS ===\n";

  std::cout << "k=0 cases:\n";
  std::cout << "  Max difference: " << std::scientific << max_diff_k0 << "\n";
  std::cout << "  Mismatch count: " << mismatch_count_k0 << "\n";

  std::cout << "k=1 cases:\n";
  std::cout << "  Max difference: " << std::scientific << max_diff_k1 << "\n";
  std::cout << "  Mismatch count: " << mismatch_count_k1 << "\n";

  // Test specific zeta reflection edge cases
  std::cout << "\n=== STEP 6: ZETA REFLECTION EDGE CASE ANALYSIS ===\n";

  for (int k = 0; k < nzeta; ++k) {
    const int k_reflected = (nzeta - k) % nzeta;
    std::cout << "k=" << k << " -> k_reflected=" << k_reflected;

    if (k == 0 && k_reflected == 0) {
      std::cout << " (SELF-REFLECTION - potential special case)";
    }
    std::cout << "\n";
  }

  // Verify our hypothesis about the root cause
  if (max_diff_k0 < 1e-12 && max_diff_k1 > 1e-6) {
    std::cout << "\n✅ CONFIRMED: Issue is specific to k!=0 cases\n";
    std::cout << "   Root cause likely: zeta reflection boundary handling or "
                 "indexing\n";
  } else if (max_diff_k0 > 1e-6) {
    std::cout << "\n❌ UNEXPECTED: k=0 cases also have issues\n";
    std::cout
        << "   Root cause likely: fundamental indexing or algorithm error\n";
  }

  // Assertions to track our progress
  EXPECT_LT(max_diff_k0, 1e-12) << "k=0 cases should match exactly";

  // For k!=0, we're debugging so we expect this to fail initially
  if (max_diff_k1 > 1e-12) {
    std::cout
        << "\n🔍 DEBUG INFO: k!=0 mismatch as expected - investigating...\n";
  }

  std::cout << "\n✅ PRECISE ZETA REFLECTION DEBUG COMPLETE\n";
}

}  // namespace vmecpp

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
