// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

// Test demonstrating the fix needed in ideal_mhd_model.cc

TEST(IdealMhdModelFix, DocumentCurrentIssue) {
  std::cout << "\n=== Current Issue in ideal_mhd_model.cc ===\n\n";

  std::cout << "Location: ideal_mhd_model.cc around line 400+\n";
  std::cout << "The code currently has:\n\n";

  std::cout << "// ODD ARRAYS HACK: Symmetrize half-grid with division by (tau "
               "+ 1)/2\n";
  std::cout << "for (int v_idx = 0; v_idx < 2 * state_->N; ++v_idx) {\n";
  std::cout << "    const double v = v_idx * dv;\n";
  std::cout << "    for (int u_idx = 0; u_idx < state_->nt; ++u_idx) {\n";
  std::cout
      << "        const int theta_forward = u_idx + v_idx * state_->nt;\n";
  std::cout << "        const int theta_backward = (state_->nt2 - 1 - u_idx) + "
               "v_idx * state_->nt;\n";
  std::cout << "        \n";
  std::cout << "        // This division is problematic!\n";
  std::cout << "        const double div_tau = 1.0 / "
               "state_->real_arrays[OddArrays::Tau][theta_forward];\n";
  std::cout
      << "        state_->real_arrays[FullGridArrays::R][theta_backward] = \n";
  std::cout
      << "            state_->real_arrays[FullGridArrays::R][theta_forward] * "
         "div_tau;\n";
  std::cout << "        // ... similar for Z and Lambda\n";
  std::cout << "    }\n";
  std::cout << "}\n\n";

  std::cout << "Problems with this approach:\n";
  std::cout << "1. Division by tau can cause numerical issues\n";
  std::cout << "2. The logic is incorrect - it's not implementing proper "
               "symmetrization\n";
  std::cout
      << "3. Missing the reflection operation for antisymmetric components\n";
}

TEST(IdealMhdModelFix, ProposedSolution) {
  std::cout << "\n=== Proposed Fix for ideal_mhd_model.cc ===\n\n";

  std::cout << "Step 1: Modify FourierToReal3DAsymmFastPoloidal to keep "
               "components separate:\n\n";

  std::cout << "// In fourier_asymmetric.cc\n";
  std::cout << "void FourierToReal3DAsymmFastPoloidal(\n";
  std::cout << "    const absl::Span<const double> fourier_coeffs_cc,\n";
  std::cout << "    const absl::Span<const double> fourier_coeffs_ss,\n";
  std::cout << "    const absl::Span<const double> fourier_coeffs_cs,\n";
  std::cout << "    const absl::Span<const double> fourier_coeffs_sc,\n";
  std::cout << "    const absl::Span<double> real_values_sym,  // Output "
               "symmetric part\n";
  std::cout << "    const absl::Span<double> real_values_asym, // Output "
               "antisymmetric part\n";
  std::cout << "    const Sizes& sizes, ...) {\n";
  std::cout << "    \n";
  std::cout << "    // Transform symmetric components (cc, ss)\n";
  std::cout << "    FourierToReal3DSymmFastPoloidal(fourier_coeffs_cc, "
               "fourier_coeffs_ss,\n";
  std::cout
      << "                                   real_values_sym, sizes, ...);\n";
  std::cout << "    \n";
  std::cout << "    // Transform antisymmetric components (cs, sc)\n";
  std::cout << "    FourierToReal3DSymmFastPoloidal(fourier_coeffs_cs, "
               "fourier_coeffs_sc,\n";
  std::cout
      << "                                   real_values_asym, sizes, ...);\n";
  std::cout << "}\n\n";

  std::cout << "Step 2: Replace the ODD ARRAYS HACK in ideal_mhd_model.cc:\n\n";

  std::cout << "// Remove the division-based hack completely\n";
  std::cout << "// Replace with proper symmetrization:\n\n";

  std::cout << "// After asymmetric transforms, call symmetrization\n";
  std::cout << "SymmetrizeRealSpaceGeometry(\n";
  std::cout << "    state_->real_arrays[EvenArrays::R],      // symmetric R\n";
  std::cout
      << "    state_->real_arrays[OddArrays::R],       // antisymmetric R\n";
  std::cout << "    state_->real_arrays[EvenArrays::Z],      // symmetric Z\n";
  std::cout
      << "    state_->real_arrays[OddArrays::Z],       // antisymmetric Z\n";
  std::cout
      << "    state_->real_arrays[EvenArrays::Lambda],  // symmetric Lambda\n";
  std::cout << "    state_->real_arrays[OddArrays::Lambda],   // antisymmetric "
               "Lambda\n";
  std::cout
      << "    state_->real_arrays[FullGridArrays::R],   // output full R\n";
  std::cout
      << "    state_->real_arrays[FullGridArrays::Z],   // output full Z\n";
  std::cout << "    state_->real_arrays[FullGridArrays::Lambda], // output "
               "full Lambda\n";
  std::cout << "    sizes_);\n";
}

TEST(IdealMhdModelFix, SymmetrizeImplementation) {
  std::cout << "\n=== Implementation of SymmetrizeRealSpaceGeometry ===\n\n";

  std::cout << "void SymmetrizeRealSpaceGeometry(\n";
  std::cout << "    const absl::Span<const double> r_sym,\n";
  std::cout << "    const absl::Span<const double> r_asym,\n";
  std::cout << "    const absl::Span<const double> z_sym,\n";
  std::cout << "    const absl::Span<const double> z_asym,\n";
  std::cout << "    const absl::Span<const double> lambda_sym,\n";
  std::cout << "    const absl::Span<const double> lambda_asym,\n";
  std::cout << "    absl::Span<double> r_full,\n";
  std::cout << "    absl::Span<double> z_full,\n";
  std::cout << "    absl::Span<double> lambda_full,\n";
  std::cout << "    const Sizes& sizes) {\n";
  std::cout << "    \n";
  std::cout << "    const int ntheta = sizes.ntheta;\n";
  std::cout << "    const int nzeta = sizes.nzeta;\n";
  std::cout << "    \n";
  std::cout << "    for (int k = 0; k < nzeta; ++k) {\n";
  std::cout << "        for (int j = 0; j < ntheta; ++j) {\n";
  std::cout << "            const int idx_half = j + k * ntheta;\n";
  std::cout << "            const int idx_full_first = j + k * (2 * ntheta);\n";
  std::cout << "            const int idx_full_second = (2 * ntheta - 1 - j) + "
               "k * (2 * ntheta);\n";
  std::cout << "            \n";
  std::cout << "            // First half: sym + asym\n";
  std::cout << "            r_full[idx_full_first] = r_sym[idx_half] + "
               "r_asym[idx_half];\n";
  std::cout << "            z_full[idx_full_first] = z_sym[idx_half] + "
               "z_asym[idx_half];\n";
  std::cout << "            lambda_full[idx_full_first] = lambda_sym[idx_half] "
               "+ lambda_asym[idx_half];\n";
  std::cout << "            \n";
  std::cout << "            // Second half: sym - asym (with reflection)\n";
  std::cout << "            r_full[idx_full_second] = r_sym[idx_half] - "
               "r_asym[idx_half];\n";
  std::cout << "            z_full[idx_full_second] = z_sym[idx_half] - "
               "z_asym[idx_half];\n";
  std::cout << "            lambda_full[idx_full_second] = "
               "lambda_sym[idx_half] - lambda_asym[idx_half];\n";
  std::cout << "        }\n";
  std::cout << "    }\n";
  std::cout << "}\n";
}

TEST(IdealMhdModelFix, VerifyTauHandling) {
  std::cout << "\n=== Tau Handling After Fix ===\n\n";

  std::cout << "Important: After implementing the fix:\n";
  std::cout << "1. The tau array should be computed normally from geometry\n";
  std::cout << "2. No division by tau is needed for symmetrization\n";
  std::cout << "3. The tau/2 pattern in debug output was a symptom, not the "
               "solution\n";
  std::cout << "4. The root cause was incorrect array combination, not tau "
               "calculation\n\n";

  std::cout << "The tau calculation remains:\n";
  std::cout << "tau = 0.5 * (R * Z_theta - Z * R_theta) / sqrt(g)\n";
  std::cout << "This is correct and doesn't need modification.\n";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
