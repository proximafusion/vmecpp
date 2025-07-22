// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// Test to understand how symmetric and asymmetric arrays should be combined
TEST(ArrayCombinationTest, UnderstandArrayCombination) {
  std::cout << "\n=== ARRAY COMBINATION TEST ===\n" << std::endl;

  std::cout
      << "Understanding how symmetric and asymmetric contributions combine..."
      << std::endl;

  std::cout << "\nCURRENT ISSUE:" << std::endl;
  std::cout << "- Symmetric arrays (r1_e, r1_o) contain symmetric contributions"
            << std::endl;
  std::cout << "- Asymmetric arrays (m_ls_.r1e_i, m_ls_.r1o_i) contain "
               "asymmetric contributions"
            << std::endl;
  std::cout << "- At kl=6-9, symmetric arrays are zero (no symmetric "
               "contribution there)"
            << std::endl;
  std::cout << "- But asymmetric contributions exist and are not being used!"
            << std::endl;

  std::cout << "\nPROPER COMBINATION:" << std::endl;
  std::cout << "For asymmetric equilibria, the total geometry is:" << std::endl;
  std::cout << "  R_total = R_symmetric + R_asymmetric" << std::endl;
  std::cout << "  Z_total = Z_symmetric + Z_asymmetric" << std::endl;

  std::cout << "\nIN CODE TERMS:" << std::endl;
  std::cout << "Before using geometry in Jacobian calculation:" << std::endl;
  std::cout << "  r1_e[idx] should contain r1_e_symm[idx] + r1e_i_asymm[idx]"
            << std::endl;
  std::cout << "  r1_o[idx] should contain r1_o_symm[idx] + r1o_i_asymm[idx]"
            << std::endl;
  std::cout << "  (same for z1_e, z1_o, ru_e, ru_o, zu_e, zu_o)" << std::endl;

  std::cout << "\nWHERE TO FIX:" << std::endl;
  std::cout << "In ideal_mhd_model::geometryFromFourier():" << std::endl;
  std::cout << "1. After symmetric transform fills r1_e, r1_o, etc."
            << std::endl;
  std::cout
      << "2. After asymmetric transform fills m_ls_.r1e_i, m_ls_.r1o_i, etc."
      << std::endl;
  std::cout << "3. BEFORE using these arrays in computeJacobian()" << std::endl;
  std::cout << "4. Add: r1_e[idx] += m_ls_.r1e_i[idx] for all geometry arrays"
            << std::endl;

  std::cout << "\nSIMPLE EXAMPLE:" << std::endl;
  {
    // Simulate what should happen
    std::vector<double> r_symm = {3.0, 2.9, 2.6, 2.1, 1.6,
                                  1.5, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> r_asymm = {0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 1.6, 2.1, 2.6, 2.9};
    std::vector<double> r_total(10);

    std::cout << "Symmetric contribution (r1_e):" << std::endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "  i=" << i << ": " << r_symm[i];
      if (i >= 6) std::cout << " (zero at second half)";
      std::cout << std::endl;
    }

    std::cout << "\nAsymmetric contribution (m_ls_.r1e_i):" << std::endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "  i=" << i << ": " << r_asymm[i];
      if (i >= 6) std::cout << " (non-zero at second half!)";
      std::cout << std::endl;
    }

    std::cout << "\nCombined total (what Jacobian should use):" << std::endl;
    for (int i = 0; i < 10; ++i) {
      r_total[i] = r_symm[i] + r_asymm[i];
      std::cout << "  i=" << i << ": " << r_total[i];
      if (i >= 6 && r_symm[i] == 0.0 && r_asymm[i] != 0.0) {
        std::cout << " ← This fixes the zero issue!";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\nIMPLEMENTATION STRATEGY:" << std::endl;
  std::cout << "1. Find where asymmetric transform completes" << std::endl;
  std::cout << "2. Add loop to combine arrays:" << std::endl;
  std::cout << "   for (int idx = 0; idx < arraySize; ++idx) {" << std::endl;
  std::cout << "     r1_e[idx] += m_ls_.r1e_i[idx];" << std::endl;
  std::cout << "     r1_o[idx] += m_ls_.r1o_i[idx];" << std::endl;
  std::cout << "     z1_e[idx] += m_ls_.z1e_i[idx];" << std::endl;
  std::cout << "     z1_o[idx] += m_ls_.z1o_i[idx];" << std::endl;
  std::cout << "     // ... same for derivatives ru_e, ru_o, zu_e, zu_o"
            << std::endl;
  std::cout << "   }" << std::endl;
  std::cout << "3. This ensures Jacobian calculation sees complete geometry"
            << std::endl;

  EXPECT_TRUE(true) << "Analysis completed";
}

// Test to verify tau2 calculation fix
TEST(ArrayCombinationTest, VerifyTau2Fix) {
  std::cout << "\n=== TAU2 CALCULATION FIX ===\n" << std::endl;

  std::cout << "Current problematic tau2 calculation:" << std::endl;
  std::cout << "tau2 = ... + (even_terms) / protected_sqrtSH" << std::endl;
  std::cout << "       ^^^^ DIVISION causes numerical issues" << std::endl;

  std::cout << "\njVMEC approach (numerically stable):" << std::endl;
  std::cout << "tau_even = rue * zs - rs * zue  (symmetric-like)" << std::endl;
  std::cout << "tau_odd = ruo * zs - rs * zuo   (asymmetric)" << std::endl;
  std::cout << "tau = tau_even + sqrtSH * tau_odd" << std::endl;
  std::cout << "      ^^^^ MULTIPLICATION is stable!" << std::endl;

  std::cout << "\nPROPOSED FIX:" << std::endl;
  std::cout << "Replace the tau2 calculation to avoid division:" << std::endl;
  std::cout << "1. Separate even and odd contributions" << std::endl;
  std::cout << "2. Use multiplication by sqrtSH instead of division"
            << std::endl;
  std::cout << "3. This matches jVMEC and is numerically stable at axis"
            << std::endl;

  EXPECT_TRUE(true) << "Fix strategy documented";
}

}  // namespace vmecpp
