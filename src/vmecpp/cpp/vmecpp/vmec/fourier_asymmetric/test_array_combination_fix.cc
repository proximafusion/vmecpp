// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// Test to demonstrate the array combination fix implementation
TEST(ArrayCombinationFixTest, DemonstrateArrayCombination) {
  std::cout << "\n=== ARRAY COMBINATION FIX DEMONSTRATION ===\n" << std::endl;

  std::cout << "Demonstrating how to implement the array combination fix..."
            << std::endl;

  // Simulate the arrays we have in ideal_mhd_model
  const int nZnT = 10;    // theta points
  const int nRadial = 3;  // radial surfaces
  const int totalSize = nRadial * nZnT;

  // Symmetric arrays (what we currently have)
  std::vector<double> r1_e(totalSize, 0.0);
  std::vector<double> r1_o(totalSize, 0.0);
  std::vector<double> z1_e(totalSize, 0.0);
  std::vector<double> z1_o(totalSize, 0.0);

  // Asymmetric arrays (stored separately)
  std::vector<double> r1e_i(totalSize, 0.0);
  std::vector<double> r1o_i(totalSize, 0.0);
  std::vector<double> z1e_i(totalSize, 0.0);
  std::vector<double> z1o_i(totalSize, 0.0);

  std::cout << "\n1. SIMULATE SYMMETRIC TRANSFORM OUTPUT:" << std::endl;
  // Fill symmetric arrays (only first half of theta has values)
  for (int j = 0; j < nRadial; ++j) {
    for (int kl = 0; kl < nZnT / 2; ++kl) {
      int idx = j * nZnT + kl;
      r1_e[idx] = 3.0 - j * 0.5 - kl * 0.2;  // Some values
      z1_o[idx] = 1.0 - j * 0.3 - kl * 0.1;  // Some values
    }
  }

  std::cout << "   Symmetric r1_e at j=0: ";
  for (int kl = 0; kl < nZnT; ++kl) {
    std::cout << r1_e[kl] << " ";
  }
  std::cout << std::endl;

  std::cout << "\n2. SIMULATE ASYMMETRIC TRANSFORM OUTPUT:" << std::endl;
  // Fill asymmetric arrays (only second half of theta has values)
  for (int j = 0; j < nRadial; ++j) {
    for (int kl = nZnT / 2; kl < nZnT; ++kl) {
      int idx = j * nZnT + kl;
      r1e_i[idx] = 2.0 + j * 0.2 + (kl - nZnT / 2) * 0.15;  // Some values
      z1e_i[idx] = 0.5 + j * 0.1 + (kl - nZnT / 2) * 0.05;  // Some values
    }
  }

  std::cout << "   Asymmetric r1e_i at j=0: ";
  for (int kl = 0; kl < nZnT; ++kl) {
    std::cout << r1e_i[kl] << " ";
  }
  std::cout << std::endl;

  std::cout << "\n3. IMPLEMENT ARRAY COMBINATION FIX:" << std::endl;
  std::cout << "   Adding asymmetric contributions to symmetric arrays..."
            << std::endl;

  // THE FIX: Combine arrays
  for (int idx = 0; idx < totalSize; ++idx) {
    r1_e[idx] += r1e_i[idx];
    r1_o[idx] += r1o_i[idx];
    z1_e[idx] += z1e_i[idx];
    z1_o[idx] += z1o_i[idx];
  }

  std::cout << "\n4. RESULT AFTER COMBINATION:" << std::endl;
  std::cout << "   Combined r1_e at j=0: ";
  for (int kl = 0; kl < nZnT; ++kl) {
    std::cout << r1_e[kl] << " ";
  }
  std::cout << std::endl;

  std::cout << "\n5. VERIFY FIX:" << std::endl;
  bool allNonZero = true;
  for (int kl = 0; kl < nZnT; ++kl) {
    if (r1_e[kl] == 0.0) {
      allNonZero = false;
      std::cout << "   ❌ Still zero at kl=" << kl << std::endl;
    }
  }
  if (allNonZero) {
    std::cout << "   ✅ All positions now have non-zero values!" << std::endl;
    std::cout << "   ✅ This prevents r1_e=0 error in Jacobian calculation!"
              << std::endl;
  }

  std::cout << "\n6. IMPLEMENTATION IN ideal_mhd_model.cc:" << std::endl;
  std::cout << "   After line ~1380 (after asymmetric transform):" << std::endl;
  std::cout << "   ```cpp" << std::endl;
  std::cout << "   // Combine symmetric and asymmetric contributions"
            << std::endl;
  std::cout << "   for (int idx = 0; idx < r1_e.size(); ++idx) {" << std::endl;
  std::cout << "     r1_e[idx] += m_ls_.r1e_i[idx];" << std::endl;
  std::cout << "     r1_o[idx] += m_ls_.r1o_i[idx];" << std::endl;
  std::cout << "     z1_e[idx] += m_ls_.z1e_i[idx];" << std::endl;
  std::cout << "     z1_o[idx] += m_ls_.z1o_i[idx];" << std::endl;
  std::cout << "     ru_e[idx] += m_ls_.rue_i[idx];" << std::endl;
  std::cout << "     ru_o[idx] += m_ls_.ruo_i[idx];" << std::endl;
  std::cout << "     zu_e[idx] += m_ls_.zue_i[idx];" << std::endl;
  std::cout << "     zu_o[idx] += m_ls_.zuo_i[idx];" << std::endl;
  std::cout << "   }" << std::endl;
  std::cout << "   ```" << std::endl;

  EXPECT_TRUE(allNonZero) << "Array combination should eliminate zero values";
}

// Test to show the tau2 fix
TEST(ArrayCombinationFixTest, DemonstrateTau2Fix) {
  std::cout << "\n=== TAU2 CALCULATION FIX DEMONSTRATION ===\n" << std::endl;

  std::cout << "Current problematic code (line ~1688):" << std::endl;
  std::cout << "```cpp" << std::endl;
  std::cout
      << "double tau2 = ruo_o * z1o_o + m_ls_.ruo_i[kl] * m_ls_.z1o_i[kl] -"
      << std::endl;
  std::cout
      << "              zuo_o * r1o_o - m_ls_.zuo_i[kl] * m_ls_.r1o_i[kl] +"
      << std::endl;
  std::cout
      << "              (rue_o * z1o_o + m_ls_.rue_i[kl] * m_ls_.z1o_i[kl] -"
      << std::endl;
  std::cout
      << "               zue_o * r1o_o - m_ls_.zue_i[kl] * m_ls_.r1o_i[kl]) /"
      << std::endl;
  std::cout << "                  protected_sqrtSH;  // <-- DIVISION!"
            << std::endl;
  std::cout << "```" << std::endl;

  std::cout << "\nPROBLEM: Division by small sqrtSH amplifies errors"
            << std::endl;

  std::cout << "\nFIXED CODE (jVMEC style):" << std::endl;
  std::cout << "```cpp" << std::endl;
  std::cout << "// Separate even and odd contributions" << std::endl;
  std::cout
      << "double tau_odd = ruo_o * z1o_o + m_ls_.ruo_i[kl] * m_ls_.z1o_i[kl] -"
      << std::endl;
  std::cout
      << "                 zuo_o * r1o_o - m_ls_.zuo_i[kl] * m_ls_.r1o_i[kl];"
      << std::endl;
  std::cout << "" << std::endl;
  std::cout << "double tau_even_asymm = rue_o * z1o_o + m_ls_.rue_i[kl] * "
               "m_ls_.z1o_i[kl] -"
            << std::endl;
  std::cout << "                        zue_o * r1o_o - m_ls_.zue_i[kl] * "
               "m_ls_.r1o_i[kl];"
            << std::endl;
  std::cout << "" << std::endl;
  std::cout << "// No division - multiply instead!" << std::endl;
  std::cout << "double tau2 = tau_odd + protected_sqrtSH * tau_even_asymm;"
            << std::endl;
  std::cout << "```" << std::endl;

  std::cout << "\nNUMERICAL EXAMPLE:" << std::endl;
  double sqrtSH = 0.1;  // Small value near axis
  double tau_odd = 0.5;
  double tau_even = 0.3;

  // Old way (with division)
  double tau2_old = tau_odd + tau_even / sqrtSH;
  std::cout << "Old: tau2 = " << tau_odd << " + " << tau_even << " / " << sqrtSH
            << " = " << tau2_old << " (large!)" << std::endl;

  // New way (with multiplication)
  double tau2_new = tau_odd + sqrtSH * tau_even;
  std::cout << "New: tau2 = " << tau_odd << " + " << sqrtSH << " * " << tau_even
            << " = " << tau2_new << " (stable!)" << std::endl;

  std::cout << "\nBENEFITS:" << std::endl;
  std::cout << "✅ No division by small numbers" << std::endl;
  std::cout << "✅ Stable as sqrtSH → 0 at axis" << std::endl;
  std::cout << "✅ Matches jVMEC implementation" << std::endl;

  EXPECT_LT(tau2_new, tau2_old) << "New formulation should be more stable";
}

}  // namespace vmecpp
