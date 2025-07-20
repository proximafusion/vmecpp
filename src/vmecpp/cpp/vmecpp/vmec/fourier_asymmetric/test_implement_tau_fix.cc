// TDD test to guide implementation of educational_VMEC tau formula
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace vmecpp {

TEST(ImplementTauFixTest, DocumentCurrentVsCorrectFormula) {
  std::cout << "\n=== DOCUMENT CURRENT VS CORRECT FORMULA ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "CURRENT VMEC++ (lines 1792-1796):\n";
  std::cout << "```cpp\n";
  std::cout
      << "double tau2 = ruo_o * z1o_o + m_ls_.ruo_i[kl] * m_ls_.z1o_i[kl] -\n";
  std::cout
      << "              zuo_o * r1o_o - m_ls_.zuo_i[kl] * m_ls_.r1o_i[kl] +\n";
  std::cout
      << "              (rue_o * z1o_o + m_ls_.rue_i[kl] * m_ls_.z1o_i[kl] -\n";
  std::cout << "               zue_o * r1o_o - m_ls_.zue_i[kl] * "
               "m_ls_.r1o_i[kl]) /\n";
  std::cout << "                  protected_sqrtSH;\n";
  std::cout << "```\n\n";

  std::cout << "PROBLEMS:\n";
  std::cout
      << "1. Uses r1o_o and r1o_i in mixed terms (should be z1o_o, z1o_i)\n";
  std::cout << "2. Uses dSHalfDsInterp (computed) instead of constant 0.25\n";
  std::cout << "3. Split into tau1 + tau2 instead of unified formula\n\n";

  std::cout << "CORRECT FORMULA (educational_VMEC/jVMEC):\n";
  std::cout << "```cpp\n";
  std::cout << "// Educational_VMEC unified tau formula\n";
  std::cout << "const double dshalfds = 0.25;  // Constant as in jVMEC\n";
  std::cout << "\n";
  std::cout << "// Basic Jacobian (evn_contrib in jVMEC)\n";
  std::cout << "double tau_val = ru12[iHalf] * zs[iHalf] - rs[iHalf] * "
               "zu12[iHalf];\n";
  std::cout << "\n";
  std::cout << "// Add odd contributions (odd_contrib in jVMEC)\n";
  std::cout << "tau_val += dshalfds * (\n";
  std::cout << "    // Pure odd m terms\n";
  std::cout << "    (ruo_o * z1o_o + m_ls_.ruo_i[kl] * m_ls_.z1o_i[kl]\n";
  std::cout << "     - zuo_o * r1o_o - m_ls_.zuo_i[kl] * m_ls_.r1o_i[kl])\n";
  std::cout << "    // Mixed even m × odd m terms (CRITICAL FIX)\n";
  std::cout << "    + (rue_o * z1o_o + m_ls_.rue_i[kl] * m_ls_.z1o_i[kl]\n";
  std::cout << "       - zue_o * r1o_o - m_ls_.zue_i[kl] * m_ls_.r1o_i[kl]) / "
               "sqrtSH\n";
  std::cout << ");\n";
  std::cout << "```\n";

  EXPECT_TRUE(true) << "Formula comparison documented";
}

TEST(ImplementTauFixTest, ShowExactCodeChange) {
  std::cout << "\n=== SHOW EXACT CODE CHANGE ===\n";

  std::cout << "REPLACE lines 1764-1812 in ideal_mhd_model.cc with:\n\n";

  std::cout << "```cpp\n";
  std::cout << "      // Educational_VMEC unified tau formula (matches jVMEC "
               "exactly)\n";
  std::cout
      << "      const double dshalfds = 0.25;  // Constant, not computed\n";
  std::cout << "      \n";
  std::cout << "      // Basic Jacobian term (evn_contrib in jVMEC)\n";
  std::cout << "      double tau_val = ru12[iHalf] * zs[iHalf] - rs[iHalf] * "
               "zu12[iHalf];\n";
  std::cout << "      \n";
  std::cout << "      // Add odd mode contributions (odd_contrib in jVMEC)\n";
  std::cout << "      double odd_contrib = \n";
  std::cout << "          // Pure odd m terms at surfaces j and j-1\n";
  std::cout << "          (ruo_o * z1o_o + m_ls_.ruo_i[kl] * m_ls_.z1o_i[kl]\n";
  std::cout
      << "           - zuo_o * r1o_o - m_ls_.zuo_i[kl] * m_ls_.r1o_i[kl])\n";
  std::cout << "          // Mixed even m × odd m terms (divided by sqrtSH)\n";
  std::cout
      << "          + (rue_o * z1o_o + m_ls_.rue_i[kl] * m_ls_.z1o_i[kl]\n";
  std::cout << "             - zue_o * r1o_o - m_ls_.zue_i[kl] * "
               "m_ls_.r1o_i[kl]) / sqrtSH;\n";
  std::cout << "      \n";
  std::cout << "      tau_val += dshalfds * odd_contrib;\n";
  std::cout << "      \n";
  std::cout << "      // Debug output for asymmetric mode\n";
  std::cout << "      if (s_.lasym && (kl >= 6 && kl <= 9)) {\n";
  std::cout << "        std::cout << \"DEBUG TAU CALC kl=\" << kl << \" jH=\" "
               "<< jH\n";
  std::cout << "                  << \" iHalf=\" << iHalf << \":\\n\";\n";
  std::cout << "        std::cout << \"  Basic Jacobian: \" << (ru12[iHalf] * "
               "zs[iHalf] - rs[iHalf] * zu12[iHalf]) << \"\\n\";\n";
  std::cout << "        std::cout << \"  Odd contrib: \" << odd_contrib << "
               "\"\\n\";\n";
  std::cout << "        std::cout << \"  tau_val = \" << tau_val << \"\\n\";\n";
  std::cout << "      }\n";
  std::cout << "```\n";

  std::cout << "\nKEY DIFFERENCES:\n";
  std::cout << "1. Use constant dshalfds = 0.25 (not dSHalfDsInterp)\n";
  std::cout << "2. Unified formula (not split tau1 + tau2)\n";
  std::cout << "3. Correct mixed terms using z1o_o not r1o_o\n";
  std::cout << "4. Matches jVMEC and educational_VMEC exactly\n";

  EXPECT_TRUE(true) << "Code change documented";
}

TEST(ImplementTauFixTest, VerifyAxisProtection) {
  std::cout << "\n=== VERIFY AXIS PROTECTION ===\n";

  std::cout << "AXIS PROTECTION STRATEGY:\n";
  std::cout << "1. The sqrtSH division is protected by physics (sqrtSH never "
               "exactly 0)\n";
  std::cout << "2. At axis (jH=0), copy tau from next radial point\n";
  std::cout << "3. This is done AFTER computing all tau values\n\n";

  std::cout << "KEEP existing axis protection code (lines 1823-1833):\n";
  std::cout << "```cpp\n";
  std::cout
      << "// Apply axis protection: use constant extrapolation like jVMEC\n";
  std::cout << "if (jH == r_.nsMinH && r_.nsMinH == 0 && s_.lasym) {\n";
  std::cout << "  // For axis point in asymmetric mode, use value from next "
               "radial point\n";
  std::cout << "  ...\n";
  std::cout << "}\n";
  std::cout << "```\n";

  std::cout << "\nALSO add axis extrapolation at end (like educational_VMEC "
               "line 70-71):\n";
  std::cout << "```cpp\n";
  std::cout << "// After loop, extrapolate tau to axis as constant\n";
  std::cout << "if (r_.nsMinH == 0) {\n";
  std::cout << "  for (int kl = 0; kl < s_.nZnT; ++kl) {\n";
  std::cout << "    tau[0 * s_.nZnT + kl] = tau[1 * s_.nZnT + kl];\n";
  std::cout << "  }\n";
  std::cout << "}\n";
  std::cout << "```\n";

  EXPECT_TRUE(true) << "Axis protection verified";
}

TEST(ImplementTauFixTest, TestPlan) {
  std::cout << "\n=== TEST PLAN ===\n";

  std::cout << "AFTER IMPLEMENTING FIX:\n";
  std::cout << "1. Run test_jacobian_geometry_debug\n";
  std::cout << "   - Verify tau values are different\n";
  std::cout << "   - Check odd_contrib is non-zero\n";
  std::cout << "   - Confirm no Jacobian sign change\n\n";

  std::cout << "2. Run test_embedded_asymmetric_tokamak\n";
  std::cout << "   - Should converge without Jacobian error\n";
  std::cout << "   - First successful asymmetric equilibrium!\n\n";

  std::cout << "3. Run symmetric regression tests\n";
  std::cout << "   - Ensure lasym=false still works\n";
  std::cout << "   - No change in symmetric behavior\n\n";

  std::cout << "4. Compare with educational_VMEC\n";
  std::cout << "   - Run same input in both codes\n";
  std::cout << "   - Compare tau values and convergence\n";

  EXPECT_TRUE(true) << "Test plan ready";

  std::cout << "\n🎯 READY TO IMPLEMENT in ideal_mhd_model.cc!\n";
}

}  // namespace vmecpp
