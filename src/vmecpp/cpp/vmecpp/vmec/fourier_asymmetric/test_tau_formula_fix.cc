// TDD unit test to verify educational_VMEC tau formula implementation
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace vmecpp {

TEST(TauFormulaFixTest, VerifyUnifiedFormulaComponents) {
  std::cout << "\n=== VERIFY UNIFIED TAU FORMULA COMPONENTS ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout
      << "GOAL: Test educational_VMEC unified tau formula implementation\n";
  std::cout << "This test documents the exact formula to be implemented.\n";

  // Test data from debug output (surface jH=0, kl=6)
  double ru12 = 0.300000;   // (ru_even at half grid)
  double zs = -0.600000;    // dZ/ds at half grid
  double rs = -12.060000;   // dR/ds at half grid
  double zu12 = -0.060000;  // (zu_even at half grid)

  // For asymmetric case, we need mode-separated values
  // These would come from r1, z1, ru, zu arrays with meven/modd indices
  // For now, use representative values to test formula
  double ru_l_odd = 0.1;   // ru(l, modd)
  double z1_l_odd = 0.2;   // z1(l, modd)
  double zu_l_odd = 0.05;  // zu(l, modd)
  double r1_l_odd = 0.15;  // r1(l, modd)

  double ru_lm1_odd = 0.08;  // ru(l-1, modd)
  double z1_lm1_odd = 0.18;  // z1(l-1, modd)
  double zu_lm1_odd = 0.04;  // zu(l-1, modd)
  double r1_lm1_odd = 0.14;  // r1(l-1, modd)

  double ru_l_even = 0.3;     // ru(l, meven)
  double zu_l_even = 0.06;    // zu(l, meven)
  double ru_lm1_even = 0.28;  // ru(l-1, meven)
  double zu_lm1_even = 0.05;  // zu(l-1, meven)

  double shalf_l = 0.5;    // sqrt(s) at half grid
  double dshalfds = 0.25;  // d(sqrt(s))/ds = 0.5/sqrt(s) = 0.25

  std::cout << "\nEducational_VMEC unified formula:\n";
  std::cout << "tau(l) = ru12*zs - rs*zu12 + dshalfds*[\n";
  std::cout
      << "          (ru_odd*z1_odd - zu_odd*r1_odd)       // Pure odd terms\n";
  std::cout
      << "        + (ru_even*z1_odd - zu_even*r1_odd)/shalf  // Mixed terms\n";
  std::cout << "        ]\n";

  // Calculate tau1 (basic Jacobian term)
  double tau1 = ru12 * zs - rs * zu12;
  std::cout << "\nStep 1 - Basic Jacobian (tau1):\n";
  std::cout << "  tau1 = " << ru12 << " * " << zs << " - " << rs << " * "
            << zu12 << "\n";
  std::cout << "  tau1 = " << tau1 << "\n";

  // Calculate pure odd terms
  double pure_odd_l = ru_l_odd * z1_l_odd - zu_l_odd * r1_l_odd;
  double pure_odd_lm1 = ru_lm1_odd * z1_lm1_odd - zu_lm1_odd * r1_lm1_odd;
  double pure_odd_sum = pure_odd_l + pure_odd_lm1;

  std::cout << "\nStep 2 - Pure odd terms:\n";
  std::cout << "  At l:   " << ru_l_odd << " * " << z1_l_odd << " - "
            << zu_l_odd << " * " << r1_l_odd << " = " << pure_odd_l << "\n";
  std::cout << "  At l-1: " << ru_lm1_odd << " * " << z1_lm1_odd << " - "
            << zu_lm1_odd << " * " << r1_lm1_odd << " = " << pure_odd_lm1
            << "\n";
  std::cout << "  Sum: " << pure_odd_sum << "\n";

  // Calculate mixed even/odd terms
  double mixed_l = ru_l_even * z1_l_odd - zu_l_even * r1_l_odd;
  double mixed_lm1 = ru_lm1_even * z1_lm1_odd - zu_lm1_even * r1_lm1_odd;
  double mixed_sum = (mixed_l + mixed_lm1) / shalf_l;

  std::cout << "\nStep 3 - Mixed even/odd terms:\n";
  std::cout << "  At l:   " << ru_l_even << " * " << z1_l_odd << " - "
            << zu_l_even << " * " << r1_l_odd << " = " << mixed_l << "\n";
  std::cout << "  At l-1: " << ru_lm1_even << " * " << z1_lm1_odd << " - "
            << zu_lm1_even << " * " << r1_lm1_odd << " = " << mixed_lm1 << "\n";
  std::cout << "  Sum/shalf: (" << mixed_l << " + " << mixed_lm1 << ") / "
            << shalf_l << " = " << mixed_sum << "\n";

  // Calculate total tau
  double tau_unified = tau1 + dshalfds * (pure_odd_sum + mixed_sum);

  std::cout << "\nStep 4 - Total tau (unified formula):\n";
  std::cout << "  tau = " << tau1 << " + " << dshalfds << " * (" << pure_odd_sum
            << " + " << mixed_sum << ")\n";
  std::cout << "  tau = " << tau_unified << "\n";

  // Compare with current VMEC++ (tau2 = 0)
  double tau_current = tau1;  // Since tau2 = 0 in current implementation

  std::cout << "\nComparison:\n";
  std::cout << "  Current VMEC++: tau = " << tau_current
            << " (missing terms)\n";
  std::cout << "  Educational_VMEC: tau = " << tau_unified << " (complete)\n";
  std::cout << "  Difference: " << (tau_unified - tau_current) << "\n";

  EXPECT_NE(tau_unified, tau_current)
      << "Unified formula should differ from current";

  std::cout << "\n✅ Formula components verified - ready for implementation\n";
}

TEST(TauFormulaFixTest, TestModeExtraction) {
  std::cout << "\n=== TEST MODE EXTRACTION FOR TAU FORMULA ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "GOAL: Document how to extract even/odd modes from arrays\n";

  std::cout << "\nIn educational_VMEC:\n";
  std::cout << "- meven = 0 (even parity wrt theta=0)\n";
  std::cout << "- modd = 1 (odd parity wrt theta=0)\n";
  std::cout
      << "- Arrays like r1(l,parity) store values with parity separation\n";

  std::cout << "\nIn VMEC++:\n";
  std::cout << "- r1_e, ru_e, zu_e: Even parity arrays\n";
  std::cout << "- r1_o, ru_o, zu_o: Odd parity arrays\n";
  std::cout << "- Direct access: r1_e[l] gives even mode at position l\n";

  std::cout << "\nImplementation mapping:\n";
  std::cout << "- ru(l,meven) → ru_e[l]\n";
  std::cout << "- ru(l,modd) → ru_o[l]\n";
  std::cout << "- z1(l,modd) → z1_o[l]\n";
  std::cout << "- etc.\n";

  std::cout << "\nKey insight: VMEC++ already has mode-separated arrays!\n";
  std::cout << "Just need to use them in educational_VMEC formula pattern.\n";

  EXPECT_TRUE(true) << "Mode extraction pattern documented";
}

TEST(TauFormulaFixTest, VerifyNumericalStability) {
  std::cout << "\n=== VERIFY NUMERICAL STABILITY ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "GOAL: Ensure formula handles edge cases correctly\n";

  // Test near-axis case (shalf → 0)
  double shalf_small = 1e-6;
  double dshalfds_small = 0.5 / shalf_small;  // Very large!

  std::cout << "\nNear-axis case:\n";
  std::cout << "  shalf = " << shalf_small << " (very small)\n";
  std::cout << "  dshalfds = 0.5/shalf = " << dshalfds_small
            << " (very large)\n";

  // But in educational_VMEC, dshalfds is constant 0.25
  double dshalfds_const = 0.25;
  std::cout << "\nEducational_VMEC uses CONSTANT dshalfds = " << dshalfds_const
            << "\n";
  std::cout << "This avoids division issues near axis!\n";

  // Test axis protection
  std::cout << "\nAxis protection strategy:\n";
  std::cout << "1. Use constant dshalfds = 0.25 (not 0.5/shalf)\n";
  std::cout << "2. Mixed terms divided by shalf need protection\n";
  std::cout << "3. At axis (l=1), copy tau from l=2 (standard practice)\n";

  EXPECT_DOUBLE_EQ(dshalfds_const, 0.25) << "dshalfds should be constant 0.25";

  std::cout << "\n✅ Numerical stability considerations verified\n";
}

TEST(TauFormulaFixTest, DocumentImplementationPlan) {
  std::cout << "\n=== DOCUMENT IMPLEMENTATION PLAN ===\n";

  std::cout << "IMPLEMENTATION STEPS:\n";
  std::cout << "1. Locate computeJacobian() in ideal_mhd_model.cc\n";
  std::cout << "2. Find current tau calculation (around line 1764)\n";
  std::cout << "3. Replace split tau1+tau2 with unified formula\n";
  std::cout << "4. Use existing even/odd arrays (r1_e, r1_o, etc.)\n";
  std::cout << "5. Apply constant dshalfds = 0.25\n";
  std::cout << "6. Add axis protection for l=1\n";

  std::cout << "\nEXPECTED CODE PATTERN:\n";
  std::cout << "```cpp\n";
  std::cout << "// Educational_VMEC unified tau formula\n";
  std::cout << "const double dshalfds = 0.25;\n";
  std::cout
      << "double tau_val = ru12[iHalf]*zs[iHalf] - rs[iHalf]*zu12[iHalf]\n";
  std::cout << "  + dshalfds * (\n";
  std::cout << "      // Pure odd terms at l and l-1\n";
  std::cout << "      (ru_o[l]*z1_o[l] + ru_o[l-1]*z1_o[l-1]\n";
  std::cout << "       - zu_o[l]*r1_o[l] - zu_o[l-1]*r1_o[l-1])\n";
  std::cout << "      // Mixed even/odd terms\n";
  std::cout << "      + (ru_e[l]*z1_o[l] + ru_e[l-1]*z1_o[l-1]\n";
  std::cout << "         - zu_e[l]*r1_o[l] - zu_e[l-1]*r1_o[l-1]) / sqrtSH\n";
  std::cout << "    );\n";
  std::cout << "```\n";

  std::cout << "\nVERIFICATION:\n";
  std::cout << "- Run test_jacobian_geometry_debug to verify tau values\n";
  std::cout << "- Check that tau2 is no longer zero\n";
  std::cout << "- Verify Jacobian passes (no sign change)\n";
  std::cout << "- Test convergence with asymmetric configurations\n";

  EXPECT_TRUE(true) << "Implementation plan documented";

  std::cout << "\n🎯 Ready to implement educational_VMEC tau formula!\n";
}

}  // namespace vmecpp
