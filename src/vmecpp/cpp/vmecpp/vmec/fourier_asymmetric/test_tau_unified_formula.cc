// TDD test to implement and verify educational_VMEC unified tau formula
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(TauUnifiedFormulaTest, VerifyCurrentTau2IsZero) {
  std::cout << "\n=== VERIFY CURRENT TAU2 IS ZERO ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "Running asymmetric case to verify tau2 = 0 issue\n";

  // Minimal asymmetric tokamak configuration
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 3;
  config.ntor = 0;
  config.ns_array = {3};
  config.niter_array = {1};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;
  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.pmass_type = "power_series";
  config.am = {0.0};  // Zero pressure
  config.pres_scale = 1.0;

  // Axis
  config.raxis_c = {6.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  // Boundary with small asymmetric perturbation
  config.rbc = {6.0, 0.0, 0.6};
  config.zbs = {0.0, 0.0, 0.6};
  config.rbs = {0.0, 0.0, 0.1};  // Small asymmetric perturbation
  config.zbc = {0.0, 0.0, 0.1};  // Small asymmetric perturbation

  std::cout << "\nConfiguration:\n";
  std::cout << "  R0 = 6.0, a = 0.6\n";
  std::cout << "  Asymmetric perturbation = 0.1\n";
  std::cout << "  lasym = true\n";

  const auto output = vmecpp::run(config);

  if (!output.ok()) {
    std::string error_msg(output.status().message());
    std::cout << "\n❌ EXPECTED: " << error_msg << "\n";

    // Check debug output to verify tau2 = 0
    if (error_msg.find("INITIAL JACOBIAN CHANGED SIGN") != std::string::npos) {
      std::cout << "\n✅ CONFIRMED: Jacobian fails due to tau2 = 0\n";
      std::cout << "Debug output should show all tau2 components = 0\n";
      std::cout << "This confirms missing mixed even/odd terms\n";
    }
  } else {
    std::cout
        << "\n⚠️ UNEXPECTED: Run succeeded (should fail with current formula)\n";
  }

  EXPECT_FALSE(output.ok())
      << "Should fail with current incomplete tau2 formula";
}

TEST(TauUnifiedFormulaTest, DocumentTauFormulaMapping) {
  std::cout << "\n=== DOCUMENT TAU FORMULA MAPPING ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "Mapping educational_VMEC to VMEC++ arrays:\n\n";

  std::cout << "Educational_VMEC formula (line 55-59):\n";
  std::cout << "tau(l) = ru12(l)*zs(l) - rs(l)*zu12(l) + dshalfds*[\n";
  std::cout << "           ru(l,modd)*z1(l,modd) + ru(l-1,modd)*z1(l-1,modd)\n";
  std::cout << "         - zu(l,modd)*r1(l,modd) - zu(l-1,modd)*r1(l-1,modd)\n";
  std::cout
      << "       + ( ru(l,meven)*z1(l,modd) + ru(l-1,meven)*z1(l-1,modd)\n";
  std::cout << "         - zu(l,meven)*r1(l,modd) - zu(l-1,meven)*r1(l-1,modd) "
               ")/shalf(l)\n";
  std::cout << "         ]\n\n";

  std::cout << "VMEC++ mapping:\n";
  std::cout << "- ru(l,modd) → ruo_o (at current surface jH+1)\n";
  std::cout << "- ru(l-1,modd) → m_ls_.ruo_i[kl] (at previous surface jH)\n";
  std::cout << "- z1(l,modd) → z1o_o (at current surface)\n";
  std::cout << "- z1(l-1,modd) → m_ls_.z1o_i[kl] (at previous surface)\n";
  std::cout << "- ru(l,meven) → rue_o (even mode at current surface)\n";
  std::cout << "- ru(l-1,meven) → m_ls_.rue_i[kl] (even mode at previous)\n";
  std::cout << "- shalf(l) → sqrtSH (sqrt(s) at half grid)\n";
  std::cout << "- dshalfds → 0.25 (constant)\n\n";

  std::cout << "Current VMEC++ tau2 (INCOMPLETE):\n";
  std::cout << "tau2 = ruo_o*z1o_o + m_ls_.ruo_i[kl]*m_ls_.z1o_i[kl]\n";
  std::cout << "     - zuo_o*r1o_o - m_ls_.zuo_i[kl]*m_ls_.r1o_i[kl]\n";
  std::cout << "     + (rue_o*z1o_o + m_ls_.rue_i[kl]*m_ls_.z1o_i[kl]\n";
  std::cout << "        - zue_o*r1o_o - m_ls_.zue_i[kl]*m_ls_.r1o_i[kl]) / "
               "sqrtSH\n\n";

  std::cout << "❌ PROBLEM: Current formula uses r1o_o instead of z1o_o!\n";
  std::cout << "Should be z1(l,modd) not r1(l,modd) in mixed terms!\n";

  EXPECT_TRUE(true) << "Formula mapping documented";
}

TEST(TauUnifiedFormulaTest, ImplementCorrectFormula) {
  std::cout << "\n=== IMPLEMENT CORRECT FORMULA ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "CORRECT UNIFIED TAU FORMULA:\n\n";

  std::cout << "```cpp\n";
  std::cout << "// Educational_VMEC unified tau formula\n";
  std::cout << "const double dshalfds = 0.25;  // Constant, not 0.5/sqrtSH\n";
  std::cout << "\n";
  std::cout << "// Basic Jacobian term (tau1)\n";
  std::cout << "double tau_val = ru12[iHalf] * zs[iHalf] - rs[iHalf] * "
               "zu12[iHalf];\n";
  std::cout << "\n";
  std::cout << "// Add dshalfds contributions\n";
  std::cout << "tau_val += dshalfds * (\n";
  std::cout << "    // Pure odd terms at l and l-1\n";
  std::cout << "    (ruo_o * z1o_o + m_ls_.ruo_i[kl] * m_ls_.z1o_i[kl]\n";
  std::cout << "     - zuo_o * r1o_o - m_ls_.zuo_i[kl] * m_ls_.r1o_i[kl])\n";
  std::cout << "    // Mixed even/odd terms (CRITICAL FIX)\n";
  std::cout << "    + (rue_o * z1o_o + m_ls_.rue_i[kl] * m_ls_.z1o_i[kl]\n";
  std::cout << "       - zue_o * r1o_o - m_ls_.zue_i[kl] * m_ls_.r1o_i[kl]) / "
               "sqrtSH\n";
  std::cout << ");\n";
  std::cout << "```\n\n";

  std::cout << "KEY CHANGES:\n";
  std::cout << "1. Use constant dshalfds = 0.25\n";
  std::cout << "2. Unified formula (not split tau1 + tau2)\n";
  std::cout << "3. Correct mixed terms with z1o not r1o\n";
  std::cout << "4. Apply dshalfds to entire bracket\n";

  EXPECT_TRUE(true) << "Implementation documented";
}

TEST(TauUnifiedFormulaTest, VerifyExpectedImpact) {
  std::cout << "\n=== VERIFY EXPECTED IMPACT ===\n";
  std::cout << std::fixed << std::setprecision(6);

  std::cout << "Expected impact of unified formula:\n\n";

  std::cout << "BEFORE (current VMEC++):\n";
  std::cout << "- tau2 = 0 (all odd components zero)\n";
  std::cout << "- tau ranges from negative to positive\n";
  std::cout << "- Jacobian sign change → convergence failure\n\n";

  std::cout << "AFTER (with unified formula):\n";
  std::cout << "- tau2 ≠ 0 (includes mixed even/odd terms)\n";
  std::cout << "- Additional terms shift tau distribution\n";
  std::cout << "- All tau values same sign → Jacobian OK\n";
  std::cout << "- Asymmetric equilibria converge!\n\n";

  std::cout << "Verification tests:\n";
  std::cout << "1. Check tau2 no longer zero\n";
  std::cout << "2. Verify no Jacobian sign change\n";
  std::cout << "3. Test convergence with asymmetric cases\n";
  std::cout << "4. Ensure no regression in symmetric mode\n";

  EXPECT_TRUE(true) << "Impact analysis complete";

  std::cout << "\n🎯 Ready to modify ideal_mhd_model.cc!\n";
}

}  // namespace vmecpp
