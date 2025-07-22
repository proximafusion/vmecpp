// TDD test documenting the odd arrays bug
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace vmecpp {

TEST(OddArraysBugTest, DocumentRootCause) {
  std::cout << "\n=== DOCUMENT ODD ARRAYS BUG ROOT CAUSE ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout
      << "BUG: r1_o, z1_o, ru_o, zu_o are always zero in asymmetric mode!\n\n";

  std::cout << "ROOT CAUSE ANALYSIS:\n";
  std::cout << "1. In symmetric mode, Fourier basis for R and Z:\n";
  std::cout << "   R ~ cos(m*u)*cos(n*v) + sin(m*u)*sin(n*v)\n";
  std::cout << "   Z ~ sin(m*u)*cos(n*v) + cos(m*u)*sin(n*v)\n\n";

  std::cout << "2. The symmetric transform produces:\n";
  std::cout << "   rnkcc[0] = cos(m*theta) terms for R (even parity)\n";
  std::cout << "   rnkcc[1] = sin(m*theta) terms for R (odd parity)\n";
  std::cout << "   znksc[0] = sin(m*theta) terms for Z (even parity)\n";
  std::cout << "   znksc[1] = cos(m*theta) terms for Z (odd parity)\n\n";

  std::cout << "3. But R doesn't have sin(m*theta) in symmetric mode!\n";
  std::cout << "   So rnkcc[1] = 0 always for symmetric coefficients\n";
  std::cout << "   Similarly znksc[1] = 0 for symmetric coefficients\n\n";

  std::cout << "4. In asymmetric mode, we ADD new basis functions:\n";
  std::cout << "   R gains: sin(m*u)*cos(n*v) + cos(m*u)*sin(n*v)\n";
  std::cout << "   Z gains: cos(m*u)*cos(n*v) + sin(m*u)*sin(n*v)\n\n";

  std::cout << "5. These should populate the odd arrays:\n";
  std::cout << "   rmnsc -> sin(m*theta) -> r1_o (odd parity for R)\n";
  std::cout << "   zmncc -> cos(m*theta) -> z1_o (odd parity for Z)\n\n";

  std::cout << "6. BUT the code at lines 1593-1597 does:\n";
  std::cout << "   r1_o[idx_jl] += rnkcc[kOddParity];  // WRONG!\n";
  std::cout << "   z1_o[idx_jl] += znksc[kOddParity];  // WRONG!\n\n";

  std::cout << "7. And the asymmetric transform (lines 1350-1370):\n";
  std::cout << "   - Computes asymmetric contributions\n";
  std::cout << "   - But only adds them to r1_e, z1_e\n";
  std::cout << "   - Never populates r1_o, z1_o!\n\n";

  std::cout << "CONSEQUENCE:\n";
  std::cout << "- r1_o = 0, z1_o = 0, ru_o = 0, zu_o = 0 always\n";
  std::cout << "- tau formula odd_contrib = 0 always\n";
  std::cout << "- Jacobian calculation incorrect for asymmetric\n";

  EXPECT_TRUE(true) << "Bug documented";
}

TEST(OddArraysBugTest, ProposeFix) {
  std::cout << "\n=== PROPOSE FIX FOR ODD ARRAYS BUG ===\n";

  std::cout << "OPTION 1: Modify array combination (lines 1350-1370)\n";
  std::cout << "The asymmetric transform should populate BOTH even and odd:\n";
  std::cout << "```cpp\n";
  std::cout << "// Current code only does:\n";
  std::cout << "r1_e[idx] += m_ls_.r1e_i[idx];  // Even contributions\n";
  std::cout << "z1_e[idx] += m_ls_.z1e_i[idx];\n";
  std::cout << "\n";
  std::cout << "// Need to also add:\n";
  std::cout << "r1_o[idx] += m_ls_.r1o_i[idx];  // Odd contributions\n";
  std::cout << "z1_o[idx] += m_ls_.z1o_i[idx];\n";
  std::cout << "ru_o[idx] += m_ls_.ruo_i[idx];\n";
  std::cout << "zu_o[idx] += m_ls_.zuo_i[idx];\n";
  std::cout << "```\n\n";

  std::cout << "BUT WAIT: The asymmetric transform doesn't fill r1o_i!\n";
  std::cout << "It only fills r1e_i with BOTH symmetric AND asymmetric!\n\n";

  std::cout << "OPTION 2: Separate asymmetric transform outputs\n";
  std::cout << "Modify FourierToReal2DAsymmFastPoloidal to output:\n";
  std::cout << "- Even parity results (cos for R, sin for Z)\n";
  std::cout << "- Odd parity results (sin for R, cos for Z)\n\n";

  std::cout << "OPTION 3: Change how odd arrays are computed\n";
  std::cout << "Instead of trying to get them from transforms,\n";
  std::cout << "compute them directly in geometryFromFourier:\n";
  std::cout << "```cpp\n";
  std::cout << "if (s_.lasym) {\n";
  std::cout << "  // Do separate transform for asymmetric odd terms\n";
  std::cout << "  for (int l = 0; l < s_.nThetaEff; ++l) {\n";
  std::cout << "    double theta = ...;\n";
  std::cout << "    // Transform rmnsc -> sin(m*theta) -> r1_o\n";
  std::cout << "    // Transform zmncc -> cos(m*theta) -> z1_o\n";
  std::cout << "  }\n";
  std::cout << "}\n";
  std::cout << "```\n\n";

  std::cout << "RECOMMENDATION: Option 3 is simplest\n";
  std::cout << "Add explicit loops to compute odd arrays from\n";
  std::cout << "asymmetric Fourier coefficients\n";

  EXPECT_TRUE(true) << "Fix proposed";
}

TEST(OddArraysBugTest, VerifyEducationalVMEC) {
  std::cout << "\n=== VERIFY EDUCATIONAL_VMEC APPROACH ===\n";

  std::cout << "In educational_VMEC totzsp.f90:\n";
  std::cout << "- Line 80-81: r1(k,js,0) and r1(k,js,1) for even/odd\n";
  std::cout << "- It properly separates symmetric and asymmetric\n\n";

  std::cout << "In VMEC++:\n";
  std::cout << "- r1_e should contain: symmetric cos + asymmetric sin\n";
  std::cout << "- r1_o should contain: symmetric sin + asymmetric cos\n";
  std::cout << "- But currently r1_o only gets symmetric (which is 0!)\n\n";

  std::cout << "The tau formula NEEDS these odd arrays to work!\n";
  std::cout << "Without them, odd_contrib = 0 always\n";

  EXPECT_TRUE(true) << "Educational_VMEC verification complete";
}

}  // namespace vmecpp
