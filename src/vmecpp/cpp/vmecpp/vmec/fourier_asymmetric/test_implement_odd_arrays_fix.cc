// TDD test showing exact implementation of odd arrays fix
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace vmecpp {

TEST(ImplementOddArraysFixTest, ShowExactCodeLocation) {
  std::cout << "\n=== SHOW EXACT CODE LOCATION ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "LOCATION: ideal_mhd_model.cc, after line 1366\n";
  std::cout << "Currently says: 'Note: r1_o and z1_o are not filled...'\n\n";

  std::cout << "This is EXACTLY where we need to add the fix!\n";
  std::cout << "The asymmetric transform has already computed values\n";
  std::cout << "We just need to populate the odd arrays\n";

  EXPECT_TRUE(true) << "Location identified";
}

TEST(ImplementOddArraysFixTest, ShowMinimalFix) {
  std::cout << "\n=== SHOW MINIMAL FIX ===\n";

  std::cout << "OPTION 1: Quick fix - reuse existing transform\n";
  std::cout << "Since asymmetric transform already ran, we can extract odd "
               "parts:\n\n";

  std::cout << "```cpp\n";
  std::cout << "// After line 1366, add:\n";
  std::cout << "if (s_.lasym) {\n";
  std::cout << "  // The asymmetric transform combined everything into r1e_i\n";
  std::cout
      << "  // But we need to separate odd contributions for tau formula\n";
  std::cout << "  // For now, just copy to odd arrays (will refine later)\n";
  std::cout << "  r1_o[idx] = m_ls_.r1e_i[idx];  // Contains asymmetric "
               "contributions\n";
  std::cout << "  z1_o[idx] = m_ls_.z1e_i[idx];\n";
  std::cout << "}\n";
  std::cout << "```\n\n";

  std::cout << "This is WRONG but will show if the tau formula works!\n";
  std::cout
      << "If odd_contrib becomes non-zero, we know we're on right track\n";

  EXPECT_TRUE(true) << "Quick test fix shown";
}

TEST(ImplementOddArraysFixTest, ShowProperFix) {
  std::cout << "\n=== SHOW PROPER FIX ===\n";

  std::cout << "The proper fix requires computing odd parity separately:\n\n";

  std::cout << "```cpp\n";
  std::cout
      << "// After current array combination (line 1379), add new section:\n";
  std::cout << "if (s_.lasym) {\n";
  std::cout
      << "  // Populate odd arrays from asymmetric Fourier coefficients\n";
  std::cout << "  // These were not computed by the asymmetric transform\n";
  std::cout << "  \n";
  std::cout << "  for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {\n";
  std::cout << "    double* src_rsc = &(physical_x.rmnsc[(jF - r_.nsMinF1) * "
               "s_.mnsize]);\n";
  std::cout << "    double* src_zcc = &(physical_x.zmncc[(jF - r_.nsMinF1) * "
               "s_.mnsize]);\n";
  std::cout << "    \n";
  std::cout << "    for (int l = 0; l < s_.nThetaEff; ++l) {\n";
  std::cout << "      double theta = 2.0 * M_PI * l / s_.nThetaEff;\n";
  std::cout << "      int idx_jl = (jF - r_.nsMinF1) * s_.nThetaEff + l;\n";
  std::cout << "      \n";
  std::cout << "      // Transform asymmetric coefficients to odd arrays\n";
  std::cout << "      for (int mn = 0; mn < s_.mnmax; ++mn) {\n";
  std::cout << "        int m = xm[mn];\n";
  std::cout << "        int n = xn[mn] / s_.nfp;\n";
  std::cout << "        \n";
  std::cout << "        // Basis functions for odd parity\n";
  std::cout << "        double sin_mu = sin(m * theta);\n";
  std::cout << "        double cos_mu = cos(m * theta);\n";
  std::cout << "        if (m > 0) {\n";
  std::cout << "          sin_mu *= sqrt(2.0);  // Normalization\n";
  std::cout << "          cos_mu *= sqrt(2.0);\n";
  std::cout << "        }\n";
  std::cout << "        \n";
  std::cout << "        // R odd: sin(m*theta) from rmnsc\n";
  std::cout << "        r1_o[idx_jl] += src_rsc[mn] * sin_mu;\n";
  std::cout
      << "        ru_o[idx_jl] += src_rsc[mn] * m * cos_mu;  // derivative\n";
  std::cout << "        \n";
  std::cout << "        // Z odd: cos(m*theta) from zmncc\n";
  std::cout << "        z1_o[idx_jl] += src_zcc[mn] * cos_mu;\n";
  std::cout << "        zu_o[idx_jl] += src_zcc[mn] * (-m) * sin_mu;  // "
               "derivative\n";
  std::cout << "      }\n";
  std::cout << "    }\n";
  std::cout << "  }\n";
  std::cout << "}\n";
  std::cout << "```\n";

  EXPECT_TRUE(true) << "Proper fix shown";
}

TEST(ImplementOddArraysFixTest, ShowIntegrationApproach) {
  std::cout << "\n=== SHOW INTEGRATION APPROACH ===\n";

  std::cout << "BETTER APPROACH: Modify existing symmetric transform loop\n";
  std::cout << "This avoids code duplication and is more efficient\n\n";

  std::cout << "In the main transform loop (around line 1540), add:\n";
  std::cout << "```cpp\n";
  std::cout << "// After accumulating symmetric contributions (line 1593)\n";
  std::cout << "if (s_.lasym && !lthreed) {  // 2D asymmetric case\n";
  std::cout << "  // Also accumulate asymmetric odd contributions\n";
  std::cout << "  // Find the asymmetric coefficients for this (m,n)\n";
  std::cout << "  double rsc_coeff = 0.0, zcc_coeff = 0.0;\n";
  std::cout << "  if (mn >= 0 && mn < s_.mnmax) {\n";
  std::cout << "    rsc_coeff = physical_x.rmnsc[(jF - r_.nsMinF1) * s_.mnsize "
               "+ mn];\n";
  std::cout << "    zcc_coeff = physical_x.zmncc[(jF - r_.nsMinF1) * s_.mnsize "
               "+ mn];\n";
  std::cout << "  }\n";
  std::cout << "  \n";
  std::cout << "  // R odd parity: sin(m*theta) terms\n";
  std::cout
      << "  double sin_basis = rnkcc[kOddParity];  // This is sin(m*theta)\n";
  std::cout << "  r1_o[idx_jl] += rsc_coeff * cosnv * sin_basis;\n";
  std::cout << "  \n";
  std::cout << "  // Z odd parity: cos(m*theta) terms\n";
  std::cout
      << "  double cos_basis = znksc[kOddParity];  // This is cos(m*theta)\n";
  std::cout << "  z1_o[idx_jl] += zcc_coeff * cosnv * cos_basis;\n";
  std::cout << "}\n";
  std::cout << "```\n";

  std::cout << "\nThis integrates naturally with existing code!\n";

  EXPECT_TRUE(true) << "Integration approach shown";
}

TEST(ImplementOddArraysFixTest, TestingStrategy) {
  std::cout << "\n=== TESTING STRATEGY ===\n";

  std::cout << "1. FIRST: Try quick fix to verify tau formula\n";
  std::cout << "   - Just copy r1e_i to r1_o (wrong but tests concept)\n";
  std::cout << "   - Run test_odd_m_modes to see if odd_contrib != 0\n";
  std::cout << "   - If yes, tau formula is working!\n\n";

  std::cout << "2. THEN: Implement proper odd array population\n";
  std::cout << "   - Use integration approach for efficiency\n";
  std::cout << "   - Verify with m=1 modes test\n";
  std::cout << "   - Check that odd arrays have expected sin/cos pattern\n\n";

  std::cout << "3. FINALLY: Full convergence test\n";
  std::cout << "   - Run test_embedded_asymmetric_tokamak\n";
  std::cout << "   - Should converge without Jacobian errors!\n";
  std::cout << "   - First successful asymmetric equilibrium\n\n";

  std::cout << "SUCCESS CRITERIA:\n";
  std::cout << "- odd_contrib != 0 in tau calculation\n";
  std::cout << "- No Jacobian sign change errors\n";
  std::cout << "- Asymmetric equilibrium converges\n";

  EXPECT_TRUE(true) << "Testing strategy defined";
}

}  // namespace vmecpp
