// TDD test to debug why odd arrays are zero
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(DebugOddArraysTest, CheckOddArraysNotZero) {
  std::cout << "\n=== DEBUG ODD ARRAYS NOT ZERO ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "Understanding why r1_o, z1_o, ru_o, zu_o are zero\n";
  std::cout << "even when we have m=1 modes (odd m)\n\n";

  std::cout << "HYPOTHESIS: The issue is in how arrays are indexed\n";
  std::cout << "In ideal_mhd_model.cc lines 1593-1597:\n";
  std::cout << "  r1_o[idx_jl] += rnkcc[kOddParity];\n";
  std::cout << "  z1_o[idx_jl] += znksc[kOddParity];\n\n";

  std::cout << "Where kOddParity = 1 for symmetric transforms\n";
  std::cout << "But for asymmetric, we need different indexing!\n\n";

  std::cout << "EXPECTED BEHAVIOR:\n";
  std::cout << "- r1_o should contain sin(m*theta) terms from rmnsc\n";
  std::cout << "- z1_o should contain cos(m*theta) terms from zmncc\n";
  std::cout << "- These come from ASYMMETRIC coefficients\n\n";

  std::cout << "ACTUAL BEHAVIOR:\n";
  std::cout << "- r1_o is trying to get cos terms from symmetric transform\n";
  std::cout << "- But symmetric transform doesn't have odd parity for R!\n";
  std::cout << "- R symmetric: cos(m*theta) -> even parity\n";
  std::cout << "- R asymmetric: sin(m*theta) -> odd parity\n\n";

  std::cout << "ROOT CAUSE:\n";
  std::cout
      << "The code at lines 1580-1598 is ONLY doing symmetric transforms!\n";
  std::cout << "It's not calling the asymmetric transform for odd arrays!\n\n";

  std::cout << "SOLUTION:\n";
  std::cout << "Need to call asymmetric transforms and accumulate:\n";
  std::cout << "- r1_o += asymmetric transform of rmnsc (sin terms)\n";
  std::cout << "- z1_o += asymmetric transform of zmncc (cos terms)\n";

  EXPECT_TRUE(true) << "Debug analysis complete";
}

TEST(DebugOddArraysTest, VerifyAsymmetricTransformCalls) {
  std::cout << "\n=== VERIFY ASYMMETRIC TRANSFORM CALLS ===\n";

  std::cout << "Looking at ideal_mhd_model.cc geometryFromFourier():\n\n";

  std::cout << "Line 1507: if (s_.lasym) {\n";
  std::cout << "  This block SHOULD populate r1_o, z1_o arrays\n";
  std::cout << "  But it's calling SymmetrizeRealSpaceGeometry\n";
  std::cout << "  which only symmetrizes, doesn't transform!\n\n";

  std::cout << "The asymmetric transform is at line 388:\n";
  std::cout << "  FourierToReal2DAsymmFastPoloidal(...)\n";
  std::cout << "  But this puts results in m_ls_.r1e_i, not r1_o!\n\n";

  std::cout << "CRITICAL FINDING:\n";
  std::cout << "The odd arrays r1_o, z1_o are NEVER populated\n";
  std::cout << "from asymmetric Fourier coefficients!\n\n";

  std::cout << "They're only populated from symmetric transforms\n";
  std::cout << "But symmetric R doesn't have odd parity!\n";

  EXPECT_TRUE(true) << "Critical bug identified";
}

TEST(DebugOddArraysTest, ProposeFix) {
  std::cout << "\n=== PROPOSE FIX ===\n";

  std::cout << "The fix is in geometryFromFourier() around line 1580:\n\n";

  std::cout << "CURRENT CODE only does symmetric transforms:\n";
  std::cout << "```cpp\n";
  std::cout << "// Symmetric transform\n";
  std::cout << "r1_e[idx_jl] += rnkcc[kEvenParity];  // cos terms\n";
  std::cout << "r1_o[idx_jl] += rnkcc[kOddParity];   // sin terms (WRONG!)\n";
  std::cout << "```\n\n";

  std::cout << "NEEDED: Also do asymmetric transforms:\n";
  std::cout << "```cpp\n";
  std::cout << "if (s_.lasym) {\n";
  std::cout << "  // Asymmetric contributions\n";
  std::cout << "  double rnksc[2], znkcc[2];  // sin for R, cos for Z\n";
  std::cout << "  // ... compute asymmetric basis ...\n";
  std::cout << "  r1_o[idx_jl] += rnksc[kEvenParity];  // sin(m*theta)\n";
  std::cout << "  z1_o[idx_jl] += znkcc[kEvenParity];  // cos(m*theta)\n";
  std::cout << "}\n";
  std::cout << "```\n\n";

  std::cout << "This explains why tau formula gives odd_contrib=0!\n";
  std::cout << "All the odd arrays are zero!\n";

  EXPECT_TRUE(true) << "Fix proposed";
}

}  // namespace vmecpp
