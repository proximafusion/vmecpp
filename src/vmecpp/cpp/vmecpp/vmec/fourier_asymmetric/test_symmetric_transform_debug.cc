// Test to add debug output to symmetric transform
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(SymmetricTransformDebugTest, AddDebugToTransform) {
  std::cout << "\n=== ADD DEBUG TO SYMMETRIC TRANSFORM ===\n";
  std::cout << std::fixed << std::setprecision(6);

  std::cout << "GOAL: Add debug output to dft_FourierToReal_2d_symm to see:\n";
  std::cout << "1. Input Fourier coefficients (src_rcc, src_zsc)\n";
  std::cout << "2. Basis functions (cosmu, sinmu) at different theta points\n";
  std::cout
      << "3. Intermediate values (rnkcc, znksc) before array combination\n";
  std::cout << "4. Final real-space values at critical points\n";

  std::cout << "\nCRITICAL DEBUG POINTS:\n";
  std::cout << "- At l=0 (θ=0): expect R ≈ R0 + a\n";
  std::cout << "- At l=6 (θ=π): expect R ≈ R0 - a\n";
  std::cout << "- If R=18 at l=6, something is wrong in the transform\n";

  std::cout << "\nTo add to ideal_mhd_model.cc dft_FourierToReal_2d_symm:\n";
  std::cout << "```cpp\n";
  std::cout << "// DEBUG: Print input coefficients\n";
  std::cout << "if (jF == 1) {  // First interior surface\n";
  std::cout << "  std::cout << \"Input coeffs: rcc[0]=\" << src_rcc[0] \n";
  std::cout << "               << \", rcc[1]=\" << src_rcc[1] << std::endl;\n";
  std::cout << "}\n";
  std::cout << "\n";
  std::cout << "// DEBUG: After m-loop, before array combination\n";
  std::cout << "if (jF == 1 && (l == 0 || l == 6)) {\n";
  std::cout
      << "  std::cout << \"l=\" << l << \" theta=\" << (2*M_PI*l/s_.ntheta)\n";
  std::cout << "           << \" rnkcc[0]=\" << rnkcc[0] \n";
  std::cout << "           << \" rnkcc[1]=\" << rnkcc[1] << std::endl;\n";
  std::cout << "}\n";
  std::cout << "```\n";

  std::cout << "\nNEXT STEPS:\n";
  std::cout << "1. Modify ideal_mhd_model.cc to add this debug output\n";
  std::cout << "2. Run test_symmetric_tau_only to see the values\n";
  std::cout << "3. Identify why R=18 at θ=π instead of R=8\n";
  std::cout << "4. Compare with expected circular tokamak geometry\n";

  EXPECT_TRUE(true) << "Debug planning complete";
}

TEST(SymmetricTransformDebugTest, AnalyzeArrayCombination) {
  std::cout << "\n=== ANALYZE ARRAY COMBINATION LOGIC ===\n";

  std::cout << "From symmetric transform:\n";
  std::cout << "- rnkcc[0] = sum of even-m modes (m=0,2,4,...)\n";
  std::cout << "- rnkcc[1] = sum of odd-m modes (m=1,3,5,...)\n";
  std::cout << "\n";

  std::cout << "Array combination in real space:\n";
  std::cout << "```cpp\n";
  std::cout << "const int kl = (jF - r_.nsMinF1) * s_.nThetaEff + l;\n";
  std::cout << "r1_e[kl] = rnkcc[0];  // even-m contribution\n";
  std::cout << "r1_o[kl] = rnkcc[1];  // odd-m contribution\n";
  std::cout << "```\n";

  std::cout << "\nLater array combination:\n";
  std::cout << "```cpp\n";
  std::cout
      << "// Array combination for symmetric half (0 ≤ l < nThetaReduced)\n";
  std::cout << "realspaceGm.r[kl] = r1_e[kl] + r1_o[kl];\n";
  std::cout << "\n";
  std::cout << "// Array combination for extended half (nThetaReduced ≤ l < "
               "nThetaEff)\n";
  std::cout << "int kl_sym = /* symmetric point */;\n";
  std::cout
      << "realspaceGm.r[kl] = r1_e[kl_sym] - r1_o[kl_sym];  // SIGN CHANGE!\n";
  std::cout << "```\n";

  std::cout << "\nTHEORY:\n";
  std::cout << "For circular tokamak R = R0 + a*cos(θ):\n";
  std::cout << "- rcc[0] = R0 (m=0 mode)\n";
  std::cout << "- rcc[1] = a (m=1 mode)\n";
  std::cout << "- At θ=0: R = R0 + a\n";
  std::cout << "- At θ=π: R = R0 - a (due to cos(π) = -1)\n";

  std::cout << "\nIF R=18 at θ=π:\n";
  std::cout << "- Expected: R = 10 - 2 = 8\n";
  std::cout << "- Actual: R = 18\n";
  std::cout << "- Suggests: R = 10 + 8 = 18\n";
  std::cout << "- WRONG SIGN in array combination!\n";

  EXPECT_TRUE(true) << "Array combination analysis complete";
}

}  // namespace vmecpp
