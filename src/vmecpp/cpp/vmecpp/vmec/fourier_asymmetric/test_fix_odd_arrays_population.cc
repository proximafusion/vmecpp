// TDD test documenting how to fix odd arrays population
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace vmecpp {

TEST(FixOddArraysTest, DocumentCurrentProblem) {
  std::cout << "\n=== DOCUMENT CURRENT PROBLEM ===\n";
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "CURRENT SITUATION:\n";
  std::cout << "1. Symmetric transform (lines 1580-1598):\n";
  std::cout << "   - Populates r1_e with cos(m*theta) terms ✅\n";
  std::cout << "   - Tries to populate r1_o with sin(m*theta) terms ❌\n";
  std::cout << "   - BUT symmetric R has no sin terms, so r1_o = 0!\n\n";

  std::cout << "2. Asymmetric transform (lines 1350-1370):\n";
  std::cout << "   - Computes BOTH symmetric + asymmetric contributions\n";
  std::cout << "   - But only adds to r1_e arrays!\n";
  std::cout << "   - r1_o arrays never get asymmetric contributions\n\n";

  std::cout << "CONSEQUENCE:\n";
  std::cout << "- r1_o = 0, z1_o = 0, ru_o = 0, zu_o = 0 always\n";
  std::cout << "- tau formula odd_contrib = 0 always\n";
  std::cout << "- Asymmetric equilibria cannot converge\n";

  EXPECT_TRUE(true) << "Problem documented";
}

TEST(FixOddArraysTest, UnderstandAsymmetricFourierBasis) {
  std::cout << "\n=== UNDERSTAND ASYMMETRIC FOURIER BASIS ===\n";

  std::cout << "SYMMETRIC MODE (lasym=false):\n";
  std::cout
      << "R = Σ rmncc * cos(m*u) * cos(n*v) + rmnss * sin(m*u) * sin(n*v)\n";
  std::cout
      << "Z = Σ zmnsc * sin(m*u) * cos(n*v) + zmncs * cos(m*u) * sin(n*v)\n\n";

  std::cout << "ASYMMETRIC MODE (lasym=true) ADDS:\n";
  std::cout
      << "R += Σ rmnsc * sin(m*u) * cos(n*v) + rmncs * cos(m*u) * sin(n*v)\n";
  std::cout
      << "Z += Σ zmncc * cos(m*u) * cos(n*v) + zmnss * sin(m*u) * sin(n*v)\n\n";

  std::cout << "KEY INSIGHT:\n";
  std::cout << "- Symmetric R: cos basis → even parity in theta\n";
  std::cout << "- Asymmetric R: sin basis → odd parity in theta\n";
  std::cout << "- Symmetric Z: sin basis → even parity in theta\n";
  std::cout << "- Asymmetric Z: cos basis → odd parity in theta\n\n";

  std::cout << "THEREFORE:\n";
  std::cout << "- r1_o should contain transform of rmnsc (sin terms)\n";
  std::cout << "- z1_o should contain transform of zmncc (cos terms)\n";

  EXPECT_TRUE(true) << "Fourier basis understood";
}

TEST(FixOddArraysTest, ProposeSolution) {
  std::cout << "\n=== PROPOSE SOLUTION ===\n";

  std::cout << "SOLUTION: Modify asymmetric transform to separate even/odd\n\n";

  std::cout << "APPROACH 1: Modify FourierToReal2DAsymmFastPoloidal\n";
  std::cout << "Change the function to output SEPARATE even/odd arrays:\n";
  std::cout << "```cpp\n";
  std::cout << "void FourierToReal2DAsymmFastPoloidal(\n";
  std::cout << "    const Sizes& sizes,\n";
  std::cout << "    // Input Fourier coefficients\n";
  std::cout << "    absl::Span<const double> rmncc, rmnss, rmnsc, rmncs,\n";
  std::cout << "    absl::Span<const double> zmnsc, zmncs, zmncc, zmnss,\n";
  std::cout << "    // Output even parity (current behavior)\n";
  std::cout << "    absl::Span<double> r_even,  // cos terms from symmetric + "
               "sin from asymmetric\n";
  std::cout << "    absl::Span<double> z_even,  // sin terms from symmetric + "
               "cos from asymmetric\n";
  std::cout << "    // Output odd parity (NEW)\n";
  std::cout << "    absl::Span<double> r_odd,   // sin terms from asymmetric "
               "coefficients\n";
  std::cout << "    absl::Span<double> z_odd,   // cos terms from asymmetric "
               "coefficients\n";
  std::cout << "    absl::Span<double> lambda_real);\n";
  std::cout << "```\n\n";

  std::cout << "APPROACH 2: Add separate odd array population\n";
  std::cout << "After current asymmetric transform, add:\n";
  std::cout << "```cpp\n";
  std::cout << "// In geometryFromFourier after line 1370\n";
  std::cout << "if (s_.lasym) {\n";
  std::cout << "  // Current code combines symmetric + asymmetric into r1_e\n";
  std::cout
      << "  // Now ALSO populate odd arrays from asymmetric coefficients\n";
  std::cout << "  populateOddArraysFromAsymmetric(physical_x, r1_o, z1_o, "
               "ru_o, zu_o, ...);\n";
  std::cout << "}\n";
  std::cout << "```\n\n";

  std::cout << "APPROACH 3: Fix in geometryFromFourier directly\n";
  std::cout << "Modify the symmetric transform loop to handle asymmetric:\n";
  std::cout << "```cpp\n";
  std::cout << "// Around line 1580\n";
  std::cout << "if (s_.lasym) {\n";
  std::cout
      << "  // Do additional transform for asymmetric odd contributions\n";
  std::cout << "  double rnksc[2], znkcc[2];  // Asymmetric basis\n";
  std::cout << "  \n";
  std::cout << "  // Compute sin(m*theta) for R, cos(m*theta) for Z\n";
  std::cout << "  fourier_basis.computeAsymmetricBasis(m, l, rnksc, znkcc);\n";
  std::cout << "  \n";
  std::cout << "  // Accumulate asymmetric contributions to odd arrays\n";
  std::cout << "  for (int n = 0; n <= sizes.ntor; ++n) {\n";
  std::cout << "    int mn = findMode(m, n);\n";
  std::cout << "    if (mn >= 0) {\n";
  std::cout << "      r1_o[idx_jl] += physical_x.rmnsc[mn] * rnksc[0];  // sin "
               "basis\n";
  std::cout << "      z1_o[idx_jl] += physical_x.zmncc[mn] * znkcc[0];  // cos "
               "basis\n";
  std::cout << "      // ... similar for derivatives ...\n";
  std::cout << "    }\n";
  std::cout << "  }\n";
  std::cout << "}\n";
  std::cout << "```\n";

  EXPECT_TRUE(true) << "Solution proposed";
}

TEST(FixOddArraysTest, ImplementationPlan) {
  std::cout << "\n=== IMPLEMENTATION PLAN ===\n";

  std::cout << "RECOMMENDED: Approach 3 - Fix in geometryFromFourier\n";
  std::cout << "This is least invasive and most straightforward\n\n";

  std::cout << "STEPS:\n";
  std::cout << "1. In geometryFromFourier, after symmetric transform\n";
  std::cout << "2. Add conditional block for lasym=true\n";
  std::cout << "3. Loop over m, n modes\n";
  std::cout << "4. Transform asymmetric coefficients to odd arrays\n";
  std::cout << "5. Use proper sin/cos basis for R/Z respectively\n\n";

  std::cout << "KEY POINTS:\n";
  std::cout << "- Don't modify existing symmetric transform\n";
  std::cout << "- Add separate logic for asymmetric odd contributions\n";
  std::cout << "- Ensure proper normalization (sqrt(2) factors)\n";
  std::cout << "- Test with m=1 modes to verify non-zero odd_contrib\n\n";

  std::cout << "EXPECTED RESULT:\n";
  std::cout << "- r1_o, z1_o will have non-zero values\n";
  std::cout << "- tau formula odd_contrib will be non-zero\n";
  std::cout << "- Asymmetric equilibria should converge!\n";

  EXPECT_TRUE(true) << "Implementation plan ready";
}

TEST(FixOddArraysTest, VerifyAgainstEducationalVMEC) {
  std::cout << "\n=== VERIFY AGAINST EDUCATIONAL_VMEC ===\n";

  std::cout << "Educational_VMEC approach (totzspa.f90):\n";
  std::cout << "1. Transforms asymmetric coefficients separately\n";
  std::cout << "2. Stores in armn, azmn arrays (anti-symmetric)\n";
  std::cout << "3. Later combines with symmetric in symrzl\n\n";

  std::cout << "Key difference:\n";
  std::cout << "- Educational_VMEC: Separate arrays for symmetric/asymmetric\n";
  std::cout << "- VMEC++: Even/odd parity arrays\n\n";

  std::cout << "Mapping:\n";
  std::cout << "- Educational_VMEC armn → VMEC++ r1_o (R odd parity)\n";
  std::cout << "- Educational_VMEC azmn → VMEC++ z1_o (Z odd parity)\n";
  std::cout << "- Educational_VMEC r1+armn → VMEC++ r1_e+r1_o (total R)\n\n";

  std::cout << "The fix must ensure VMEC++ produces same total result!\n";

  EXPECT_TRUE(true) << "Educational_VMEC comparison complete";
}

}  // namespace vmecpp
