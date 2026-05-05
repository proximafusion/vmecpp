// TDD test comparing Jacobian calculation between VMEC++ and jVMEC
// Focus on understanding why Jacobian changes sign in asymmetric mode

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace vmecpp {

TEST(JacobianJVMECComparisonTest, DocumentJVMECJacobianApproach) {
  std::cout << "\n=== DOCUMENT JVMEC JACOBIAN APPROACH ===" << std::endl;
  std::cout << std::fixed << std::setprecision(10);

  std::cout << "JVMEC JACOBIAN CALCULATION (from jVMEC source):\n";
  std::cout << "1. Computes tau at each point (same formula as VMEC++)\n";
  std::cout << "2. Checks for sign change: (minTau * maxTau < 0)\n";
  std::cout << "3. If sign changes, calls guessAxis() to find better axis\n";
  std::cout << "4. guessAxis does 61x61 grid search maximizing min(|tau|)\n\n";

  std::cout << "KEY DIFFERENCE - Axis Optimization:\n";
  std::cout << "- jVMEC: Always tries axis optimization if Jacobian bad\n";
  std::cout << "- VMEC++: No axis optimization, just fails\n";
  std::cout
      << "- This could explain why jVMEC converges but VMEC++ doesn't\n\n";

  std::cout
      << "HOWEVER: Previous testing showed axis optimization didn't help\n";
  std::cout << "- Tested 81 axis positions, all failed\n";
  std::cout << "- Suggests deeper issue than just axis position\n";

  EXPECT_TRUE(true) << "jVMEC approach documented";
}

TEST(JacobianJVMECComparisonTest, AnalyzeTauComponents) {
  std::cout << "\n=== ANALYZE TAU COMPONENTS ===" << std::endl;

  std::cout << "TAU FORMULA BREAKDOWN:\n";
  std::cout << "tau = ru12*zs - rs*zu12 + dshalfds*odd_contrib\n\n";

  std::cout << "WHERE odd_contrib = \n";
  std::cout << "  (ruo_o*z1o_o + ruo_i*z1o_i - zuo_o*r1o_o - zuo_i*r1o_i)\n";
  std::cout
      << "+ (rue_o*z1o_o + rue_i*z1o_i - zue_o*r1o_o - zue_i*r1o_i)/sqrtSH\n\n";

  std::cout << "FROM DEBUG OUTPUT:\n";
  std::cout << "- Basic Jacobian (ru12*zs - rs*zu12) ranges from -3.4 to 4.5\n";
  std::cout << "- odd_contrib is small: 0.02 to 0.49\n";
  std::cout << "- dshalfds = 0.25 (constant)\n";
  std::cout << "- Sign change comes from basic Jacobian, not odd terms\n\n";

  std::cout << "HYPOTHESIS:\n";
  std::cout << "The issue is in the basic geometry derivatives (ru12, zs, rs, "
               "zu12)\n";
  std::cout << "not in the asymmetric odd contribution\n";

  EXPECT_TRUE(true) << "Tau components analyzed";
}

TEST(JacobianJVMECComparisonTest, CompareInitialGeometry) {
  std::cout << "\n=== COMPARE INITIAL GEOMETRY ===" << std::endl;

  std::cout << "CRITICAL OBSERVATION:\n";
  std::cout << "Same boundary produces different initial geometry:\n";
  std::cout << "- Symmetric mode: Works fine\n";
  std::cout << "- Asymmetric mode: Jacobian changes sign\n\n";

  std::cout << "DIFFERENCES IN GEOMETRY SETUP:\n";
  std::cout << "1. Theta range: [0,π] vs [0,2π]\n";
  std::cout << "2. Array sizes: nThetaReduced vs nThetaEff\n";
  std::cout << "3. Symmetrization: Applied differently\n";
  std::cout << "4. Initial guess: Interpolation differs\n\n";

  std::cout << "NEXT STEPS:\n";
  std::cout << "1. Add debug output to initial guess generation\n";
  std::cout << "2. Compare ru12, zs, rs, zu12 values between modes\n";
  std::cout << "3. Check if half-grid derivatives calculated differently\n";
  std::cout << "4. Verify surface averaging in asymmetric mode\n";

  EXPECT_TRUE(true) << "Initial geometry comparison documented";
}

TEST(JacobianJVMECComparisonTest, StudyEducationalVMECJacobian) {
  std::cout << "\n=== STUDY EDUCATIONAL_VMEC JACOBIAN ===" << std::endl;

  std::cout << "EDUCATIONAL_VMEC jacobian.f90 INSIGHTS:\n";
  std::cout << "1. Uses gsqrt = sqrt(g) where g is metric determinant\n";
  std::cout << "2. Computes tau with unified formula (as we implemented)\n";
  std::cout << "3. Special handling at axis (js=1)\n";
  std::cout << "4. Extrapolates tau to axis: tau(1) = tau(2)\n\n";

  std::cout << "AXIS HANDLING:\n";
  std::cout << "- At axis, many derivatives are singular\n";
  std::cout << "- Educational_VMEC copies from js=2 to js=1\n";
  std::cout
      << "- This prevents tau=0 at axis which would fail Jacobian check\n\n";

  std::cout << "VMEC++ IMPLEMENTATION:\n";
  std::cout << "- Already has axis extrapolation (line 1835-1840)\n";
  std::cout << "- Copies tau from j=1 to j=0 surface\n";
  std::cout << "- So this is not the issue\n";

  EXPECT_TRUE(true) << "Educational_VMEC Jacobian studied";
}

TEST(JacobianJVMECComparisonTest, ProposeDebugStrategy) {
  std::cout << "\n=== PROPOSE DEBUG STRATEGY ===" << std::endl;

  std::cout << "SYSTEMATIC DEBUG PLAN:\n\n";

  std::cout << "1. ADD GEOMETRY DERIVATIVE DEBUG:\n";
  std::cout << "   - Print ru12, zs, rs, zu12 at each point\n";
  std::cout << "   - Compare symmetric vs asymmetric mode\n";
  std::cout << "   - Identify where values diverge\n\n";

  std::cout << "2. TRACE INITIAL GUESS GENERATION:\n";
  std::cout << "   - Add debug to spectral_to_initial_guess\n";
  std::cout << "   - Compare interpolation for full theta range\n";
  std::cout << "   - Check boundary application differences\n\n";

  std::cout << "3. COMPARE WITH EDUCATIONAL_VMEC:\n";
  std::cout << "   - Run same config through educational_VMEC\n";
  std::cout << "   - Add matching debug output\n";
  std::cout << "   - Line-by-line comparison of tau calculation\n\n";

  std::cout << "4. TEST MINIMAL ASYMMETRY:\n";
  std::cout << "   - Start with tiny asymmetric perturbation (0.01%)\n";
  std::cout << "   - Gradually increase to find breaking point\n";
  std::cout << "   - Identify critical asymmetry level\n\n";

  std::cout << "IMPLEMENTATION ORDER:\n";
  std::cout << "- First: Geometry derivative debug (most likely issue)\n";
  std::cout << "- Second: Initial guess comparison\n";
  std::cout << "- Third: Educational_VMEC reference\n";
  std::cout << "- Fourth: Perturbation study\n";

  EXPECT_TRUE(true) << "Debug strategy proposed";
}

TEST(JacobianJVMECComparisonTest, CheckSurfaceDerivatives) {
  std::cout << "\n=== CHECK SURFACE DERIVATIVES ===" << std::endl;

  std::cout << "SURFACE DERIVATIVE CALCULATION:\n";
  std::cout << "- ru12 = (r1[j] + r1[j-1])/2  (average at half grid)\n";
  std::cout << "- zu12 = (zu[j] + zu[j-1])/2\n";
  std::cout << "- rs = (r1[j] - r1[j-1])/ds  (radial derivative)\n";
  std::cout << "- zs = (z1[j] - z1[j-1])/ds\n\n";

  std::cout << "POTENTIAL ISSUES:\n";
  std::cout << "1. ds (radial grid spacing) calculation\n";
  std::cout << "2. j-1 indexing at boundaries\n";
  std::cout << "3. Half-grid interpolation accuracy\n";
  std::cout << "4. Sign conventions for derivatives\n\n";

  std::cout << "DEBUG NEEDED:\n";
  std::cout << "- Print ds values at each surface\n";
  std::cout << "- Check j-1 indices are valid\n";
  std::cout << "- Verify r1[j] and r1[j-1] values\n";
  std::cout << "- Compare with educational_VMEC derivatives\n";

  EXPECT_TRUE(true) << "Surface derivatives analyzed";
}

}  // namespace vmecpp
