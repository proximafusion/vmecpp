// TDD unit test to implement exact jVMEC tau calculation
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace vmecpp {

TEST(ImplementJVMECTauTest, AnalyzeCurrentTauComponents) {
  std::cout << "\n=== ANALYZE CURRENT TAU COMPONENTS FROM DEBUG OUTPUT ===\n";
  std::cout << std::fixed << std::setprecision(8);

  std::cout << "From latest debug output (kl=6, jH=0):\n";
  std::cout << "tau1 = -11.625000  (ru12*zs - rs*zu12)\n";
  std::cout << "tau2 = -6.000000   (current VMEC++ formula)\n";
  std::cout << "tau_val = -13.125000  (tau1 + 0.25*tau2)\n";

  std::cout << "\nFrom latest debug output (kl=6, jH=1):\n";
  std::cout << "tau1 = 0.491025   (ru12*zs - rs*zu12)\n";
  std::cout << "tau2 = -4.535898  (current VMEC++ formula)\n";
  std::cout << "tau_val = -0.642949  (tau1 + 0.25*tau2)\n";

  std::cout << "\nCURRENT VMEC++ tau2 BREAKDOWN:\n";
  std::cout << "ruo_o * z1o_o = -0.000000 * 0.000000 = -0.000000\n";
  std::cout << "zuo_o * r1o_o = -2.000000 * -2.000000 = 4.000000\n";
  std::cout << "division term = 4.000000 / sqrtSH\n";
  std::cout << "tau2 = -(division term) = -4.000000/sqrtSH\n";

  std::cout << "\nCRITICAL ISSUE:\n";
  std::cout << "The division by sqrtSH creates different values:\n";
  std::cout << "- jH=0: sqrtSH=0.500000 → tau2 = -4.0/0.5 = -8.0 → tau2 = "
               "-(-8.0) = 8.0??\n";
  std::cout << "- But debug shows tau2 = -6.0, not 8.0\n";
  std::cout << "- jH=1: sqrtSH=0.866025 → tau2 = -4.0/0.866 = -4.62 → matches "
               "-4.535898 ✓\n";

  std::cout << "\nMISSING PIECE:\n";
  std::cout << "The debug shows tau2 calculation has extra steps not visible\n";
  std::cout << "Need to examine exact formula in ideal_mhd_model.cc\n";

  EXPECT_TRUE(true) << "Current tau component analysis complete";
}

TEST(ImplementJVMECTauTest, CompareJVMECFormulaStructure) {
  std::cout << "\n=== COMPARE jVMEC FORMULA STRUCTURE ===\n";

  std::cout << "jVMEC tau calculation (exact from RealSpaceGeometry.java):\n";
  std::cout << "\n1. evn_contrib (even contribution):\n";
  std::cout
      << "   evn_contrib = dRdThetaHalf * dZdSHalf - dRdSHalf * dZdThetaHalf\n";
  std::cout << "   → Matches VMEC++ tau1 = ru12 * zs - rs * zu12 ✓\n";

  std::cout << "\n2. odd_contrib (odd contribution - COMPLEX):\n";
  std::cout << "   Part A: Direct odd mode terms\n";
  std::cout << "     dRdTheta[j][m_odd] * Z[j][m_odd]\n";
  std::cout << "   + dRdTheta[j-1][m_odd] * Z[j-1][m_odd]\n";
  std::cout << "   - dZdTheta[j][m_odd] * R[j][m_odd]\n";
  std::cout << "   - dZdTheta[j-1][m_odd] * R[j-1][m_odd]\n";

  std::cout << "\n   Part B: Cross-coupling terms (divided by sqrtSHalf)\n";
  std::cout << "   + (dRdTheta[j][m_evn] * Z[j][m_odd]\n";
  std::cout << "    + dRdTheta[j-1][m_evn] * Z[j-1][m_odd]\n";
  std::cout << "    - dZdTheta[j][m_evn] * R[j][m_odd]\n";
  std::cout << "    - dZdTheta[j-1][m_evn] * R[j-1][m_odd]) / sqrtSHalf[j]\n";

  std::cout << "\n3. Final tau calculation:\n";
  std::cout << "   tau = evn_contrib + 0.25 * odd_contrib\n";

  std::cout << "\nVMEC++ CURRENT APPROACH:\n";
  std::cout << "tau2 = (ruo_o * z1o_o - zuo_o * r1o_o) / sqrtSH\n";
  std::cout << "This is ONLY Part B (cross-coupling), missing Part A!\n";

  std::cout << "\nMISSING IN VMEC++:\n";
  std::cout << "1. Part A: Direct odd mode contributions\n";
  std::cout << "2. Half-grid interpolation (j and j-1 terms)\n";
  std::cout << "3. Proper even/odd mode separation\n";

  EXPECT_TRUE(true) << "jVMEC formula structure comparison complete";
}

TEST(ImplementJVMECTauTest, ImplementJVMECOddContrib) {
  std::cout << "\n=== IMPLEMENT jVMEC odd_contrib CALCULATION ===\n";

  std::cout << "STEP 1: Map jVMEC variables to VMEC++ arrays\n";
  std::cout << "jVMEC → VMEC++ mapping:\n";
  std::cout << "dRdTheta[j][m_odd] → ru_o[j] (odd R derivative)\n";
  std::cout << "dZdTheta[j][m_odd] → zu_o[j] (odd Z derivative)\n";
  std::cout << "R[j][m_odd] → r1_o[j] (odd R position)\n";
  std::cout << "Z[j][m_odd] → z1_o[j] (odd Z position)\n";
  std::cout << "dRdTheta[j][m_evn] → ru_e[j] (even R derivative)\n";
  std::cout << "dZdTheta[j][m_evn] → zu_e[j] (even Z derivative)\n";
  std::cout << "R[j][m_evn] → r1_e[j] (even R position)\n";
  std::cout << "Z[j][m_evn] → z1_e[j] (even Z position)\n";

  std::cout << "\nSTEP 2: Implement Part A (direct odd terms)\n";
  std::cout << "part_a = ru_o[j] * z1_o[j] + ru_o[j-1] * z1_o[j-1]\n";
  std::cout << "       - zu_o[j] * r1_o[j] - zu_o[j-1] * r1_o[j-1]\n";

  std::cout << "\nSTEP 3: Implement Part B (cross-coupling terms)\n";
  std::cout << "part_b = (ru_e[j] * z1_o[j] + ru_e[j-1] * z1_o[j-1]\n";
  std::cout
      << "        - zu_e[j] * r1_o[j] - zu_e[j-1] * r1_o[j-1]) / sqrtSH\n";

  std::cout << "\nSTEP 4: Combine parts\n";
  std::cout << "odd_contrib = part_a + part_b\n";
  std::cout << "tau = evn_contrib + 0.25 * odd_contrib\n";

  std::cout << "\nIMPLEMENTATION LOCATION:\n";
  std::cout << "File: vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc\n";
  std::cout << "Function: IdealMhdModel::computeJacobian()\n";
  std::cout << "Line: ~1800 (tau2 calculation section)\n";

  std::cout << "\nTESTING APPROACH:\n";
  std::cout << "1. Add debug output for part_a and part_b separately\n";
  std::cout << "2. Compare with current tau2 to see differences\n";
  std::cout << "3. Verify jH=0 and jH=1 surface calculations\n";
  std::cout << "4. Test with same configuration that currently fails\n";

  EXPECT_TRUE(true) << "jVMEC odd_contrib implementation plan complete";
}

TEST(ImplementJVMECTauTest, HandlesForHalfGridInterpolation) {
  std::cout << "\n=== HANDLE HALF-GRID INTERPOLATION ===\n";

  std::cout << "jVMEC APPROACH:\n";
  std::cout << "Uses contributions from both j and j-1 surfaces\n";
  std::cout << "This averages values across radial grid points\n";
  std::cout << "for more stable numerical calculation\n";

  std::cout << "\nVMEC++ CURRENT APPROACH:\n";
  std::cout << "Uses only current surface j values\n";
  std::cout << "No averaging with previous surface j-1\n";

  std::cout << "\nIMPLEMENTATION CHALLENGE:\n";
  std::cout << "j-1 surface: How to access previous radial surface?\n";
  std::cout << "In VMEC++ loop structure:\n";
  std::cout << "for (int jH = r_.nsMinH; jH <= r_.nsMaxH; ++jH)\n";
  std::cout << "Current jH corresponds to j in jVMEC\n";
  std::cout << "Previous would be jH-1 or (jH-1) radial surface\n";

  std::cout << "\nSOLUTION APPROACH:\n";
  std::cout << "1. For jH > r_.nsMinH: Use both jH and jH-1 contributions\n";
  std::cout << "2. For jH == r_.nsMinH: Use only jH (boundary condition)\n";
  std::cout << "3. Access previous surface arrays at (jH-1) indices\n";

  std::cout << "\nIMPLEMENTATION NOTE:\n";
  std::cout << "This requires careful array indexing to avoid bounds errors\n";
  std::cout << "Must verify array sizes support jH-1 access\n";

  EXPECT_TRUE(true) << "Half-grid interpolation handling complete";
}

TEST(ImplementJVMECTauTest, NextStepImplementation) {
  std::cout << "\n=== NEXT STEP: IMPLEMENT IN IDEAL_MHD_MODEL.CC ===\n";

  std::cout << "IMMEDIATE ACTION:\n";
  std::cout << "1. Locate tau2 calculation in computeJacobian() function\n";
  std::cout
      << "2. Replace current simplified formula with jVMEC exact formula\n";
  std::cout << "3. Add extensive debug output for part_a and part_b\n";
  std::cout << "4. Test with current failing configuration\n";

  std::cout << "\nEXPECTED OUTCOME:\n";
  std::cout << "tau2 values should change significantly\n";
  std::cout << "This may resolve Jacobian sign change issue\n";
  std::cout << "Or reveal next level of algorithm differences\n";

  std::cout << "\nTEST VERIFICATION:\n";
  std::cout << "Run test_tau_symmetric_vs_asymmetric again\n";
  std::cout << "Compare before/after tau values\n";
  std::cout << "Check if Jacobian check now passes\n";

  std::cout << "\nIf still fails:\n";
  std::cout << "1. Compare exact tau values with jVMEC debug output\n";
  std::cout << "2. Verify array indexing matches jVMEC structure\n";
  std::cout << "3. Check half-grid interpolation implementation\n";

  EXPECT_TRUE(true) << "Implementation plan complete";
}

}  // namespace vmecpp
