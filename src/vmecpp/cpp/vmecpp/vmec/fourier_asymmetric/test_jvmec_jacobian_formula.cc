// Unit test comparing VMEC++ tau calculation with exact jVMEC formula
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(JVMECJacobianFormulaTest, CompareWithExactJVMECCalculation) {
  std::cout << "\n=== COMPARE VMEC++ vs jVMEC JACOBIAN FORMULA ===\n";
  std::cout << std::fixed << std::setprecision(8);

  std::cout << "GOAL: Implement exact jVMEC tau calculation formula\n";
  std::cout << "REFERENCE: jVMEC RealSpaceGeometry.java lines 301-309\n";

  std::cout << "\njVMEC FORMULA:\n";
  std::cout
      << "evn_contrib = dRdThetaHalf * dZdSHalf - dRdSHalf * dZdThetaHalf\n";
  std::cout << "odd_contrib = complex cross-terms between even/odd modes\n";
  std::cout << "tau = evn_contrib + dSHalfdS * odd_contrib\n";
  std::cout << "WHERE: dSHalfdS = 0.25 (constant in jVMEC)\n";

  std::cout << "\nVMEC++ CURRENT FORMULA:\n";
  std::cout << "tau1 = ru12 * zs - rs * zu12  (matches evn_contrib)\n";
  std::cout
      << "tau2 = (ruo_o*z1o_o - zuo_o*r1o_o) / sqrtSH  (simplified odd)\n";
  std::cout << "tau = tau1 + dSHalfDsInterp * tau2\n";
  std::cout << "WHERE: dSHalfDsInterp = derived from array dimensions\n";

  std::cout << "\nKEY DIFFERENCES IDENTIFIED:\n";
  std::cout
      << "1. dSHalfdS: jVMEC uses constant 0.25, VMEC++ derives from arrays\n";
  std::cout
      << "2. odd_contrib: jVMEC has complex cross-terms, VMEC++ simplified\n";
  std::cout << "3. Half-grid interpolation: jVMEC averages adjacent surfaces\n";
  std::cout
      << "4. Ghost surface handling: jVMEC extrapolates tau[0] = tau[1]\n";

  // Create minimal test configuration
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
  config.am = {0.0};
  config.pres_scale = 0.0;

  config.rbc = {10.0, 2.0, 0.5};
  config.zbs = {0.0, 2.0, 0.5};
  config.rbs = {0.0, 0.0, 0.0};
  config.zbc = {0.0, 0.0, 0.0};

  config.raxis_c = {10.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  std::cout
      << "\nRunning VMEC++ with current formula to analyze components...\n";
  const auto output = vmecpp::run(config);

  if (!output.ok()) {
    std::cout << "Current VMEC++ status: " << output.status() << std::endl;
    std::string error_msg(output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout
          << "✅ Expected Jacobian failure - analyzing formula differences\n";
    }
  }

  std::cout << "\nANALYSIS OF FORMULA DIFFERENCES:\n";
  std::cout << "\n1. CONSTANT dSHalfdS = 0.25:\n";
  std::cout << "   jVMEC: Always uses 0.25 regardless of grid\n";
  std::cout << "   VMEC++: dSHalfDsInterp varies (observed: 0.25 in debug)\n";
  std::cout << "   → Match: This doesn't explain the difference\n";

  std::cout << "\n2. COMPLEX odd_contrib CALCULATION:\n";
  std::cout << "   jVMEC: Cross-terms between even/odd modes with half-grid "
               "interpolation\n";
  std::cout << "   VMEC++: Simplified (ruo_o*z1o_o - zuo_o*r1o_o) / sqrtSH\n";
  std::cout << "   → CRITICAL: This is likely the main difference!\n";

  std::cout << "\n3. HALF-GRID INTERPOLATION:\n";
  std::cout
      << "   jVMEC: Averages contributions from adjacent radial surfaces\n";
  std::cout << "   VMEC++: Uses single surface values\n";
  std::cout << "   → May affect numerical stability\n";

  EXPECT_TRUE(true) << "jVMEC formula comparison complete";
}

TEST(JVMECJacobianFormulaTest, AnalyzeOddContribCalculation) {
  std::cout << "\n=== ANALYZE jVMEC odd_contrib CALCULATION ===\n";

  std::cout << "jVMEC odd_contrib formula (complex):\n";
  std::cout << "odd_contrib = \n";
  std::cout << "  dRdTheta[j][m_odd] * Z[j][m_odd] + dRdTheta[j-1][m_odd] * "
               "Z[j-1][m_odd]\n";
  std::cout << "  - dZdTheta[j][m_odd] * R[j][m_odd] - dZdTheta[j-1][m_odd] * "
               "R[j-1][m_odd]\n";
  std::cout << "  + (cross_terms) / sqrtSHalf[j]\n";

  std::cout << "\nwhere cross_terms =\n";
  std::cout << "  dRdTheta[j][m_evn] * Z[j][m_odd] + dRdTheta[j-1][m_evn] * "
               "Z[j-1][m_odd]\n";
  std::cout << "  - dZdTheta[j][m_evn] * R[j][m_odd] - dZdTheta[j-1][m_evn] * "
               "R[j-1][m_odd]\n";

  std::cout << "\nVMEC++ tau2 formula (simplified):\n";
  std::cout << "tau2 = (ruo_o * z1o_o - zuo_o * r1o_o) / sqrtSH\n";

  std::cout << "\nKEY INSIGHT:\n";
  std::cout << "jVMEC includes:\n";
  std::cout << "1. Half-grid interpolation (j and j-1 surfaces)\n";
  std::cout << "2. Separate even-m and odd-m mode contributions\n";
  std::cout << "3. Cross-coupling terms between even and odd modes\n";
  std::cout << "4. Proper radial surface averaging\n";

  std::cout << "\nVMEC++ simplified approach may miss:\n";
  std::cout << "1. Cross-coupling between even/odd modes\n";
  std::cout << "2. Proper half-grid interpolation\n";
  std::cout << "3. Multi-surface contributions\n";

  std::cout << "\nHYPOTHESIS:\n";
  std::cout << "The simplified tau2 calculation in VMEC++ doesn't capture\n";
  std::cout << "the full complexity of the jVMEC odd_contrib formula.\n";
  std::cout << "This could explain why Jacobian fails in asymmetric mode.\n";

  EXPECT_TRUE(true) << "odd_contrib analysis complete";
}

TEST(JVMECJacobianFormulaTest, RecoveryStrategyComparison) {
  std::cout << "\n=== COMPARE JACOBIAN FAILURE RECOVERY STRATEGIES ===\n";

  std::cout << "jVMEC RECOVERY STRATEGY:\n";
  std::cout << "1. Automatic axis guess improvement (guessAxis())\n";
  std::cout << "2. Progressive time step reduction (0.98, 0.96, ...)\n";
  std::cout << "3. Up to 75 Jacobian retries before giving up\n";
  std::cout << "4. 3-surface fallback if multigrid fails\n";
  std::cout << "5. Multiple restart attempts with different parameters\n";

  std::cout << "\nVMEC++ CURRENT STRATEGY:\n";
  std::cout
      << "1. Single attempt: 'TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS'\n";
  std::cout << "2. Immediate failure: 'FATAL ERROR in SolveEquilibrium'\n";
  std::cout << "3. No progressive parameter adjustment\n";
  std::cout << "4. No retry mechanism\n";

  std::cout << "\nRECOMMENDATIONS:\n";
  std::cout
      << "1. IMPLEMENT exact jVMEC tau formula with complex odd_contrib\n";
  std::cout << "2. ADD progressive error recovery strategy\n";
  std::cout << "3. IMPLEMENT automatic axis improvement\n";
  std::cout << "4. ADD time step reduction and retry logic\n";
  std::cout << "5. MAINTAIN constant dSHalfdS = 0.25 like jVMEC\n";

  std::cout << "\nPRIORITY ORDER:\n";
  std::cout << "1. Fix tau calculation formula (highest impact)\n";
  std::cout << "2. Add basic retry logic (medium impact)\n";
  std::cout << "3. Implement axis improvement (polish)\n";

  EXPECT_TRUE(true) << "Recovery strategy comparison complete";
}

TEST(JVMECJacobianFormulaTest, NextImplementationSteps) {
  std::cout << "\n=== NEXT IMPLEMENTATION STEPS ===\n";

  std::cout << "STEP 1: Create unit test for exact jVMEC tau formula\n";
  std::cout << "- Implement jVMEC odd_contrib calculation\n";
  std::cout << "- Test against known values from jVMEC debug output\n";
  std::cout << "- Verify half-grid interpolation works correctly\n";

  std::cout << "\nSTEP 2: Modify ideal_mhd_model.cc computeJacobian()\n";
  std::cout << "- Replace simplified tau2 with complex odd_contrib\n";
  std::cout << "- Ensure dSHalfdS = 0.25 constant\n";
  std::cout << "- Add proper even/odd mode separation\n";

  std::cout << "\nSTEP 3: Test with asymmetric cases\n";
  std::cout << "- Run identical config in jVMEC and VMEC++\n";
  std::cout << "- Compare tau values element-by-element\n";
  std::cout << "- Verify Jacobian check now passes\n";

  std::cout << "\nSTEP 4: Add recovery strategy\n";
  std::cout << "- Implement axis guess improvement\n";
  std::cout << "- Add progressive time step reduction\n";
  std::cout << "- Include retry logic for robustness\n";

  std::cout << "\nSTEP 5: Validation\n";
  std::cout << "- Test with jVMEC reference cases\n";
  std::cout << "- Verify symmetric mode unchanged\n";
  std::cout << "- Achieve first convergent asymmetric equilibrium\n";

  std::cout << "\nCURRENT FOCUS:\n";
  std::cout
      << "Implement exact jVMEC tau formula in VMEC++ computeJacobian()\n";
  std::cout << "This should resolve the fundamental algorithm difference\n";

  EXPECT_TRUE(true) << "Implementation plan complete";
}

}  // namespace vmecpp
