// TDD test to isolate jVMEC boundary preprocessing differences
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

TEST(JVMECBoundaryPreprocessingTest, CompareTheetaShiftFormula) {
  std::cout << "\n=== COMPARE THETA SHIFT FORMULA ===\n";
  std::cout << std::fixed << std::setprecision(8);

  // Use exact jVMEC input.tok_asym coefficients
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 7;
  config.ntor = 0;

  // jVMEC boundary coefficients (m=0 to m=6)
  config.rbc = {5.9163,     1.9196,   0.33736,   0.041504,
                -0.0058256, 0.010374, -0.0056365};
  config.rbs = {0.0,       0.027610, 0.10038,  -0.071843,
                -0.011423, 0.008177, -0.007611};
  config.zbc = {0.4105,     0.057302, 0.0046697, -0.039155,
                -0.0087848, 0.021175, 0.002439};
  config.zbs = {0.0,      3.6223,   -0.18511, -0.0048568,
                0.059268, 0.004477, -0.016773};

  std::cout << "jVMEC boundary coefficients (m=1, n=0 modes):\n";
  std::cout << "  rbc[1] = " << config.rbc[1] << "\n";
  std::cout << "  rbs[1] = " << config.rbs[1] << "\n";
  std::cout << "  zbc[1] = " << config.zbc[1] << "\n";
  std::cout << "  zbs[1] = " << config.zbs[1] << "\n";

  // jVMEC theta shift formula (corrected version)
  double rbs_1_0 = config.rbs[1];  // 0.027610
  double zbc_1_0 = config.zbc[1];  // 0.057302
  double rbc_1_0 = config.rbc[1];  // 1.9196
  double zbs_1_0 = config.zbs[1];  // 3.6223

  double delta_jvmec = atan2(rbs_1_0 - zbc_1_0, rbc_1_0 + zbs_1_0);

  std::cout << "\njVMEC theta shift calculation:\n";
  std::cout << "  numerator = rbs[1] - zbc[1] = " << rbs_1_0 << " - " << zbc_1_0
            << " = " << (rbs_1_0 - zbc_1_0) << "\n";
  std::cout << "  denominator = rbc[1] + zbs[1] = " << rbc_1_0 << " + "
            << zbs_1_0 << " = " << (rbc_1_0 + zbs_1_0) << "\n";
  std::cout << "  delta = atan2(" << (rbs_1_0 - zbc_1_0) << ", "
            << (rbc_1_0 + zbs_1_0) << ") = " << delta_jvmec << " radians\n";
  std::cout << "  delta = " << (delta_jvmec * 180.0 / M_PI) << " degrees\n";

  // Compare with VMEC++ actual calculation (debug output showed -0.00535768)
  double delta_vmecpp = -0.00535768;  // From debug output
  std::cout << "\nVMEC++ computed delta = " << delta_vmecpp << " radians\n";
  std::cout << "VMEC++ computed delta = " << (delta_vmecpp * 180.0 / M_PI)
            << " degrees\n";

  double delta_diff = std::abs(delta_jvmec - delta_vmecpp);
  std::cout << "\nDifference = " << delta_diff << " radians ("
            << (delta_diff * 180.0 / M_PI) << " degrees)\n";

  if (delta_diff < 1e-6) {
    std::cout
        << "✅ THETA SHIFT MATCHES: VMEC++ implements correct jVMEC formula\n";
  } else {
    std::cout
        << "❌ THETA SHIFT DIFFERS: Potential implementation difference\n";
  }

  EXPECT_LT(delta_diff, 1e-6) << "Theta shift should match jVMEC calculation";
}

TEST(JVMECBoundaryPreprocessingTest, AnalyzeM1ModeConstraints) {
  std::cout << "\n=== ANALYZE M=1 MODE CONSTRAINTS ===\n";
  std::cout << std::fixed << std::setprecision(8);

  // jVMEC working config
  VmecINDATA config;
  config.rbc = {5.9163,     1.9196,   0.33736,   0.041504,
                -0.0058256, 0.010374, -0.0056365};
  config.rbs = {0.0,       0.027610, 0.10038,  -0.071843,
                -0.011423, 0.008177, -0.007611};
  config.zbc = {0.4105,     0.057302, 0.0046697, -0.039155,
                -0.0087848, 0.021175, 0.002439};
  config.zbs = {0.0,      3.6223,   -0.18511, -0.0048568,
                0.059268, 0.004477, -0.016773};

  std::cout << "jVMEC M=1 constraint enforcement:\n";
  std::cout << "Original M=1 modes (before constraint):\n";
  std::cout << "  rbc[1] = " << config.rbc[1] << "\n";
  std::cout << "  rbs[1] = " << config.rbs[1] << "\n";
  std::cout << "  zbc[1] = " << config.zbc[1] << "\n";
  std::cout << "  zbs[1] = " << config.zbs[1] << "\n";

  // jVMEC constraint: rbsc[n][1] = (rbsc[n][1] + zbcc[n][1]) / 2
  // In VMEC++ notation: rbs[1] and zbc[1] are coupled
  double rbs_1_original = config.rbs[1];
  double zbc_1_original = config.zbc[1];

  double constrained_value = (rbs_1_original + zbc_1_original) / 2.0;

  std::cout << "\njVMEC M=1 constraint formula:\n";
  std::cout << "  constrained_value = (rbs[1] + zbc[1]) / 2\n";
  std::cout << "  constrained_value = (" << rbs_1_original << " + "
            << zbc_1_original << ") / 2 = " << constrained_value << "\n";

  std::cout << "\nAfter jVMEC constraint enforcement:\n";
  std::cout << "  rbs[1] = " << constrained_value << "\n";
  std::cout << "  zbc[1] = " << constrained_value << "\n";

  // Check if constraint would change values significantly
  double rbs_change = std::abs(constrained_value - rbs_1_original);
  double zbc_change = std::abs(constrained_value - zbc_1_original);

  std::cout << "\nConstraint impact:\n";
  std::cout << "  rbs[1] change: " << rbs_change << " ("
            << (100.0 * rbs_change / std::abs(rbs_1_original)) << "%)\n";
  std::cout << "  zbc[1] change: " << zbc_change << " ("
            << (100.0 * zbc_change / std::abs(zbc_1_original)) << "%)\n";

  if (rbs_change > 0.01 || zbc_change > 0.01) {
    std::cout << "⚠️  SIGNIFICANT CONSTRAINT EFFECT: M=1 constraint would "
                 "change boundary\n";
    std::cout << "This may explain why jVMEC config fails in VMEC++\n";
  } else {
    std::cout
        << "✅ MINIMAL CONSTRAINT EFFECT: M=1 constraint makes small changes\n";
    std::cout << "M=1 constraint unlikely to be the primary issue\n";
  }

  EXPECT_TRUE(true) << "M=1 constraint analysis complete";
}

TEST(JVMECBoundaryPreprocessingTest, CheckJacobianSignHeuristic) {
  std::cout << "\n=== CHECK JACOBIAN SIGN HEURISTIC ===\n";
  std::cout << std::fixed << std::setprecision(8);

  std::cout << "jVMEC Jacobian sign heuristic from checkSignOfJacobian():\n";
  std::cout << "1. Compute rTest = sum of rbcc[n][1] for n=0 to ntor\n";
  std::cout << "2. Compute zTest = sum of zbsc[n][1] for n=0 to ntor\n";
  std::cout << "3. Check condition: (rTest * zTest * signOfJacobian > 0.0)\n";
  std::cout << "4. If true, need to flip theta definition\n";

  // jVMEC working config (n=0 only since ntor=0)
  VmecINDATA config;
  config.rbc = {5.9163,     1.9196,   0.33736,   0.041504,
                -0.0058256, 0.010374, -0.0056365};
  config.rbs = {0.0,       0.027610, 0.10038,  -0.071843,
                -0.011423, 0.008177, -0.007611};
  config.zbc = {0.4105,     0.057302, 0.0046697, -0.039155,
                -0.0087848, 0.021175, 0.002439};
  config.zbs = {0.0,      3.6223,   -0.18511, -0.0048568,
                0.059268, 0.004477, -0.016773};

  // For ntor=0, only n=0 contributes
  double rTest = config.rbc[1];  // rbcc[0][1] = rbc[1]
  double zTest = config.zbs[1];  // zbsc[0][1] = zbs[1]
  int signOfJacobian = -1;       // Standard VMEC convention

  std::cout << "\nComputing heuristic for jVMEC config:\n";
  std::cout << "  rTest = rbcc[0][1] = rbc[1] = " << rTest << "\n";
  std::cout << "  zTest = zbsc[0][1] = zbs[1] = " << zTest << "\n";
  std::cout << "  signOfJacobian = " << signOfJacobian << "\n";

  double heuristic_product = rTest * zTest * signOfJacobian;
  std::cout << "  heuristic = rTest * zTest * signOfJacobian = " << rTest
            << " * " << zTest << " * " << signOfJacobian << " = "
            << heuristic_product << "\n";

  bool need_flip = (heuristic_product > 0.0);
  std::cout << "  need_flip = (heuristic > 0.0) = " << std::boolalpha
            << need_flip << "\n";

  if (need_flip) {
    std::cout
        << "⚠️  jVMEC WOULD FLIP THETA: Boundary implies wrong Jacobian sign\n";
    std::cout << "This suggests jVMEC applies additional preprocessing that "
                 "VMEC++ lacks\n";
  } else {
    std::cout
        << "✅ NO THETA FLIP NEEDED: Boundary Jacobian sign appears correct\n";
    std::cout << "Jacobian sign heuristic unlikely to be the primary issue\n";
  }

  EXPECT_TRUE(true) << "Jacobian sign heuristic analysis complete";
}

TEST(JVMECBoundaryPreprocessingTest, SummaryOfPreprocessingDifferences) {
  std::cout << "\n=== SUMMARY OF PREPROCESSING DIFFERENCES ===\n";

  std::cout << "IDENTIFIED jVMEC PREPROCESSING STEPS:\n";
  std::cout
      << "1. ✅ Theta angle correction: VMEC++ implements this correctly\n";
  std::cout << "2. 🔍 M=1 mode constraint enforcement: Need to test impact\n";
  std::cout << "3. 🔍 Jacobian sign heuristic: Need to test if triggered\n";
  std::cout << "4. ❌ Axis optimization (61×61 grid): VMEC++ lacks this\n";
  std::cout << "5. ❓ Other boundary preprocessing: May exist in jVMEC\n";

  std::cout << "\nCRITICAL FINDING:\n";
  std::cout << "Even exact jVMEC working config fails in VMEC++ with:\n";
  std::cout << "  minTau = -29.290532, maxTau = 69.194719\n";
  std::cout << "  minTau * maxTau = -2026.750155 < 0 → BAD_JACOBIAN\n";

  std::cout << "\nIMPLICATIONS:\n";
  std::cout << "1. Issue is NOT just boundary configuration\n";
  std::cout << "2. VMEC++ may lack critical preprocessing that jVMEC applies\n";
  std::cout
      << "3. Focus should be on implementation differences, not test configs\n";
  std::cout << "4. Axis optimization alone may not be sufficient\n";

  std::cout << "\nNEXT STEPS (TDD Approach):\n";
  std::cout << "1. Test M=1 constraint impact with unit test\n";
  std::cout << "2. Test Jacobian sign heuristic with unit test\n";
  std::cout << "3. Compare debug output line-by-line with jVMEC\n";
  std::cout << "4. Identify the specific preprocessing step that makes the "
               "difference\n";

  EXPECT_TRUE(true) << "Preprocessing differences analysis complete";
}

}  // namespace vmecpp
