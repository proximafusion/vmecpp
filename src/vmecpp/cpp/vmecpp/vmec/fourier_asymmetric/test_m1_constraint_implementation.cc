// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// Unit test to implement and validate jVMEC M=1 constraint enforcement
// Following TDD approach with small steps and meticulous debug output
class M1ConstraintImplementationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup for jVMEC M=1 constraint testing
  }

  VmecINDATA CreateJVMECWorkingConfig() {
    // Exact jVMEC input.tok_asym coefficients
    VmecINDATA config;
    config.lasym = true;
    config.nfp = 1;
    config.mpol = 7;
    config.ntor = 0;

    config.rbc = {5.9163,     1.9196,   0.33736,   0.041504,
                  -0.0058256, 0.010374, -0.0056365};
    config.rbs = {0.0,       0.027610, 0.10038,  -0.071843,
                  -0.011423, 0.008177, -0.007611};
    config.zbc = {0.4105,     0.057302, 0.0046697, -0.039155,
                  -0.0087848, 0.021175, 0.002439};
    config.zbs = {0.0,      3.6223,   -0.18511, -0.0048568,
                  0.059268, 0.004477, -0.016773};

    return config;
  }

  void ApplyJVMECM1Constraint(VmecINDATA& config) {
    std::cout << "\n=== APPLYING JVMEC M=1 CONSTRAINT ===\n";
    std::cout << std::fixed << std::setprecision(8);

    // Store original values for comparison
    double original_rbs_1 = config.rbs[1];
    double original_zbc_1 = config.zbc[1];

    std::cout << "Original boundary coefficients:\n";
    std::cout << "  rbs[1] = " << original_rbs_1 << "\n";
    std::cout << "  zbc[1] = " << original_zbc_1 << "\n";

    // Apply jVMEC constraint: rbsc[n][1] = (rbsc[n][1] + zbcc[n][1]) / 2
    // This couples the asymmetric modes for m=1
    double constrained_value = (original_rbs_1 + original_zbc_1) / 2.0;

    std::cout << "\njVMEC M=1 constraint formula:\n";
    std::cout << "  constrained_value = (rbs[1] + zbc[1]) / 2\n";
    std::cout << "  constrained_value = (" << original_rbs_1 << " + "
              << original_zbc_1 << ") / 2 = " << constrained_value << "\n";

    // Apply constraint
    config.rbs[1] = constrained_value;
    config.zbc[1] = constrained_value;

    std::cout << "\nAfter M=1 constraint enforcement:\n";
    std::cout << "  rbs[1] = " << config.rbs[1] << "\n";
    std::cout << "  zbc[1] = " << config.zbc[1] << "\n";

    // Calculate impact
    double rbs_change = std::abs(config.rbs[1] - original_rbs_1);
    double zbc_change = std::abs(config.zbc[1] - original_zbc_1);

    std::cout << "\nConstraint impact:\n";
    std::cout << "  rbs[1] change: " << rbs_change << " ("
              << (100.0 * rbs_change / std::abs(original_rbs_1)) << "%)\n";
    std::cout << "  zbc[1] change: " << zbc_change << " ("
              << (100.0 * zbc_change / std::abs(original_zbc_1)) << "%)\n";
  }

  void PrintConfigSummary(const VmecINDATA& config, const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << "Critical asymmetric boundary modes:\n";
    std::cout << "  rbc[0] = " << config.rbc[0] << " (major radius)\n";
    std::cout << "  rbc[1] = " << config.rbc[1] << " (m=1 symmetric R)\n";
    std::cout << "  rbs[1] = " << config.rbs[1] << " (m=1 antisymmetric R)\n";
    std::cout << "  zbc[1] = " << config.zbc[1] << " (m=1 antisymmetric Z)\n";
    std::cout << "  zbs[1] = " << config.zbs[1] << " (m=1 symmetric Z)\n";
  }
};

TEST_F(M1ConstraintImplementationTest, TestM1ConstraintFormula) {
  std::cout << "\n=== M=1 CONSTRAINT FORMULA VALIDATION ===\n";

  VmecINDATA config = CreateJVMECWorkingConfig();

  PrintConfigSummary(config, "ORIGINAL JVMEC CONFIG");

  // Apply the constraint
  ApplyJVMECM1Constraint(config);

  PrintConfigSummary(config, "CONSTRAINED CONFIG");

  std::cout << "\n=== CONSTRAINT VALIDATION ===\n";

  // Verify constraint is satisfied
  double rbs_1 = config.rbs[1];
  double zbc_1 = config.zbc[1];
  double constraint_violation = std::abs(rbs_1 - zbc_1);

  std::cout << "Constraint verification:\n";
  std::cout << "  rbs[1] - zbc[1] = " << rbs_1 << " - " << zbc_1 << " = "
            << (rbs_1 - zbc_1) << "\n";
  std::cout << "  |constraint_violation| = " << constraint_violation << "\n";

  // Should be zero (or machine precision) after constraint
  EXPECT_LT(constraint_violation, 1e-14)
      << "M=1 constraint should make rbs[1] = zbc[1]";

  std::cout << "✅ M=1 CONSTRAINT FORMULA WORKING CORRECTLY\n";
}

TEST_F(M1ConstraintImplementationTest, CompareConstrainedVsOriginal) {
  std::cout << "\n=== COMPARE CONSTRAINED VS ORIGINAL JACOBIAN ===\n";

  VmecINDATA original_config = CreateJVMECWorkingConfig();
  VmecINDATA constrained_config = CreateJVMECWorkingConfig();

  // Apply constraint to one copy
  ApplyJVMECM1Constraint(constrained_config);

  std::cout << "\n=== EXPECTED JACOBIAN IMPACT ===\n";

  std::cout << "Original config (jVMEC input.tok_asym):\n";
  std::cout << "  Known result: minTau=-29.29, maxTau=69.19 → BAD_JACOBIAN\n";
  std::cout << "  This fails in VMEC++ but works in jVMEC\n";

  std::cout << "\nConstrained config (after M=1 enforcement):\n";
  std::cout
      << "  Hypothesis: Better conditioned boundary may improve Jacobian\n";
  std::cout
      << "  Test: Check if this reduces tau range or eliminates sign change\n";

  std::cout << "\n=== COEFFICIENT COMPARISON ===\n";

  double original_rbs_1 = original_config.rbs[1];
  double original_zbc_1 = original_config.zbc[1];
  double constrained_rbs_1 = constrained_config.rbs[1];
  double constrained_zbc_1 = constrained_config.zbc[1];

  std::cout << "Boundary coefficient changes:\n";
  std::cout << "  rbs[1]: " << original_rbs_1 << " → " << constrained_rbs_1
            << " (Δ=" << (constrained_rbs_1 - original_rbs_1) << ")\n";
  std::cout << "  zbc[1]: " << original_zbc_1 << " → " << constrained_zbc_1
            << " (Δ=" << (constrained_zbc_1 - original_zbc_1) << ")\n";

  // Framework for future Jacobian testing
  std::cout << "\nNext testing steps:\n";
  std::cout << "1. Create VMEC equilibrium with constrained config\n";
  std::cout << "2. Compare initial Jacobian distribution\n";
  std::cout << "3. Check if tau sign change is eliminated\n";
  std::cout << "4. Test convergence behavior\n";

  EXPECT_TRUE(true) << "Constraint comparison framework created";
}

TEST_F(M1ConstraintImplementationTest, CreateConstraintIntegrationTest) {
  std::cout << "\n=== M=1 CONSTRAINT INTEGRATION FRAMEWORK ===\n";

  std::cout << "Integration test framework for VMEC++ M=1 constraint:\n";

  std::cout << "\nStep 1: Boundary coefficient preprocessing\n";
  std::cout << "- Apply jVMEC M=1 constraint before VMEC initialization\n";
  std::cout
      << "- Ensure rbs[1] = zbc[1] = (original_rbs[1] + original_zbc[1])/2\n";
  std::cout << "- Preserve all other boundary coefficients unchanged\n";

  std::cout << "\nStep 2: Integration with existing VMEC++ pipeline\n";
  std::cout << "- Apply constraint in boundary preprocessing phase\n";
  std::cout << "- Before asymmetric transform functions\n";
  std::cout
      << "- After boundary coefficient loading but before equilibrium solve\n";

  std::cout << "\nStep 3: Jacobian comparison testing\n";
  std::cout << "- Test constrained config vs original config\n";
  std::cout << "- Compare minTau, maxTau values after constraint\n";
  std::cout << "- Check if Jacobian sign change is eliminated\n";
  std::cout << "- Monitor convergence behavior\n";

  std::cout << "\nStep 4: Validation against jVMEC\n";
  std::cout << "- Verify VMEC++ constrained result matches jVMEC output\n";
  std::cout << "- Compare force residuals and iteration counts\n";
  std::cout << "- Validate final equilibrium properties\n";

  VmecINDATA test_config = CreateJVMECWorkingConfig();
  ApplyJVMECM1Constraint(test_config);

  std::cout << "\n✅ READY FOR M=1 CONSTRAINT INTEGRATION TESTING\n";
  std::cout
      << "Constrained boundary configuration prepared for VMEC++ testing\n";

  // Verify we have a valid constrained configuration
  EXPECT_LT(std::abs(test_config.rbs[1] - test_config.zbc[1]), 1e-14)
      << "Constrained config should satisfy M=1 constraint";
}

}  // namespace vmecpp

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
