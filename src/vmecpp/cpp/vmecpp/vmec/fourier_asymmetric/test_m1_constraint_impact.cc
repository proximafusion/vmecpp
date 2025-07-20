// TDD test to evaluate M=1 constraint impact on Jacobian failures
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(M1ConstraintImpactTest, TestOriginalVsConstrainedBoundary) {
  std::cout << "\n=== TEST ORIGINAL VS CONSTRAINED BOUNDARY ===\n";
  std::cout << std::fixed << std::setprecision(6);

  // Base jVMEC configuration (from input.tok_asym)
  VmecINDATA base_config;
  base_config.lasym = true;
  base_config.nfp = 1;
  base_config.mpol = 7;
  base_config.ntor = 0;
  base_config.ns_array = {5};
  base_config.niter_array = {1};
  base_config.ftol_array = {1e-12};
  base_config.return_outputs_even_if_not_converged = true;
  base_config.delt = 0.25;
  base_config.tcon0 = 1.0;
  base_config.phiedge = 119.15;
  base_config.pmass_type = "power_series";
  base_config.am = {1.0, -2.0, 1.0};
  base_config.pres_scale = 100000.0;

  // jVMEC axis
  base_config.raxis_c = {6.676};
  base_config.raxis_s = {0.0};
  base_config.zaxis_s = {0.0};
  base_config.zaxis_c = {0.47};

  // Original jVMEC boundary coefficients
  base_config.rbc = {5.9163,     1.9196,   0.33736,   0.041504,
                     -0.0058256, 0.010374, -0.0056365};
  base_config.rbs = {0.0,       0.027610, 0.10038,  -0.071843,
                     -0.011423, 0.008177, -0.007611};
  base_config.zbc = {0.4105,     0.057302, 0.0046697, -0.039155,
                     -0.0087848, 0.021175, 0.002439};
  base_config.zbs = {0.0,      3.6223,   -0.18511, -0.0048568,
                     0.059268, 0.004477, -0.016773};

  std::cout << "Testing original jVMEC boundary configuration:\n";
  std::cout << "  Original M=1 modes: rbs[1]=" << base_config.rbs[1]
            << ", zbc[1]=" << base_config.zbc[1] << "\n";

  const auto original_output = vmecpp::run(base_config);

  if (original_output.ok()) {
    std::cout << "  ✅ SUCCESS: Original config converges!\n";
  } else {
    std::string error_msg(original_output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout << "  ❌ JACOBIAN: Original config fails with Jacobian error\n";
    } else {
      std::cout << "  ❌ OTHER: " << error_msg.substr(0, 50) << "...\n";
    }
  }

  // Apply jVMEC M=1 constraint
  VmecINDATA constrained_config = base_config;

  // Constraint: rbsc[n][1] = (rbsc[n][1] + zbcc[n][1]) / 2
  // For n=0: rbs[1] = (rbs[1] + zbc[1]) / 2
  double rbs_1_original = constrained_config.rbs[1];
  double zbc_1_original = constrained_config.zbc[1];
  double constrained_value = (rbs_1_original + zbc_1_original) / 2.0;

  constrained_config.rbs[1] = constrained_value;
  constrained_config.zbc[1] = constrained_value;

  std::cout << "\nTesting jVMEC M=1 constrained boundary:\n";
  std::cout << "  Constrained M=1 modes: rbs[1]=" << constrained_config.rbs[1]
            << ", zbc[1]=" << constrained_config.zbc[1] << "\n";
  std::cout << "  Change: rbs[1] "
            << ((constrained_value > rbs_1_original) ? "+" : "")
            << (constrained_value - rbs_1_original) << "\n";
  std::cout << "  Change: zbc[1] "
            << ((constrained_value > zbc_1_original) ? "+" : "")
            << (constrained_value - zbc_1_original) << "\n";

  const auto constrained_output = vmecpp::run(constrained_config);

  if (constrained_output.ok()) {
    std::cout << "  ✅ SUCCESS: Constrained config converges!\n";
  } else {
    std::string error_msg(constrained_output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout
          << "  ❌ JACOBIAN: Constrained config fails with Jacobian error\n";
    } else {
      std::cout << "  ❌ OTHER: " << error_msg.substr(0, 50) << "...\n";
    }
  }

  // Summary
  std::cout << "\nM=1 CONSTRAINT IMPACT SUMMARY:\n";
  bool original_success = original_output.ok();
  bool constrained_success = constrained_output.ok();

  if (!original_success && constrained_success) {
    std::cout << "🎉 BREAKTHROUGH: M=1 constraint fixes Jacobian failure!\n";
    std::cout << "This confirms jVMEC preprocessing is critical for asymmetric "
                 "VMEC\n";
  } else if (original_success && !constrained_success) {
    std::cout << "⚠️  M=1 constraint breaks working configuration\n";
    std::cout << "VMEC++ may not need this constraint\n";
  } else if (!original_success && !constrained_success) {
    std::cout << "❌ M=1 constraint doesn't fix the issue\n";
    std::cout << "Need to investigate other jVMEC preprocessing steps\n";
  } else {
    std::cout << "✅ Both configurations work - M=1 constraint not critical\n";
  }

  EXPECT_TRUE(true) << "M=1 constraint impact test complete";
}

TEST(M1ConstraintImpactTest, TestConstraintImplementation) {
  std::cout << "\n=== TEST CONSTRAINT IMPLEMENTATION ===\n";
  std::cout << std::fixed << std::setprecision(8);

  std::cout << "jVMEC M=1 constraint implementation details:\n";
  std::cout << "From jVMEC Boundaries.java line 335-341:\n";
  std::cout << "```java\n";
  std::cout << "if (lasym) {\n";
  std::cout << "  for (int n = 0; n <= ntor; ++n) {\n";
  std::cout << "    final double backup_rbsc = rbsc[n][m];\n";
  std::cout << "    rbsc[n][m] = (backup_rbsc + zbcc[n][m]) / 2;\n";
  std::cout << "    zbcc[n][m] = (backup_rbsc - zbcc[n][m]) / 2;\n";
  std::cout << "  }\n";
  std::cout << "}\n";
  std::cout << "```\n";

  std::cout << "\nThis shows jVMEC uses a DIFFERENT constraint formula:\n";
  std::cout << "1. Store original rbsc value\n";
  std::cout << "2. rbsc[n][1] = (rbsc[n][1] + zbcc[n][1]) / 2\n";
  std::cout << "3. zbcc[n][1] = (rbsc_original - zbcc[n][1]) / 2\n";
  std::cout << "\nThis is NOT the same as setting both to the average!\n";

  // Test the actual jVMEC constraint formula
  double rbs_1 = 0.027610;  // From jVMEC config
  double zbc_1 = 0.057302;  // From jVMEC config

  std::cout << "\nApplying actual jVMEC constraint:\n";
  std::cout << "  Original: rbs[1] = " << rbs_1 << ", zbc[1] = " << zbc_1
            << "\n";

  double backup_rbs = rbs_1;
  double new_rbs = (backup_rbs + zbc_1) / 2.0;
  double new_zbc = (backup_rbs - zbc_1) / 2.0;

  std::cout << "  After constraint: rbs[1] = " << new_rbs
            << ", zbc[1] = " << new_zbc << "\n";
  std::cout << "  Sum check: rbs[1] + zbc[1] = " << (new_rbs + new_zbc)
            << " (should equal original rbs[1] = " << backup_rbs << ")\n";
  std::cout << "  Difference: rbs[1] - zbc[1] = " << (new_rbs - new_zbc)
            << " (should equal original zbc[1] = " << zbc_1 << ")\n";

  // Verify the constraint properties
  double sum_check = new_rbs + new_zbc;
  double diff_check = new_rbs - new_zbc;

  if (std::abs(sum_check - backup_rbs) < 1e-12 &&
      std::abs(diff_check - zbc_1) < 1e-12) {
    std::cout << "✅ CONSTRAINT VERIFICATION: jVMEC formula preserves sum and "
                 "difference\n";
  } else {
    std::cout << "❌ CONSTRAINT ERROR: Formula doesn't preserve expected "
                 "properties\n";
  }

  std::cout << "\nCONCLUSION:\n";
  std::cout
      << "Need to test the CORRECT jVMEC constraint formula in next test\n";
  std::cout << "Previous test used wrong averaging approach\n";

  EXPECT_TRUE(true) << "Constraint implementation analysis complete";
}

TEST(M1ConstraintImpactTest, TestCorrectJVMECConstraint) {
  std::cout << "\n=== TEST CORRECT jVMEC CONSTRAINT ===\n";
  std::cout << std::fixed << std::setprecision(6);

  // Base jVMEC configuration
  VmecINDATA base_config;
  base_config.lasym = true;
  base_config.nfp = 1;
  base_config.mpol = 7;
  base_config.ntor = 0;
  base_config.ns_array = {5};
  base_config.niter_array = {1};
  base_config.ftol_array = {1e-12};
  base_config.return_outputs_even_if_not_converged = true;
  base_config.delt = 0.25;
  base_config.tcon0 = 1.0;
  base_config.phiedge = 119.15;
  base_config.pmass_type = "power_series";
  base_config.am = {1.0, -2.0, 1.0};
  base_config.pres_scale = 100000.0;

  base_config.raxis_c = {6.676};
  base_config.raxis_s = {0.0};
  base_config.zaxis_s = {0.0};
  base_config.zaxis_c = {0.47};

  base_config.rbc = {5.9163,     1.9196,   0.33736,   0.041504,
                     -0.0058256, 0.010374, -0.0056365};
  base_config.rbs = {0.0,       0.027610, 0.10038,  -0.071843,
                     -0.011423, 0.008177, -0.007611};
  base_config.zbc = {0.4105,     0.057302, 0.0046697, -0.039155,
                     -0.0087848, 0.021175, 0.002439};
  base_config.zbs = {0.0,      3.6223,   -0.18511, -0.0048568,
                     0.059268, 0.004477, -0.016773};

  // Apply CORRECT jVMEC M=1 constraint
  VmecINDATA corrected_config = base_config;

  // For n=0, m=1: rbsc[0][1] corresponds to rbs[1], zbcc[0][1] corresponds to
  // zbc[1]
  double backup_rbs = corrected_config.rbs[1];
  double zbc_1 = corrected_config.zbc[1];

  corrected_config.rbs[1] = (backup_rbs + zbc_1) / 2.0;
  corrected_config.zbc[1] = (backup_rbs - zbc_1) / 2.0;

  std::cout << "Testing CORRECT jVMEC M=1 constraint:\n";
  std::cout << "  Original: rbs[1]=" << backup_rbs << ", zbc[1]=" << zbc_1
            << "\n";
  std::cout << "  Corrected: rbs[1]=" << corrected_config.rbs[1]
            << ", zbc[1]=" << corrected_config.zbc[1] << "\n";

  const auto corrected_output = vmecpp::run(corrected_config);

  if (corrected_output.ok()) {
    std::cout << "  ✅ SUCCESS: Corrected jVMEC constraint works!\n";
    std::cout
        << "  🎉 BREAKTHROUGH: This may be the missing preprocessing step!\n";
  } else {
    std::string error_msg(corrected_output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout << "  ❌ JACOBIAN: Corrected constraint still fails\n";
      std::cout << "  Need to investigate other preprocessing differences\n";
    } else {
      std::cout << "  ❌ OTHER: " << error_msg.substr(0, 50) << "...\n";
    }
  }

  EXPECT_TRUE(true) << "Correct jVMEC constraint test complete";
}

}  // namespace vmecpp
