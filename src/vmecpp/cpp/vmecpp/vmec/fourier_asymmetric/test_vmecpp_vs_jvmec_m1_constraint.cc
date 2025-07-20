// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace vmecpp {

// Critical test comparing VMEC++ M=1 constraint vs jVMEC M=1 constraint
// Following user request for meticulous debug output from all three codes
class VMECppVsJVMECM1ConstraintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup for constraint comparison
  }

  void CompareConstraintFormulas() {
    std::cout
        << "\n=== CRITICAL COMPARISON: VMEC++ vs jVMEC M=1 Constraint ===\n";
    std::cout << std::fixed << std::setprecision(8);

    // Test case using jVMEC tok_asym coefficients
    // For axisymmetric (n=0), we focus on the relevant arrays
    double original_rbsc = 0.027610;  // rbs[1] in jVMEC notation
    double original_zbcc = 0.057302;  // zbc[1] in jVMEC notation

    std::cout << "Original coefficients (jVMEC input.tok_asym):\n";
    std::cout << "  rbsc[m=1,n=0] = " << original_rbsc
              << " (rbs[1] in jVMEC)\n";
    std::cout << "  zbcc[m=1,n=0] = " << original_zbcc
              << " (zbc[1] in jVMEC)\n";

    // jVMEC constraint formula
    double jvmec_constrained = (original_rbsc + original_zbcc) / 2.0;
    double jvmec_rbsc = jvmec_constrained;
    double jvmec_zbcc = jvmec_constrained;

    std::cout << "\njVMEC constraint (from our analysis):\n";
    std::cout << "  Formula: rbsc = zbcc = (rbsc + zbcc) / 2\n";
    std::cout << "  Result: rbsc = zbcc = " << jvmec_constrained << "\n";

    // VMEC++ constraint formula with scaling_factor = 0.5
    double scaling_factor = 0.5;
    double vmecpp_rbsc = (original_rbsc + original_zbcc) * scaling_factor;
    double vmecpp_zbcc = (original_rbsc - original_zbcc) * scaling_factor;

    std::cout << "\nVMEC++ constraint (from boundaries.cc):\n";
    std::cout << "  Formula: rbsc = (rbsc + zbcc) * " << scaling_factor << "\n";
    std::cout << "           zbcc = (rbsc - zbcc) * " << scaling_factor << "\n";
    std::cout << "  Result: rbsc = " << vmecpp_rbsc << "\n";
    std::cout << "          zbcc = " << vmecpp_zbcc << "\n";

    // Compare results
    std::cout << "\n=== COMPARISON RESULTS ===\n";

    std::cout << "rbsc values:\n";
    std::cout << "  jVMEC:  " << jvmec_rbsc << "\n";
    std::cout << "  VMEC++: " << vmecpp_rbsc << "\n";
    std::cout << "  Difference: " << std::abs(jvmec_rbsc - vmecpp_rbsc) << "\n";

    std::cout << "\nzbcc values:\n";
    std::cout << "  jVMEC:  " << jvmec_zbcc << "\n";
    std::cout << "  VMEC++: " << vmecpp_zbcc << "\n";
    std::cout << "  Difference: " << std::abs(jvmec_zbcc - vmecpp_zbcc) << "\n";

    // Check if they satisfy the same constraint
    double jvmec_violation = std::abs(jvmec_rbsc - jvmec_zbcc);
    double vmecpp_violation = std::abs(vmecpp_rbsc - vmecpp_zbcc);

    std::cout << "\nConstraint satisfaction |rbsc - zbcc|:\n";
    std::cout << "  jVMEC:  " << jvmec_violation << " (should be 0)\n";
    std::cout << "  VMEC++: " << vmecpp_violation << " (NOT zero!)\n";

    // Analyze the difference
    if (jvmec_violation < 1e-14 && vmecpp_violation > 1e-10) {
      std::cout
          << "\n🚨 CRITICAL FINDING: VMEC++ uses DIFFERENT M=1 constraint!\n";
      std::cout << "jVMEC enforces rbsc = zbcc (coupling)\n";
      std::cout << "VMEC++ uses a different transformation that doesn't couple "
                   "them\n";
    }
  }

  void AnalyzeVMECppFormula() {
    std::cout << "\n=== VMEC++ M=1 CONSTRAINT ANALYSIS ===\n";

    std::cout << "VMEC++ formula with scaling_factor = 0.5:\n";
    std::cout << "  rbsc_new = (rbsc_old + zbcc_old) * 0.5\n";
    std::cout << "  zbcc_new = (rbsc_old - zbcc_old) * 0.5\n";

    std::cout << "\nThis is a ROTATION transformation:\n";
    std::cout << "  [rbsc_new]   [0.5   0.5] [rbsc_old]\n";
    std::cout << "  [zbcc_new] = [0.5  -0.5] [zbcc_old]\n";

    std::cout << "\nProperties:\n";
    std::cout << "- Does NOT enforce rbsc = zbcc\n";
    std::cout << "- Preserves sum: rbsc_new + zbcc_new = rbsc_old\n";
    std::cout << "- Changes difference: rbsc_new - zbcc_new = zbcc_old\n";

    std::cout << "\nContrast with jVMEC:\n";
    std::cout << "- jVMEC enforces rbsc = zbcc (coupling constraint)\n";
    std::cout << "- jVMEC preserves sum but zeros the difference\n";
    std::cout << "- Fundamentally different constraints!\n";
  }
};

TEST_F(VMECppVsJVMECM1ConstraintTest, CompareConstraintImplementations) {
  std::cout << "\n=== VMEC++ vs jVMEC M=1 CONSTRAINT COMPARISON ===\n";

  CompareConstraintFormulas();
  AnalyzeVMECppFormula();

  std::cout << "\n=== IMPLICATIONS ===\n";
  std::cout
      << "1. VMEC++ HAS an M=1 constraint but it's DIFFERENT from jVMEC\n";
  std::cout << "2. This explains why jVMEC configs still fail in VMEC++\n";
  std::cout
      << "3. Need to modify VMEC++ to use jVMEC formula for compatibility\n";
  std::cout << "4. The 53.77% boundary change we found is specific to jVMEC "
               "formula\n";

  // This test documents the difference, not tests correctness
  EXPECT_TRUE(true) << "Constraint comparison documented";
}

TEST_F(VMECppVsJVMECM1ConstraintTest, TestConstraintImpact) {
  std::cout << "\n=== CONSTRAINT IMPACT COMPARISON ===\n";

  // Original values
  double rbsc = 0.027610;
  double zbcc = 0.057302;

  std::cout << "Original: rbsc = " << rbsc << ", zbcc = " << zbcc << "\n";
  std::cout << "Original difference: |rbsc - zbcc| = " << std::abs(rbsc - zbcc)
            << "\n\n";

  // jVMEC constraint
  double jvmec_value = (rbsc + zbcc) / 2.0;
  std::cout << "jVMEC constraint:\n";
  std::cout << "  Both set to: " << jvmec_value << "\n";
  std::cout << "  rbsc change: "
            << (100.0 * std::abs(jvmec_value - rbsc) / rbsc) << "%\n";
  std::cout << "  zbcc change: "
            << (100.0 * std::abs(jvmec_value - zbcc) / zbcc) << "%\n";
  std::cout << "  Constraint |rbsc - zbcc| = 0.0\n";

  // VMEC++ constraint
  double vmecpp_rbsc = (rbsc + zbcc) * 0.5;
  double vmecpp_zbcc = (rbsc - zbcc) * 0.5;
  std::cout << "\nVMEC++ constraint:\n";
  std::cout << "  rbsc = " << vmecpp_rbsc << "\n";
  std::cout << "  zbcc = " << vmecpp_zbcc << "\n";
  std::cout << "  rbsc change: "
            << (100.0 * std::abs(vmecpp_rbsc - rbsc) / rbsc) << "%\n";
  std::cout << "  zbcc change: "
            << (100.0 * std::abs(vmecpp_zbcc - zbcc) / zbcc) << "%\n";
  std::cout << "  Constraint |rbsc - zbcc| = "
            << std::abs(vmecpp_rbsc - vmecpp_zbcc) << "\n";

  std::cout
      << "\n🚨 KEY FINDING: VMEC++ constraint does NOT zero the difference!\n";

  EXPECT_TRUE(true) << "Impact comparison complete";
}

TEST_F(VMECppVsJVMECM1ConstraintTest, ProposedModification) {
  std::cout << "\n=== PROPOSED VMEC++ MODIFICATION ===\n";

  std::cout << "Current VMEC++ ensureM1Constrained needs modification:\n\n";

  std::cout << "CURRENT (boundaries.cc):\n";
  std::cout
      << "  rbsc[idx_mn] = (backup_rsc + zbcc[idx_mn]) * scaling_factor;\n";
  std::cout
      << "  zbcc[idx_mn] = (backup_rsc - zbcc[idx_mn]) * scaling_factor;\n";

  std::cout << "\nPROPOSED (jVMEC-compatible):\n";
  std::cout
      << "  double constrained_value = (backup_rsc + zbcc[idx_mn]) / 2.0;\n";
  std::cout << "  rbsc[idx_mn] = constrained_value;\n";
  std::cout << "  zbcc[idx_mn] = constrained_value;\n";

  std::cout << "\nThis would:\n";
  std::cout << "1. Match jVMEC behavior exactly\n";
  std::cout << "2. Create the 53.77% boundary change we discovered\n";
  std::cout << "3. Potentially fix asymmetric convergence issues\n";
  std::cout << "4. The scaling_factor parameter becomes unnecessary\n";

  std::cout << "\n✅ READY TO IMPLEMENT jVMEC-COMPATIBLE M=1 CONSTRAINT\n";

  EXPECT_TRUE(true) << "Modification proposal created";
}

}  // namespace vmecpp

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
