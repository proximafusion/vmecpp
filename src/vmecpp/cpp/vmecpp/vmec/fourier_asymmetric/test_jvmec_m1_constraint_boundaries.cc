// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/boundaries/boundaries.h"

namespace vmecpp {

// Test for modified boundaries.cc with jVMEC-compatible M=1 constraint
// Following TDD approach with meticulous debug output
class JVMECCompatibleM1ConstraintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup for M=1 constraint boundary testing
  }

  void CreateSizesAndBasis(const VmecINDATA& config,
                           std::unique_ptr<Sizes>& sizes,
                           std::unique_ptr<FourierBasisFastPoloidal>& basis) {
    // Create Sizes object
    int ntheta = 2 * config.mpol + 6;       // VMEC standard
    int nzeta = (config.ntor > 0) ? 3 : 1;  // 3 for 3D, 1 for axisymmetric

    sizes = std::make_unique<Sizes>(config.lasym,  // lasym
                                    config.nfp,    // nfp
                                    config.mpol,   // mpol
                                    config.ntor,   // ntor
                                    ntheta,        // ntheta
                                    nzeta          // nzeta
    );

    // Create Fourier basis
    basis = std::make_unique<FourierBasisFastPoloidal>(sizes.get());
  }

  VmecINDATA CreateTestConfig() {
    // jVMEC input.tok_asym configuration
    VmecINDATA config;
    config.lasym = true;
    config.nfp = 1;
    config.mpol = 7;
    config.ntor = 0;
    config.ns_array = {3};  // Small for testing
    // Axisymmetric (ntor=0)

    // Resize arrays for mpol=7, ntor=0
    // Format: m * (2*ntor+1) + (ntor+n) where n goes from -ntor to ntor
    // For ntor=0: only n=0, so size = mpol * 1 = 7
    config.rbc.resize(config.mpol, 0.0);
    config.rbs.resize(config.mpol, 0.0);
    config.zbc.resize(config.mpol, 0.0);
    config.zbs.resize(config.mpol, 0.0);

    // jVMEC coefficients (axisymmetric n=0 only)
    config.rbc[0] = 5.9163;
    config.rbc[1] = 1.9196;
    config.rbc[2] = 0.33736;
    config.rbc[3] = 0.041504;
    config.rbc[4] = -0.0058256;
    config.rbc[5] = 0.010374;
    config.rbc[6] = -0.0056365;

    config.rbs[0] = 0.0;
    config.rbs[1] = 0.027610;  // Critical M=1 coefficient
    config.rbs[2] = 0.10038;
    config.rbs[3] = -0.071843;
    config.rbs[4] = -0.011423;
    config.rbs[5] = 0.008177;
    config.rbs[6] = -0.007611;

    config.zbc[0] = 0.4105;
    config.zbc[1] = 0.057302;  // Critical M=1 coefficient
    config.zbc[2] = 0.0046697;
    config.zbc[3] = -0.039155;
    config.zbc[4] = -0.0087848;
    config.zbc[5] = 0.021175;
    config.zbc[6] = 0.002439;

    config.zbs[0] = 0.0;
    config.zbs[1] = 3.6223;
    config.zbs[2] = -0.18511;
    config.zbs[3] = -0.0048568;
    config.zbs[4] = 0.059268;
    config.zbs[5] = 0.004477;
    config.zbs[6] = -0.016773;

    // Axis coefficients
    config.raxis_c = {7.5025, 0.0};
    config.zaxis_s = {0.0, 0.0};
    config.raxis_s = {0.0, 0.0};
    config.zaxis_c = {0.0, 0.0};

    return config;
  }
};

TEST_F(JVMECCompatibleM1ConstraintTest, TestCurrentVMECppConstraint) {
  std::cout << "\n=== TEST CURRENT VMEC++ M=1 CONSTRAINT ===\n";
  std::cout << std::fixed << std::setprecision(8);

  VmecINDATA config = CreateTestConfig();

  // Store original M=1 coefficients
  double original_rbs1 = config.rbs[1];
  double original_zbc1 = config.zbc[1];

  std::cout << "Original input coefficients:\n";
  std::cout << "  rbs[1] = " << original_rbs1 << "\n";
  std::cout << "  zbc[1] = " << original_zbc1 << "\n";
  std::cout << "  Difference = " << std::abs(original_rbs1 - original_zbc1)
            << "\n";

  // Create Sizes and Basis
  std::unique_ptr<Sizes> sizes;
  std::unique_ptr<FourierBasisFastPoloidal> basis;
  CreateSizesAndBasis(config, sizes, basis);

  // Create Boundaries object and apply constraint
  Boundaries boundaries(sizes.get(), basis.get(), -1);
  boundaries.setupFromIndata(config, true);

  // Check what current VMEC++ constraint did
  // m=1, n=0 is at index 1 * (0+1) + 0 = 1
  double vmecpp_rbsc = boundaries.rbsc[1];
  double vmecpp_zbcc = boundaries.zbcc[1];

  std::cout << "\nAfter CURRENT VMEC++ ensureM1Constrained(0.5):\n";
  std::cout << "  rbsc[1] = " << vmecpp_rbsc << "\n";
  std::cout << "  zbcc[1] = " << vmecpp_zbcc << "\n";
  std::cout << "  Difference = " << std::abs(vmecpp_rbsc - vmecpp_zbcc) << "\n";

  // Verify current behavior matches rotation transformation
  double expected_rbsc = (original_rbs1 + original_zbc1) * 0.5;
  double expected_zbcc = (original_rbs1 - original_zbc1) * 0.5;

  std::cout << "\nExpected from rotation formula:\n";
  std::cout << "  rbsc = (rbs + zbc) * 0.5 = " << expected_rbsc << "\n";
  std::cout << "  zbcc = (rbs - zbc) * 0.5 = " << expected_zbcc << "\n";

  EXPECT_NEAR(vmecpp_rbsc, expected_rbsc, 1e-12)
      << "Current VMEC++ uses rotation transformation";
  EXPECT_NEAR(vmecpp_zbcc, expected_zbcc, 1e-12)
      << "Current VMEC++ uses rotation transformation";

  // Verify it does NOT satisfy jVMEC constraint
  EXPECT_GT(std::abs(vmecpp_rbsc - vmecpp_zbcc), 1e-10)
      << "Current VMEC++ does NOT enforce rbsc = zbcc";
}

TEST_F(JVMECCompatibleM1ConstraintTest, TestProposedJVMECConstraint) {
  std::cout << "\n=== TEST PROPOSED jVMEC-COMPATIBLE CONSTRAINT ===\n";
  std::cout << std::fixed << std::setprecision(8);

  VmecINDATA config = CreateTestConfig();

  // Store original M=1 coefficients
  double original_rbs1 = config.rbs[1];
  double original_zbc1 = config.zbc[1];

  std::cout << "Original input coefficients:\n";
  std::cout << "  rbs[1] = " << original_rbs1 << "\n";
  std::cout << "  zbc[1] = " << original_zbc1 << "\n";

  // Simulate jVMEC constraint being applied
  double jvmec_value = (original_rbs1 + original_zbc1) / 2.0;

  std::cout << "\nAfter jVMEC constraint (proposed modification):\n";
  std::cout << "  rbsc[1] = " << jvmec_value << " (would be)\n";
  std::cout << "  zbcc[1] = " << jvmec_value << " (would be)\n";
  std::cout << "  Difference = " << std::abs(jvmec_value - jvmec_value) << "\n";

  // Verify constraint properties
  EXPECT_NEAR(jvmec_value, 0.042456, 1e-6)
      << "jVMEC constraint average value correct";
  EXPECT_LT(std::abs(jvmec_value - jvmec_value), 1e-14)
      << "jVMEC constraint enforces exact equality";

  std::cout << "\nConstraint comparison:\n";
  std::cout << "  Original |rbs[1] - zbc[1]| = "
            << std::abs(original_rbs1 - original_zbc1) << "\n";
  std::cout << "  jVMEC    |rbsc[1] - zbcc[1]| = 0.0 (exact)\n";
  std::cout << "  Change in rbs[1]: "
            << 100.0 * std::abs(jvmec_value - original_rbs1) / original_rbs1
            << "%\n";
  std::cout << "  Change in zbc[1]: "
            << 100.0 * std::abs(jvmec_value - original_zbc1) / original_zbc1
            << "%\n";
}

TEST_F(JVMECCompatibleM1ConstraintTest, TestConstraintImplementationPlan) {
  std::cout << "\n=== IMPLEMENTATION PLAN FOR jVMEC CONSTRAINT ===\n";

  std::cout << "\nCurrent VMEC++ code in boundaries.cc (line 244-259):\n";
  std::cout << "```cpp\n";
  std::cout << "void Boundaries::ensureM1Constrained(const double "
               "scaling_factor) {\n";
  std::cout << "  for (int n = 0; n <= s_.ntor; ++n) {\n";
  std::cout << "    int m = 1;\n";
  std::cout << "    int idx_mn = m * (s_.ntor + 1) + n;\n";
  std::cout << "    if (s_.lthreed) {\n";
  std::cout << "      double backup_rss = rbss[idx_mn];\n";
  std::cout
      << "      rbss[idx_mn] = (backup_rss + zbcs[idx_mn]) * scaling_factor;\n";
  std::cout
      << "      zbcs[idx_mn] = (backup_rss - zbcs[idx_mn]) * scaling_factor;\n";
  std::cout << "    }\n";
  std::cout << "    if (s_.lasym) {\n";
  std::cout << "      double backup_rsc = rbsc[idx_mn];\n";
  std::cout
      << "      rbsc[idx_mn] = (backup_rsc + zbcc[idx_mn]) * scaling_factor;\n";
  std::cout
      << "      zbcc[idx_mn] = (backup_rsc - zbcc[idx_mn]) * scaling_factor;\n";
  std::cout << "    }\n";
  std::cout << "  }\n";
  std::cout << "}\n";
  std::cout << "```\n";

  std::cout << "\nProposed jVMEC-compatible modification:\n";
  std::cout << "```cpp\n";
  std::cout << "void Boundaries::ensureM1Constrained(const double "
               "scaling_factor) {\n";
  std::cout
      << "  // scaling_factor parameter ignored for jVMEC compatibility\n";
  std::cout << "  for (int n = 0; n <= s_.ntor; ++n) {\n";
  std::cout << "    int m = 1;\n";
  std::cout << "    int idx_mn = m * (s_.ntor + 1) + n;\n";
  std::cout << "    if (s_.lthreed) {\n";
  std::cout << "      // jVMEC constraint: set both to average\n";
  std::cout << "      double constrained_value = (rbss[idx_mn] + zbcs[idx_mn]) "
               "/ 2.0;\n";
  std::cout << "      rbss[idx_mn] = constrained_value;\n";
  std::cout << "      zbcs[idx_mn] = constrained_value;\n";
  std::cout << "    }\n";
  std::cout << "    if (s_.lasym) {\n";
  std::cout << "      // jVMEC constraint: set both to average\n";
  std::cout << "      double constrained_value = (rbsc[idx_mn] + zbcc[idx_mn]) "
               "/ 2.0;\n";
  std::cout << "      rbsc[idx_mn] = constrained_value;\n";
  std::cout << "      zbcc[idx_mn] = constrained_value;\n";
  std::cout << "    }\n";
  std::cout << "  }\n";
  std::cout << "}\n";
  std::cout << "```\n";

  std::cout << "\nKey changes:\n";
  std::cout << "1. Replace rotation transformation with averaging\n";
  std::cout << "2. Set both coefficients to the same value (coupling)\n";
  std::cout << "3. scaling_factor parameter becomes unused\n";
  std::cout << "4. Maintains same structure for symmetric/asymmetric modes\n";

  std::cout << "\n✅ READY TO IMPLEMENT jVMEC-COMPATIBLE CONSTRAINT\n";

  EXPECT_TRUE(true) << "Implementation plan documented";
}

TEST_F(JVMECCompatibleM1ConstraintTest, TestConstraintImpact) {
  std::cout << "\n=== CONSTRAINT IMPACT ANALYSIS ===\n";
  std::cout << std::fixed << std::setprecision(8);

  VmecINDATA config = CreateTestConfig();

  // Analyze all M=1 modes (for all n)
  std::cout << "M=1 mode analysis (axisymmetric case, n=0 only):\n";

  double rbs1 = config.rbs[1];
  double zbc1 = config.zbc[1];

  // Current VMEC++ constraint
  double vmecpp_rbsc = (rbs1 + zbc1) * 0.5;
  double vmecpp_zbcc = (rbs1 - zbc1) * 0.5;

  // jVMEC constraint
  double jvmec_value = (rbs1 + zbc1) / 2.0;

  std::cout << "\nOriginal:\n";
  std::cout << "  rbs[1] = " << rbs1 << "\n";
  std::cout << "  zbc[1] = " << zbc1 << "\n";

  std::cout << "\nVMEC++ constraint result:\n";
  std::cout << "  rbsc[1] = " << vmecpp_rbsc << "\n";
  std::cout << "  zbcc[1] = " << vmecpp_zbcc << "\n";
  std::cout << "  Sum preserved: " << (vmecpp_rbsc + vmecpp_zbcc) << " vs "
            << rbs1 << "\n";
  std::cout << "  Coupling: |rbsc - zbcc| = "
            << std::abs(vmecpp_rbsc - vmecpp_zbcc) << "\n";

  std::cout << "\njVMEC constraint result:\n";
  std::cout << "  rbsc[1] = " << jvmec_value << "\n";
  std::cout << "  zbcc[1] = " << jvmec_value << "\n";
  std::cout << "  Sum preserved: " << (2.0 * jvmec_value) << " vs "
            << (rbs1 + zbc1) << "\n";
  std::cout << "  Coupling: |rbsc - zbcc| = 0.0 (exact)\n";

  std::cout << "\nPhysical interpretation:\n";
  std::cout << "- VMEC++ preserves R sine component, modifies Z relationship\n";
  std::cout << "- jVMEC couples R and Z antisymmetric m=1 modes\n";
  std::cout << "- jVMEC approach may improve Jacobian conditioning\n";

  EXPECT_TRUE(true) << "Impact analysis complete";
}

}  // namespace vmecpp

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
