// Test jVMEC boundary theta shift correction
// Based on analysis showing jVMEC uses corrected formula for theta shift

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

using vmecpp::VmecINDATA;

namespace vmecpp {

TEST(JVMECBoundaryShiftTest, AnalyzeTheTaShiftFormula) {
  std::cout << "\n=== JVMEC BOUNDARY THETA SHIFT ANALYSIS ===" << std::endl;

  // Create configuration matching up_down_asymmetric_tokamak.json
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 5;
  config.ntor = 0;

  // Set up boundary coefficients from proven working configuration
  config.rbc.resize(5, 0.0);
  config.zbs.resize(5, 0.0);
  config.rbs.resize(5, 0.0);  // Asymmetric
  config.zbc.resize(5, 0.0);  // Asymmetric

  // From up_down_asymmetric_tokamak.json:
  // rbc: [6.0, 0.0, 0.6, 0.0, 0.12]
  // rbs: [0.0, 0.0, 0.189737, 0.0, 0.0]
  // zbc: [0.0, 0.0, 0.6, 0.0, 0.12]  // Note: this should be zbc
  // zbs: [0.0, 0.0, 0.0, 0.0, 0.0]

  config.rbc[0] = 6.0;
  config.rbc[2] = 0.6;
  config.rbc[4] = 0.12;

  config.rbs[2] = 0.189737;

  config.zbc[2] = 0.6;  // Note: JSON has this as zbc, not zbs
  config.zbc[4] = 0.12;

  // config.zbs remains all zeros

  std::cout << "Boundary coefficients analysis:" << std::endl;
  std::cout << "  Symmetric:" << std::endl;
  std::cout << "    rbc[0] = " << config.rbc[0] << " (major radius)"
            << std::endl;
  std::cout << "    rbc[2] = " << config.rbc[2] << " (minor radius)"
            << std::endl;
  std::cout << "    rbc[4] = " << config.rbc[4] << " (higher order)"
            << std::endl;
  std::cout << "    zbs[2] = " << config.zbs[2] << " (elongation)" << std::endl;

  std::cout << "  Asymmetric:" << std::endl;
  std::cout << "    rbs[2] = " << config.rbs[2] << " (R up-down asymmetry)"
            << std::endl;
  std::cout << "    zbc[2] = " << config.zbc[2] << " (Z up-down asymmetry)"
            << std::endl;
  std::cout << "    zbc[4] = " << config.zbc[4] << " (Z higher order)"
            << std::endl;

  // Calculate jVMEC boundary theta shift formula
  // From jVMEC Boundaries.java:
  // delta = Math.atan2(Rbs[ntord][1] - Zbc[ntord][1], Rbc[ntord][1] +
  // Zbs[ntord][1]); where ntord = ntor and mode index 1 corresponds to m=1

  // For our case: ntor=0, so ntord=0, and we need m=1 mode (index 1)
  // But we don't have m=1 modes in this configuration, let's check m=2 (index
  // 2)

  double rbs_m2 = config.rbs[2];  // m=2 asymmetric R
  double zbc_m2 = config.zbc[2];  // m=2 asymmetric Z
  double rbc_m2 = config.rbc[2];  // m=2 symmetric R
  double zbs_m2 = config.zbs[2];  // m=2 symmetric Z

  std::cout << "\njVMEC theta shift formula analysis (m=2 mode):" << std::endl;
  std::cout << "  rbs[m=2] = " << rbs_m2 << std::endl;
  std::cout << "  zbc[m=2] = " << zbc_m2 << std::endl;
  std::cout << "  rbc[m=2] = " << rbc_m2 << std::endl;
  std::cout << "  zbs[m=2] = " << zbs_m2 << std::endl;

  // Calculate jVMEC formula
  double numerator = rbs_m2 - zbc_m2;    // Rbs[ntord][m] - Zbc[ntord][m]
  double denominator = rbc_m2 + zbs_m2;  // Rbc[ntord][m] + Zbs[ntord][m]

  std::cout << "\njVMEC theta shift calculation:" << std::endl;
  std::cout << "  numerator = rbs[2] - zbc[2] = " << rbs_m2 << " - " << zbc_m2
            << " = " << numerator << std::endl;
  std::cout << "  denominator = rbc[2] + zbs[2] = " << rbc_m2 << " + " << zbs_m2
            << " = " << denominator << std::endl;

  if (std::abs(denominator) > 1e-12) {
    double delta = std::atan2(numerator, denominator);
    std::cout << "  delta = atan2(" << numerator << ", " << denominator
              << ") = " << delta << " radians" << std::endl;
    std::cout << "  delta = " << (delta * 180.0 / M_PI) << " degrees"
              << std::endl;

    // Check if this is a significant correction
    if (std::abs(delta) > 1e-6) {
      std::cout << "  ⚠️  SIGNIFICANT theta shift detected!" << std::endl;
      std::cout << "  This could be a source of convergence differences!"
                << std::endl;
    } else {
      std::cout << "  ✅ theta shift is negligible" << std::endl;
    }
  } else {
    std::cout << "  ✅ denominator ≈ 0, no theta shift needed" << std::endl;
  }

  // Also check if we have any m=1 modes that would be more relevant
  std::cout << "\nChecking for m=1 modes (more relevant for theta shift):"
            << std::endl;
  if (config.rbc.size() > 1) {
    std::cout << "  rbc[1] = " << config.rbc[1] << std::endl;
  }
  if (config.rbs.size() > 1) {
    std::cout << "  rbs[1] = " << config.rbs[1] << std::endl;
  }
  if (config.zbs.size() > 1) {
    std::cout << "  zbs[1] = " << config.zbs[1] << std::endl;
  }
  if (config.zbc.size() > 1) {
    std::cout << "  zbc[1] = " << config.zbc[1] << std::endl;
  }

  // Test passes - this is an analysis test
  EXPECT_TRUE(true) << "jVMEC boundary theta shift analysis";
}

TEST(JVMECBoundaryShiftTest, CompareWithOriginalVMECFormula) {
  std::cout << "\n=== ORIGINAL VMEC vs JVMEC THETA SHIFT COMPARISON ==="
            << std::endl;

  // This test explores whether VMEC++ uses the original (potentially buggy)
  // formula or the corrected jVMEC formula

  // Create a simple asymmetric configuration with m=1 modes for clear testing
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 3;
  config.ntor = 0;

  config.rbc.resize(3, 0.0);
  config.zbs.resize(3, 0.0);
  config.rbs.resize(3, 0.0);
  config.zbc.resize(3, 0.0);

  // Set up simple m=1 mode for testing
  config.rbc[0] = 10.0;  // Major radius
  config.rbc[1] = 2.0;   // Minor radius m=1
  config.zbs[1] = 2.0;   // Z symmetric m=1

  // Add small asymmetric perturbations
  config.rbs[1] = 0.1;   // Small R asymmetric m=1
  config.zbc[1] = 0.05;  // Small Z asymmetric m=1

  std::cout << "Test configuration with m=1 modes:" << std::endl;
  std::cout << "  rbc[1] = " << config.rbc[1] << " (symmetric R)" << std::endl;
  std::cout << "  zbs[1] = " << config.zbs[1] << " (symmetric Z)" << std::endl;
  std::cout << "  rbs[1] = " << config.rbs[1] << " (asymmetric R)" << std::endl;
  std::cout << "  zbc[1] = " << config.zbc[1] << " (asymmetric Z)" << std::endl;

  // jVMEC corrected formula
  double jvmec_num = config.rbs[1] - config.zbc[1];
  double jvmec_den = config.rbc[1] + config.zbs[1];

  std::cout << "\njVMEC corrected formula:" << std::endl;
  std::cout << "  numerator = rbs[1] - zbc[1] = " << config.rbs[1] << " - "
            << config.zbc[1] << " = " << jvmec_num << std::endl;
  std::cout << "  denominator = rbc[1] + zbs[1] = " << config.rbc[1] << " + "
            << config.zbs[1] << " = " << jvmec_den << std::endl;

  if (std::abs(jvmec_den) > 1e-12) {
    double jvmec_delta = std::atan2(jvmec_num, jvmec_den);
    std::cout << "  jVMEC delta = " << jvmec_delta
              << " radians = " << (jvmec_delta * 180.0 / M_PI) << " degrees"
              << std::endl;
  }

  // Note: Without access to original VMEC formula, we document what jVMEC uses
  std::cout << "\nKey insight: jVMEC corrects theta shift calculation"
            << std::endl;
  std::cout << "This could explain convergence differences if VMEC++ uses "
               "different formula"
            << std::endl;

  // Test passes - analysis test
  EXPECT_TRUE(true) << "Theta shift comparison analysis";
}

}  // namespace vmecpp
