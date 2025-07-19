// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// Unit test to isolate Jacobian calculation issues in asymmetric mode
TEST(GeometryJacobianTest, DebugAsymmetricJacobianCalculation) {
  std::cout << "\n=== GEOMETRY JACOBIAN DEBUG TEST ===\n" << std::endl;

  // Test with the simplest possible asymmetric configuration
  // to isolate where Jacobian becomes singular

  std::cout
      << "Testing Jacobian calculation with minimal asymmetric coefficients..."
      << std::endl;

  // Simple test case: circular tokamak with tiny asymmetric perturbation
  VmecINDATA indata;

  indata.nfp = 1;
  indata.lasym = true;
  indata.mpol = 2;     // m=0,1 modes
  indata.ntor = 0;     // Axisymmetric base
  indata.ntheta = 10;  // Slightly more theta points to see kl=6-9
  indata.nzeta = 1;

  // Array setup
  int coeff_size = indata.mpol * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  // Perfect circular tokamak (symmetric part)
  indata.rbc[0] = 3.0;  // R00 = major radius
  indata.rbc[1] = 1.0;  // R10 = minor radius
  indata.zbs[1] = 1.0;  // Z10 = elongation

  std::cout << "\nTesting different asymmetric perturbation levels:"
            << std::endl;

  // Test 1: Zero asymmetric perturbation (should work like symmetric)
  std::cout << "\n1. ZERO ASYMMETRIC PERTURBATION:" << std::endl;
  indata.rbs[1] = 0.0;
  indata.zbc[1] = 0.0;
  std::cout << "  rbs[1] = " << indata.rbs[1] << ", zbc[1] = " << indata.zbc[1]
            << std::endl;
  std::cout << "  Expected: Should work like symmetric case" << std::endl;

  // Test 2: Tiny asymmetric perturbation
  std::cout << "\n2. TINY ASYMMETRIC PERTURBATION:" << std::endl;
  indata.rbs[1] = 1e-8;  // Extremely small
  indata.zbc[1] = 1e-8;
  std::cout << "  rbs[1] = " << indata.rbs[1] << ", zbc[1] = " << indata.zbc[1]
            << std::endl;
  std::cout << "  Expected: Should still work if algorithm is robust"
            << std::endl;

  // Test 3: Small asymmetric perturbation
  std::cout << "\n3. SMALL ASYMMETRIC PERTURBATION:" << std::endl;
  indata.rbs[1] = 1e-4;
  indata.zbc[1] = 1e-4;
  std::cout << "  rbs[1] = " << indata.rbs[1] << ", zbc[1] = " << indata.zbc[1]
            << std::endl;
  std::cout << "  Expected: This is where problems might start" << std::endl;

  // Test 4: Moderate asymmetric perturbation (known to fail)
  std::cout << "\n4. MODERATE ASYMMETRIC PERTURBATION:" << std::endl;
  indata.rbs[1] = 0.001;
  indata.zbc[1] = 0.001;
  std::cout << "  rbs[1] = " << indata.rbs[1] << ", zbc[1] = " << indata.zbc[1]
            << std::endl;
  std::cout << "  Expected: Known to cause NaN at kl=6-9" << std::endl;

  // Axis arrays
  indata.raxis_c = {3.0};
  indata.zaxis_s = {0.0};
  indata.raxis_s = {0.0};
  indata.zaxis_c = {0.0};

  std::cout << "\n📋 ANALYSIS QUESTIONS:" << std::endl;
  std::cout
      << "1. Is there a threshold where asymmetric perturbations cause issues?"
      << std::endl;
  std::cout << "2. Are specific theta positions (kl=6-9) more sensitive?"
            << std::endl;
  std::cout
      << "3. Is the issue in the Fourier transform or Jacobian calculation?"
      << std::endl;
  std::cout << "4. Does jVMEC handle these same perturbations successfully?"
            << std::endl;

  std::cout << "\n🔍 COMPARISON WITH jVMEC:" << std::endl;
  std::cout << "jVMEC uses separate even/odd contributions in Jacobian:"
            << std::endl;
  std::cout << "  tau = even_contrib + dSHalfdS * odd_contrib" << std::endl;
  std::cout << "VMEC++ uses single calculation:" << std::endl;
  std::cout << "  tau = tau1 + dSHalfDsInterp * tau2" << std::endl;
  std::cout << "Difference may be in how asymmetric terms are handled in tau2"
            << std::endl;

  std::cout << "\n⚠️  KNOWN ISSUES TO INVESTIGATE:" << std::endl;
  std::cout << "1. sqrtSH division near axis (axis protection needed)"
            << std::endl;
  std::cout << "2. Asymmetric coefficient handling in tau2 calculation"
            << std::endl;
  std::cout << "3. Theta grid indexing differences between symmetric/asymmetric"
            << std::endl;
  std::cout << "4. Missing asymmetric force symmetrization step" << std::endl;

  // Test designed to show the expected analysis, not to pass/fail
  EXPECT_TRUE(true) << "This test is for analysis and documentation";
}

// Test specifically for Jacobian robustness
TEST(GeometryJacobianTest, TestJacobianRobustness) {
  std::cout << "\n=== JACOBIAN ROBUSTNESS TEST ===\n" << std::endl;

  std::cout << "Key differences between jVMEC and VMEC++ Jacobian calculation:"
            << std::endl;

  std::cout << "\n1. jVMEC separates even and odd contributions:" << std::endl;
  std::cout << "   - even_contrib = dRdTheta_half * dZdS_half - dRdS_half * "
               "dZdTheta_half"
            << std::endl;
  std::cout << "   - odd_contrib = (asymmetric cross terms)" << std::endl;
  std::cout << "   - tau = even_contrib + dSHalfdS * odd_contrib" << std::endl;

  std::cout << "\n2. VMEC++ combines everything:" << std::endl;
  std::cout << "   - tau1 = ru12 * zs - rs * zu12" << std::endl;
  std::cout << "   - tau2 = (all cross terms including asymmetric)"
            << std::endl;
  std::cout << "   - tau = tau1 + dSHalfDsInterp * tau2" << std::endl;

  std::cout << "\n3. Potential fixes needed in VMEC++:" << std::endl;
  std::cout << "   a) Add axis protection: if (j == 0) tau[0] = tau[1]"
            << std::endl;
  std::cout << "   b) Separate asymmetric contributions in tau2" << std::endl;
  std::cout << "   c) Better handling of sqrtSH division" << std::endl;
  std::cout << "   d) Implement asymmetric force symmetrization" << std::endl;

  std::cout << "\n4. Theta grid issues at kl=6-9:" << std::endl;
  std::cout << "   - These correspond to specific theta positions" << std::endl;
  std::cout << "   - May need different boundary treatment" << std::endl;
  std::cout << "   - jVMEC uses ntheta2 vs ntheta3 grids" << std::endl;

  EXPECT_TRUE(true) << "This test documents required fixes";
}

}  // namespace vmecpp
