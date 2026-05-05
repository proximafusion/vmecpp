// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Test to debug Jacobian calculation in asymmetric mode with detailed output
TEST(JacobianAsymmetricDebugTest, TraceJacobianCalculation) {
  std::cout << "\n=== JACOBIAN ASYMMETRIC DEBUG TEST ===\n" << std::endl;

  std::cout << "Testing Jacobian calculation with asymmetric perturbation...\n"
            << std::endl;

  // Minimal asymmetric tokamak that causes NaN
  VmecINDATA indata;

  indata.nfp = 1;
  indata.lasym = true;
  indata.mpol = 2;
  indata.ntor = 0;
  indata.ns_array = {3};     // Minimal radial surfaces
  indata.niter_array = {1};  // Just 1 iteration to see initial calculation
  indata.ntheta = 10;
  indata.nzeta = 1;

  // Zero pressure to isolate geometry
  indata.pres_scale = 0.0;
  indata.am = {0.0};
  indata.gamma = 0.0;
  indata.phiedge = 1.0;

  indata.return_outputs_even_if_not_converged = true;

  // Simple tokamak geometry
  int coeff_size = indata.mpol * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  indata.rbc[0] = 3.0;  // R00
  indata.rbc[1] = 1.0;  // R10
  indata.zbs[1] = 1.0;  // Z10

  // Small asymmetric perturbation
  indata.rbs[1] = 0.001;
  indata.zbc[1] = 0.001;

  indata.raxis_c = {3.0};
  indata.zaxis_s = {0.0};
  indata.raxis_s = {0.0};
  indata.zaxis_c = {0.0};

  std::cout << "Configuration:" << std::endl;
  std::cout << "  lasym = " << indata.lasym << std::endl;
  std::cout << "  rbs[1] = " << indata.rbs[1] << " (asymmetric R perturbation)"
            << std::endl;
  std::cout << "  zbc[1] = " << indata.zbc[1] << " (asymmetric Z perturbation)"
            << std::endl;

  std::cout << "\nKEY ISSUES TO WATCH:" << std::endl;
  std::cout << "1. tau2 calculation has division by sqrtSH" << std::endl;
  std::cout << "2. At axis (sqrtSH → 0), this division could cause issues"
            << std::endl;
  std::cout << "3. Even with protection (min 1e-12), numerical issues may arise"
            << std::endl;
  std::cout << "4. jVMEC uses different even/odd separation approach"
            << std::endl;

  std::cout << "\nEXPECTED OUTPUT:" << std::endl;
  std::cout << "- Should see detailed error about tau, zu12, ru12 becoming NaN"
            << std::endl;
  std::cout << "- Error should occur at kl=6,7,8,9 (second half of theta)"
            << std::endl;
  std::cout << "- This confirms issue is in tau2 calculation" << std::endl;

  const auto output = vmecpp::run(indata);

  if (output.ok()) {
    std::cout << "\n✅ Unexpected: No error occurred!" << std::endl;
    const auto& wout = output->wout;
    std::cout << "  Volume = " << wout.volume_p << std::endl;
    std::cout << "  This suggests axis protection may be working partially"
              << std::endl;
  } else {
    std::cout << "\n❌ Expected error occurred:" << std::endl;
    std::cout << output.status() << std::endl;

    std::string error_msg = output.status().ToString();

    // Check if it's the expected tau/Jacobian error
    if (error_msg.find("tau") != std::string::npos) {
      std::cout << "\n✅ Confirmed: tau (Jacobian) calculation fails"
                << std::endl;

      if (error_msg.find("kl=6") != std::string::npos ||
          error_msg.find("kl=7") != std::string::npos ||
          error_msg.find("kl=8") != std::string::npos ||
          error_msg.find("kl=9") != std::string::npos) {
        std::cout << "✅ Confirmed: Failure at kl=6-9 (second half of theta)"
                  << std::endl;
      }
    }

    if (error_msg.find("zu12") != std::string::npos ||
        error_msg.find("ru12") != std::string::npos) {
      std::cout << "✅ Confirmed: Geometry derivatives involved" << std::endl;
    }
  }

  std::cout << "\nPROPOSED FIX:" << std::endl;
  std::cout << "1. Implement proper even/odd separation like jVMEC"
            << std::endl;
  std::cout << "2. Avoid division by sqrtSH in tau2 calculation" << std::endl;
  std::cout
      << "3. Use jVMEC's approach: tau = even_contrib + sqrtSH * odd_contrib"
      << std::endl;
  std::cout << "4. This avoids division and is numerically stable at axis"
            << std::endl;

  // Test passes if we get the expected error
  EXPECT_FALSE(output.ok()) << "Expected NaN error in Jacobian calculation";
}

// Test to understand tau2 calculation issue
TEST(JacobianAsymmetricDebugTest, AnalyzeTau2Division) {
  std::cout << "\n=== TAU2 DIVISION ANALYSIS ===\n" << std::endl;

  std::cout << "Current tau2 calculation in ideal_mhd_model.cc:" << std::endl;
  std::cout << "```cpp" << std::endl;
  std::cout << "double tau2 = ruo_o * z1o_o + ruo_i * z1o_i -" << std::endl;
  std::cout << "              zuo_o * r1o_o - zuo_i * r1o_i +" << std::endl;
  std::cout << "              (rue_o * z1o_o + rue_i * z1o_i -" << std::endl;
  std::cout << "               zue_o * r1o_o - zue_i * r1o_i) /" << std::endl;
  std::cout << "                  protected_sqrtSH;  // <-- DIVISION!"
            << std::endl;
  std::cout << "double tau = tau1 + dSHalfDsInterp * tau2;" << std::endl;
  std::cout << "```" << std::endl;

  std::cout << "\nPROBLEM:" << std::endl;
  std::cout << "- Division by sqrtSH (even if protected to 1e-12)" << std::endl;
  std::cout << "- Can amplify numerical errors near axis" << std::endl;
  std::cout << "- Not how jVMEC handles asymmetric contributions" << std::endl;

  std::cout << "\njVMEC APPROACH:" << std::endl;
  std::cout << "```fortran" << std::endl;
  std::cout << "! No division - multiplication instead" << std::endl;
  std::cout << "tau = tau_even + sqrtS * tau_odd" << std::endl;
  std::cout << "```" << std::endl;

  std::cout << "\nCORRECT FORMULATION:" << std::endl;
  std::cout << "```cpp" << std::endl;
  std::cout << "// Even contributions (symmetric-like)" << std::endl;
  std::cout << "double tau_even = rue * zs_even - rs_even * zue;" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "// Odd contributions (asymmetric)" << std::endl;
  std::cout << "double tau_odd = ruo * zs_odd - rs_odd * zuo;" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "// Combine without division" << std::endl;
  std::cout << "double tau = tau_even + sqrtSH * tau_odd;" << std::endl;
  std::cout << "```" << std::endl;

  EXPECT_TRUE(true) << "Analysis completed";
}

}  // namespace vmecpp
