// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Test to debug force calculation issues in asymmetric tokamak
TEST(ForceDebugTest, DebugAsymmetricForceCalculation) {
  std::cout << "\n=== FORCE CALCULATION DEBUG TEST ===\n" << std::endl;

  // Create minimal asymmetric tokamak with very simple geometry
  VmecINDATA indata;

  // Basic parameters - minimal configuration
  indata.nfp = 1;
  indata.lasym = true;
  indata.mpol = 2;           // Very simple: m=0,1 only
  indata.ntor = 0;           // Axisymmetric
  indata.ns_array = {3};     // Only 3 radial surfaces
  indata.niter_array = {5};  // Just 5 iterations to see where NaN appears
  indata.ntheta = 9;         // Minimal theta grid
  indata.nzeta = 1;          // Axisymmetric

  // Very mild pressure to test if it causes NaN
  indata.pres_scale = 1000.0;  // Reduced pressure scale
  indata.am = {1.0};           // Simple parabolic pressure
  indata.gamma = 0.0;
  indata.phiedge = 1.0;  // Small flux

  indata.return_outputs_even_if_not_converged = true;

  // Simple coefficient arrays - tokamak geometry
  int coeff_size = indata.mpol * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  // Simple circular tokamak geometry
  indata.rbc[0] = 3.0;  // R00 - major radius
  indata.rbc[1] = 1.0;  // R10 - minor radius
  indata.zbs[1] = 1.0;  // Z10 - elongation

  // Small asymmetric perturbations
  indata.rbs[1] = 0.01;  // 1% asymmetric perturbation in R
  indata.zbc[1] = 0.01;  // 1% asymmetric perturbation in Z

  // Axis arrays
  indata.raxis_c = {3.0};  // Match R00
  indata.zaxis_s = {0.0};
  indata.raxis_s = {0.0};  // No asymmetric axis
  indata.zaxis_c = {0.0};

  std::cout << "Configuration:" << std::endl;
  std::cout << "  lasym = " << indata.lasym << std::endl;
  std::cout << "  mpol = " << indata.mpol << ", ntor = " << indata.ntor
            << std::endl;
  std::cout << "  Pressure scale = " << indata.pres_scale << std::endl;
  std::cout << "  R00 = " << indata.rbc[0] << ", R10 = " << indata.rbc[1]
            << std::endl;
  std::cout << "  Asymmetric R10 = " << indata.rbs[1] << std::endl;
  std::cout << "  Asymmetric Z10 = " << indata.zbc[1] << std::endl;

  std::cout << "\nRunning minimal asymmetric tokamak (5 iterations only)..."
            << std::endl;

  const auto output = vmecpp::run(indata);

  if (output.ok()) {
    std::cout << "\n✅ SUCCESS: Force calculation completed without NaN!"
              << std::endl;
    const auto& wout = output->wout;
    std::cout << "  Volume = " << wout.volume_p << std::endl;
    std::cout << "  Aspect ratio = " << wout.aspect << std::endl;

    // Check for finite values
    EXPECT_TRUE(std::isfinite(wout.volume_p)) << "Volume should be finite";
    EXPECT_TRUE(std::isfinite(wout.aspect)) << "Aspect ratio should be finite";
    EXPECT_GT(wout.volume_p, 0.0) << "Volume should be positive";

  } else {
    std::cout << "\n❌ FAILED: Force calculation failed" << std::endl;
    std::cout << "Error: " << output.status() << std::endl;

    // This will help us debug the exact failure mode
    std::string error_msg = output.status().ToString();

    if (error_msg.find("Non-finite") != std::string::npos) {
      std::cout << "\n🔍 ANALYSIS: NaN detected in force calculations"
                << std::endl;
      std::cout << "This confirms the issue is in MHD force computation, not "
                   "transforms"
                << std::endl;
    }

    if (error_msg.find("blmn_e") != std::string::npos) {
      std::cout << "Lambda forces (blmn_e) are the source of NaN values"
                << std::endl;
    }

    if (error_msg.find("taup") != std::string::npos ||
        error_msg.find("zup") != std::string::npos) {
      std::cout
          << "Pressure-related derivatives (taup/zup) are NaN - pressure issue"
          << std::endl;
    }

    // Still fail the test but with diagnostic info
    FAIL() << "Force calculation should not produce NaN values: "
           << output.status();
  }
}

}  // namespace vmecpp
