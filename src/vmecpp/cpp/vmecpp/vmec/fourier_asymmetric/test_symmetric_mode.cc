// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Test symmetric mode to ensure no regression
TEST(SymmetricModeTest, BasicSymmetricTokamak) {
  std::cout << "\n=== SYMMETRIC MODE TEST ===\n" << std::endl;

  // Simple symmetric tokamak (MUST work)
  VmecINDATA indata;

  // Symmetric parameters
  indata.nfp = 1;
  indata.lasym = false;  // SYMMETRIC MODE
  indata.mpol = 3;
  indata.ntor = 0;  // Axisymmetric
  indata.ns_array = {5};
  indata.niter_array = {100};
  indata.ftol_array = {1e-08};
  indata.ntheta = 17;
  indata.nzeta = 1;

  // Zero pressure vacuum case
  indata.pres_scale = 0.0;
  indata.am = {0.4, 0.0};
  indata.gamma = 0.0;
  indata.phiedge = 1.0;

  indata.return_outputs_even_if_not_converged = true;

  // Simple symmetric boundary configuration
  const int array_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
  indata.rbc.resize(array_size, 0.0);
  indata.zbs.resize(array_size, 0.0);
  
  // Only symmetric coefficients
  indata.rbc[0] = 1.0;   // R00 - major radius
  indata.rbc[1] = 0.3;   // R10 - minor radius  
  indata.zbs[1] = 0.3;   // Z10 - elongation

  // Axis arrays (symmetric only)
  indata.raxis_c = {1.0};  // Match R00
  indata.zaxis_s = {0.0};

  std::cout << "Symmetric Configuration:" << std::endl;
  std::cout << "  lasym = " << indata.lasym << " (SYMMETRIC)" << std::endl;
  std::cout << "  mpol = " << indata.mpol << ", ntor = " << indata.ntor
            << std::endl;
  std::cout << "  RBC(0,0) = " << indata.rbc[0] << " (major radius)" << std::endl;
  std::cout << "  RBC(0,1) = " << indata.rbc[1] << " (minor radius)" << std::endl;
  std::cout << "  ZBS(0,1) = " << indata.zbs[1] << " (elongation)" << std::endl;

  std::cout << "\nRunning symmetric tokamak (MUST PASS)..."
            << std::endl;

  const auto output = vmecpp::run(indata);

  if (output.ok()) {
    std::cout << "\n✅ SUCCESS: Symmetric mode works correctly!"
              << std::endl;
    const auto& wout = output->wout;
    std::cout << "  Volume = " << wout.volume_p << std::endl;
    std::cout << "  Aspect ratio = " << wout.aspect << std::endl;
    std::cout << "  No regression detected!" << std::endl;

    EXPECT_GT(wout.volume_p, 0.0) << "Volume should be positive";
    EXPECT_GT(wout.aspect, 0.0) << "Aspect ratio should be positive";

  } else {
    std::cout << "\n❌ CRITICAL: Symmetric mode failed!"
              << std::endl;
    std::cout << "Error: " << output.status() << std::endl;

    // This is a critical regression - symmetric mode MUST work
    FAIL() << "🚨 REGRESSION: Symmetric mode must work: "
           << output.status();
  }
}

}  // namespace vmecpp