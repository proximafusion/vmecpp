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

// Test with very simple asymmetric configuration to check basic functionality
TEST(SimpleAsymmetricTest, BasicAsymmetricTokamak) {
  std::cout << "\n=== SIMPLE ASYMMETRIC TEST ===\n" << std::endl;

  // Start with absolute minimum asymmetric tokamak
  VmecINDATA indata;

  // Minimal parameters
  indata.nfp = 1;
  indata.lasym = true;
  indata.mpol = 1;  // Just m=0,1
  indata.ntor = 0;  // Axisymmetric
  indata.ns_array = {3};
  indata.niter_array = {50};
  indata.ntheta = 17;
  indata.nzeta = 1;

  // Zero pressure to avoid pressure calculation issues
  indata.pres_scale = 0.0;
  indata.am = {0.0};
  indata.gamma = 0.0;
  indata.phiedge = 1.0;  // Small flux

  indata.return_outputs_even_if_not_converged = true;

  // Simple coefficient arrays
  int coeff_size = indata.mpol * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  // For ntor=0, mpol=1: indices are just m (0, 1)

  // m=0: Major radius
  indata.rbc[0] = 10.0;  // R00
  indata.zbc[0] = 0.0;   // Z00 (should be zero)
  indata.rbs[0] = 0.0;   // Asymmetric R00 (should be zero)
  indata.zbs[0] = 0.0;   // Asymmetric Z00 (should be zero)

  // m=1: Minor radius and elongation
  indata.rbc[1] = 1.0;  // R10 - minor radius
  indata.zbs[1] = 1.0;  // Z10 - elongation
  indata.rbs[1] = 0.1;  // Asymmetric R10 - small perturbation
  indata.zbc[1] = 0.1;  // Asymmetric Z10 - small perturbation

  // Axis arrays
  indata.raxis_c = {10.0};  // Match R00
  indata.zaxis_s = {0.0};
  indata.raxis_s = {0.0};  // Small asymmetric axis
  indata.zaxis_c = {0.0};

  std::cout << "Configuration:" << std::endl;
  std::cout << "  lasym = " << indata.lasym << std::endl;
  std::cout << "  mpol = " << indata.mpol << ", ntor = " << indata.ntor
            << std::endl;
  std::cout << "  R00 = " << indata.rbc[0] << ", R10 = " << indata.rbc[1]
            << std::endl;
  std::cout << "  Z10 = " << indata.zbs[1] << std::endl;
  std::cout << "  Asymmetric R10 = " << indata.rbs[1] << std::endl;
  std::cout << "  Asymmetric Z10 = " << indata.zbc[1] << std::endl;

  std::cout << "\nRunning simple asymmetric tokamak with zero pressure..."
            << std::endl;

  const auto output = vmecpp::run(indata);

  if (output.ok()) {
    std::cout << "\n✅ SUCCESS: Simple asymmetric configuration works!"
              << std::endl;
    const auto& wout = output->wout;
    std::cout << "  Volume = " << wout.volume_p << std::endl;
    std::cout << "  Aspect ratio = " << wout.aspect << std::endl;

    EXPECT_GT(wout.volume_p, 0.0) << "Volume should be positive";
    EXPECT_GT(wout.aspect, 0.0) << "Aspect ratio should be positive";

  } else {
    std::cout << "\n❌ FAILED: Simple asymmetric configuration failed"
              << std::endl;
    std::cout << "Error: " << output.status() << std::endl;

    // Even the simplest asymmetric case fails - this shows the fundamental
    // issue
    FAIL() << "Simple asymmetric configuration should work: "
           << output.status();
  }
}

}  // namespace vmecpp
