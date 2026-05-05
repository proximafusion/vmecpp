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

// Test to check dVds initialization differences between symmetric and
// asymmetric
TEST(DVdsInitTest, CheckDVdsInitialization) {
  std::cout << "\n=== DVDS INITIALIZATION TEST ===\n" << std::endl;

  // Create a very simple configuration
  VmecINDATA indata;
  indata.lasym = true;  // Start with asymmetric
  indata.ns_array = {3};
  indata.niter_array = {1};  // Just one iteration to see initial values
  indata.mpol = 1;
  indata.ntor = 0;  // Axisymmetric for simplicity
  indata.nfp = 1;
  indata.return_outputs_even_if_not_converged = true;

  // Zero pressure to isolate geometry effects
  indata.pmass_type = "power_series";
  indata.pres_scale = 0.0;  // No pressure
  indata.am = {0.0};
  indata.gamma = 0.0;

  // Very simple circular cross-section
  int coeff_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  // (m=0, n=0) - major radius
  int idx_00 = 0 * (2 * indata.ntor + 1) + (0 + indata.ntor);
  indata.rbc[idx_00] = 10.0;

  // (m=1, n=0) - circular cross-section
  int idx_10 = 1 * (2 * indata.ntor + 1) + (0 + indata.ntor);
  indata.rbc[idx_10] = 1.0;
  indata.zbs[idx_10] = 1.0;

  // Small asymmetric perturbation
  indata.rbs[idx_10] = 0.01;
  indata.zbc[idx_10] = 0.01;

  // Axis arrays
  indata.raxis_c.resize(1, 10.0);
  indata.zaxis_s.resize(1, 0.0);
  indata.raxis_s.resize(1, 0.0);
  indata.zaxis_c.resize(1, 0.0);

  std::cout << "Running asymmetric case with zero pressure..." << std::endl;
  std::cout << "Configuration: R0=" << indata.rbc[0] << ", a=" << indata.rbc[1]
            << ", asymmetric perturbation=" << indata.rbs[1] << std::endl;

  const auto output = vmecpp::run(indata);

  if (output.ok()) {
    std::cout << "\nAsymmetric case completed!" << std::endl;
    const auto& wout = output->wout;
    std::cout << "  Volume = " << wout.volume_p << std::endl;
    std::cout << "  Aspect ratio = " << wout.aspect << std::endl;

    // Check some basic output values
    std::cout << "  Beta total = " << wout.betaxis << std::endl;
    std::cout << "  R axis = " << wout.Rmajor_p << std::endl;
  } else {
    std::cout << "\nAsymmetric case FAILED: " << output.status() << std::endl;
  }

  // Now try symmetric case
  indata.lasym = false;
  indata.rbs.clear();
  indata.zbc.clear();
  indata.raxis_s.clear();
  indata.zaxis_c.clear();

  std::cout << "\nRunning symmetric case with zero pressure..." << std::endl;

  const auto output_sym = vmecpp::run(indata);

  if (output_sym.ok()) {
    std::cout << "\nSymmetric case completed!" << std::endl;
    const auto& wout = output_sym->wout;
    std::cout << "  Volume = " << wout.volume_p << std::endl;
    std::cout << "  Aspect ratio = " << wout.aspect << std::endl;

    std::cout << "  Beta total = " << wout.betaxis << std::endl;
    std::cout << "  R axis = " << wout.Rmajor_p << std::endl;
  } else {
    std::cout << "\nSymmetric case FAILED: " << output_sym.status()
              << std::endl;
  }

  // Both should succeed with zero pressure
  EXPECT_TRUE(output_sym.ok())
      << "Symmetric case should work with zero pressure";
  EXPECT_TRUE(output.ok()) << "Asymmetric case should work with zero pressure";
}

}  // namespace vmecpp
