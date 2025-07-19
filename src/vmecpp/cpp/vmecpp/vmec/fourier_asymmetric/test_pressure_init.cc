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

// Test to isolate pressure initialization issue
TEST(PressureInitTest, CompareSymmetricVsAsymmetric) {
  std::cout << "\n=== PRESSURE INITIALIZATION TEST ===\n" << std::endl;

  // Create identical configurations except for lasym
  VmecINDATA indata_sym;
  indata_sym.lasym = false;
  indata_sym.ns_array = {3};
  indata_sym.niter_array = {5};
  indata_sym.mpol = 2;
  indata_sym.ntor = 1;
  indata_sym.nfp = 1;
  indata_sym.return_outputs_even_if_not_converged = true;

  // Simple pressure profile
  indata_sym.pmass_type = "power_series";
  indata_sym.pres_scale = 0.01;
  indata_sym.am = {1.0, -1.0};  // Linear profile
  indata_sym.gamma = 0.0;

  // Simple geometry
  int coeff_size = indata_sym.mpol * (2 * indata_sym.ntor + 1);
  indata_sym.rbc.resize(coeff_size, 0.0);
  indata_sym.zbs.resize(coeff_size, 0.0);

  // (m=0, n=0) - major radius
  int idx_00 = 0 * (2 * indata_sym.ntor + 1) + (0 + indata_sym.ntor);
  indata_sym.rbc[idx_00] = 1.0;

  // (m=1, n=0) - minor radius
  int idx_10 = 1 * (2 * indata_sym.ntor + 1) + (0 + indata_sym.ntor);
  indata_sym.rbc[idx_10] = 0.1;
  indata_sym.zbs[idx_10] = 0.1;

  // Axis arrays
  indata_sym.raxis_c.resize(indata_sym.ntor + 1, 0.0);
  indata_sym.zaxis_s.resize(indata_sym.ntor + 1, 0.0);
  indata_sym.raxis_c[0] = 1.0;

  // Clear asymmetric arrays
  indata_sym.rbs.clear();
  indata_sym.zbc.clear();
  indata_sym.raxis_s.clear();
  indata_sym.zaxis_c.clear();

  // Create asymmetric version
  VmecINDATA indata_asym = indata_sym;
  indata_asym.lasym = true;

  // Add asymmetric arrays
  indata_asym.rbs.resize(coeff_size, 0.0);
  indata_asym.zbc.resize(coeff_size, 0.0);
  indata_asym.rbs[idx_10] = 0.001;  // Tiny perturbation
  indata_asym.zbc[idx_10] = 0.001;

  indata_asym.raxis_s.resize(indata_asym.ntor + 1, 0.0);
  indata_asym.zaxis_c.resize(indata_asym.ntor + 1, 0.0);

  // Run both cases
  std::cout << "\nRunning symmetric case..." << std::endl;
  const auto output_sym = vmecpp::run(indata_sym);

  std::cout << "\nRunning asymmetric case..." << std::endl;
  const auto output_asym = vmecpp::run(indata_asym);

  if (output_sym.ok()) {
    std::cout << "\nSymmetric case: SUCCESS" << std::endl;
    std::cout << "  Volume = " << output_sym->wout.volume_p << std::endl;
    std::cout << "  Beta = " << output_sym->wout.beta << std::endl;
  } else {
    std::cout << "\nSymmetric case: FAILED - " << output_sym.status()
              << std::endl;
  }

  if (output_asym.ok()) {
    std::cout << "\nAsymmetric case: SUCCESS" << std::endl;
    std::cout << "  Volume = " << output_asym->wout.volume_p << std::endl;
    std::cout << "  Beta = " << output_asym->wout.beta << std::endl;
  } else {
    std::cout << "\nAsymmetric case: FAILED - " << output_asym.status()
              << std::endl;
  }

  EXPECT_TRUE(output_sym.ok()) << "Symmetric case should succeed";
  EXPECT_TRUE(output_asym.ok()) << "Asymmetric case should succeed";
}

}  // namespace vmecpp
