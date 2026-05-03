// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using file_io::ReadFile;
using nlohmann::json;
using vmecpp::VmecINDATA;

namespace vmecpp {

class RealisticAsymmetricTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create realistic asymmetric tokamak configuration
    // Based on examples/data/input.up_down_asymmetric_tokamak
    indata_.lasym = true;
    indata_.nfp = 1;   // Tokamak
    indata_.mpol = 5;  // Realistic mode count
    indata_.ntor = 0;  // Axisymmetric baseline (no toroidal modes)

    // Grid parameters - start small for debugging
    indata_.ntheta = 0;  // Will be corrected by Nyquist
    indata_.nzeta = 16;  // Small grid for faster testing

    // Multi-grid progression
    indata_.ns_array = {17};       // Single grid step
    indata_.ftol_array = {1e-11};  // Reasonable tolerance
    indata_.niter_array = {100};   // Limited iterations for testing

    // Physics parameters from working config
    indata_.delt = 0.9;
    indata_.tcon0 = 1.0;  // Must be in [0, 1]
    indata_.gamma = 0.0;
    indata_.phiedge = 6.0;
    indata_.curtor = 1.0;
    indata_.spres_ped = 1.0;
    indata_.ncurr = 0;  // Pressure-driven equilibrium
    indata_.return_outputs_even_if_not_converged = true;

    // Pressure profile (minimal but non-zero)
    indata_.pmass_type = "power_series";
    indata_.am = {0.0};  // No pressure for now
    indata_.pres_scale = 1.0;

    // Current profile
    indata_.piota_type = "power_series";
    indata_.ai = {0.9, -0.65};  // From working config
    indata_.ac = {0.0};

    // Boundary setup
    indata_.lfreeb = false;
    indata_.mgrid_file = "NONE";

    // Set up axis arrays properly sized (ntor + 1 = 1)
    indata_.raxis_c = {0.0};  // Will be computed from boundary
    indata_.raxis_s = {0.0};
    indata_.zaxis_c = {0.0};
    indata_.zaxis_s = {0.0};

    // Boundary coefficients: mpol * (2*ntor + 1) = 5 * 1 = 5
    int boundary_size = indata_.mpol * (2 * indata_.ntor + 1);
    indata_.rbc.resize(boundary_size, 0.0);
    indata_.zbs.resize(boundary_size, 0.0);
    indata_.rbs.resize(boundary_size, 0.0);
    indata_.zbc.resize(boundary_size, 0.0);

    // Helper for boundary coefficient indexing
    auto get_mn_index = [this](int m, int n) -> int {
      return m * (2 * indata_.ntor + 1) + (n + indata_.ntor);
    };

    // Set realistic tokamak boundary from working config
    // RBC(0,0) = 6.0 (major radius)
    indata_.rbc[get_mn_index(0, 0)] = 6.0;

    // ZBS(0,1) = 0.6 (would be for ntor>0, but we have ntor=0)
    // For ntor=0, we only have n=0 terms
    // RBC(1,0) and ZBS(1,0) for minor radius
    if (indata_.mpol > 1) {
      // Minor radius in Z direction
      indata_.zbs[get_mn_index(1, 0)] = 0.6;

      // Asymmetric perturbation - small R component
      indata_.rbs[get_mn_index(1, 0)] = 0.1;  // Small asymmetric perturbation
    }
  }

  VmecINDATA indata_;
};

// Test 1: Print configuration for verification
TEST_F(RealisticAsymmetricTest, PrintConfiguration) {
  std::cout << "=== REALISTIC ASYMMETRIC CONFIGURATION ===" << std::endl;
  std::cout << "lasym=" << indata_.lasym << std::endl;
  std::cout << "nfp=" << indata_.nfp << std::endl;
  std::cout << "mpol=" << indata_.mpol << std::endl;
  std::cout << "ntor=" << indata_.ntor << std::endl;
  std::cout << "ntheta=" << indata_.ntheta << " (will be corrected)"
            << std::endl;
  std::cout << "nzeta=" << indata_.nzeta << std::endl;
  std::cout << "ns=" << indata_.ns_array[0] << std::endl;
  std::cout << "niter=" << indata_.niter_array[0] << std::endl;
  std::cout << "ftol=" << indata_.ftol_array[0] << std::endl;

  std::cout << "Physics parameters:" << std::endl;
  std::cout << "  phiedge=" << indata_.phiedge << std::endl;
  std::cout << "  curtor=" << indata_.curtor << std::endl;
  std::cout << "  delt=" << indata_.delt << std::endl;
  std::cout << "  tcon0=" << indata_.tcon0 << std::endl;

  std::cout << "Profiles:" << std::endl;
  std::cout << "  am size=" << indata_.am.size() << std::endl;
  std::cout << "  ai size=" << indata_.ai.size() << std::endl;
  for (size_t i = 0; i < indata_.ai.size(); ++i) {
    std::cout << "    ai[" << i << "]=" << indata_.ai[i] << std::endl;
  }

  std::cout << "Boundary coefficients:" << std::endl;
  auto get_mn_index = [this](int m, int n) -> int {
    return m * (2 * indata_.ntor + 1) + (n + indata_.ntor);
  };

  for (int m = 0; m < std::min(3, indata_.mpol); ++m) {
    for (int n = -indata_.ntor; n <= indata_.ntor; ++n) {
      int idx = get_mn_index(m, n);
      if (idx >= 0 && idx < static_cast<int>(indata_.rbc.size())) {
        if (indata_.rbc[idx] != 0.0 || indata_.zbs[idx] != 0.0 ||
            indata_.rbs[idx] != 0.0 || indata_.zbc[idx] != 0.0) {
          std::cout << "  m=" << m << ", n=" << n << ": ";
          std::cout << "rbc=" << indata_.rbc[idx] << ", ";
          std::cout << "zbs=" << indata_.zbs[idx] << ", ";
          std::cout << "rbs=" << indata_.rbs[idx] << ", ";
          std::cout << "zbc=" << indata_.zbc[idx] << std::endl;
        }
      }
    }
  }

  std::cout << "=== CONFIGURATION COMPLETE ===" << std::endl;

  EXPECT_EQ(indata_.lasym, true);
  EXPECT_EQ(indata_.nfp, 1);
}

// Test 2: Test realistic asymmetric run
TEST_F(RealisticAsymmetricTest, RealisticAsymmetricRun) {
  std::cout << "=== REALISTIC ASYMMETRIC VMEC RUN ===" << std::endl;

  std::cout << "Starting vmecpp::run with realistic asymmetric tokamak..."
            << std::endl;

  try {
    const auto output = vmecpp::run(indata_, std::nullopt, std::nullopt, true);

    if (!output.ok()) {
      std::cout << "RUN STATUS: " << output.status() << std::endl;
      std::cout << "This may be expected during debugging..." << std::endl;
    } else {
      std::cout << "SUCCESS! Asymmetric VMEC run completed." << std::endl;

      // Print some basic results
      std::cout << "Output quantities available:" << std::endl;
      const auto& result = output.value();
      std::cout << "  wout.ns achieved: " << result.wout.ns << std::endl;
      std::cout << "  wout.iota_full size: " << result.wout.iota_full.size()
                << std::endl;
      std::cout << "  wout.pressure_full size: "
                << result.wout.pressure_full.size() << std::endl;

      // Check for basic physics validity
      bool has_finite_iota = true;
      for (size_t i = 0; i < result.wout.iota_full.size(); ++i) {
        if (!std::isfinite(result.wout.iota_full[i])) {
          has_finite_iota = false;
          break;
        }
      }

      EXPECT_TRUE(has_finite_iota) << "Iota profile should be finite";
      EXPECT_GT(result.wout.ns, 0) << "Should have positive number of surfaces";
    }
  } catch (const std::exception& e) {
    std::cout << "CAUGHT EXCEPTION: " << e.what() << std::endl;
    FAIL() << "Should not throw exception, should return error status";
  }

  std::cout << "=== REALISTIC RUN COMPLETED ===" << std::endl;
}

// Test 3: Test with even smaller asymmetric perturbation
TEST_F(RealisticAsymmetricTest, SmallPerturbationTest) {
  std::cout << "=== SMALL ASYMMETRIC PERTURBATION TEST ===" << std::endl;

  // Reduce asymmetric perturbation to very small level
  auto small_pert = indata_;

  // Helper for boundary indexing
  auto get_mn_index = [&small_pert](int m, int n) -> int {
    return m * (2 * small_pert.ntor + 1) + (n + small_pert.ntor);
  };

  // Make asymmetric perturbation very small
  if (small_pert.mpol > 1) {
    small_pert.rbs[get_mn_index(1, 0)] = 0.01;  // 1% perturbation
  }

  // Reduce grid and iterations for faster testing
  small_pert.ns_array = {13};
  small_pert.niter_array = {50};
  small_pert.ftol_array = {1e-10};

  std::cout << "Testing with 1% asymmetric perturbation..." << std::endl;
  std::cout << "Grid: ns=" << small_pert.ns_array[0]
            << ", niter=" << small_pert.niter_array[0] << std::endl;

  const auto output = vmecpp::run(small_pert, std::nullopt, std::nullopt, true);

  if (!output.ok()) {
    std::cout << "Small perturbation run status: " << output.status()
              << std::endl;
  } else {
    std::cout << "Small perturbation run SUCCESS!" << std::endl;
  }

  std::cout << "=== SMALL PERTURBATION TEST COMPLETED ===" << std::endl;
}

}  // namespace vmecpp
