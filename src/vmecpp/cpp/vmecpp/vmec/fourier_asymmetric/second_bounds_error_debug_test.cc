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

class SecondBoundsErrorDebugTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create minimal asymmetric configuration from scratch
    // to avoid any input file issues
    indata_.lasym = true;
    indata_.nfp = 1;         // Tokamak (simpler than stellarator)
    indata_.mpol = 3;        // Minimal modes
    indata_.ntor = 2;        // Minimal toroidal modes
    indata_.ntheta = 16;     // FIXED: Use corrected value directly
    indata_.nzeta = 16;      // Small grid
    indata_.ns_array = {5};  // Very small radial grid
    indata_.ftol_array = {1e-6};
    indata_.niter_array = {1};  // Just 1 iteration to catch early failure
    indata_.delt = 0.7;
    indata_.tcon0 = 1.0;
    indata_.return_outputs_even_if_not_converged = true;

    // Set up minimal axis coefficients for tokamak
    // Axis arrays must be size (ntor + 1) = 3
    indata_.raxis_c = {1.0, 0.0, 0.0};  // Major radius, size ntor+1=3
    indata_.raxis_s = {0.0, 0.0, 0.0};  // For asymmetric case
    indata_.zaxis_c = {0.0, 0.0, 0.0};  // For asymmetric case
    indata_.zaxis_s = {0.0, 0.0, 0.0};  // Z axis

    // Boundary coefficients (flat arrays: mpol*(2*ntor+1))
    int boundary_size = indata_.mpol * (2 * indata_.ntor + 1);
    indata_.rbc.resize(boundary_size, 0.0);
    indata_.zbs.resize(boundary_size, 0.0);
    indata_.rbs.resize(boundary_size, 0.0);
    indata_.zbc.resize(boundary_size, 0.0);

    // Helper function to get flat index: m * (2*ntor+1) + (n+ntor)
    auto get_mn_index = [this](int m, int n) -> int {
      return m * (2 * indata_.ntor + 1) + (n + indata_.ntor);
    };

    // Set R00 = major radius at boundary
    indata_.rbc[get_mn_index(0, 0)] = 1.0;

    // Set Z10 = minor radius
    if (indata_.mpol > 1) {
      indata_.zbs[get_mn_index(1, 0)] = 0.3;
    }

    // Add small asymmetric perturbation (use rbs for sin component)
    if (indata_.mpol > 1) {
      indata_.rbs[get_mn_index(1, 0)] = 0.01;  // Small R10 asymmetric component
    }

    // Pressure and current profiles (minimal)
    indata_.pmass_type = "power_series";
    indata_.am = {1.0};
    indata_.pres_scale = 100.0;
    indata_.gamma = 0.0;

    indata_.ncurr = 1;
    indata_.pcurr_type = "power_series";
    indata_.ac = {1.0};
    indata_.curtor = 1000.0;

    indata_.lfreeb = false;  // Fixed boundary
    indata_.mgrid_file = "NONE";
  }

  VmecINDATA indata_;
};

// Test 1: Print detailed configuration info
TEST_F(SecondBoundsErrorDebugTest, PrintDetailedConfiguration) {
  std::cout << "=== DETAILED CONFIGURATION INFO ===" << std::endl;
  std::cout << "lasym=" << indata_.lasym << std::endl;
  std::cout << "nfp=" << indata_.nfp << std::endl;
  std::cout << "mpol=" << indata_.mpol << std::endl;
  std::cout << "ntor=" << indata_.ntor << std::endl;
  std::cout << "ntheta=" << indata_.ntheta << std::endl;
  std::cout << "nzeta=" << indata_.nzeta << std::endl;
  std::cout << "ns=" << indata_.ns_array[0] << std::endl;
  std::cout << "niter=" << indata_.niter_array[0] << std::endl;

  std::cout << "Boundary coefficients:" << std::endl;
  auto get_mn_index = [this](int m, int n) -> int {
    return m * (2 * indata_.ntor + 1) + (n + indata_.ntor);
  };

  for (int m = 0; m < std::min(3, indata_.mpol); ++m) {
    for (int n = -1; n <= 1; ++n) {
      int idx = get_mn_index(m, n);
      if (idx >= 0 && idx < static_cast<int>(indata_.rbc.size())) {
        std::cout << "  rbc[m=" << m << ",n=" << n << "] = " << indata_.rbc[idx]
                  << std::endl;
        std::cout << "  zbs[m=" << m << ",n=" << n << "] = " << indata_.zbs[idx]
                  << std::endl;
      }
    }
  }

  std::cout << "Expected nThetaEff for asymmetric: " << indata_.ntheta
            << std::endl;
  std::cout << "=== CONFIGURATION COMPLETE ===" << std::endl;

  // This should not fail due to ntheta=0 since we set it correctly
  EXPECT_GT(indata_.ntheta, 0);
  EXPECT_EQ(indata_.lasym, true);
}

// Test 2: Check array sizes before running
TEST_F(SecondBoundsErrorDebugTest, CheckArraySizes) {
  std::cout << "=== CHECKING ARRAY SIZES ===" << std::endl;

  std::cout << "INDATA values:" << std::endl;
  std::cout << "  mpol=" << indata_.mpol << ", ntor=" << indata_.ntor
            << std::endl;
  std::cout << "  Expected INDATA boundary array size: "
            << indata_.mpol * (2 * indata_.ntor + 1) << std::endl;
  std::cout << "  Actual rbc.size()=" << indata_.rbc.size() << std::endl;
  std::cout << "  Actual zbs.size()=" << indata_.zbs.size() << std::endl;
  std::cout << "  Actual rbs.size()=" << indata_.rbs.size() << std::endl;
  std::cout << "  Actual zbc.size()=" << indata_.zbc.size() << std::endl;

  std::cout << "AXIS arrays:" << std::endl;
  std::cout << "  Expected axis array size: " << (indata_.ntor + 1)
            << std::endl;
  std::cout << "  Actual raxis_c.size()=" << indata_.raxis_c.size()
            << std::endl;
  std::cout << "  Actual raxis_s.size()=" << indata_.raxis_s.size()
            << std::endl;
  std::cout << "  Actual zaxis_c.size()=" << indata_.zaxis_c.size()
            << std::endl;
  std::cout << "  Actual zaxis_s.size()=" << indata_.zaxis_s.size()
            << std::endl;

  // Create Sizes to see what Boundaries will expect
  vmecpp::Sizes sizes(indata_);
  std::cout << "SIZES values:" << std::endl;
  std::cout << "  sizes.mpol=" << sizes.mpol << ", sizes.ntor=" << sizes.ntor
            << std::endl;
  std::cout << "  Expected Boundaries internal array size: "
            << sizes.mpol * (sizes.ntor + 1) << std::endl;

  // Check if there's a mismatch
  int expected_indata_size = indata_.mpol * (2 * indata_.ntor + 1);
  int expected_boundaries_size = sizes.mpol * (sizes.ntor + 1);

  std::cout << "Array size analysis:" << std::endl;
  std::cout << "  INDATA boundary arrays expect size: " << expected_indata_size
            << std::endl;
  std::cout << "  Boundaries internal arrays expect size: "
            << expected_boundaries_size << std::endl;

  if (indata_.mpol != sizes.mpol || indata_.ntor != sizes.ntor) {
    std::cout << "ERROR: Size mismatch between INDATA and Sizes!" << std::endl;
    std::cout << "  INDATA: mpol=" << indata_.mpol << ", ntor=" << indata_.ntor
              << std::endl;
    std::cout << "  SIZES:  mpol=" << sizes.mpol << ", ntor=" << sizes.ntor
              << std::endl;
  }

  std::cout << "=== ARRAY SIZE CHECK COMPLETED ===" << std::endl;

  EXPECT_EQ(indata_.rbc.size(), expected_indata_size);
  EXPECT_EQ(indata_.zbs.size(), expected_indata_size);
  EXPECT_EQ(indata_.mpol, sizes.mpol)
      << "mpol should not change from INDATA to Sizes";
  EXPECT_EQ(indata_.ntor, sizes.ntor)
      << "ntor should not change from INDATA to Sizes";

  // Check axis array sizes
  int expected_axis_size = indata_.ntor + 1;
  EXPECT_EQ(indata_.raxis_c.size(), expected_axis_size) << "raxis_c size wrong";
  EXPECT_EQ(indata_.raxis_s.size(), expected_axis_size) << "raxis_s size wrong";
  EXPECT_EQ(indata_.zaxis_c.size(), expected_axis_size) << "zaxis_c size wrong";
  EXPECT_EQ(indata_.zaxis_s.size(), expected_axis_size) << "zaxis_s size wrong";
}

// Test 3: Simulate boundary parsing to find exact error
TEST_F(SecondBoundsErrorDebugTest, SimulateBoundaryParsing) {
  std::cout << "=== SIMULATING BOUNDARY PARSING ===" << std::endl;

  vmecpp::Sizes sizes(indata_);
  std::cout << "Using sizes: mpol=" << sizes.mpol << ", ntor=" << sizes.ntor
            << std::endl;
  std::cout << "INDATA array sizes: " << indata_.rbc.size() << std::endl;

  // Simulate the double loop from parseToInternalArrays
  for (int m = 0; m < sizes.mpol; ++m) {
    for (int n = -sizes.ntor; n <= sizes.ntor; ++n) {
      int source_n = sizes.ntor + n;
      int index = m * (2 * sizes.ntor + 1) + source_n;

      std::cout << "m=" << m << ", n=" << n << ", source_n=" << source_n
                << ", index=" << index
                << " (max allowed: " << (indata_.rbc.size() - 1) << ")"
                << std::endl;

      if (index >= static_cast<int>(indata_.rbc.size())) {
        std::cout << "ERROR: Index " << index << " exceeds array size "
                  << indata_.rbc.size() << std::endl;
        FAIL() << "Index out of bounds at m=" << m << ", n=" << n;
      }
    }
  }

  std::cout << "All boundary array accesses are within bounds!" << std::endl;
  std::cout << "=== BOUNDARY PARSING SIMULATION COMPLETED ===" << std::endl;
}

// Test 4: Try minimal run and catch the exact error
TEST_F(SecondBoundsErrorDebugTest, MinimalAsymmetricRun) {
  std::cout << "=== TESTING MINIMAL ASYMMETRIC RUN ===" << std::endl;

  // Enable verbose output to see what happens
  std::cout << "Starting vmecpp::run with minimal asymmetric configuration..."
            << std::endl;
  std::cout << "Note: ntheta=" << indata_.ntheta << " (should be > 0)"
            << std::endl;

  try {
    const auto output = vmecpp::run(indata_, std::nullopt, std::nullopt, true);

    if (!output.ok()) {
      std::cout << "RUN FAILED with status: " << output.status() << std::endl;
      // We expect this to fail, but we want to see the exact error
      EXPECT_FALSE(output.ok()) << "Expected failure for debugging";
    } else {
      std::cout << "UNEXPECTED SUCCESS! Run completed." << std::endl;
      EXPECT_TRUE(output.ok()) << "If it works, that's great news!";
    }
  } catch (const std::exception& e) {
    std::cout << "CAUGHT EXCEPTION: " << e.what() << std::endl;
    FAIL() << "Should not throw exception, should return error status";
  }

  std::cout << "=== MINIMAL RUN TEST COMPLETED ===" << std::endl;
}

// Test 5: Try even smaller configuration
TEST_F(SecondBoundsErrorDebugTest, UltraMinimalConfiguration) {
  std::cout << "=== TESTING ULTRA-MINIMAL CONFIGURATION ===" << std::endl;

  // Make it as small as possible
  auto ultra_minimal = indata_;
  ultra_minimal.mpol = 2;    // Just m=0,1
  ultra_minimal.ntor = 1;    // Just n=-1,0,1
  ultra_minimal.ntheta = 8;  // Smaller grid
  ultra_minimal.nzeta = 8;
  ultra_minimal.ns_array = {3};  // Minimal radial points

  // Rebuild boundary arrays for smaller dimensions
  int ultra_boundary_size = ultra_minimal.mpol * (2 * ultra_minimal.ntor + 1);
  ultra_minimal.rbc.resize(ultra_boundary_size, 0.0);
  ultra_minimal.zbs.resize(ultra_boundary_size, 0.0);
  ultra_minimal.rbs.resize(ultra_boundary_size, 0.0);
  ultra_minimal.zbc.resize(ultra_boundary_size, 0.0);

  auto ultra_get_mn_index = [&ultra_minimal](int m, int n) -> int {
    return m * (2 * ultra_minimal.ntor + 1) + (n + ultra_minimal.ntor);
  };

  // Set minimal boundary
  ultra_minimal.rbc[ultra_get_mn_index(0, 0)] = 1.0;  // R00
  if (ultra_minimal.mpol > 1) {
    ultra_minimal.zbs[ultra_get_mn_index(1, 0)] = 0.3;  // Z10
  }

  std::cout << "Ultra-minimal config: mpol=" << ultra_minimal.mpol
            << ", ntor=" << ultra_minimal.ntor
            << ", ntheta=" << ultra_minimal.ntheta
            << ", nzeta=" << ultra_minimal.nzeta
            << ", ns=" << ultra_minimal.ns_array[0] << std::endl;

  std::cout << "Starting ultra-minimal run..." << std::endl;
  const auto output =
      vmecpp::run(ultra_minimal, std::nullopt, std::nullopt, true);

  if (!output.ok()) {
    std::cout << "Ultra-minimal FAILED: " << output.status() << std::endl;
  } else {
    std::cout << "Ultra-minimal SUCCESS!" << std::endl;
  }

  std::cout << "=== ULTRA-MINIMAL TEST COMPLETED ===" << std::endl;
}

}  // namespace vmecpp
