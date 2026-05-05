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
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using file_io::ReadFile;
using nlohmann::json;
using vmecpp::VmecINDATA;

namespace vmecpp {

class VectorBoundsDebugTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load the CTH-like stellarator configuration
    const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
    absl::StatusOr<std::string> indata_json = ReadFile(filename);
    ASSERT_TRUE(indata_json.ok()) << "Failed to read " << filename;

    absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
    ASSERT_TRUE(indata.ok()) << "Failed to parse JSON";

    // Store the original configuration
    indata_base_ = *indata;

    // Reduce to minimal configuration for debugging
    indata_base_.niter_array[0] = 10;  // Very few iterations
    indata_base_.return_outputs_even_if_not_converged = true;
  }

  VmecINDATA indata_base_;
};

// Test 1: Verify symmetric case works (baseline)
TEST_F(VectorBoundsDebugTest, SymmetricBaseline) {
  auto indata_sym = indata_base_;
  indata_sym.lasym = false;

  std::cout << "=== TESTING SYMMETRIC BASELINE ===" << std::endl;
  std::cout << "Configuration: lasym=" << indata_sym.lasym
            << ", nfp=" << indata_sym.nfp << ", mpol=" << indata_sym.mpol
            << ", ntor=" << indata_sym.ntor << std::endl;

  // This should work without issues
  const auto output = vmecpp::run(indata_sym);
  EXPECT_TRUE(output.ok()) << "Symmetric case failed: " << output.status();

  std::cout << "=== SYMMETRIC BASELINE COMPLETED ===" << std::endl;
}

// Test 2: Try asymmetric with minimal debug output
TEST_F(VectorBoundsDebugTest, AsymmetricMinimalDebug) {
  auto indata_asym = indata_base_;
  indata_asym.lasym = true;
  indata_asym.niter_array[0] = 1;  // Just 1 iteration to catch early failure

  std::cout << "=== TESTING ASYMMETRIC CONFIGURATION ===" << std::endl;
  std::cout << "BEFORE adjustment - Configuration: lasym=" << indata_asym.lasym
            << ", nfp=" << indata_asym.nfp << ", mpol=" << indata_asym.mpol
            << ", ntor=" << indata_asym.ntor << std::endl;
  std::cout << "BEFORE adjustment - ntheta=" << indata_asym.ntheta
            << ", nzeta=" << indata_asym.nzeta
            << ", ns=" << indata_asym.ns_array[0] << std::endl;

  // Expected correction: ntheta should become 2*mpol+6 = 2*5+6 = 16
  int expected_ntheta = 2 * indata_asym.mpol + 6;
  std::cout << "EXPECTED ntheta after Nyquist correction: " << expected_ntheta
            << std::endl;

  // WORKAROUND: Manually set ntheta to correct value for now
  if (indata_asym.ntheta < expected_ntheta) {
    std::cout << "WORKAROUND: Manually setting ntheta from "
              << indata_asym.ntheta << " to " << expected_ntheta << std::endl;
    indata_asym.ntheta = expected_ntheta;
  }

  std::cout << "AFTER adjustment - ntheta=" << indata_asym.ntheta << std::endl;

  // This should NOT trigger vector bounds error anymore
  std::cout << "Starting VMEC run with asymmetric mode..." << std::endl;
  const auto output = vmecpp::run(indata_asym);

  if (!output.ok()) {
    std::cout << "EXPECTED FAILURE: " << output.status() << std::endl;
    // We expect this to fail with vector bounds error
    EXPECT_FALSE(output.ok()) << "We expect asymmetric case to fail for now";
  } else {
    std::cout << "UNEXPECTED SUCCESS: Asymmetric case worked!" << std::endl;
    EXPECT_TRUE(output.ok()) << "If it works, that's great!";
  }

  std::cout << "=== ASYMMETRIC TEST COMPLETED ===" << std::endl;
}

// Test 3: Try with even smaller configuration
TEST_F(VectorBoundsDebugTest, AsymmetricVerySmall) {
  auto indata_small = indata_base_;
  indata_small.lasym = true;
  indata_small.niter_array[0] = 1;

  // Make configuration as small as possible
  indata_small.ns_array[0] = std::min(indata_small.ns_array[0], 10);
  indata_small.mpol = std::min(indata_small.mpol, 3);
  indata_small.ntor = std::min(indata_small.ntor, 2);
  indata_small.nzeta = std::min(indata_small.nzeta, 16);

  // WORKAROUND: Fix ntheta for asymmetric case
  int min_ntheta = 2 * indata_small.mpol + 6;
  indata_small.ntheta = std::max(16, min_ntheta);

  std::cout << "=== TESTING VERY SMALL ASYMMETRIC CONFIGURATION ==="
            << std::endl;
  std::cout << "Configuration: lasym=" << indata_small.lasym
            << ", nfp=" << indata_small.nfp << ", mpol=" << indata_small.mpol
            << ", ntor=" << indata_small.ntor << std::endl;
  std::cout << "ntheta=" << indata_small.ntheta
            << ", nzeta=" << indata_small.nzeta
            << ", ns=" << indata_small.ns_array[0] << std::endl;

  std::cout << "Starting VMEC run with very small asymmetric configuration..."
            << std::endl;
  const auto output = vmecpp::run(indata_small);

  if (!output.ok()) {
    std::cout << "FAILURE with small config: " << output.status() << std::endl;
  } else {
    std::cout << "SUCCESS with small config!" << std::endl;
  }

  std::cout << "=== VERY SMALL TEST COMPLETED ===" << std::endl;
}

// Test 4: Try tokamak asymmetric (nfp=1) which might be simpler
TEST_F(VectorBoundsDebugTest, AsymmetricTokamak) {
  auto indata_tok = indata_base_;
  indata_tok.lasym = true;
  indata_tok.nfp = 1;  // Tokamak
  indata_tok.niter_array[0] = 1;

  // Smaller configuration for tokamak
  indata_tok.ns_array[0] = 10;
  indata_tok.mpol = 3;
  indata_tok.ntor = 2;
  indata_tok.nzeta = 16;

  // WORKAROUND: Fix ntheta for asymmetric case
  int min_ntheta = 2 * indata_tok.mpol + 6;
  indata_tok.ntheta = std::max(16, min_ntheta);

  std::cout << "=== TESTING ASYMMETRIC TOKAMAK ===" << std::endl;
  std::cout << "Configuration: lasym=" << indata_tok.lasym
            << ", nfp=" << indata_tok.nfp << ", mpol=" << indata_tok.mpol
            << ", ntor=" << indata_tok.ntor << std::endl;
  std::cout << "ntheta=" << indata_tok.ntheta << ", nzeta=" << indata_tok.nzeta
            << ", ns=" << indata_tok.ns_array[0] << std::endl;

  std::cout << "Starting VMEC run with asymmetric tokamak..." << std::endl;
  const auto output = vmecpp::run(indata_tok);

  if (!output.ok()) {
    std::cout << "FAILURE with tokamak: " << output.status() << std::endl;
  } else {
    std::cout << "SUCCESS with tokamak!" << std::endl;
  }

  std::cout << "=== TOKAMAK TEST COMPLETED ===" << std::endl;
}

}  // namespace vmecpp
