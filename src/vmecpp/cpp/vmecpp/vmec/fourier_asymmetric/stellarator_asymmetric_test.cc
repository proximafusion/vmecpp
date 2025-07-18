// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <fstream>
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

class StellaratorAsymmetricTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load the CTH-like stellarator configuration
    const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
    absl::StatusOr<std::string> indata_json = ReadFile(filename);
    ASSERT_TRUE(indata_json.ok()) << "Failed to read " << filename;

    absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
    ASSERT_TRUE(indata.ok()) << "Failed to parse JSON";

    // Store the original symmetric configuration
    indata_symmetric_ = *indata;

    // Create asymmetric configuration
    indata_asymmetric_ = *indata;
    indata_asymmetric_.lasym = true;  // Enable asymmetric mode

    // Reduce iterations for testing
    indata_symmetric_.niter_array[0] = 100;
    indata_asymmetric_.niter_array[0] = 100;

    // Allow returning outputs even if not fully converged
    indata_symmetric_.return_outputs_even_if_not_converged = true;
    indata_asymmetric_.return_outputs_even_if_not_converged = true;
  }

  VmecINDATA indata_symmetric_;
  VmecINDATA indata_asymmetric_;
};

TEST_F(StellaratorAsymmetricTest, TestSymmetricBaseline) {
  // Test that the symmetric case still works properly
  std::cout << "Testing symmetric baseline configuration..." << std::endl;

  // This should work without errors
  const auto output = vmecpp::run(indata_symmetric_);
  EXPECT_TRUE(output.ok()) << "Symmetric VMEC run failed: " << output.status();

  // Check basic properties
  EXPECT_FALSE(indata_symmetric_.lasym);
  EXPECT_EQ(indata_symmetric_.nfp, 5);
  EXPECT_EQ(indata_symmetric_.mpol, 5);
  EXPECT_EQ(indata_symmetric_.ntor, 4);

  // Check that we got valid output
  if (output.ok()) {
    const auto& wout = output->wout;
    EXPECT_GT(wout.ns, 0);
    EXPECT_GT(wout.volume_p, 0.0);
  }

  std::cout << "Symmetric baseline test completed successfully" << std::endl;
}

TEST_F(StellaratorAsymmetricTest, TestAsymmetricConfiguration) {
  // Test the asymmetric configuration
  std::cout << "Testing asymmetric stellarator configuration..." << std::endl;

  // This should work with the new asymmetric transforms
  const auto output = vmecpp::run(indata_asymmetric_);
  EXPECT_TRUE(output.ok()) << "Asymmetric VMEC run failed: " << output.status();

  // Check that asymmetric mode is enabled
  EXPECT_TRUE(indata_asymmetric_.lasym);
  EXPECT_EQ(indata_asymmetric_.nfp, 5);
  EXPECT_EQ(indata_asymmetric_.mpol, 5);
  EXPECT_EQ(indata_asymmetric_.ntor, 4);

  // Check that we got valid output
  if (output.ok()) {
    const auto& wout = output->wout;
    EXPECT_GT(wout.ns, 0);
    EXPECT_GT(wout.volume_p, 0.0);
  }

  std::cout << "Asymmetric configuration test completed successfully"
            << std::endl;
}

TEST_F(StellaratorAsymmetricTest, CompareSymmetricVsAsymmetric) {
  // Compare results between symmetric and asymmetric configurations
  std::cout << "Comparing symmetric vs asymmetric configurations..."
            << std::endl;

  // Run symmetric case
  const auto output_sym = vmecpp::run(indata_symmetric_);
  ASSERT_TRUE(output_sym.ok()) << "Symmetric VMEC run failed";

  // Run asymmetric case
  const auto output_asym = vmecpp::run(indata_asymmetric_);
  ASSERT_TRUE(output_asym.ok()) << "Asymmetric VMEC run failed";

  // Compare basic quantities
  // Note: Since we started from the same initial guess, the asymmetric case
  // should have similar overall properties but with additional degrees of
  // freedom

  std::cout << "Symmetric  - Volume: " << output_sym->wout.volume_p
            << std::endl;
  std::cout << "Asymmetric - Volume: " << output_asym->wout.volume_p
            << std::endl;

  // Volume should be similar (within reasonable tolerance)
  EXPECT_NEAR(output_sym->wout.volume_p, output_asym->wout.volume_p,
              0.1 * std::abs(output_sym->wout.volume_p))
      << "Volume difference between symmetric and asymmetric too large";

  // Check that force residuals are reasonable
  EXPECT_LT(output_asym->wout.fsqr, 1e-3) << "Radial force residual too large";
  EXPECT_LT(output_asym->wout.fsqz, 1e-3)
      << "Vertical force residual too large";
  EXPECT_LT(output_asym->wout.fsql, 1e-3) << "Lambda force residual too large";

  std::cout << "Symmetric vs asymmetric comparison completed" << std::endl;
}

TEST_F(StellaratorAsymmetricTest, TestBasicAsymmetricFunctionality) {
  // Test basic asymmetric functionality
  std::cout << "Testing basic asymmetric functionality..." << std::endl;

  // Just test that the asymmetric run completes without crashing
  const auto output = vmecpp::run(indata_asymmetric_);

  // We expect this to work, even if it doesn't converge perfectly
  EXPECT_TRUE(output.ok()) << "Asymmetric VMEC run failed: " << output.status();

  if (output.ok()) {
    std::cout << "Asymmetric run completed successfully" << std::endl;
    std::cout << "Volume: " << output->wout.volume_p << std::endl;
    std::cout << "Force residuals - R: " << output->wout.fsqr
              << ", Z: " << output->wout.fsqz << ", L: " << output->wout.fsql
              << std::endl;
  }

  std::cout << "Basic asymmetric functionality test completed" << std::endl;
}

}  // namespace vmecpp
