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
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using nlohmann::json;
using vmecpp::VmecINDATA;

namespace vmecpp {

class UpDownAsymmetricTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create the up-down asymmetric tokamak configuration
    // Based on educational_VMEC/up_down_asymmetric_tokamak.json

    // This configuration has:
    // - LASYM = T (asymmetric mode)
    // - NFP = 1 (tokamak)
    // - MPOL = 5, NTOR = 0 (axisymmetric)
    // - RBS(0,1) = 0.6, RBS(0,2) = 0.12 (up-down asymmetric components)

    json config = {{"lasym", true},
                   {"nfp", 1},
                   {"mpol", 5},
                   {"ntor", 0},
                   {"ntheta", 0},
                   {"nzeta", 0},
                   {"ns_array", {17}},
                   {"ftol_array", {1e-11}},
                   {"niter_array", {2000}},
                   {"delt", 0.9},
                   {"tcon0", 1.0},
                   {"aphi", {1.0}},
                   {"phiedge", 6.0},
                   {"nstep", 200},
                   {"pmass_type", "power_series"},
                   {"am", {0.0}},
                   {"pres_scale", 1.0},
                   {"gamma", 0.0},
                   {"spres_ped", 1.0},
                   {"ncurr", 0},
                   {"piota_type", "power_series"},
                   {"ai", {0.9, -0.65}},
                   {"lfreeb", false},
                   {"mgrid_file", "NONE"},
                   {"nvacskip", 3},
                   {"lforbal", false},
                   {"raxis_c", {0.0}},
                   {"zaxis_s", {0.0}},
                   {"raxis_s", {0.0}},
                   {"zaxis_c", {0.0}},
                   {"rbc",
                    {
                        {{"n", 0}, {"m", 0}, {"value", 6.0}},  // Major radius
                        {{"n", 0}, {"m", 1}, {"value", 0.6}}   // Minor radius
                    }},
                   {"zbs",
                    {
                        {{"n", 0},
                         {"m", 1},
                         {"value", 0.6}}  // Standard tokamak Z component
                    }},
                   {"rbs",
                    {
                        {{"n", 0},
                         {"m", 1},
                         {"value", 0.6}},  // Up-down asymmetric R component
                        {{"n", 0},
                         {"m", 2},
                         {"value", 0.12}}  // Additional asymmetric component
                    }}};

    // Parse the configuration
    auto indata_result = VmecINDATA::FromJson(config.dump());
    ASSERT_TRUE(indata_result.ok())
        << "Failed to parse up-down asymmetric tokamak config";

    indata_asymmetric_ = *indata_result;

    // Create symmetric baseline (same config but with lasym=false)
    auto config_symmetric = config;
    config_symmetric["lasym"] = false;

    auto indata_symmetric_result =
        VmecINDATA::FromJson(config_symmetric.dump());
    ASSERT_TRUE(indata_symmetric_result.ok())
        << "Failed to parse symmetric baseline config";

    indata_symmetric_ = *indata_symmetric_result;

    // Reduce iterations for testing
    indata_asymmetric_.niter_array[0] = 100;
    indata_symmetric_.niter_array[0] = 100;

    // Allow returning outputs even if not fully converged
    indata_asymmetric_.return_outputs_even_if_not_converged = true;
    indata_symmetric_.return_outputs_even_if_not_converged = true;
  }

  VmecINDATA indata_symmetric_;
  VmecINDATA indata_asymmetric_;
};

TEST_F(UpDownAsymmetricTest, TestConfigurationSetup) {
  // Test that the configuration was set up correctly
  std::cout << "Testing up-down asymmetric tokamak configuration setup..."
            << std::endl;

  // Check asymmetric configuration
  EXPECT_TRUE(indata_asymmetric_.lasym);
  EXPECT_EQ(indata_asymmetric_.nfp, 1);
  EXPECT_EQ(indata_asymmetric_.mpol, 5);
  EXPECT_EQ(indata_asymmetric_.ntor, 0);
  EXPECT_EQ(indata_asymmetric_.phiedge, 6.0);

  // Check symmetric baseline
  EXPECT_FALSE(indata_symmetric_.lasym);
  EXPECT_EQ(indata_symmetric_.nfp, 1);
  EXPECT_EQ(indata_symmetric_.mpol, 5);
  EXPECT_EQ(indata_symmetric_.ntor, 0);
  EXPECT_EQ(indata_symmetric_.phiedge, 6.0);

  // Check boundary components
  EXPECT_GT(indata_asymmetric_.rbc.size(), 0);
  EXPECT_GT(indata_asymmetric_.zbs.size(), 0);
  EXPECT_GT(indata_asymmetric_.rbs.size(),
            0);  // This is the key asymmetric component

  std::cout << "Configuration setup test passed" << std::endl;
}

TEST_F(UpDownAsymmetricTest, TestSymmetricBaseline) {
  // Test symmetric baseline to ensure the configuration works
  std::cout << "Testing symmetric baseline (lasym=false)..." << std::endl;

  const auto output = vmecpp::run(indata_symmetric_);
  EXPECT_TRUE(output.ok()) << "Symmetric baseline failed: " << output.status();

  if (output.ok()) {
    const auto& wout = output->wout;
    EXPECT_GT(wout.ns, 0);
    EXPECT_GT(wout.volume_p, 0.0);
    EXPECT_EQ(wout.lasym, false);

    std::cout << "Symmetric baseline - Volume: " << wout.volume_p << std::endl;
    std::cout << "Symmetric baseline - Force residuals: R=" << wout.fsqr
              << ", Z=" << wout.fsqz << ", L=" << wout.fsql << std::endl;
  }

  std::cout << "Symmetric baseline test completed" << std::endl;
}

TEST_F(UpDownAsymmetricTest, TestAsymmetricMode) {
  // Test the asymmetric mode with our implementation
  std::cout << "Testing up-down asymmetric mode (lasym=true)..." << std::endl;

  const auto output = vmecpp::run(indata_asymmetric_);
  EXPECT_TRUE(output.ok()) << "Asymmetric mode failed: " << output.status();

  if (output.ok()) {
    const auto& wout = output->wout;
    EXPECT_GT(wout.ns, 0);
    EXPECT_GT(wout.volume_p, 0.0);
    EXPECT_EQ(wout.lasym, true);

    std::cout << "Asymmetric mode - Volume: " << wout.volume_p << std::endl;
    std::cout << "Asymmetric mode - Force residuals: R=" << wout.fsqr
              << ", Z=" << wout.fsqz << ", L=" << wout.fsql << std::endl;

    // Check that force residuals are reasonable
    EXPECT_LT(wout.fsqr, 1e-3) << "Radial force residual too large";
    EXPECT_LT(wout.fsqz, 1e-3) << "Vertical force residual too large";
    EXPECT_LT(wout.fsql, 1e-3) << "Lambda force residual too large";
  }

  std::cout << "Asymmetric mode test completed" << std::endl;
}

TEST_F(UpDownAsymmetricTest, CompareSymmetricVsAsymmetric) {
  // Compare symmetric and asymmetric results
  std::cout << "Comparing symmetric vs asymmetric results..." << std::endl;

  // Run both cases
  const auto output_sym = vmecpp::run(indata_symmetric_);
  const auto output_asym = vmecpp::run(indata_asymmetric_);

  ASSERT_TRUE(output_sym.ok()) << "Symmetric run failed";
  ASSERT_TRUE(output_asym.ok()) << "Asymmetric run failed";

  // Compare key quantities
  const auto& wout_sym = output_sym->wout;
  const auto& wout_asym = output_asym->wout;

  std::cout << "Symmetric  - Volume: " << wout_sym.volume_p << std::endl;
  std::cout << "Asymmetric - Volume: " << wout_asym.volume_p << std::endl;

  // Volume should be similar but not identical due to asymmetric components
  double volume_diff = std::abs(wout_sym.volume_p - wout_asym.volume_p);
  double volume_rel_diff = volume_diff / std::abs(wout_sym.volume_p);

  std::cout << "Volume relative difference: " << volume_rel_diff << std::endl;

  // The asymmetric case should have some difference but not too large
  EXPECT_LT(volume_rel_diff, 0.2)
      << "Volume difference too large between symmetric and asymmetric";
  EXPECT_GT(volume_rel_diff, 0.001)
      << "Volume difference too small - asymmetric components may not be "
         "working";

  std::cout << "Symmetric vs asymmetric comparison completed" << std::endl;
}

TEST_F(UpDownAsymmetricTest, TestAsymmetricBoundaryComponents) {
  // Test that asymmetric boundary components are handled correctly
  std::cout << "Testing asymmetric boundary components..." << std::endl;

  // This test focuses on the RBS components that create up-down asymmetry
  EXPECT_GT(indata_asymmetric_.rbs.size(), 0);

  // RBS array has layout: index = m * (2*ntor + 1) + (n + ntor)
  // For tokamak (ntor=0): index = m * 1 + (n + 0) = m (since n=0)
  int ntor = indata_asymmetric_.ntor;
  int mpol = indata_asymmetric_.mpol;

  // Check RBS(0,1) component (m=1, n=0)
  int idx_01 = 1 * (2 * ntor + 1) + (0 + ntor);  // m=1, n=0
  if (idx_01 < indata_asymmetric_.rbs.size()) {
    double rbs_01 = indata_asymmetric_.rbs[idx_01];
    std::cout << "RBS(0,1) = " << rbs_01 << std::endl;
    EXPECT_NEAR(rbs_01, 0.6, 1e-6);
  }

  // Check RBS(0,2) component (m=2, n=0)
  int idx_02 = 2 * (2 * ntor + 1) + (0 + ntor);  // m=2, n=0
  if (idx_02 < indata_asymmetric_.rbs.size()) {
    double rbs_02 = indata_asymmetric_.rbs[idx_02];
    std::cout << "RBS(0,2) = " << rbs_02 << std::endl;
    EXPECT_NEAR(rbs_02, 0.12, 1e-6);
  }

  std::cout << "Asymmetric boundary components test completed" << std::endl;
}

TEST_F(UpDownAsymmetricTest, TestAxiSymmetricConstraints) {
  // Test that axisymmetric constraints are handled correctly
  std::cout << "Testing axisymmetric constraints (NTOR=0)..." << std::endl;

  // For a tokamak (NTOR=0), even with asymmetric mode, we should only have n=0
  // modes
  EXPECT_EQ(indata_asymmetric_.ntor, 0);

  // Run the asymmetric case
  const auto output = vmecpp::run(indata_asymmetric_);
  ASSERT_TRUE(output.ok()) << "Asymmetric run failed";

  // The key test is that the code runs without errors
  // The asymmetric transforms should handle the n=0 case correctly
  const auto& wout = output->wout;
  EXPECT_EQ(wout.ntor, 0);
  EXPECT_EQ(wout.nfp, 1);

  std::cout << "Axisymmetric constraints test completed" << std::endl;
}

}  // namespace vmecpp
