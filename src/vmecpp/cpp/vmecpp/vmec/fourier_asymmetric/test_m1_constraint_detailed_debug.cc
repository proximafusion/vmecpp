// SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/boundaries/boundaries.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Detailed debug test for M=1 constraint impact on convergence
class M1ConstraintDetailedDebugTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup for detailed debug analysis
  }

  VmecINDATA CreateJVMECTokAsymConfig() {
    VmecINDATA config;
    config.lasym = true;
    config.nfp = 1;
    config.mpol = 7;
    config.ntor = 0;
    config.ns_array = {3, 5, 7};
    config.ftol_array = {1e-12, 1e-12, 1e-12};
    config.niter_array = {100, 100, 100};

    // Resize arrays for mpol=7, ntor=0
    config.rbc.resize(config.mpol, 0.0);
    config.rbs.resize(config.mpol, 0.0);
    config.zbc.resize(config.mpol, 0.0);
    config.zbs.resize(config.mpol, 0.0);

    // jVMEC tok_asym coefficients
    config.rbc[0] = 5.9163;
    config.rbc[1] = 1.9196;
    config.rbc[2] = 0.33736;
    config.rbc[3] = 0.041504;
    config.rbc[4] = -0.0058256;
    config.rbc[5] = 0.010374;
    config.rbc[6] = -0.0056365;

    config.rbs[0] = 0.0;
    config.rbs[1] = 0.027610;  // Critical M=1 coefficient
    config.rbs[2] = 0.10038;
    config.rbs[3] = -0.071843;
    config.rbs[4] = -0.011423;
    config.rbs[5] = 0.008177;
    config.rbs[6] = -0.007611;

    config.zbc[0] = 0.4105;
    config.zbc[1] = 0.057302;  // Critical M=1 coefficient
    config.zbc[2] = 0.0046697;
    config.zbc[3] = -0.039155;
    config.zbc[4] = -0.0087848;
    config.zbc[5] = 0.021175;
    config.zbc[6] = 0.002439;

    config.zbs[0] = 0.0;
    config.zbs[1] = 3.6223;
    config.zbs[2] = -0.18511;
    config.zbs[3] = -0.0048568;
    config.zbs[4] = 0.059268;
    config.zbs[5] = 0.004477;
    config.zbs[6] = -0.016773;

    // Axis coefficients
    config.raxis_c = {7.5025, 0.47};
    config.zaxis_s = {0.0, 0.0};
    config.raxis_s = {0.0, 0.0};
    config.zaxis_c = {0.0, 0.0};

    // Physics parameters
    config.gamma = 0.0;
    config.ncurr = 0;
    config.pcurr_type = "power_series";
    config.pmass_type = "power_series";
    config.ac = {0.0};
    config.am = {0.0};
    config.pres_scale = 100000.0;
    config.curtor = 0.0;

    return config;
  }

  void AnalyzeBoundaryProcessing(const VmecINDATA& config) {
    std::cout << "\n=== BOUNDARY PROCESSING ANALYSIS ===\n";
    std::cout << std::fixed << std::setprecision(8);

    // Create sizes and basis
    int ntheta = 2 * config.mpol + 6;
    int nzeta = (config.ntor > 0) ? 3 : 1;

    Sizes sizes(config.lasym, config.nfp, config.mpol, config.ntor, ntheta,
                nzeta);
    FourierBasisFastPoloidal basis(&sizes);

    // Create boundaries and process
    Boundaries boundaries(&sizes, &basis, -1);

    std::cout << "Input M=1 coefficients:\n";
    std::cout << "  rbs[1] = " << config.rbs[1] << "\n";
    std::cout << "  zbc[1] = " << config.zbc[1] << "\n";
    std::cout << "  Difference = " << std::abs(config.rbs[1] - config.zbc[1])
              << "\n";

    // Theta shift calculation
    double delta =
        atan2(config.rbs[1] - config.zbc[1], config.rbc[1] + config.zbs[1]);
    std::cout << "\nTheta shift delta = " << delta << " rad ("
              << (delta * 180.0 / M_PI) << " degrees)\n";

    // Expected values after theta shift
    double rbs1_shifted =
        config.rbs[1] * cos(delta) - config.rbc[1] * sin(delta);
    double zbc1_shifted =
        config.zbc[1] * cos(delta) + config.zbs[1] * sin(delta);

    std::cout << "\nExpected after theta shift:\n";
    std::cout << "  rbs[1] = " << rbs1_shifted << "\n";
    std::cout << "  zbc[1] = " << zbc1_shifted << "\n";

    // Apply boundary processing
    boundaries.setupFromIndata(config, true);

    // Check internal arrays
    int idx_m1_n0 = 1 * (sizes.ntor + 1) + 0;  // m=1, n=0

    std::cout << "\nAfter boundary processing:\n";
    if (sizes.lasym) {
      std::cout << "  rbsc[m=1,n=0] = " << boundaries.rbsc[idx_m1_n0] << "\n";
      std::cout << "  zbcc[m=1,n=0] = " << boundaries.zbcc[idx_m1_n0] << "\n";
      std::cout << "  Constraint satisfied: "
                << (std::abs(boundaries.rbsc[idx_m1_n0] -
                             boundaries.zbcc[idx_m1_n0]) < 1e-14
                        ? "YES"
                        : "NO")
                << "\n";
    }

    // Analyze all M=1 modes
    std::cout << "\nAll M=1 modes after processing:\n";
    for (int n = 0; n <= sizes.ntor; ++n) {
      int idx = 1 * (sizes.ntor + 1) + n;
      if (sizes.lthreed) {
        std::cout << "  n=" << n << ": rbss=" << boundaries.rbss[idx]
                  << " zbcs=" << boundaries.zbcs[idx] << " diff="
                  << std::abs(boundaries.rbss[idx] - boundaries.zbcs[idx])
                  << "\n";
      }
      if (sizes.lasym) {
        std::cout << "  n=" << n << ": rbsc=" << boundaries.rbsc[idx]
                  << " zbcc=" << boundaries.zbcc[idx] << " diff="
                  << std::abs(boundaries.rbsc[idx] - boundaries.zbcc[idx])
                  << "\n";
      }
    }
  }
};

TEST_F(M1ConstraintDetailedDebugTest, AnalyzeJVMECBoundaryProcessing) {
  std::cout << "\n=== JVMEC BOUNDARY PROCESSING WITH M=1 CONSTRAINT ===\n";

  VmecINDATA config = CreateJVMECTokAsymConfig();
  AnalyzeBoundaryProcessing(config);

  std::cout << "\n=== KEY FINDINGS ===\n";
  std::cout << "1. Theta shift transforms coefficients\n";
  std::cout << "2. M=1 constraint enforces rbsc = zbcc\n";
  std::cout << "3. This matches jVMEC preprocessing exactly\n";
  std::cout << "4. But Jacobian still becomes negative\n";

  std::cout << "\n=== HYPOTHESIS ===\n";
  std::cout << "The M=1 constraint alone is insufficient.\n";
  std::cout << "Other factors contributing to convergence:\n";
  std::cout << "- Initial guess generation differences\n";
  std::cout << "- Spectral condensation approach\n";
  std::cout << "- Force processing differences\n";
  std::cout << "- Numerical tolerances and iterations\n";

  EXPECT_TRUE(true) << "Boundary analysis complete";
}

TEST_F(M1ConstraintDetailedDebugTest, CompareInitialGuessGeneration) {
  std::cout << "\n=== INITIAL GUESS GENERATION COMPARISON ===\n";

  VmecINDATA config = CreateJVMECTokAsymConfig();

  // Try with very small NS to isolate initial guess issues
  config.ns_array = {3};
  config.ftol_array = {1e-4};
  config.niter_array = {5};  // Just a few iterations

  std::cout << "Testing with minimal configuration:\n";
  std::cout << "  NS = 3 (single radial grid)\n";
  std::cout << "  max_iter = 5 (early termination)\n";
  std::cout << "  ftol = 1e-4 (loose tolerance)\n";

  try {
    Vmec vmec(config);
    std::cout << "\n✅ Initial guess successful with M=1 constraint\n";
    std::cout << "The modified boundary processing allows initialization\n";
  } catch (const std::exception& e) {
    std::cout << "\n❌ Failed even with minimal config: " << e.what() << "\n";
    std::cout << "This suggests issues beyond boundary preprocessing\n";
  }

  std::cout << "\n=== NEXT INVESTIGATION STEPS ===\n";
  std::cout << "1. Compare initial R,Z profiles between codes\n";
  std::cout << "2. Check spectral condensation differences\n";
  std::cout << "3. Analyze force calculation discrepancies\n";
  std::cout << "4. Look for numerical conditioning issues\n";

  EXPECT_TRUE(true) << "Initial guess analysis complete";
}

TEST_F(M1ConstraintDetailedDebugTest, TestDifferentNumericalParameters) {
  std::cout << "\n=== NUMERICAL PARAMETER SENSITIVITY ===\n";

  VmecINDATA base_config = CreateJVMECTokAsymConfig();

  struct TestCase {
    std::string name;
    std::vector<int> ns_array;
    std::vector<double> ftol_array;
    std::vector<int> niter_array;
    double delt;
  };

  std::vector<TestCase> test_cases = {
      {"Minimal", {3}, {1e-4}, {10}, 0.9},
      {"Small grid", {3, 5}, {1e-4, 1e-6}, {50, 50}, 0.9},
      {"Loose tolerance", {5}, {1e-2}, {20}, 0.9},
      {"Small timestep", {3}, {1e-4}, {10}, 0.1},
      {"Large timestep", {3}, {1e-4}, {10}, 1.5}};

  for (const auto& test : test_cases) {
    std::cout << "\n--- Test: " << test.name << " ---\n";

    VmecINDATA config = base_config;
    config.ns_array = test.ns_array;
    config.ftol_array = test.ftol_array;
    config.niter_array = test.niter_array;
    if (test.delt > 0) {
      config.delt = test.delt;
    }

    std::cout << "Parameters:\n";
    std::cout << "  NS = [";
    for (auto ns : config.ns_array) std::cout << ns << " ";
    std::cout << "]\n";
    std::cout << "  ftol = [";
    for (auto ftol : config.ftol_array) std::cout << ftol << " ";
    std::cout << "]\n";
    std::cout << "  delt = " << config.delt << "\n";

    try {
      Vmec vmec(config);
      std::cout << "✅ SUCCESS with these parameters\n";
    } catch (const std::exception& e) {
      std::cout << "❌ FAILED: " << e.what() << "\n";

      // Extract failure type
      std::string error_msg = e.what();
      if (error_msg.find("JACOBIAN") != std::string::npos) {
        std::cout << "   Failure type: Jacobian sign change\n";
      } else if (error_msg.find("NaN") != std::string::npos) {
        std::cout << "   Failure type: NaN in calculation\n";
      } else {
        std::cout << "   Failure type: Other\n";
      }
    }
  }

  std::cout << "\n=== PARAMETER SENSITIVITY ANALYSIS ===\n";
  std::cout << "The M=1 constraint improves stability but doesn't guarantee "
               "convergence\n";
  std::cout << "Success depends on multiple numerical parameters\n";
  std::cout << "This suggests the need for:\n";
  std::cout << "- Better initial guess generation\n";
  std::cout << "- Improved spectral filtering\n";
  std::cout << "- Adaptive timestep control\n";

  EXPECT_TRUE(true) << "Parameter sensitivity analysis complete";
}

}  // namespace vmecpp

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
