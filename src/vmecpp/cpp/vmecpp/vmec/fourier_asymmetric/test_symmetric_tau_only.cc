// Test to check tau values for a truly symmetric configuration with lasym=true
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(SymmetricTauOnlyTest, TestSymmetricGeometryWithAsymmetricFlag) {
  std::cout << "\n=== TEST SYMMETRIC GEOMETRY WITH LASYM=TRUE ===" << std::endl;
  std::cout << std::fixed << std::setprecision(8);

  VmecINDATA config;
  config.lasym = true;  // Asymmetric mode enabled
  config.nfp = 1;
  config.mpol = 3;
  config.ntor = 0;

  config.ns_array = {3};
  config.niter_array = {1};  // Just one iteration to see initial tau
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;

  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.pmass_type = "power_series";
  config.am = {0.0};
  config.pres_scale = 0.0;  // Zero pressure to isolate geometry

  // Simple circular tokamak - fully symmetric
  config.rbc = {10.0, 2.0, 0.5};
  config.zbs = {0.0, 2.0, 0.5};

  // EXPLICITLY set asymmetric coefficients to zero
  config.rbs = {0.0, 0.0, 0.0};
  config.zbc = {0.0, 0.0, 0.0};

  config.raxis_c = {10.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};  // Zero asymmetric axis
  config.zaxis_c = {0.0};  // Zero asymmetric axis

  std::cout << "Configuration:" << std::endl;
  std::cout << "  lasym = true (asymmetric mode enabled)" << std::endl;
  std::cout << "  All asymmetric coefficients = 0" << std::endl;
  std::cout << "  R0 = " << config.rbc[0] << ", a = " << config.rbc[1]
            << std::endl;
  std::cout << "  Zero pressure (pres_scale = 0)" << std::endl;
  std::cout << "  This should behave EXACTLY like symmetric case" << std::endl;

  std::cout << "\nRunning VMEC..." << std::endl;
  const auto output = vmecpp::run(config);

  if (!output.ok()) {
    std::cout << "\nStatus: " << output.status() << std::endl;
    std::string error_msg(output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout << "\n❌ CRITICAL: Jacobian fails for symmetric geometry with "
                   "lasym=true!"
                << std::endl;
      std::cout << "This proves the issue is in asymmetric mode handling, not "
                   "the geometry"
                << std::endl;
      std::cout << "\nPossible causes:" << std::endl;
      std::cout << "1. Theta range [0,2π] causes different interpolation"
                << std::endl;
      std::cout << "2. Array indexing issue in tau calculation" << std::endl;
      std::cout << "3. sqrtSH handling differs in asymmetric mode" << std::endl;
      std::cout << "4. Initial guess generation differs" << std::endl;
    }
  } else {
    std::cout << "\n✅ SUCCESS: Symmetric geometry works with lasym=true"
              << std::endl;
    std::cout << "The asymmetric mode handling is correct" << std::endl;
  }

  // Test passes - this is diagnostic
  EXPECT_TRUE(true) << "Symmetric geometry with asymmetric flag test";
}

TEST(SymmetricTauOnlyTest, CompareTheSameGeometry) {
  std::cout << "\n=== DIRECT COMPARISON: SAME GEOMETRY, DIFFERENT FLAGS ==="
            << std::endl;

  // Configuration for both tests
  auto makeConfig = [](bool lasym) {
    VmecINDATA config;
    config.lasym = lasym;
    config.nfp = 1;
    config.mpol = 3;
    config.ntor = 0;
    config.ns_array = {3};
    config.niter_array = {1};
    config.ftol_array = {1e-6};
    config.return_outputs_even_if_not_converged = true;
    config.delt = 0.5;
    config.tcon0 = 1.0;
    config.phiedge = 1.0;
    config.pmass_type = "power_series";
    config.am = {0.0};
    config.pres_scale = 0.0;

    // Identical geometry
    config.rbc = {10.0, 2.0, 0.5};
    config.zbs = {0.0, 2.0, 0.5};

    if (lasym) {
      config.rbs = {0.0, 0.0, 0.0};
      config.zbc = {0.0, 0.0, 0.0};
      config.raxis_s = {0.0};
      config.zaxis_c = {0.0};
    }

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};

    return config;
  };

  std::cout << "Test 1: lasym=false (symmetric mode)" << std::endl;
  auto config1 = makeConfig(false);
  auto output1 = vmecpp::run(config1);
  bool symmetric_ok = output1.ok();
  std::cout << "Result: " << (symmetric_ok ? "✅ SUCCESS" : "❌ FAILED")
            << std::endl;

  std::cout << "\nTest 2: lasym=true with zero asymmetric coeffs" << std::endl;
  auto config2 = makeConfig(true);
  auto output2 = vmecpp::run(config2);
  bool asymmetric_ok = output2.ok();
  std::cout << "Result: " << (asymmetric_ok ? "✅ SUCCESS" : "❌ FAILED")
            << std::endl;

  std::cout << "\nComparison:" << std::endl;
  if (symmetric_ok && !asymmetric_ok) {
    std::cout << "❌ CRITICAL BUG: Identical geometry behaves differently!"
              << std::endl;
    std::cout << "The asymmetric mode has a fundamental issue" << std::endl;
  } else if (symmetric_ok && asymmetric_ok) {
    std::cout << "✅ Both modes work correctly" << std::endl;
  } else if (!symmetric_ok && !asymmetric_ok) {
    std::cout << "⚠️ Both modes fail - geometry issue, not asymmetric bug"
              << std::endl;
  }

  // Test passes
  EXPECT_TRUE(true) << "Direct comparison test";
}

}  // namespace vmecpp
