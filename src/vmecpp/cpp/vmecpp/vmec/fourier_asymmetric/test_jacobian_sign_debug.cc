// Debug test for Jacobian sign check in asymmetric mode
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(JacobianSignDebugTest, CheckJacobianCalculation) {
  std::cout << "\n=== JACOBIAN SIGN DEBUG TEST ===" << std::endl;

  // The checkSignOfJacobian function uses:
  // rTest = sum of rbcc[m=1,n] for all n
  // zTest = sum of zbsc[m=1,n] for all n
  // Returns true if rTest * zTest * sign_of_jacobian > 0

  std::cout << "Understanding Jacobian sign check:" << std::endl;
  std::cout << "- Uses only m=1 modes (first poloidal harmonic)" << std::endl;
  std::cout << "- rTest = sum(rbcc[m=1,n]), zTest = sum(zbsc[m=1,n])"
            << std::endl;
  std::cout << "- Flips theta if rTest * zTest * signJac > 0" << std::endl;

  // Test with up_down_asymmetric_tokamak configuration
  {
    VmecINDATA config;
    config.lasym = true;
    config.nfp = 1;
    config.mpol = 5;
    config.ntor = 0;

    config.rbc.resize(5, 0.0);
    config.zbs.resize(5, 0.0);
    config.rbs.resize(5, 0.0);
    config.zbc.resize(5, 0.0);

    // From up_down_asymmetric_tokamak.json
    config.rbc[0] = 6.0;
    config.rbc[2] = 0.6;
    config.zbs[2] = 0.6;
    config.rbs[2] = 0.189737;
    config.zbc[2] = 0.189737;

    // m=1 modes are ZERO!
    std::cout << "\nConfiguration 1: up_down_asymmetric_tokamak (m=1 = 0)"
              << std::endl;
    std::cout << "  rbc[1] = " << config.rbc[1] << " (m=1 symmetric R)"
              << std::endl;
    std::cout << "  zbs[1] = " << config.zbs[1] << " (m=1 symmetric Z)"
              << std::endl;

    // After theta shift (which is 0), rbcc = rbc, zbsc = zbs
    double rTest = config.rbc[1];  // 0
    double zTest = config.zbs[1];  // 0

    std::cout << "  rTest = " << rTest << ", zTest = " << zTest << std::endl;
    std::cout << "  rTest * zTest = " << (rTest * zTest) << std::endl;
    std::cout << "  ⚠️  Both are zero - Jacobian check is undefined!"
              << std::endl;
  }

  // Test with non-zero m=1 modes
  {
    VmecINDATA config;
    config.lasym = true;
    config.nfp = 1;
    config.mpol = 5;
    config.ntor = 0;

    config.rbc.resize(5, 0.0);
    config.zbs.resize(5, 0.0);
    config.rbs.resize(5, 0.0);
    config.zbc.resize(5, 0.0);

    config.rbc[0] = 6.0;
    config.rbc[1] = 0.5;  // Non-zero m=1
    config.rbc[2] = 0.6;
    config.zbs[1] = 0.5;  // Non-zero m=1
    config.zbs[2] = 0.6;
    config.rbs[1] = 0.05;  // Small asymmetric
    config.zbc[1] = 0.05;  // Small asymmetric

    std::cout << "\nConfiguration 2: With m=1 modes" << std::endl;
    std::cout << "  rbc[1] = " << config.rbc[1] << " (m=1 symmetric R)"
              << std::endl;
    std::cout << "  zbs[1] = " << config.zbs[1] << " (m=1 symmetric Z)"
              << std::endl;

    // After theta shift, coefficients get modified
    // But for initial check, use original values
    double rTest = config.rbc[1];  // 0.5
    double zTest = config.zbs[1];  // 0.5

    std::cout << "  rTest = " << rTest << ", zTest = " << zTest << std::endl;
    std::cout << "  rTest * zTest = " << (rTest * zTest) << std::endl;
    std::cout << "  ✅ Both positive - normal tokamak orientation" << std::endl;
  }

  // Test passes
  EXPECT_TRUE(true) << "Jacobian sign debug test";
}

TEST(JacobianSignDebugTest, WhyJacobianFailsForAsymmetric) {
  std::cout << "\n=== WHY JACOBIAN FAILS FOR ASYMMETRIC ===" << std::endl;

  std::cout << "Hypothesis 1: Zero m=1 modes cause undefined behavior"
            << std::endl;
  std::cout << "- When rbc[1] = zbs[1] = 0, rTest * zTest = 0" << std::endl;
  std::cout << "- The sign check becomes ambiguous" << std::endl;
  std::cout << "- This might trigger false positive for 'bad Jacobian'"
            << std::endl;

  std::cout << "\nHypothesis 2: Asymmetric adds complexity" << std::endl;
  std::cout << "- With lasym=true, we also have rbsc, zbcc terms" << std::endl;
  std::cout << "- These might affect Jacobian but aren't in sign check"
            << std::endl;
  std::cout << "- The simple m=1 check might not be sufficient" << std::endl;

  std::cout << "\nHypothesis 3: Initial guess interpolation" << std::endl;
  std::cout << "- Interpolation from axis to boundary might fail" << std::endl;
  std::cout << "- Asymmetric geometry might create self-intersections"
            << std::endl;
  std::cout << "- Even small perturbations could cause issues" << std::endl;

  std::cout << "\nNext debugging steps:" << std::endl;
  std::cout << "1. Add m=1 modes to up_down_asymmetric_tokamak" << std::endl;
  std::cout << "2. Print actual Jacobian values during initialization"
            << std::endl;
  std::cout << "3. Check if flipTheta is being called incorrectly" << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Analysis test";
}

TEST(JacobianSignDebugTest, TestWithAddedM1Modes) {
  std::cout << "\n=== TEST WITH ADDED M=1 MODES ===" << std::endl;

  // Take up_down_asymmetric_tokamak and add m=1 modes
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 5;
  config.ntor = 0;

  config.ns_array = {3};
  config.niter_array = {20};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;

  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.pmass_type = "power_series";
  config.am = {0.0};
  config.pres_scale = 0.0;

  config.rbc.resize(5, 0.0);
  config.zbs.resize(5, 0.0);
  config.rbs.resize(5, 0.0);
  config.zbc.resize(5, 0.0);

  // Original configuration
  config.rbc[0] = 6.0;
  config.rbc[2] = 0.6;
  config.zbs[2] = 0.6;
  config.rbs[2] = 0.189737;
  config.zbc[2] = 0.189737;

  // ADD m=1 modes to avoid zero rTest/zTest
  config.rbc[1] = 0.3;  // Half of minor radius
  config.zbs[1] = 0.3;  // Same as R to maintain shape

  config.raxis_c = {6.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  std::cout << "Modified configuration with m=1 modes:" << std::endl;
  std::cout << "  rbc[1] = " << config.rbc[1] << " (added)" << std::endl;
  std::cout << "  zbs[1] = " << config.zbs[1] << " (added)" << std::endl;
  std::cout << "  rbc[2] = " << config.rbc[2] << " (original)" << std::endl;
  std::cout << "  zbs[2] = " << config.zbs[2] << " (original)" << std::endl;
  std::cout << "  rbs[2] = " << config.rbs[2] << " (asymmetric)" << std::endl;
  std::cout << "  zbc[2] = " << config.zbc[2] << " (asymmetric)" << std::endl;

  std::cout << "\nRunning VMEC with added m=1 modes..." << std::endl;

  const auto output = vmecpp::run(config);

  std::cout << "\nResult: " << (output.ok() ? "SUCCESS" : "FAILED")
            << std::endl;

  if (!output.ok()) {
    std::cout << "Status: " << output.status() << std::endl;

    // Check if it's still Jacobian issue
    std::string error_msg(output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout << "❌ Still fails with Jacobian issue even with m=1 modes!"
                << std::endl;
      std::cout << "This suggests the problem is deeper than just zero m=1"
                << std::endl;
    }
  } else {
    std::cout << "✅ SUCCESS: Adding m=1 modes fixed the issue!" << std::endl;
  }

  // Test passes
  EXPECT_TRUE(true) << "M=1 modes addition test";
}

}  // namespace vmecpp
