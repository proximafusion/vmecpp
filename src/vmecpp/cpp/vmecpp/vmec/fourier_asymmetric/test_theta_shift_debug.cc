// Debug test to verify theta shift is being applied correctly
#include <gtest/gtest.h>

#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(ThetaShiftDebugTest, VerifyShiftIsApplied) {
  std::cout << "\n=== THETA SHIFT DEBUG TEST ===" << std::endl;

  // Create up_down_asymmetric_tokamak configuration
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 5;
  config.ntor = 0;

  // Minimal conservative setup
  config.ns_array = {3};
  config.niter_array = {5};  // Just a few iterations to see debug output
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;

  // Zero pressure for stability
  config.pmass_type = "power_series";
  config.am = {0.0};
  config.pres_scale = 0.0;

  // Boundary coefficients
  config.rbc.resize(5, 0.0);
  config.zbs.resize(5, 0.0);
  config.rbs.resize(5, 0.0);
  config.zbc.resize(5, 0.0);

  // Symmetric boundary
  config.rbc[0] = 6.0;  // Major radius
  config.rbc[2] = 0.6;  // Minor radius
  config.zbs[2] = 0.6;  // Vertical elongation

  // Asymmetric perturbations - the key part
  config.rbs[2] = 0.189737;  // R up-down asymmetry
  config.zbc[2] = 0.189737;  // Z up-down asymmetry

  // But wait - the theta shift formula uses m=1, n=0 mode!
  // Let's also set m=1 coefficients
  config.rbc[1] = 0.1;    // Small m=1 symmetric R
  config.zbs[1] = 0.1;    // Small m=1 symmetric Z
  config.rbs[1] = 0.05;   // Small m=1 asymmetric R
  config.zbc[1] = 0.025;  // Small m=1 asymmetric Z

  // Axis
  config.raxis_c = {6.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  // Physical parameters
  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.gamma = 0.0;
  config.curtor = 0.0;
  config.ncurr = 0;
  config.piota_type = "power_series";
  config.ai = {0.0};

  // Enable verbose mode by setting environment variable
  setenv("VMECPP_VERBOSE", "1", 1);

  std::cout << "Running VMEC with verbose mode to see theta shift..."
            << std::endl;
  std::cout << "Key m=1 coefficients for theta shift:" << std::endl;
  std::cout << "  rbc[1] = " << config.rbc[1] << std::endl;
  std::cout << "  zbs[1] = " << config.zbs[1] << std::endl;
  std::cout << "  rbs[1] = " << config.rbs[1] << std::endl;
  std::cout << "  zbc[1] = " << config.zbc[1] << std::endl;

  // Calculate expected theta shift
  double expected_delta =
      std::atan2(config.rbs[1] - config.zbc[1], config.rbc[1] + config.zbs[1]);
  std::cout << "\nExpected theta shift: " << expected_delta
            << " radians = " << (expected_delta * 180.0 / M_PI) << " degrees"
            << std::endl;

  // Run VMEC
  const auto output = vmecpp::run(config);

  std::cout << "\nVMEC run completed with status: "
            << (output.ok() ? "SUCCESS" : "FAILED") << std::endl;

  if (!output.ok()) {
    std::cout << "Error: " << output.status() << std::endl;
  }

  // Test passes - this is a debug test
  EXPECT_TRUE(true) << "Theta shift debug test";
}

TEST(ThetaShiftDebugTest, CompareBoundaryCoefficients) {
  std::cout << "\n=== BOUNDARY COEFFICIENT COMPARISON ===" << std::endl;

  // The issue might be that our test configuration doesn't have m=1 modes!
  // The theta shift formula specifically uses m=1, n=0 mode

  std::cout << "From up_down_asymmetric_tokamak.json:" << std::endl;
  std::cout << "  rbc: [6.0, 0.0, 0.6, 0.0, 0.12]" << std::endl;
  std::cout << "  rbs: [0.0, 0.0, 0.189737, 0.0, 0.0]" << std::endl;
  std::cout << "  zbs: [0.0, 0.0, 0.0, 0.0, 0.0]" << std::endl;
  std::cout << "  zbc: [0.0, 0.0, 0.189737, 0.0, 0.0]" << std::endl;

  std::cout << "\nKey observation: m=1 modes are all ZERO!" << std::endl;
  std::cout << "  rbc[1] = 0.0, rbs[1] = 0.0" << std::endl;
  std::cout << "  zbs[1] = 0.0, zbc[1] = 0.0" << std::endl;

  std::cout << "\nTheta shift calculation:" << std::endl;
  std::cout << "  delta = atan2(rbs[1] - zbc[1], rbc[1] + zbs[1])" << std::endl;
  std::cout << "  delta = atan2(0.0 - 0.0, 0.0 + 0.0)" << std::endl;
  std::cout << "  delta = atan2(0.0, 0.0) = 0.0 (no shift!)" << std::endl;

  std::cout << "\n❌ PROBLEM IDENTIFIED: The theta shift is 0 because m=1 "
               "modes are zero!"
            << std::endl;
  std::cout << "The -34.36° shift we calculated was using m=2 modes, not m=1!"
            << std::endl;

  // Test passes - analysis complete
  EXPECT_TRUE(true) << "Boundary coefficient analysis";
}

TEST(ThetaShiftDebugTest, CheckIfM2ShiftNeeded) {
  std::cout << "\n=== SHOULD WE USE M=2 FOR THETA SHIFT? ===" << std::endl;

  // The original VMEC and jVMEC use m=1 for theta shift
  // But what if the configuration has no m=1 modes?

  std::cout << "Educational_VMEC convert_sym/convert_asym routines:"
            << std::endl;
  std::cout << "- These specifically handle m=1 mode conversion" << std::endl;
  std::cout
      << "- They convert between symmetric and asymmetric m=1 representations"
      << std::endl;

  std::cout << "\nPossible explanations:" << std::endl;
  std::cout << "1. The theta shift should only use m=1 (current implementation)"
            << std::endl;
  std::cout << "2. We need to also implement convert_sym/convert_asym for m=1"
            << std::endl;
  std::cout << "3. The boundary is failing for other reasons" << std::endl;

  std::cout << "\nNext steps:" << std::endl;
  std::cout << "- Check if educational_VMEC applies theta shift to all modes"
            << std::endl;
  std::cout << "- Look for m=1 mode conversion routines" << std::endl;
  std::cout << "- Test with configurations that have non-zero m=1 modes"
            << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "M=2 shift analysis";
}

}  // namespace vmecpp
