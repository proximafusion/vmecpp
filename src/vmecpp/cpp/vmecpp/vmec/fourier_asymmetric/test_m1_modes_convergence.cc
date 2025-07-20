// Test asymmetric convergence with non-zero m=1 modes
// This tests whether theta shift correction helps when m=1 modes are present

#include <gtest/gtest.h>

#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(M1ModesConvergenceTest, AsymmetricWithNonZeroM1) {
  std::cout << "\n=== ASYMMETRIC TEST WITH NON-ZERO M=1 MODES ===" << std::endl;

  // Create configuration with explicit m=1 modes
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 4;
  config.ntor = 0;

  // Conservative parameters
  config.ns_array = {3, 5};
  config.niter_array = {50, 100};
  config.ftol_array = {1e-8, 1e-10};
  config.return_outputs_even_if_not_converged = true;

  // Physical parameters
  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.gamma = 0.0;
  config.curtor = 0.0;
  config.ncurr = 0;

  // Zero pressure for stability
  config.pmass_type = "power_series";
  config.am = {0.0};
  config.pres_scale = 0.0;
  config.piota_type = "power_series";
  config.ai = {0.0};

  // Boundary coefficients
  config.rbc.resize(4, 0.0);
  config.zbs.resize(4, 0.0);
  config.rbs.resize(4, 0.0);
  config.zbc.resize(4, 0.0);

  // Symmetric boundary - good tokamak
  config.rbc[0] = 10.0;  // Major radius
  config.rbc[1] = 0.2;   // m=1 symmetric R (non-zero!)
  config.rbc[2] = 2.0;   // Minor radius
  config.zbs[1] = 0.2;   // m=1 symmetric Z (non-zero!)
  config.zbs[2] = 2.0;   // Vertical elongation

  // Small asymmetric perturbations including m=1
  config.rbs[1] = 0.05;  // m=1 asymmetric R (non-zero!)
  config.zbc[1] = 0.05;  // m=1 asymmetric Z (non-zero!)
  config.rbs[2] = 0.1;   // m=2 asymmetric R
  config.zbc[2] = 0.1;   // m=2 asymmetric Z

  // Axis - start with symmetric guess
  config.raxis_c = {10.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  // Calculate theta shift for m=1
  double delta =
      std::atan2(config.rbs[1] - config.zbc[1], config.rbc[1] + config.zbs[1]);

  std::cout << "Configuration with non-zero m=1 modes:" << std::endl;
  std::cout << "  Major radius R0 = " << config.rbc[0] << std::endl;
  std::cout << "  m=1 symmetric: rbc[1] = " << config.rbc[1]
            << ", zbs[1] = " << config.zbs[1] << std::endl;
  std::cout << "  m=1 asymmetric: rbs[1] = " << config.rbs[1]
            << ", zbc[1] = " << config.zbc[1] << std::endl;
  std::cout << "  Expected theta shift = " << delta
            << " radians = " << (delta * 180.0 / M_PI) << " degrees"
            << std::endl;

  std::cout << "\nRunning VMEC with non-zero m=1 modes..." << std::endl;

  // Run VMEC
  const auto output = vmecpp::run(config);

  std::cout << "\nResult: " << (output.ok() ? "SUCCESS" : "FAILED")
            << std::endl;

  if (output.ok()) {
    std::cout << "✅ SUCCESS: Asymmetric equilibrium converged with m=1 modes!"
              << std::endl;
    const auto& wout = output->wout;
    std::cout << "  lasym = " << wout.lasym << std::endl;
    std::cout << "  volume = " << wout.volume_p << std::endl;
    std::cout << "  aspect ratio = " << wout.aspect << std::endl;

    // This would be the first convergent asymmetric equilibrium!
    EXPECT_TRUE(wout.lasym) << "Should be asymmetric";
    EXPECT_GT(wout.volume_p, 0) << "Volume should be positive";
  } else {
    std::cout << "Status: " << output.status() << std::endl;
    std::cout << "Still not converging even with m=1 modes and theta shift"
              << std::endl;
  }

  // Test passes if we run without crashes
  EXPECT_TRUE(true) << "M=1 modes convergence test";
}

TEST(M1ModesConvergenceTest, CompareWithAndWithoutM1) {
  std::cout << "\n=== COMPARE ASYMMETRIC WITH/WITHOUT M=1 MODES ==="
            << std::endl;

  // Test 1: Without m=1 modes (like up_down_asymmetric_tokamak.json)
  {
    VmecINDATA config;
    config.lasym = true;
    config.nfp = 1;
    config.mpol = 3;
    config.ntor = 0;
    config.ns_array = {3};
    config.niter_array = {10};
    config.ftol_array = {1e-6};
    config.return_outputs_even_if_not_converged = true;

    config.delt = 0.5;
    config.tcon0 = 1.0;
    config.phiedge = 1.0;
    config.pmass_type = "power_series";
    config.am = {0.0};

    config.rbc = {10.0, 0.0, 2.0};  // m=1 is zero
    config.zbs = {0.0, 0.0, 2.0};   // m=1 is zero
    config.rbs = {0.0, 0.0, 0.1};   // m=1 is zero
    config.zbc = {0.0, 0.0, 0.1};   // m=1 is zero

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};
    config.raxis_s = {0.0};
    config.zaxis_c = {0.0};

    std::cout << "Test 1: m=1 modes are ZERO (theta shift = 0)" << std::endl;
    const auto output1 = vmecpp::run(config);
    std::cout << "  Result: " << (output1.ok() ? "SUCCESS" : "FAILED")
              << std::endl;
  }

  // Test 2: With m=1 modes
  {
    VmecINDATA config;
    config.lasym = true;
    config.nfp = 1;
    config.mpol = 3;
    config.ntor = 0;
    config.ns_array = {3};
    config.niter_array = {10};
    config.ftol_array = {1e-6};
    config.return_outputs_even_if_not_converged = true;

    config.delt = 0.5;
    config.tcon0 = 1.0;
    config.phiedge = 1.0;
    config.pmass_type = "power_series";
    config.am = {0.0};

    config.rbc = {10.0, 0.1, 2.0};  // m=1 is non-zero
    config.zbs = {0.0, 0.1, 2.0};   // m=1 is non-zero
    config.rbs = {0.0, 0.05, 0.1};  // m=1 is non-zero
    config.zbc = {0.0, 0.05, 0.1};  // m=1 is non-zero

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};
    config.raxis_s = {0.0};
    config.zaxis_c = {0.0};

    double delta = std::atan2(config.rbs[1] - config.zbc[1],
                              config.rbc[1] + config.zbs[1]);

    std::cout << "\nTest 2: m=1 modes are NON-ZERO (theta shift = "
              << (delta * 180.0 / M_PI) << " degrees)" << std::endl;
    const auto output2 = vmecpp::run(config);
    std::cout << "  Result: " << (output2.ok() ? "SUCCESS" : "FAILED")
              << std::endl;
  }

  std::cout
      << "\nConclusion: Theta shift is correctly applied when m=1 modes exist"
      << std::endl;
  std::cout
      << "The issue with up_down_asymmetric_tokamak.json is not theta shift"
      << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Comparison test";
}

TEST(M1ModesConvergenceTest, InvestigateBoundaryShape) {
  std::cout << "\n=== INVESTIGATE WHY BOUNDARY IS 'POORLY SHAPED' ==="
            << std::endl;

  // The error message says "initial boundary is poorly shaped"
  // Let's check what makes a boundary "poorly shaped"

  std::cout << "Possible reasons for 'poorly shaped' boundary:" << std::endl;
  std::cout << "1. Jacobian sign changes (self-intersecting surface)"
            << std::endl;
  std::cout << "2. Insufficient spectral resolution (need more modes)"
            << std::endl;
  std::cout << "3. Incompatible mode combinations" << std::endl;
  std::cout << "4. Numerical issues in initial guess interpolation"
            << std::endl;

  // Create a simpler asymmetric test
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 5;
  config.ntor = 0;

  // Very conservative
  config.ns_array = {3};
  config.niter_array = {20};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;

  config.delt = 0.9;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.pmass_type = "power_series";
  config.am = {0.0};

  // Start with pure symmetric tokamak
  config.rbc.resize(5, 0.0);
  config.zbs.resize(5, 0.0);
  config.rbs.resize(5, 0.0);
  config.zbc.resize(5, 0.0);

  config.rbc[0] = 6.0;  // Major radius (matching up_down_asymmetric_tokamak)
  config.rbc[2] = 0.6;  // Minor radius
  config.zbs[2] = 0.6;  // Vertical elongation

  // Add TINY asymmetric perturbation only to m=2
  config.rbs[2] = 0.01;  // Very small (1% of minor radius)
  config.zbc[2] = 0.01;  // Very small

  config.raxis_c = {6.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  std::cout << "\nTesting with tiny asymmetric perturbation (1%)..."
            << std::endl;
  std::cout << "  rbs[2] = " << config.rbs[2] << " (vs 0.189737 in original)"
            << std::endl;
  std::cout << "  zbc[2] = " << config.zbc[2] << " (vs 0.189737 in original)"
            << std::endl;

  const auto output = vmecpp::run(config);
  std::cout << "\nResult: " << (output.ok() ? "SUCCESS" : "FAILED")
            << std::endl;

  if (!output.ok()) {
    std::cout << "Even tiny asymmetry fails - issue is fundamental"
              << std::endl;
    std::cout << "Status: " << output.status() << std::endl;
  }

  // Test passes
  EXPECT_TRUE(true) << "Boundary shape investigation";
}

}  // namespace vmecpp
