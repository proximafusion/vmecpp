// Debug test to trace tau calculation in asymmetric mode
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Forward declaration to access internals for debugging
namespace internal {
void EnableTauDebugOutput(bool enable);
}  // namespace internal

TEST(TauCalculationDebugTest, DetailedTauAnalysis) {
  std::cout << "\n=== TAU CALCULATION DEBUG TEST ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6);

  // Create simple asymmetric configuration that should work
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
  config.pres_scale = 0.0;

  // Large circular tokamak with non-zero m=1 modes
  config.rbc = {10.0, 2.0, 1.0};  // R0=10, m=1 radius=2, m=2 radius=1
  config.zbs = {0.0, 2.0, 1.0};   // Circular cross-section

  // Tiny asymmetric perturbation (1%)
  config.rbs = {0.0, 0.02, 0.01};
  config.zbc = {0.0, 0.02, 0.01};

  config.raxis_c = {10.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  std::cout << "Configuration:" << std::endl;
  std::cout << "  R0 = " << config.rbc[0] << ", a = " << config.rbc[1]
            << std::endl;
  std::cout << "  Asymmetric perturbation: " << config.rbs[1] << " (1%)"
            << std::endl;
  std::cout << "  m=1 modes present: rbc[1] = " << config.rbc[1]
            << ", zbs[1] = " << config.zbs[1] << std::endl;

  std::cout << "\nExpected behavior:" << std::endl;
  std::cout << "  - tau = (ru * zs - rs * zu) / R" << std::endl;
  std::cout << "  - Should be positive for normal Jacobian" << std::endl;
  std::cout << "  - If tau changes sign, surfaces self-intersect" << std::endl;

  std::cout << "\nAnalysis of tau components:" << std::endl;
  std::cout << "  tau1 = ru12 * zs - rs * zu12" << std::endl;
  std::cout << "  tau2 involves division by sqrtSH" << std::endl;
  std::cout << "  tau = tau1 + dSHalfDsInterp * tau2" << std::endl;

  std::cout << "\nKey questions:" << std::endl;
  std::cout << "1. Why does tau change sign for ALL asymmetric cases?"
            << std::endl;
  std::cout << "2. Is it the tau1 term or tau2 term causing issues?"
            << std::endl;
  std::cout << "3. Are the geometry derivatives (rs, zs) correct?" << std::endl;
  std::cout << "4. Is the initial guess interpolation creating problems?"
            << std::endl;

  // Test passes - this is just analysis
  EXPECT_TRUE(true) << "Tau calculation analysis";
}

TEST(TauCalculationDebugTest, CompareTauSymmetricVsAsymmetric) {
  std::cout << "\n=== COMPARE TAU: SYMMETRIC VS ASYMMETRIC ===" << std::endl;

  // First test: Symmetric geometry with lasym=false
  {
    VmecINDATA config;
    config.lasym = false;  // SYMMETRIC
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

    // Simple circular tokamak
    config.rbc = {10.0, 2.0, 0.5};
    config.zbs = {0.0, 2.0, 0.5};

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};

    std::cout << "Test 1: Symmetric circular tokamak (lasym=false)"
              << std::endl;
    std::cout << "  Expected: tau should be positive everywhere" << std::endl;

    // We would run VMEC here and capture tau values
    // For now, just document the test structure
  }

  // Second test: Same geometry with lasym=true but no asymmetric coeffs
  {
    VmecINDATA config;
    config.lasym = true;  // ASYMMETRIC mode
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

    // Same circular tokamak
    config.rbc = {10.0, 2.0, 0.5};
    config.zbs = {0.0, 2.0, 0.5};
    config.rbs = {0.0, 0.0, 0.0};  // No asymmetry
    config.zbc = {0.0, 0.0, 0.0};  // No asymmetry

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};
    config.raxis_s = {0.0};
    config.zaxis_c = {0.0};

    std::cout << "\nTest 2: Same geometry with lasym=true (but zero asymmetric "
                 "coeffs)"
              << std::endl;
    std::cout << "  Expected: Should behave identically to Test 1" << std::endl;
    std::cout << "  If this fails but Test 1 works: Issue in asymmetric setup"
              << std::endl;
  }

  // Test passes
  EXPECT_TRUE(true) << "Tau comparison structure";
}

TEST(TauCalculationDebugTest, InvestigateTauDivision) {
  std::cout << "\n=== INVESTIGATE TAU2 DIVISION ISSUE ===" << std::endl;

  std::cout << "From code analysis:" << std::endl;
  std::cout << "- tau2 involves division by sqrtSH" << std::endl;
  std::cout << "- At axis (jH=0), sqrtSH = 0" << std::endl;
  std::cout << "- This causes division by zero!" << std::endl;

  std::cout << "\nAxis protection attempt:" << std::endl;
  std::cout << "- Code has: protected_sqrtSH = max(sqrtSH, 1e-12)" << std::endl;
  std::cout << "- But this may not be sufficient" << std::endl;
  std::cout << "- jVMEC uses extrapolation from next radial point" << std::endl;

  std::cout << "\nHypothesis:" << std::endl;
  std::cout << "1. Division by sqrtSH in tau2 is problematic near axis"
            << std::endl;
  std::cout << "2. The 1e-12 protection may create very large tau2 values"
            << std::endl;
  std::cout << "3. This could cause tau to change sign" << std::endl;

  std::cout << "\nNext steps:" << std::endl;
  std::cout << "1. Add debug output to print tau1, tau2 separately"
            << std::endl;
  std::cout << "2. Check sqrtSH values at each radial position" << std::endl;
  std::cout << "3. Implement proper jVMEC-style axis extrapolation"
            << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Tau division analysis";
}

}  // namespace vmecpp
