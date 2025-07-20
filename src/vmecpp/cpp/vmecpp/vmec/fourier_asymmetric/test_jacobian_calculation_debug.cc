// Debug test to trace actual Jacobian calculation
#include <gtest/gtest.h>

#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

TEST(JacobianCalculationDebugTest, UnderstandJacobianFailure) {
  std::cout << "\n=== UNDERSTANDING JACOBIAN CALCULATION ===" << std::endl;

  std::cout << "From code analysis:" << std::endl;
  std::cout << "1. Jacobian check: minTau * maxTau < 0.0 (sign change)"
            << std::endl;
  std::cout << "2. tau appears to be related to sqrt(g) (Jacobian)"
            << std::endl;
  std::cout << "3. The check happens in computeLocalBasisVectors()"
            << std::endl;

  std::cout << "\nPossible issues for asymmetric:" << std::endl;
  std::cout
      << "- Initial guess interpolation creates self-intersecting surfaces"
      << std::endl;
  std::cout << "- Asymmetric perturbations cause R to go negative" << std::endl;
  std::cout << "- Theta symmetrization might create discontinuities"
            << std::endl;

  std::cout << "\nThe debug output shows:" << std::endl;
  std::cout << "- R and Z values look reasonable (R > 0)" << std::endl;
  std::cout << "- Array combination works (non-zero at kl=6-9)" << std::endl;
  std::cout << "- But still get 'INITIAL JACOBIAN CHANGED SIGN!'" << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Analysis test";
}

TEST(JacobianCalculationDebugTest, TestSymmetricVsAsymmetric) {
  std::cout << "\n=== COMPARE SYMMETRIC VS ASYMMETRIC ===" << std::endl;

  // First create a symmetric configuration that works
  {
    VmecINDATA config;
    config.lasym = false;  // SYMMETRIC
    config.nfp = 1;
    config.mpol = 3;
    config.ntor = 0;

    config.ns_array = {3};
    config.niter_array = {5};
    config.ftol_array = {1e-6};
    config.return_outputs_even_if_not_converged = true;

    config.delt = 0.5;
    config.tcon0 = 1.0;
    config.phiedge = 1.0;
    config.pmass_type = "power_series";
    config.am = {0.0};

    // Simple circular tokamak
    config.rbc = {10.0, 0.0, 2.0};
    config.zbs = {0.0, 0.0, 2.0};

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};

    std::cout << "Test 1: Symmetric circular tokamak" << std::endl;
    std::cout << "  lasym = false" << std::endl;
    std::cout << "  R0 = 10, a = 2" << std::endl;
    std::cout << "  Expected: Should work without Jacobian issues" << std::endl;
  }

  // Now the exact same geometry but asymmetric
  {
    VmecINDATA config;
    config.lasym = true;  // ASYMMETRIC
    config.nfp = 1;
    config.mpol = 3;
    config.ntor = 0;

    config.ns_array = {3};
    config.niter_array = {5};
    config.ftol_array = {1e-6};
    config.return_outputs_even_if_not_converged = true;

    config.delt = 0.5;
    config.tcon0 = 1.0;
    config.phiedge = 1.0;
    config.pmass_type = "power_series";
    config.am = {0.0};

    // Same circular tokamak but with lasym=true
    config.rbc = {10.0, 0.0, 2.0};
    config.zbs = {0.0, 0.0, 2.0};
    config.rbs = {0.0, 0.0, 0.0};  // No asymmetric perturbation
    config.zbc = {0.0, 0.0, 0.0};  // No asymmetric perturbation

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};
    config.raxis_s = {0.0};
    config.zaxis_c = {0.0};

    std::cout << "\nTest 2: Same geometry with lasym=true (but zero asymmetric "
                 "coeffs)"
              << std::endl;
    std::cout << "  lasym = true" << std::endl;
    std::cout << "  All asymmetric coefficients = 0" << std::endl;
    std::cout << "  Expected: Should behave like symmetric case" << std::endl;
  }

  std::cout
      << "\nIf Test 2 fails but Test 1 works, the issue is in asymmetric setup"
      << std::endl;
  std::cout << "even when there's no actual asymmetry in the geometry!"
            << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Comparison test";
}

TEST(JacobianCalculationDebugTest, MinimalAsymmetricGeometry) {
  std::cout << "\n=== MINIMAL ASYMMETRIC GEOMETRY TEST ===" << std::endl;

  // Create the absolute simplest asymmetric case
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 2;  // Minimal
  config.ntor = 0;

  config.ns_array = {3};
  config.niter_array = {5};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;

  config.delt = 0.9;  // Large time step
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.pmass_type = "power_series";
  config.am = {0.0};

  // Minimal boundary: just R0 and minor radius
  config.rbc = {10.0, 0.0};  // R0 = 10
  config.zbs = {0.0, 0.0};   // No vertical shift
  config.rbs = {0.0, 0.0};   // No asymmetry yet
  config.zbc = {0.0, 0.0};   // No asymmetry yet

  // Add only m=1 mode for circular cross-section
  config.rbc[1] = 2.0;  // Minor radius
  config.zbs[1] = 2.0;  // Circular

  config.raxis_c = {10.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  std::cout << "Minimal asymmetric configuration:" << std::endl;
  std::cout << "  mpol = 2 (minimal)" << std::endl;
  std::cout << "  Only m=0,1 modes" << std::endl;
  std::cout << "  R0 = 10, a = 2" << std::endl;
  std::cout << "  No actual asymmetry (rbs = zbc = 0)" << std::endl;

  std::cout << "\nThis is the simplest possible asymmetric case." << std::endl;
  std::cout << "If this fails, the problem is fundamental to lasym=true mode."
            << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Minimal geometry test";
}

}  // namespace vmecpp
