// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Test to debug Jacobian and pressure initialization in asymmetric mode
TEST(JacobianDebugTest, CompareSymmetricVsAsymmetricJacobian) {
  std::cout << "\n=== JACOBIAN DEBUG TEST ===\n" << std::endl;

  std::cout
      << "Testing if asymmetric perturbations cause Jacobian singularities...\n"
      << std::endl;

  // Test 1: Pure symmetric case (should work)
  std::cout << "1. SYMMETRIC CASE:" << std::endl;
  {
    VmecINDATA indata;

    // Symmetric configuration
    indata.nfp = 1;
    indata.lasym = false;  // Symmetric
    indata.mpol = 2;
    indata.ntor = 0;
    indata.ns_array = {3};
    indata.niter_array = {3};  // Just 3 iterations
    indata.ntheta = 9;
    indata.nzeta = 1;

    // Zero pressure to isolate geometry issues
    indata.pres_scale = 0.0;
    indata.am = {0.0};
    indata.gamma = 0.0;
    indata.phiedge = 1.0;

    indata.return_outputs_even_if_not_converged = true;

    // Simple tokamak arrays
    int coeff_size = indata.mpol * (2 * indata.ntor + 1);
    indata.rbc.resize(coeff_size, 0.0);
    indata.zbs.resize(coeff_size, 0.0);

    indata.rbc[0] = 3.0;  // R00
    indata.rbc[1] = 1.0;  // R10
    indata.zbs[1] = 1.0;  // Z10

    indata.raxis_c = {3.0};
    indata.zaxis_s = {0.0};

    const auto output = vmecpp::run(indata);

    if (output.ok()) {
      std::cout << "  ✅ Symmetric case works correctly" << std::endl;
      const auto& wout = output->wout;
      std::cout << "  Volume = " << wout.volume_p << std::endl;
    } else {
      std::cout << "  ❌ Symmetric case failed: " << output.status()
                << std::endl;
    }
  }

  // Test 2: Asymmetric case with ZERO pressure (isolate geometry)
  std::cout << "\n2. ASYMMETRIC CASE (ZERO PRESSURE):" << std::endl;
  {
    VmecINDATA indata;

    // Asymmetric configuration
    indata.nfp = 1;
    indata.lasym = true;  // Asymmetric
    indata.mpol = 2;
    indata.ntor = 0;
    indata.ns_array = {3};
    indata.niter_array = {3};  // Just 3 iterations
    indata.ntheta = 9;
    indata.nzeta = 1;

    // Zero pressure to isolate geometry issues
    indata.pres_scale = 0.0;
    indata.am = {0.0};
    indata.gamma = 0.0;
    indata.phiedge = 1.0;

    indata.return_outputs_even_if_not_converged = true;

    // Same tokamak arrays + small asymmetric perturbation
    int coeff_size = indata.mpol * (2 * indata.ntor + 1);
    indata.rbc.resize(coeff_size, 0.0);
    indata.zbs.resize(coeff_size, 0.0);
    indata.rbs.resize(coeff_size, 0.0);
    indata.zbc.resize(coeff_size, 0.0);

    // Symmetric part (same as above)
    indata.rbc[0] = 3.0;  // R00
    indata.rbc[1] = 1.0;  // R10
    indata.zbs[1] = 1.0;  // Z10

    // VERY small asymmetric perturbation
    indata.rbs[1] = 0.001;  // 0.1% asymmetric perturbation
    indata.zbc[1] = 0.001;  // 0.1% asymmetric perturbation

    indata.raxis_c = {3.0};
    indata.zaxis_s = {0.0};
    indata.raxis_s = {0.0};
    indata.zaxis_c = {0.0};

    const auto output = vmecpp::run(indata);

    if (output.ok()) {
      std::cout << "  ✅ Asymmetric case works correctly" << std::endl;
      const auto& wout = output->wout;
      std::cout << "  Volume = " << wout.volume_p << std::endl;
    } else {
      std::cout << "  ❌ Asymmetric case failed: " << output.status()
                << std::endl;

      std::string error_msg = output.status().ToString();
      if (error_msg.find("taup") != std::string::npos ||
          error_msg.find("zup") != std::string::npos) {
        std::cout << "  🔍 DIAGNOSIS: Jacobian derivatives (tau, zu12) are "
                     "problematic"
                  << std::endl;
      }
      if (error_msg.find("gbvbv") != std::string::npos) {
        std::cout << "  🔍 DIAGNOSIS: Magnetic field calculations (gsqrt, "
                     "bsupv) are problematic"
                  << std::endl;
      }
    }
  }

  // Test 3: Even smaller asymmetric perturbation
  std::cout << "\n3. MINIMAL ASYMMETRIC PERTURBATION:" << std::endl;
  {
    VmecINDATA indata;

    indata.nfp = 1;
    indata.lasym = true;
    indata.mpol = 2;
    indata.ntor = 0;
    indata.ns_array = {3};
    indata.niter_array = {2};  // Even fewer iterations
    indata.ntheta = 9;
    indata.nzeta = 1;

    // Zero pressure
    indata.pres_scale = 0.0;
    indata.am = {0.0};
    indata.gamma = 0.0;
    indata.phiedge = 1.0;

    indata.return_outputs_even_if_not_converged = true;

    int coeff_size = indata.mpol * (2 * indata.ntor + 1);
    indata.rbc.resize(coeff_size, 0.0);
    indata.zbs.resize(coeff_size, 0.0);
    indata.rbs.resize(coeff_size, 0.0);
    indata.zbc.resize(coeff_size, 0.0);

    // Symmetric part
    indata.rbc[0] = 3.0;
    indata.rbc[1] = 1.0;
    indata.zbs[1] = 1.0;

    // TINY asymmetric perturbation
    indata.rbs[1] = 1e-6;  // 0.0001% perturbation
    indata.zbc[1] = 1e-6;  // 0.0001% perturbation

    indata.raxis_c = {3.0};
    indata.zaxis_s = {0.0};
    indata.raxis_s = {0.0};
    indata.zaxis_c = {0.0};

    const auto output = vmecpp::run(indata);

    if (output.ok()) {
      std::cout << "  ✅ Minimal asymmetric perturbation works" << std::endl;
      std::cout << "  → Issue appears with larger asymmetric perturbations"
                << std::endl;
    } else {
      std::cout << "  ❌ Even minimal asymmetric perturbation fails"
                << std::endl;
      std::cout << "  → Fundamental issue with asymmetric geometry calculation"
                << std::endl;
    }
  }

  std::cout << "\n📊 CONCLUSION:" << std::endl;
  std::cout << "This test isolates whether the issue is:" << std::endl;
  std::cout << "- Asymmetric coefficient handling in geometry calculations"
            << std::endl;
  std::cout << "- Jacobian singularities from asymmetric perturbations"
            << std::endl;
  std::cout << "- Pressure-independent vs pressure-dependent failures"
            << std::endl;

  // Test always fails - we expect failures and want to analyze them
  FAIL() << "Test designed to analyze failure modes, not to pass";
}

}  // namespace vmecpp
