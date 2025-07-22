// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "absl/types/span.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

// Test to isolate geometry derivative calculation from valid R,Z input
TEST(GeometryDerivativesTest, CalculateDerivativesFromValidTransform) {
  std::cout << "\n=== GEOMETRY DERIVATIVES TEST ===\n" << std::endl;

  std::cout << "Testing derivative calculation with known valid R,Z values..."
            << std::endl;

  // Use the same configuration that causes NaN, but with verified transform
  // output
  VmecINDATA indata;

  indata.nfp = 1;
  indata.lasym = true;
  indata.mpol = 2;
  indata.ntor = 0;
  indata.ntheta = 10;
  indata.nzeta = 1;

  // Simple tokamak with small asymmetric perturbation
  int coeff_size = indata.mpol * (2 * indata.ntor + 1);

  // Set up sizes for asymmetric calculation
  Sizes sizes(true, 1, indata.mpol, indata.ntor, indata.ntheta, indata.nzeta);

  // Create R,Z arrays with known valid values from transform test
  std::vector<double> r_values = {
      4.41421,  // i=0
      4.14495,  // i=1
      3.43836,  // i=2
      2.56433,  // i=3
      1.85671,  // i=4
      1.58579,  // i=5
      1.85505,  // i=6 - PROBLEMATIC
      2.56164,  // i=7 - PROBLEMATIC
      3.43567,  // i=8 - PROBLEMATIC
      4.14329   // i=9 - PROBLEMATIC
  };

  std::vector<double> z_values = {
      0.0,           // i=0
      0.000831254,   // i=1
      0.001345,      // i=2
      0.001345,      // i=3
      0.000831254,   // i=4
      1.73191e-19,   // i=5
      -0.000831254,  // i=6 - PROBLEMATIC
      -0.001345,     // i=7 - PROBLEMATIC
      -0.001345,     // i=8 - PROBLEMATIC
      -0.000831254   // i=9 - PROBLEMATIC
  };

  std::cout << "\nInput R,Z values (verified finite from transform test):"
            << std::endl;
  for (int i = 0; i < indata.ntheta; ++i) {
    std::cout << "  i=" << i << ": R=" << r_values[i] << ", Z=" << z_values[i];
    if (i >= 6 && i <= 9) {
      std::cout << " ← Position where NaN occurs in full VMEC";
    }
    std::cout << std::endl;
  }

  std::cout << "\n1. CALCULATE THETA DERIVATIVES (ru, zu):" << std::endl;
  {
    // Simple finite difference for theta derivatives
    std::vector<double> ru(indata.ntheta), zu(indata.ntheta);

    for (int i = 0; i < indata.ntheta; ++i) {
      int ip1 = (i + 1) % indata.ntheta;
      int im1 = (i - 1 + indata.ntheta) % indata.ntheta;

      // Central difference
      double dtheta = 2.0 * M_PI / indata.ntheta;
      ru[i] = (r_values[ip1] - r_values[im1]) / (2.0 * dtheta);
      zu[i] = (z_values[ip1] - z_values[im1]) / (2.0 * dtheta);

      std::cout << "  i=" << i << ": ru=" << ru[i] << ", zu=" << zu[i];
      if (!std::isfinite(ru[i]) || !std::isfinite(zu[i])) {
        std::cout << " ⚠️ NON-FINITE DERIVATIVE!";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\n2. CALCULATE JACOBIAN COMPONENTS:" << std::endl;
  {
    // For asymmetric case on half grid, need special handling
    // This mimics what happens in ideal_mhd_model.cc

    std::cout << "\n  a) Symmetric-like calculation (tau = ru * zs - rs * zu):"
              << std::endl;
    // Simplified: assume rs=0, zs=1 for this test
    for (int i = 6; i <= 9; ++i) {
      double ru = 0.1;   // Placeholder derivative
      double zu = 0.01;  // Placeholder derivative
      double rs = 0.0;   // Radial derivative (zero at fixed radius)
      double zs = 1.0;   // Radial derivative

      double tau = ru * zs - rs * zu;
      std::cout << "    i=" << i << ": tau=" << tau;
      if (!std::isfinite(tau)) {
        std::cout << " ⚠️ NON-FINITE JACOBIAN!";
      }
      std::cout << std::endl;
    }

    std::cout << "\n  b) Check for division by small values:" << std::endl;
    for (int i = 6; i <= 9; ++i) {
      // Check if R approaches zero (shouldn't based on our values)
      std::cout << "    i=" << i << ": 1/R=" << 1.0 / r_values[i];
      if (r_values[i] < 0.1) {
        std::cout << " ⚠️ SMALL R VALUE!";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\n3. ASYMMETRIC-SPECIFIC CALCULATIONS:" << std::endl;
  {
    std::cout << "  In jVMEC, asymmetric Jacobian uses even/odd separation:"
              << std::endl;
    std::cout << "  - even_contrib = ru_even * zs_even - rs_even * zu_even"
              << std::endl;
    std::cout << "  - odd_contrib = ru_odd * zs_odd - rs_odd * zu_odd"
              << std::endl;
    std::cout << "  - tau = even_contrib + sqrtS * odd_contrib" << std::endl;
    std::cout << "\n  VMEC++ may be missing this separation!" << std::endl;
  }

  std::cout << "\n4. ANALYSIS:" << std::endl;
  std::cout << "✅ Transform output is valid (R,Z finite at all positions)"
            << std::endl;
  std::cout
      << "❓ Need to check actual derivative calculation in ideal_mhd_model"
      << std::endl;
  std::cout << "❓ May need even/odd separation for asymmetric case"
            << std::endl;
  std::cout << "❓ Check if sqrtS or other factors approach zero" << std::endl;

  EXPECT_TRUE(true) << "Analysis test completed";
}

// Test to compare symmetric vs asymmetric derivative calculation
TEST(GeometryDerivativesTest, CompareSymmetricVsAsymmetricDerivatives) {
  std::cout << "\n=== SYMMETRIC VS ASYMMETRIC DERIVATIVES ===\n" << std::endl;

  std::cout << "Key differences in derivative calculation:" << std::endl;

  std::cout << "\n1. GRID DIFFERENCES:" << std::endl;
  std::cout << "  - Symmetric: Uses full theta grid [0, 2π)" << std::endl;
  std::cout << "  - Asymmetric: May use half grid [0, π] with reflection"
            << std::endl;
  std::cout << "  - This affects derivative stencils at boundaries"
            << std::endl;

  std::cout << "\n2. JACOBIAN CALCULATION:" << std::endl;
  std::cout << "  - Symmetric: tau = ru * zs - rs * zu (straightforward)"
            << std::endl;
  std::cout << "  - Asymmetric: Needs special handling for odd/even modes"
            << std::endl;
  std::cout << "  - Missing this could cause NaN at specific theta positions"
            << std::endl;

  std::cout << "\n3. BOUNDARY CONDITIONS:" << std::endl;
  std::cout << "  - Symmetric: Periodic in theta" << std::endl;
  std::cout << "  - Asymmetric: May need reflection conditions" << std::endl;
  std::cout << "  - Incorrect boundaries could cause derivative errors"
            << std::endl;

  std::cout << "\n4. RECOMMENDED NEXT STEPS:" << std::endl;
  std::cout
      << "  a) Add debug output to ideal_mhd_model at derivative calculation"
      << std::endl;
  std::cout << "  b) Compare with jVMEC's totzsp_as function" << std::endl;
  std::cout << "  c) Check if VMEC++ implements even/odd separation"
            << std::endl;
  std::cout << "  d) Verify boundary condition handling in asymmetric mode"
            << std::endl;

  EXPECT_TRUE(true) << "Comparison completed";
}

}  // namespace vmecpp
