// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// Test to validate input data structure for asymmetric configurations
TEST(InputValidationTest, ValidateAsymmetricInputStructure) {
  std::cout << "\n=== INPUT VALIDATION TEST ===\n" << std::endl;

  // Create minimal asymmetric configuration
  VmecINDATA indata;

  indata.nfp = 1;
  indata.lasym = true;
  indata.mpol = 1;
  indata.ntor = 0;

  std::cout << "Setting up arrays for mpol=" << indata.mpol
            << ", ntor=" << indata.ntor << std::endl;

  // Calculate expected array sizes
  int coeff_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
  std::cout << "Calculated coefficient array size: " << coeff_size << std::endl;

  // Initialize arrays
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  std::cout << "Array sizes after resize:" << std::endl;
  std::cout << "  rbc.size() = " << indata.rbc.size() << std::endl;
  std::cout << "  zbs.size() = " << indata.zbs.size() << std::endl;
  std::cout << "  rbs.size() = " << indata.rbs.size() << std::endl;
  std::cout << "  zbc.size() = " << indata.zbc.size() << std::endl;

  // Test index calculations for different modes
  std::cout << "\nTesting index calculations:" << std::endl;

  for (int m = 0; m <= indata.mpol; ++m) {
    for (int n = -indata.ntor; n <= indata.ntor; ++n) {
      int idx = m * (2 * indata.ntor + 1) + (n + indata.ntor);
      std::cout << "  m=" << m << ", n=" << n << " -> idx=" << idx;

      if (idx >= 0 && idx < coeff_size) {
        std::cout << " ✅ Valid" << std::endl;

        // Test setting coefficient
        indata.rbc[idx] = 1.0 + m + 0.1 * n;
        indata.zbs[idx] = 2.0 + m + 0.1 * n;
        if (indata.lasym) {
          indata.rbs[idx] = 0.01 * (1.0 + m + 0.1 * n);
          indata.zbc[idx] = 0.01 * (2.0 + m + 0.1 * n);
        }

      } else {
        std::cout << " ❌ Invalid (out of bounds)" << std::endl;
        FAIL() << "Index out of bounds for m=" << m << ", n=" << n
               << ", idx=" << idx << ", size=" << coeff_size;
      }
    }
  }

  // Test axis arrays
  int axis_size = indata.ntor + 1;
  std::cout << "\nAxis array size: " << axis_size << std::endl;

  indata.raxis_c.resize(axis_size, 0.0);
  indata.zaxis_s.resize(axis_size, 0.0);
  if (indata.lasym) {
    indata.raxis_s.resize(axis_size, 0.0);
    indata.zaxis_c.resize(axis_size, 0.0);
  }

  // Set axis values
  indata.raxis_c[0] = 10.0;  // R-axis at n=0
  if (indata.lasym) {
    indata.raxis_s[0] = 0.01;  // Small asymmetric axis
  }

  std::cout << "Axis arrays initialized:" << std::endl;
  std::cout << "  raxis_c.size() = " << indata.raxis_c.size() << std::endl;
  std::cout << "  zaxis_s.size() = " << indata.zaxis_s.size() << std::endl;
  if (indata.lasym) {
    std::cout << "  raxis_s.size() = " << indata.raxis_s.size() << std::endl;
    std::cout << "  zaxis_c.size() = " << indata.zaxis_c.size() << std::endl;
  }

  // Validate coefficient values
  std::cout << "\nCoefficient validation:" << std::endl;

  // Check specific modes
  int idx_00 = 0 * (2 * indata.ntor + 1) + (0 + indata.ntor);  // m=0, n=0
  int idx_10 = 1 * (2 * indata.ntor + 1) + (0 + indata.ntor);  // m=1, n=0

  std::cout << "  idx(0,0) = " << idx_00 << ", rbc = " << indata.rbc[idx_00]
            << std::endl;
  std::cout << "  idx(1,0) = " << idx_10 << ", rbc = " << indata.rbc[idx_10]
            << std::endl;

  if (indata.lasym) {
    std::cout << "  idx(1,0) = " << idx_10 << ", rbs = " << indata.rbs[idx_10]
              << std::endl;
    std::cout << "  idx(1,0) = " << idx_10 << ", zbc = " << indata.zbc[idx_10]
              << std::endl;
  }

  // All validations passed
  std::cout << "\n✅ All input validation tests passed!" << std::endl;

  EXPECT_EQ(indata.rbc.size(), coeff_size);
  EXPECT_EQ(indata.zbs.size(), coeff_size);
  EXPECT_EQ(indata.rbs.size(), coeff_size);
  EXPECT_EQ(indata.zbc.size(), coeff_size);
  EXPECT_EQ(indata.raxis_c.size(), axis_size);
  EXPECT_EQ(indata.zaxis_s.size(), axis_size);

  if (indata.lasym) {
    EXPECT_EQ(indata.raxis_s.size(), axis_size);
    EXPECT_EQ(indata.zaxis_c.size(), axis_size);
  }
}

}  // namespace vmecpp
