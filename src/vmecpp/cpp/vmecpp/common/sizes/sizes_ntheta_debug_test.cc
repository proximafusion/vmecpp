// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iostream>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

class SizesNthetaDebugTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// Test 1: Verify Nyquist correction works for direct Sizes constructor
TEST_F(SizesNthetaDebugTest, DirectSizesConstructorNyquist) {
  std::cout << "=== TESTING DIRECT SIZES CONSTRUCTOR ===" << std::endl;

  // Test with ntheta=0, should be corrected to 2*mpol+6
  bool lasym = true;
  int nfp = 1;
  int mpol = 5;
  int ntor = 4;
  int ntheta = 0;  // This should be corrected
  int nzeta = 36;

  int expected_ntheta = 2 * mpol + 6;  // 2*5+6 = 16

  std::cout << "BEFORE: ntheta=" << ntheta << ", expected=" << expected_ntheta
            << std::endl;

  // Create Sizes object - this should trigger Nyquist correction
  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "AFTER: sizes.ntheta=" << sizes.ntheta << std::endl;
  std::cout << "sizes.nThetaEven=" << sizes.nThetaEven << std::endl;
  std::cout << "sizes.nThetaReduced=" << sizes.nThetaReduced << std::endl;
  std::cout << "sizes.nThetaEff=" << sizes.nThetaEff << std::endl;

  // This should pass if Nyquist correction works
  EXPECT_GE(sizes.ntheta, expected_ntheta)
      << "Nyquist correction should set ntheta >= " << expected_ntheta;
  EXPECT_GT(sizes.nThetaEff, 0)
      << "nThetaEff should be > 0 for asymmetric case";

  std::cout << "=== DIRECT CONSTRUCTOR TEST COMPLETED ===" << std::endl;
}

// Test 2: Test with VmecINDATA constructor
TEST_F(SizesNthetaDebugTest, VmecINDATAConstructorNyquist) {
  std::cout << "=== TESTING VMEC INDATA CONSTRUCTOR ===" << std::endl;

  // Create VmecINDATA with ntheta=0
  VmecINDATA indata;
  indata.lasym = true;
  indata.nfp = 1;
  indata.mpol = 5;
  indata.ntor = 4;
  indata.ntheta = 0;  // This should be corrected
  indata.nzeta = 36;
  indata.ns_array = {10};
  indata.ftol_array = {1e-6};
  indata.niter_array = {100};

  int expected_ntheta = 2 * indata.mpol + 6;  // 2*5+6 = 16

  std::cout << "BEFORE: indata.ntheta=" << indata.ntheta
            << ", expected=" << expected_ntheta << std::endl;

  // Create Sizes from INDATA - this should trigger Nyquist correction
  Sizes sizes(indata);

  std::cout << "AFTER: sizes.ntheta=" << sizes.ntheta << std::endl;
  std::cout << "sizes.nThetaEven=" << sizes.nThetaEven << std::endl;
  std::cout << "sizes.nThetaReduced=" << sizes.nThetaReduced << std::endl;
  std::cout << "sizes.nThetaEff=" << sizes.nThetaEff << std::endl;

  // This should pass if Nyquist correction works
  EXPECT_GE(sizes.ntheta, expected_ntheta)
      << "Nyquist correction should set ntheta >= " << expected_ntheta;
  EXPECT_GT(sizes.nThetaEff, 0)
      << "nThetaEff should be > 0 for asymmetric case";

  // Also check if the original INDATA was modified
  std::cout << "INDATA after Sizes creation: indata.ntheta=" << indata.ntheta
            << std::endl;

  std::cout << "=== INDATA CONSTRUCTOR TEST COMPLETED ===" << std::endl;
}

// Test 3: Test different mpol values to verify Nyquist formula
TEST_F(SizesNthetaDebugTest, NyquistFormulaVerification) {
  std::cout << "=== TESTING NYQUIST FORMULA FOR DIFFERENT MPOL ==="
            << std::endl;

  for (int mpol = 1; mpol <= 6; ++mpol) {
    int expected_ntheta = 2 * mpol + 6;

    std::cout << "Testing mpol=" << mpol
              << ", expected ntheta=" << expected_ntheta << std::endl;

    // Test with ntheta=0 (should be corrected)
    Sizes sizes(true, 1, mpol, 2, 0, 16);

    std::cout << "  Result: sizes.ntheta=" << sizes.ntheta
              << ", nThetaEff=" << sizes.nThetaEff << std::endl;

    EXPECT_GE(sizes.ntheta, expected_ntheta)
        << "For mpol=" << mpol << ", ntheta should be >= " << expected_ntheta;
    EXPECT_GT(sizes.nThetaEff, 0)
        << "For mpol=" << mpol << ", nThetaEff should be > 0";
  }

  std::cout << "=== NYQUIST FORMULA TEST COMPLETED ===" << std::endl;
}

// Test 4: Test symmetric vs asymmetric behavior
TEST_F(SizesNthetaDebugTest, SymmetricVsAsymmetricBehavior) {
  std::cout << "=== TESTING SYMMETRIC VS ASYMMETRIC BEHAVIOR ===" << std::endl;

  int mpol = 5;
  int expected_ntheta = 2 * mpol + 6;  // 16

  // Test symmetric case
  std::cout << "SYMMETRIC case (lasym=false):" << std::endl;
  Sizes sizes_sym(false, 1, mpol, 4, 0, 36);
  std::cout << "  ntheta=" << sizes_sym.ntheta
            << ", nThetaEff=" << sizes_sym.nThetaEff
            << ", nThetaReduced=" << sizes_sym.nThetaReduced << std::endl;

  // Test asymmetric case
  std::cout << "ASYMMETRIC case (lasym=true):" << std::endl;
  Sizes sizes_asym(true, 1, mpol, 4, 0, 36);
  std::cout << "  ntheta=" << sizes_asym.ntheta
            << ", nThetaEff=" << sizes_asym.nThetaEff
            << ", nThetaReduced=" << sizes_asym.nThetaReduced << std::endl;

  // Both should have corrected ntheta
  EXPECT_GE(sizes_sym.ntheta, expected_ntheta);
  EXPECT_GE(sizes_asym.ntheta, expected_ntheta);

  // But nThetaEff should differ
  EXPECT_EQ(sizes_sym.nThetaEff, sizes_sym.nThetaReduced)
      << "Symmetric case should use nThetaReduced";
  EXPECT_EQ(sizes_asym.nThetaEff, sizes_asym.nThetaEven)
      << "Asymmetric case should use nThetaEven";
  EXPECT_GT(sizes_asym.nThetaEff, sizes_sym.nThetaEff)
      << "Asymmetric should need more theta points";

  std::cout << "=== SYMMETRIC VS ASYMMETRIC TEST COMPLETED ===" << std::endl;
}

}  // namespace vmecpp
