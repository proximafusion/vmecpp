// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

namespace {
const double PI = 3.14159265358979323846;
const double TOLERANCE = 1e-10;
}  // namespace

class FourierTransformUnitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Standard test configuration
    lasym_ = true;
    nfp_ = 1;
    mpol_ = 3;  // m=0,1,2
    ntor_ = 2;  // n=-2,-1,0,1,2
    ntheta_ = 16;
    nzeta_ = 8;

    sizes_ =
        std::make_unique<Sizes>(lasym_, nfp_, mpol_, ntor_, ntheta_, nzeta_);

    // Initialize coefficient arrays
    InitializeArrays();
  }

  void InitializeArrays() {
    const int mnmax = sizes_->mnmax;
    const int nznt = sizes_->nZnT;

    // Fourier coefficient arrays
    rmncc_.assign(mnmax, 0.0);
    rmnss_.assign(mnmax, 0.0);
    rmnsc_.assign(mnmax, 0.0);
    rmncs_.assign(mnmax, 0.0);
    zmnsc_.assign(mnmax, 0.0);
    zmncs_.assign(mnmax, 0.0);
    zmncc_.assign(mnmax, 0.0);
    zmnss_.assign(mnmax, 0.0);
    lmnsc_.assign(mnmax, 0.0);
    lmncs_.assign(mnmax, 0.0);
    lmncc_.assign(mnmax, 0.0);
    lmnss_.assign(mnmax, 0.0);

    // Real space arrays
    r_real_.assign(nznt, 0.0);
    z_real_.assign(nznt, 0.0);
    lambda_real_.assign(nznt, 0.0);
  }

  int FindModeIndex(int m, int n) {
    FourierBasisFastPoloidal fourier_basis(sizes_.get());
    for (int mn = 0; mn < sizes_->mnmax; ++mn) {
      if (fourier_basis.xm[mn] == m &&
          fourier_basis.xn[mn] / sizes_->nfp == n) {
        return mn;
      }
    }
    return -1;
  }

 protected:
  bool lasym_;
  int nfp_, mpol_, ntor_, ntheta_, nzeta_;
  std::unique_ptr<Sizes> sizes_;

  std::vector<double> rmncc_, rmnss_, rmnsc_, rmncs_;
  std::vector<double> zmnsc_, zmncs_, zmncc_, zmnss_;
  std::vector<double> lmnsc_, lmncs_, lmncc_, lmnss_;
  std::vector<double> r_real_, z_real_, lambda_real_;
};

TEST_F(FourierTransformUnitTest, TestConstantMode) {
  // Test m=0, n=0 (constant) mode in isolation
  int mn = FindModeIndex(0, 0);
  ASSERT_GE(mn, 0) << "Mode (0,0) not found";

  // Set only constant coefficient
  rmncc_[mn] = 2.0;

  // Forward transform
  FourierToReal3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(rmncc_), absl::MakeSpan(rmnss_),
      absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_), absl::MakeSpan(zmnsc_),
      absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_), absl::MakeSpan(zmnss_),
      absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_));

  // Check all points should equal the constant
  for (int i = 0; i < sizes_->nZnT; ++i) {
    EXPECT_NEAR(r_real_[i], 2.0, TOLERANCE)
        << "Constant mode failed at index " << i;
    EXPECT_NEAR(z_real_[i], 0.0, TOLERANCE)
        << "Z should be zero for R-only constant mode at index " << i;
  }
}

TEST_F(FourierTransformUnitTest, TestSingleCosineMode) {
  // Test m=1, n=0 mode: R = cos(u)
  int mn = FindModeIndex(1, 0);
  ASSERT_GE(mn, 0) << "Mode (1,0) not found";

  // Set only rmncc coefficient for cos(u)
  rmncc_[mn] = 1.0;

  // Forward transform
  FourierToReal3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(rmncc_), absl::MakeSpan(rmnss_),
      absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_), absl::MakeSpan(zmnsc_),
      absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_), absl::MakeSpan(zmnss_),
      absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_));

  // Check specific angles
  for (int i = 0; i < sizes_->nThetaEff; ++i) {
    double u = 2.0 * PI * i / sizes_->nThetaEff;
    int idx = i * sizes_->nZeta;  // k=0

    // Expected: cos(u) with normalization factor
    double expected_r = cos(u);
    if (mn > 0) expected_r *= sqrt(2.0);  // m>0 normalization

    EXPECT_NEAR(r_real_[idx], expected_r, TOLERANCE)
        << "cos(u) mode failed at theta index " << i << ", u=" << u
        << ", expected=" << expected_r;
  }
}

TEST_F(FourierTransformUnitTest, TestSingleSineMode) {
  // Test m=1, n=0 mode: Z = sin(u)
  int mn = FindModeIndex(1, 0);
  ASSERT_GE(mn, 0) << "Mode (1,0) not found";

  // Set only zmnsc coefficient for sin(u)
  zmnsc_[mn] = 1.0;

  // Forward transform
  FourierToReal3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(rmncc_), absl::MakeSpan(rmnss_),
      absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_), absl::MakeSpan(zmnsc_),
      absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_), absl::MakeSpan(zmnss_),
      absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_));

  // Check specific angles
  for (int i = 0; i < sizes_->nThetaEff; ++i) {
    double u = 2.0 * PI * i / sizes_->nThetaEff;
    int idx = i * sizes_->nZeta;  // k=0

    // Expected: sin(u) with normalization factor
    double expected_z = sin(u);
    if (mn > 0) expected_z *= sqrt(2.0);  // m>0 normalization

    EXPECT_NEAR(z_real_[idx], expected_z, TOLERANCE)
        << "sin(u) mode failed at theta index " << i << ", u=" << u
        << ", expected=" << expected_z;
  }
}

TEST_F(FourierTransformUnitTest, TestAsymmetricSineMode) {
  // Test asymmetric m=1, n=0 mode: R = sin(u)
  int mn = FindModeIndex(1, 0);
  ASSERT_GE(mn, 0) << "Mode (1,0) not found";

  // Set only rmnsc coefficient for asymmetric sin(u)
  rmnsc_[mn] = 1.0;

  // Forward transform
  FourierToReal3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(rmncc_), absl::MakeSpan(rmnss_),
      absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_), absl::MakeSpan(zmnsc_),
      absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_), absl::MakeSpan(zmnss_),
      absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_));

  // Check specific angles
  for (int i = 0; i < sizes_->nThetaEff; ++i) {
    double u = 2.0 * PI * i / sizes_->nThetaEff;
    int idx = i * sizes_->nZeta;  // k=0

    // Expected: sin(u) with normalization factor
    double expected_r = sin(u);
    if (mn > 0) expected_r *= sqrt(2.0);  // m>0 normalization

    EXPECT_NEAR(r_real_[idx], expected_r, TOLERANCE)
        << "asymmetric sin(u) mode failed at theta index " << i << ", u=" << u
        << ", expected=" << expected_r;
  }
}

// NOTE: TestNegativeNMode removed - negative toroidal modes are not used
// in VMEC due to 2D half-sided Fourier series compression

TEST_F(FourierTransformUnitTest, TestInverseTransformConstant) {
  // Test inverse transform of constant field
  for (int i = 0; i < sizes_->nZnT; ++i) {
    r_real_[i] = 3.5;  // Constant value
  }

  // Inverse transform
  RealToFourier3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_), absl::MakeSpan(rmncc_),
      absl::MakeSpan(rmnss_), absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_),
      absl::MakeSpan(zmnsc_), absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_),
      absl::MakeSpan(zmnss_), absl::MakeSpan(lmnsc_), absl::MakeSpan(lmncs_),
      absl::MakeSpan(lmncc_), absl::MakeSpan(lmnss_));

  // Check only (0,0) mode should be non-zero
  int mn00 = FindModeIndex(0, 0);
  ASSERT_GE(mn00, 0);

  EXPECT_NEAR(rmncc_[mn00], 3.5, TOLERANCE)
      << "Constant inverse transform failed";

  // All other modes should be zero
  for (int mn = 0; mn < sizes_->mnmax; ++mn) {
    if (mn != mn00) {
      EXPECT_NEAR(rmncc_[mn], 0.0, TOLERANCE)
          << "Non-constant mode " << mn << " should be zero";
    }
  }
}

TEST_F(FourierTransformUnitTest, TestRoundTripConstant) {
  // Test round-trip for constant mode
  int mn = FindModeIndex(0, 0);
  ASSERT_GE(mn, 0);

  double original_value = 1.7;
  rmncc_[mn] = original_value;

  // Forward transform
  FourierToReal3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(rmncc_), absl::MakeSpan(rmnss_),
      absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_), absl::MakeSpan(zmnsc_),
      absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_), absl::MakeSpan(zmnss_),
      absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_));

  // Reset coefficients
  std::fill(rmncc_.begin(), rmncc_.end(), 0.0);

  // Inverse transform
  RealToFourier3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_), absl::MakeSpan(rmncc_),
      absl::MakeSpan(rmnss_), absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_),
      absl::MakeSpan(zmnsc_), absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_),
      absl::MakeSpan(zmnss_), absl::MakeSpan(lmnsc_), absl::MakeSpan(lmncs_),
      absl::MakeSpan(lmncc_), absl::MakeSpan(lmnss_));

  // Check round-trip accuracy
  EXPECT_NEAR(rmncc_[mn], original_value, TOLERANCE)
      << "Round-trip failed for constant mode";
}

TEST_F(FourierTransformUnitTest, TestRoundTripSingleCosine) {
  // Test round-trip for single cosine mode
  int mn = FindModeIndex(1, 0);
  ASSERT_GE(mn, 0);

  double original_value = 0.8;
  rmncc_[mn] = original_value;

  // Forward transform
  FourierToReal3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(rmncc_), absl::MakeSpan(rmnss_),
      absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_), absl::MakeSpan(zmnsc_),
      absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_), absl::MakeSpan(zmnss_),
      absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_));

  // Reset coefficients
  std::fill(rmncc_.begin(), rmncc_.end(), 0.0);

  // Inverse transform
  RealToFourier3DAsymmFastPoloidal(
      *sizes_, absl::MakeSpan(r_real_), absl::MakeSpan(z_real_),
      absl::MakeSpan(lambda_real_), absl::MakeSpan(rmncc_),
      absl::MakeSpan(rmnss_), absl::MakeSpan(rmnsc_), absl::MakeSpan(rmncs_),
      absl::MakeSpan(zmnsc_), absl::MakeSpan(zmncs_), absl::MakeSpan(zmncc_),
      absl::MakeSpan(zmnss_), absl::MakeSpan(lmnsc_), absl::MakeSpan(lmncs_),
      absl::MakeSpan(lmncc_), absl::MakeSpan(lmnss_));

  // Check round-trip accuracy
  EXPECT_NEAR(rmncc_[mn], original_value, TOLERANCE)
      << "Round-trip failed for cosine mode";
}

}  // namespace vmecpp
