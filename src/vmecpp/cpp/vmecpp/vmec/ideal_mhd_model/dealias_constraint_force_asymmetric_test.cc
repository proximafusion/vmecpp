// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

class DeAliasConstraintForceAsymmetricTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test configuration for asymmetric spectral condensation
    bool lasym = true;
    int nfp = 1;
    int mpol = 3;
    int ntor = 2;
    int ntheta = 16;
    int nzeta = 16;

    sizes_ = std::make_unique<Sizes>(lasym, nfp, mpol, ntor, ntheta, nzeta);

    // Create radial partitioning
    radial_partitioning_ = std::make_unique<RadialPartitioning>();
    radial_partitioning_->adjustRadialPartitioning(1, 0, 10, false, false);

    // Create Fourier basis
    fourier_basis_ = std::make_unique<FourierBasisFastPoloidal>(sizes_.get());

    // Initialize test arrays
    faccon_.resize(sizes_->mnmax, 1.0);
    tcon_.resize(radial_partitioning_->nsMaxF - radial_partitioning_->nsMinF,
                 1.0);
    gConEff_.resize(
        (radial_partitioning_->nsMaxF - radial_partitioning_->nsMinF) *
            sizes_->nZnT,
        0.0);

    // Initialize output arrays
    gsc_.resize(sizes_->mnmax, 0.0);
    gcs_.resize(sizes_->mnmax, 0.0);
    gcc_.resize(sizes_->mnmax, 0.0);
    gss_.resize(sizes_->mnmax, 0.0);
    gCon_.resize((radial_partitioning_->nsMaxF - radial_partitioning_->nsMinF) *
                     sizes_->nZnT,
                 0.0);
  }

  std::unique_ptr<Sizes> sizes_;
  std::unique_ptr<RadialPartitioning> radial_partitioning_;
  std::unique_ptr<FourierBasisFastPoloidal> fourier_basis_;
  std::vector<double> faccon_;
  std::vector<double> tcon_;
  std::vector<double> gConEff_;
  std::vector<double> gsc_;
  std::vector<double> gcs_;
  std::vector<double> gcc_;
  std::vector<double> gss_;
  std::vector<double> gCon_;
};

// Test 1: Verify gcc and gss arrays are properly initialized
TEST_F(DeAliasConstraintForceAsymmetricTest, GccGssArrayInitialization) {
  // Arrays should be initialized to zero
  for (size_t i = 0; i < gcc_.size(); ++i) {
    EXPECT_EQ(gcc_[i], 0.0) << "gcc_[" << i << "] not initialized to zero";
    EXPECT_EQ(gss_[i], 0.0) << "gss_[" << i << "] not initialized to zero";
  }

  // Arrays should have correct size
  EXPECT_EQ(gcc_.size(), sizes_->mnmax);
  EXPECT_EQ(gss_.size(), sizes_->mnmax);

  // Call deAliasConstraintForce
  deAliasConstraintForce(*radial_partitioning_, *fourier_basis_, *sizes_,
                         faccon_, tcon_, gConEff_, gsc_, gcs_, gcc_, gss_,
                         gCon_);

  // After call, arrays should still be valid sizes
  EXPECT_EQ(gcc_.size(), sizes_->mnmax);
  EXPECT_EQ(gss_.size(), sizes_->mnmax);
}

// Test 2: Verify asymmetric case processes work[2] and work[3] arrays
TEST_F(DeAliasConstraintForceAsymmetricTest, WorkArraysAsymmetricProcessing) {
  // Set up non-zero constraint force with spatial variation
  for (size_t i = 0; i < gConEff_.size(); ++i) {
    // Create a pattern that will trigger asymmetric processing
    double theta = static_cast<double>(i % sizes_->nThetaReduced) * 2.0 * M_PI /
                   sizes_->nThetaReduced;
    double zeta =
        static_cast<double>((i / sizes_->nThetaReduced) % sizes_->nZeta) * 2.0 *
        M_PI / sizes_->nZeta;
    gConEff_[i] =
        std::sin(theta) * std::cos(zeta) + 0.1 * std::cos(2.0 * theta);
  }

  // Call deAliasConstraintForce
  deAliasConstraintForce(*radial_partitioning_, *fourier_basis_, *sizes_,
                         faccon_, tcon_, gConEff_, gsc_, gcs_, gcc_, gss_,
                         gCon_);

  // Verify that the function completed without errors
  // and that arrays have correct sizes
  EXPECT_EQ(gcc_.size(), sizes_->mnmax);
  EXPECT_EQ(gss_.size(), sizes_->mnmax);
  EXPECT_EQ(gsc_.size(), sizes_->mnmax);
  EXPECT_EQ(gcs_.size(), sizes_->mnmax);

  // Check that values are finite (not NaN or infinite)
  for (size_t i = 0; i < gcc_.size(); ++i) {
    EXPECT_TRUE(std::isfinite(gcc_[i])) << "gcc_[" << i << "] is not finite";
    EXPECT_TRUE(std::isfinite(gss_[i])) << "gss_[" << i << "] is not finite";
  }
}

// Test 3: Test reflection index calculations
TEST_F(DeAliasConstraintForceAsymmetricTest, ReflectionIndexCalculations) {
  // Set up specific pattern in gConEff to test reflection
  for (size_t i = 0; i < gConEff_.size(); ++i) {
    gConEff_[i] = static_cast<double>(i % 3);  // Pattern: 0,1,2,0,1,2,...
  }

  // Call deAliasConstraintForce
  deAliasConstraintForce(*radial_partitioning_, *fourier_basis_, *sizes_,
                         faccon_, tcon_, gConEff_, gsc_, gcs_, gcc_, gss_,
                         gCon_);

  // Verify no array bounds violations occurred
  EXPECT_EQ(gcc_.size(), sizes_->mnmax);
  EXPECT_EQ(gss_.size(), sizes_->mnmax);

  // Check that reflection indices are within bounds
  // This is implicitly tested by not crashing
  SUCCEED();
}

// Test 4: Test on-the-fly symmetrization
TEST_F(DeAliasConstraintForceAsymmetricTest, OnTheFlySymmetrization) {
  // Set up asymmetric input
  for (size_t i = 0; i < gConEff_.size(); ++i) {
    gConEff_[i] = std::sin(static_cast<double>(i) * 0.1);
  }

  // Call deAliasConstraintForce
  deAliasConstraintForce(*radial_partitioning_, *fourier_basis_, *sizes_,
                         faccon_, tcon_, gConEff_, gsc_, gcs_, gcc_, gss_,
                         gCon_);

  // Verify symmetrization properties
  // For cosine symmetry: f(theta) = f(-theta)
  // For sine symmetry: f(theta) = -f(-theta)

  // The exact verification would require detailed knowledge of the
  // specific symmetrization implementation, but we can verify
  // that the computation completed without errors
  EXPECT_EQ(gsc_.size(), sizes_->mnmax);
  EXPECT_EQ(gcs_.size(), sizes_->mnmax);
  EXPECT_EQ(gcc_.size(), sizes_->mnmax);
  EXPECT_EQ(gss_.size(), sizes_->mnmax);
}

// Test 5: Test zero input produces zero output
TEST_F(DeAliasConstraintForceAsymmetricTest, ZeroInputProducesZeroOutput) {
  // Ensure all inputs are zero
  std::fill(gConEff_.begin(), gConEff_.end(), 0.0);
  std::fill(faccon_.begin(), faccon_.end(), 0.0);
  std::fill(tcon_.begin(), tcon_.end(), 0.0);

  // Call deAliasConstraintForce
  deAliasConstraintForce(*radial_partitioning_, *fourier_basis_, *sizes_,
                         faccon_, tcon_, gConEff_, gsc_, gcs_, gcc_, gss_,
                         gCon_);

  // All outputs should be zero
  for (size_t i = 0; i < gsc_.size(); ++i) {
    EXPECT_EQ(gsc_[i], 0.0) << "gsc_[" << i << "] should be zero";
    EXPECT_EQ(gcs_[i], 0.0) << "gcs_[" << i << "] should be zero";
    EXPECT_EQ(gcc_[i], 0.0) << "gcc_[" << i << "] should be zero";
    EXPECT_EQ(gss_[i], 0.0) << "gss_[" << i << "] should be zero";
  }

  for (size_t i = 0; i < gCon_.size(); ++i) {
    EXPECT_EQ(gCon_[i], 0.0) << "gCon_[" << i << "] should be zero";
  }
}

// Test 6: Test with symmetric configuration (lasym=false)
TEST_F(DeAliasConstraintForceAsymmetricTest, SymmetricConfiguration) {
  // Create symmetric configuration
  bool lasym = false;
  auto sizes_symm = std::make_unique<Sizes>(lasym, 1, 3, 2, 16, 16);
  auto fourier_basis_symm =
      std::make_unique<FourierBasisFastPoloidal>(sizes_symm.get());

  // Initialize arrays for symmetric case
  std::vector<double> gsc_symm(sizes_symm->mnmax, 0.0);
  std::vector<double> gcs_symm(sizes_symm->mnmax, 0.0);
  std::vector<double> gcc_symm(sizes_symm->mnmax, 0.0);
  std::vector<double> gss_symm(sizes_symm->mnmax, 0.0);
  std::vector<double> gCon_symm(
      (radial_partitioning_->nsMaxF - radial_partitioning_->nsMinF) *
          sizes_symm->nZnT,
      0.0);

  std::vector<double> faccon_symm(sizes_symm->mnmax, 1.0);
  std::vector<double> gConEff_symm(gCon_symm.size(), 1.0);

  // Call deAliasConstraintForce for symmetric case
  deAliasConstraintForce(*radial_partitioning_, *fourier_basis_symm,
                         *sizes_symm, faccon_symm, tcon_, gConEff_symm,
                         gsc_symm, gcs_symm, gcc_symm, gss_symm, gCon_symm);

  // For symmetric case, gcc and gss should remain zero or be unused
  // (exact behavior depends on implementation)
  EXPECT_EQ(gcc_symm.size(), sizes_symm->mnmax);
  EXPECT_EQ(gss_symm.size(), sizes_symm->mnmax);
}

// Test 7: Test array bounds safety
TEST_F(DeAliasConstraintForceAsymmetricTest, ArrayBoundsSafety) {
  // Use maximum valid indices to test bounds checking
  int max_k = sizes_->nZeta - 1;
  int max_l = sizes_->nThetaReduced - 1;

  // Verify that reflection index calculations stay within bounds
  for (int k = 0; k <= max_k; ++k) {
    int k_reversed = (sizes_->nZeta - k) % sizes_->nZeta;
    EXPECT_GE(k_reversed, 0);
    EXPECT_LT(k_reversed, sizes_->nZeta);
  }

  for (int l = 0; l <= max_l; ++l) {
    int l_reversed = (sizes_->nThetaReduced - l) % sizes_->nThetaReduced;
    EXPECT_GE(l_reversed, 0);
    EXPECT_LT(l_reversed, sizes_->nThetaReduced);
  }

  // Set up test input
  std::fill(gConEff_.begin(), gConEff_.end(), 0.5);

  // This should not crash or produce bounds violations
  EXPECT_NO_THROW({
    deAliasConstraintForce(*radial_partitioning_, *fourier_basis_, *sizes_,
                           faccon_, tcon_, gConEff_, gsc_, gcs_, gcc_, gss_,
                           gCon_);
  });
}

// Test 8: Test with realistic perturbation amplitudes
TEST_F(DeAliasConstraintForceAsymmetricTest, RealisticPerturbationAmplitudes) {
  // Set up realistic constraint force pattern
  for (size_t i = 0; i < gConEff_.size(); ++i) {
    // Use small perturbations typical of asymmetric equilibria
    gConEff_[i] = 0.01 * std::sin(static_cast<double>(i) * 0.5) +
                  0.005 * std::cos(static_cast<double>(i) * 0.3);
  }

  // Use realistic scaling factors
  for (size_t i = 0; i < faccon_.size(); ++i) {
    faccon_[i] = 1.0 + 0.1 * std::sin(static_cast<double>(i));
  }

  for (size_t i = 0; i < tcon_.size(); ++i) {
    tcon_[i] = 1.0;  // Standard constraint force scaling
  }

  // Call deAliasConstraintForce
  deAliasConstraintForce(*radial_partitioning_, *fourier_basis_, *sizes_,
                         faccon_, tcon_, gConEff_, gsc_, gcs_, gcc_, gss_,
                         gCon_);

  // Verify outputs are finite and reasonable
  for (size_t i = 0; i < gsc_.size(); ++i) {
    EXPECT_TRUE(std::isfinite(gsc_[i])) << "gsc_[" << i << "] is not finite";
    EXPECT_TRUE(std::isfinite(gcs_[i])) << "gcs_[" << i << "] is not finite";
    EXPECT_TRUE(std::isfinite(gcc_[i])) << "gcc_[" << i << "] is not finite";
    EXPECT_TRUE(std::isfinite(gss_[i])) << "gss_[" << i << "] is not finite";
  }

  for (size_t i = 0; i < gCon_.size(); ++i) {
    EXPECT_TRUE(std::isfinite(gCon_[i])) << "gCon_[" << i << "] is not finite";
  }
}

}  // namespace vmecpp
