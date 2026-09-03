// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/boundaries/guess_magnetic_axis.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/boundaries/boundaries.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

namespace vmecpp {

namespace {
using ::testing::TestWithParam;
using ::testing::Values;

struct AxisGuessCase {
  std::string identifier;
  int number_of_flux_surfaces;
};
}  // namespace

// The axis guess is reached from Vmec::SolveEquilibrium when the initial
// Jacobian changes sign, on whatever boundary the run was given. The set-up
// here is the solver's own: parse the input, build the basis, and let
// Boundaries fill the boundary and initial-axis coefficient arrays.
class GuessMagneticAxisTest : public TestWithParam<AxisGuessCase> {
 protected:
  void SetUp() override {
    const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromFile(
        absl::StrFormat("vmecpp/test_data/%s.json", GetParam().identifier));
    ASSERT_TRUE(indata.ok()) << indata.status();
    indata_ = *indata;

    sizes_.emplace(indata_);
    fourier_basis_.emplace(&sizes_.value());
    boundaries_.emplace(&sizes_.value(), &fourier_basis_.value(),
                        vmec_algorithm_constants::kSignOfJacobian);
    // The return value says whether theta had to be flipped, not whether the
    // set-up succeeded.
    boundaries_->setupFromIndata(indata_, /*verbose=*/false);
  }

  RecomputeAxisWorkspace Recompute() const {
    return RecomputeMagneticAxisToFixJacobianSign(
        GetParam().number_of_flux_surfaces,
        vmec_algorithm_constants::kSignOfJacobian, sizes_.value(),
        fourier_basis_.value(), boundaries_->rbcc, boundaries_->rbss,
        boundaries_->rbsc, boundaries_->rbcs, boundaries_->zbsc,
        boundaries_->zbcs, boundaries_->zbcc, boundaries_->zbss,
        boundaries_->raxis_c, boundaries_->raxis_s, boundaries_->zaxis_s,
        boundaries_->zaxis_c);
  }

  // The same boundary declared asymmetric, with every non-stellarator-symmetric
  // coefficient zero, which is the same equilibrium.
  RecomputeAxisWorkspace RecomputeAsAsymmetric() const {
    VmecINDATA asym = indata_;
    asym.lasym = true;
    asym.raxis_s = Eigen::VectorXd::Zero(asym.raxis_c.size());
    asym.zaxis_c = Eigen::VectorXd::Zero(asym.zaxis_s.size());
    asym.rbs = RowMatrixXd::Zero(asym.rbc.rows(), asym.rbc.cols());
    asym.zbc = RowMatrixXd::Zero(asym.zbs.rows(), asym.zbs.cols());

    Sizes sizes(asym);
    FourierBasisFastPoloidal fourier_basis(&sizes);
    Boundaries boundaries(&sizes, &fourier_basis,
                          vmec_algorithm_constants::kSignOfJacobian);
    boundaries.setupFromIndata(asym, /*verbose=*/false);

    return RecomputeMagneticAxisToFixJacobianSign(
        GetParam().number_of_flux_surfaces,
        vmec_algorithm_constants::kSignOfJacobian, sizes, fourier_basis,
        boundaries.rbcc, boundaries.rbss, boundaries.rbsc, boundaries.rbcs,
        boundaries.zbsc, boundaries.zbcs, boundaries.zbcc, boundaries.zbss,
        boundaries.raxis_c, boundaries.raxis_s, boundaries.zaxis_s,
        boundaries.zaxis_c);
  }

  VmecINDATA indata_;
  std::optional<Sizes> sizes_;
  std::optional<FourierBasisFastPoloidal> fourier_basis_;
  std::optional<Boundaries> boundaries_;
};

TEST_P(GuessMagneticAxisTest, CheckAxisGuessLiesInsideTheBoundary) {
  const RecomputeAxisWorkspace w = Recompute();

  for (int k = 0; k < sizes_->nZeta; ++k) {
    const double min_r =
        *std::min_element(w.r_lcfs[k].begin(), w.r_lcfs[k].end());
    const double max_r =
        *std::max_element(w.r_lcfs[k].begin(), w.r_lcfs[k].end());
    const double min_z =
        *std::min_element(w.z_lcfs[k].begin(), w.z_lcfs[k].end());
    const double max_z =
        *std::max_element(w.z_lcfs[k].begin(), w.z_lcfs[k].end());

    EXPECT_GT(w.new_r_axis[k], min_r) << "plane " << k;
    EXPECT_LT(w.new_r_axis[k], max_r) << "plane " << k;
    EXPECT_GE(w.new_z_axis[k], min_z) << "plane " << k;
    EXPECT_LE(w.new_z_axis[k], max_z) << "plane " << k;
  }  // k
}

TEST_P(GuessMagneticAxisTest, CheckStellaratorSymmetryMirrorsTheAxisGuess) {
  ASSERT_FALSE(sizes_->lasym);
  const RecomputeAxisWorkspace w = Recompute();

  // Only the planes from 0 to nZeta / 2 are searched; the rest follow from
  // zeta -> -zeta, which is index nZeta - k.
  for (int k = 1; k < sizes_->nZeta; ++k) {
    const int k_reversed = sizes_->nZeta - k;
    if (k >= k_reversed) {
      continue;
    }
    EXPECT_EQ(w.new_r_axis[k_reversed], w.new_r_axis[k]) << "plane " << k;
    EXPECT_EQ(w.new_z_axis[k_reversed], -w.new_z_axis[k]) << "plane " << k;
  }  // k

  // The zeta = 0 plane is its own mirror image, so the axis sits in the
  // symmetry plane there.
  EXPECT_EQ(w.new_z_axis[0], 0.0);
}

TEST_P(GuessMagneticAxisTest, CheckAsymmetricRunSearchesEveryPlane) {
  ASSERT_FALSE(sizes_->lasym);
  const RecomputeAxisWorkspace w = RecomputeAsAsymmetric();

  // The mirror that fills the second half of the toroidal range for a
  // symmetric run does not hold for an asymmetric one, so every plane has to
  // be searched. A plane left out keeps the axis at the origin, outside the
  // boundary.
  for (int k = 0; k < sizes_->nZeta; ++k) {
    const double min_r =
        *std::min_element(w.r_lcfs[k].begin(), w.r_lcfs[k].end());
    const double max_r =
        *std::max_element(w.r_lcfs[k].begin(), w.r_lcfs[k].end());
    const double min_z =
        *std::min_element(w.z_lcfs[k].begin(), w.z_lcfs[k].end());
    const double max_z =
        *std::max_element(w.z_lcfs[k].begin(), w.z_lcfs[k].end());

    EXPECT_GT(w.new_r_axis[k], min_r) << "plane " << k;
    EXPECT_LT(w.new_r_axis[k], max_r) << "plane " << k;
    EXPECT_GE(w.new_z_axis[k], min_z) << "plane " << k;
    EXPECT_LE(w.new_z_axis[k], max_z) << "plane " << k;
  }  // k
}

INSTANTIATE_TEST_SUITE_P(
    TestGuessMagneticAxis, GuessMagneticAxisTest,
    Values(
        // axisymmetric: ntor = 0, so nZeta = 1 and there is one plane
        AxisGuessCase{.identifier = "solovev", .number_of_flux_surfaces = 5},
        // nzeta left to the default 2 * ntor + 4, so nZeta = 16
        AxisGuessCase{.identifier = "cma", .number_of_flux_surfaces = 15},
        // even nZeta = 36, with a grid point on the half-period plane
        AxisGuessCase{.identifier = "cth_like_fixed_bdy",
                      .number_of_flux_surfaces = 15},
        // odd nZeta = 37, with no grid point on the half-period plane
        AxisGuessCase{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .number_of_flux_surfaces = 15}));

}  // namespace vmecpp
