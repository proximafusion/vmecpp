// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/boundaries/guess_magnetic_axis.h"

#include <algorithm>
#include <cmath>
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

// The routine evaluates the boundary against a poloidal basis tabulated on
// [0, pi] and assembles the rest of the interval from parity. Evaluating the
// series directly is independent of that assembly. cosmu and cosnv carry the
// mscale and nscale factors that basis_norm divides out again, so what is left
// is the plain series.
void ExpectGeometryMatchesTheFourierSeries(const Sizes& s, const Boundaries& b,
                                           const RecomputeAxisWorkspace& w) {
  static constexpr double kTolerance = 1.0e-12;
  static constexpr double kTwoPi = 2.0 * M_PI;

  for (int k = 0; k < s.nZeta; ++k) {
    for (int l = 0; l < s.nThetaEven; ++l) {
      const double theta = kTwoPi * l / s.nThetaEven;

      double r = 0.0;
      double z = 0.0;
      double d_r_d_theta = 0.0;
      double d_z_d_theta = 0.0;
      for (int m = 0; m < s.mpol; ++m) {
        const double cos_m_theta = std::cos(m * theta);
        const double sin_m_theta = std::sin(m * theta);
        for (int n = 0; n <= s.ntor; ++n) {
          const int idx_mn = m * (s.ntor + 1) + n;
          const double cos_n_zeta = std::cos(n * kTwoPi * k / s.nZeta);
          const double sin_n_zeta = std::sin(n * kTwoPi * k / s.nZeta);

          r += b.rbcc[idx_mn] * cos_m_theta * cos_n_zeta;
          z += b.zbsc[idx_mn] * sin_m_theta * cos_n_zeta;
          d_r_d_theta -= m * b.rbcc[idx_mn] * sin_m_theta * cos_n_zeta;
          d_z_d_theta += m * b.zbsc[idx_mn] * cos_m_theta * cos_n_zeta;
          if (s.lthreed) {
            // Boundaries allocates these only for lthreed, and the m = 1
            // constraint is undone before the transform.
            const double rss =
                m == 1 ? b.rbss[idx_mn] + b.zbcs[idx_mn] : b.rbss[idx_mn];
            const double zcs =
                m == 1 ? b.rbss[idx_mn] - b.zbcs[idx_mn] : b.zbcs[idx_mn];
            r += rss * sin_m_theta * sin_n_zeta;
            z += zcs * cos_m_theta * sin_n_zeta;
            d_r_d_theta += m * rss * cos_m_theta * sin_n_zeta;
            d_z_d_theta -= m * zcs * sin_m_theta * sin_n_zeta;
          }
          if (s.lasym) {
            // likewise allocated only for lasym
            const double rsc =
                m == 1 ? b.rbsc[idx_mn] + b.zbcc[idx_mn] : b.rbsc[idx_mn];
            const double zcc =
                m == 1 ? b.rbsc[idx_mn] - b.zbcc[idx_mn] : b.zbcc[idx_mn];
            r += rsc * sin_m_theta * cos_n_zeta;
            z += zcc * cos_m_theta * cos_n_zeta;
            d_r_d_theta += m * rsc * cos_m_theta * cos_n_zeta;
            d_z_d_theta -= m * zcc * sin_m_theta * cos_n_zeta;
            if (s.lthreed) {
              r += b.rbcs[idx_mn] * cos_m_theta * sin_n_zeta;
              z += b.zbss[idx_mn] * sin_m_theta * sin_n_zeta;
              d_r_d_theta -= m * b.rbcs[idx_mn] * sin_m_theta * sin_n_zeta;
              d_z_d_theta += m * b.zbss[idx_mn] * cos_m_theta * sin_n_zeta;
            }
          }
        }  // n
      }  // m

      EXPECT_NEAR(w.r_lcfs[k][l], r, kTolerance)
          << "plane " << k << ", l " << l;
      EXPECT_NEAR(w.z_lcfs[k][l], z, kTolerance)
          << "plane " << k << ", l " << l;
      EXPECT_NEAR(w.d_r_d_theta_lcfs[k][l], d_r_d_theta, kTolerance)
          << "plane " << k << ", l " << l;
      EXPECT_NEAR(w.d_z_d_theta_lcfs[k][l], d_z_d_theta, kTolerance)
          << "plane " << k << ", l " << l;
    }  // l
  }  // k
}
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
  if (sizes_->lasym) {
    GTEST_SKIP() << "there is no symmetry to mirror across";
  }
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

TEST_P(GuessMagneticAxisTest, CheckBoundaryGeometryMatchesTheFourierSeries) {
  ExpectGeometryMatchesTheFourierSeries(sizes_.value(), boundaries_.value(),
                                        Recompute());
}

TEST_P(GuessMagneticAxisTest, CheckGeometryMatchesTheSeriesWithBothParities) {
  if (sizes_->lasym) {
    GTEST_SKIP() << "the input is already asymmetric";
  }

  // The bundled asymmetric inputs carry antisymmetric content in Z only, which
  // leaves the R side of the assembly unexercised. Give the boundary a
  // deterministic antisymmetric part in both, so that all four coefficient
  // families contribute.
  VmecINDATA indata = indata_;
  indata.lasym = true;
  indata.raxis_s = Eigen::VectorXd::Zero(indata_.raxis_c.size());
  indata.zaxis_c = Eigen::VectorXd::Zero(indata_.zaxis_s.size());
  RowMatrixXd rbs = RowMatrixXd::Zero(indata_.rbc.rows(), indata_.rbc.cols());
  RowMatrixXd zbc = RowMatrixXd::Zero(indata_.zbs.rows(), indata_.zbs.cols());
  for (int row = 0; row < rbs.rows(); ++row) {
    for (int col = 0; col < rbs.cols(); ++col) {
      rbs(row, col) = 1.0e-3 * std::cos(1.0 + row + 3.0 * col);
      zbc(row, col) = 1.0e-3 * std::sin(2.0 + row + 5.0 * col);
    }  // col
  }  // row
  indata.rbs = rbs;
  indata.zbc = zbc;

  const Sizes sizes(indata);
  const FourierBasisFastPoloidal fourier_basis(&sizes);
  Boundaries boundaries(&sizes, &fourier_basis,
                        vmec_algorithm_constants::kSignOfJacobian);
  boundaries.setupFromIndata(indata, /*verbose=*/false);

  const RecomputeAxisWorkspace w = RecomputeMagneticAxisToFixJacobianSign(
      GetParam().number_of_flux_surfaces,
      vmec_algorithm_constants::kSignOfJacobian, sizes, fourier_basis,
      boundaries.rbcc, boundaries.rbss, boundaries.rbsc, boundaries.rbcs,
      boundaries.zbsc, boundaries.zbcs, boundaries.zbcc, boundaries.zbss,
      boundaries.raxis_c, boundaries.raxis_s, boundaries.zaxis_s,
      boundaries.zaxis_c);

  ExpectGeometryMatchesTheFourierSeries(sizes, boundaries, w);
}

TEST_P(GuessMagneticAxisTest, CheckTheAsymmetricPathReproducesTheSymmetricOne) {
  if (sizes_->lasym) {
    GTEST_SKIP() << "the input is already asymmetric";
  }
  const RecomputeAxisWorkspace symmetric = Recompute();

  // A stellarator-symmetric boundary carries no antisymmetric content, so
  // running it through the lasym path has to land on the same axis. The two
  // paths share no shortcut: with symmetry the geometry is evaluated on the
  // reduced poloidal interval and mirrored, and only the planes up to the half
  // period are searched; without it both cover their full range.
  VmecINDATA asymmetric_indata = indata_;
  asymmetric_indata.lasym = true;
  asymmetric_indata.raxis_s = Eigen::VectorXd::Zero(indata_.raxis_c.size());
  asymmetric_indata.zaxis_c = Eigen::VectorXd::Zero(indata_.zaxis_s.size());
  asymmetric_indata.rbs =
      RowMatrixXd::Zero(indata_.rbc.rows(), indata_.rbc.cols());
  asymmetric_indata.zbc =
      RowMatrixXd::Zero(indata_.zbs.rows(), indata_.zbs.cols());

  const Sizes asymmetric_sizes(asymmetric_indata);
  const FourierBasisFastPoloidal asymmetric_basis(&asymmetric_sizes);
  Boundaries asymmetric_boundaries(&asymmetric_sizes, &asymmetric_basis,
                                   vmec_algorithm_constants::kSignOfJacobian);
  asymmetric_boundaries.setupFromIndata(asymmetric_indata, /*verbose=*/false);

  const RecomputeAxisWorkspace asymmetric =
      RecomputeMagneticAxisToFixJacobianSign(
          GetParam().number_of_flux_surfaces,
          vmec_algorithm_constants::kSignOfJacobian, asymmetric_sizes,
          asymmetric_basis, asymmetric_boundaries.rbcc,
          asymmetric_boundaries.rbss, asymmetric_boundaries.rbsc,
          asymmetric_boundaries.rbcs, asymmetric_boundaries.zbsc,
          asymmetric_boundaries.zbcs, asymmetric_boundaries.zbcc,
          asymmetric_boundaries.zbss, asymmetric_boundaries.raxis_c,
          asymmetric_boundaries.raxis_s, asymmetric_boundaries.zaxis_s,
          asymmetric_boundaries.zaxis_c);

  static constexpr double kTolerance = 1.0e-10;
  for (int k = 0; k < sizes_->nZeta; ++k) {
    EXPECT_NEAR(asymmetric.new_r_axis[k], symmetric.new_r_axis[k], kTolerance)
        << "plane " << k;
    EXPECT_NEAR(asymmetric.new_z_axis[k], symmetric.new_z_axis[k], kTolerance)
        << "plane " << k;
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
                      .number_of_flux_surfaces = 15},
        // a genuinely asymmetric boundary, where every plane is independent
        AxisGuessCase{.identifier = "cth_like_free_bdy_asym",
                      .number_of_flux_surfaces = 15}));

}  // namespace vmecpp
