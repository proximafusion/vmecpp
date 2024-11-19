#include "vmecpp/vmec/boundaries/boundaries.h"

#include <fstream>
#include <string>

#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/boundaries/guess_magnetic_axis.h"

namespace vmecpp {

namespace {
using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;
}  // namespace

// used to specify case-specific tolerances
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
};

class BondariesTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(BondariesTest, CheckBoundaries) {
  const double tolerance = data_source_.tolerance;

  std::string filename = absl::StrFormat(
      "vmecpp/test_data/%s/readin_boundary/"
      "readin_boundary_00000_000001_01.%s.json",
      data_source_.identifier, data_source_.identifier);
  std::ifstream ifs(filename);
  ASSERT_TRUE(ifs.is_open());
  json readin_boundary = json::parse(ifs);

  filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Sizes sizes(*vmec_indata);
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // VMEC convention:
  // The flux coordiantes (s, u, v) form a left-handed coordinate system.
  // This implies that the Jacobian of the coordinate system is negative.
  static constexpr int kSignOfJacobian = -1;

  Boundaries boundaries(&sizes, &fourier_basis, kSignOfJacobian);
  boundaries.setupFromIndata(*vmec_indata);

  for (int m = 0; m < sizes.mpol; ++m) {
    for (int n = 0; n <= sizes.ntor; ++n) {
      int idx_mn = m * (sizes.ntor + 1) + n;

      EXPECT_TRUE(IsCloseRelAbs(readin_boundary["rbcc"][n][m],
                                boundaries.rbcc[idx_mn], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(readin_boundary["zbsc"][n][m],
                                boundaries.zbsc[idx_mn], tolerance));
    }  // m
  }    // n
}  // CheckBoundaries

INSTANTIATE_TEST_SUITE_P(
    TestBoundaries, BondariesTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-30},
           DataSource{.identifier = "solovev_analytical", .tolerance = 1.0e-30},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-30},
           DataSource{.identifier = "cma", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-30}));

class RecomputeMagneticAxisToFixJacobianSignTest
    : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(RecomputeMagneticAxisToFixJacobianSignTest,
       CheckRecomputeMagneticAxisToFixJacobianSign) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // guess_axis is only called in the first multi-grid step
  const int number_of_flux_surfaces = vmec_indata->ns_array[0];

  Sizes sizes(*vmec_indata);
  FourierBasisFastPoloidal fourier_basis(&sizes);

  // VMEC convention:
  // The flux coordiantes (s, u, v) form a left-handed coordinate system.
  // This implies that the Jacobian of the coordinate system is negative.
  static constexpr int kSignOfJacobian = -1;

  Boundaries boundaries(&sizes, &fourier_basis, kSignOfJacobian);
  boundaries.setupFromIndata(*vmec_indata);

  // Assume that RecomputeMagneticAxisToFixJacobianSign will be triggered for
  // this test case, so directly proceed to testing it here.
  const RecomputeAxisWorkspace w =
      vmecpp::RecomputeMagneticAxisToFixJacobianSign(
          number_of_flux_surfaces, kSignOfJacobian, sizes, fourier_basis,
          boundaries.rbcc, boundaries.rbss, boundaries.rbsc, boundaries.rbcs,
          boundaries.zbsc, boundaries.zbcs, boundaries.zbcc, boundaries.zbss,
          boundaries.raxis_c, boundaries.raxis_s, boundaries.zaxis_s,
          boundaries.zaxis_c);

  filename = absl::StrFormat(
      "vmecpp/test_data/%s/guess_axis/guess_axis_%05d_000001_01.%s.json",
      data_source_.identifier, number_of_flux_surfaces,
      data_source_.identifier);
  std::ifstream ifs_guess_axis(filename);
  ASSERT_TRUE(ifs_guess_axis.is_open());
  json guess_axis = json::parse(ifs_guess_axis);

  // Now check that correct axis was guessed.
  // First check intermediate data.
  for (int k = 0; k < sizes.nZeta / 2 + 1; ++k) {
    EXPECT_TRUE(
        IsCloseRelAbs(guess_axis["raxis_in"][k], w.r_axis[k], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(guess_axis["zaxis_in"][k], w.z_axis[k], tolerance));
    for (int l = 0; l < sizes.nThetaEven; ++l) {
      EXPECT_TRUE(
          IsCloseRelAbs(guess_axis["r1b"][k][l], w.r_lcfs[k][l], tolerance));
      EXPECT_TRUE(
          IsCloseRelAbs(guess_axis["z1b"][k][l], w.z_lcfs[k][l], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(guess_axis["rub"][k][l],
                                w.d_r_d_theta_lcfs[k][l], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(guess_axis["zub"][k][l],
                                w.d_z_d_theta_lcfs[k][l], tolerance));

      EXPECT_TRUE(
          IsCloseRelAbs(guess_axis["r12"][k][l], w.r_half[k][l], tolerance));
      EXPECT_TRUE(
          IsCloseRelAbs(guess_axis["z12"][k][l], w.z_half[k][l], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(guess_axis["ru12"][k][l],
                                w.d_r_d_theta_half[k][l], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(guess_axis["zu12"][k][l],
                                w.d_z_d_theta_half[k][l], tolerance));

      EXPECT_TRUE(IsCloseRelAbs(guess_axis["rs"][k][l], w.d_r_d_s_half[k][l],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(guess_axis["zs"][k][l], w.d_z_d_s_half[k][l],
                                tolerance));

      EXPECT_TRUE(
          IsCloseRelAbs(guess_axis["tau0"][k][l], w.tau0[k][l], tolerance));

      EXPECT_TRUE(
          IsCloseRelAbs(guess_axis["tau"][k][l], w.tau[k][l], tolerance));
    }  // l
  }    // k

  // Now check the realspace geometry of the new axis.
  for (int k = 0; k < sizes.nZeta; ++k) {
    EXPECT_TRUE(
        IsCloseRelAbs(guess_axis["rcom"][k], w.new_r_axis[k], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(guess_axis["zcom"][k], w.new_z_axis[k], tolerance));
  }  // k

  // Now check the Fourier coefficients of the new axis.
  for (int n = 0; n <= sizes.ntor; ++n) {
    EXPECT_TRUE(
        IsCloseRelAbs(guess_axis["raxis_cc"][n], w.new_raxis_c[n], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(guess_axis["zaxis_cs"][n], w.new_zaxis_s[n], tolerance));
    if (sizes.lasym) {
      EXPECT_TRUE(IsCloseRelAbs(guess_axis["raxis_cs"][n], w.new_raxis_s[n],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(guess_axis["zaxis_cc"][n], w.new_zaxis_c[n],
                                tolerance));
    }
  }  // n
}  // CheckRecomputeMagneticAxisToFixJacobianSign

INSTANTIATE_TEST_SUITE_P(
    TestBoundaries, RecomputeMagneticAxisToFixJacobianSignTest,
    Values(DataSource{.identifier = "solovev_analytical", .tolerance = 1.0e-14},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-15},
           DataSource{.identifier = "cma", .tolerance = 1.0e-15}));

}  // namespace vmecpp
