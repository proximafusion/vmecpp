// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/sizes/sizes.h"

#include <fstream>
#include <string>

#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"

namespace vmecpp {

namespace {
using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;
}  // namespace

// Check that the Sizes setup from the JSON input file agrees with the
// corresponding parameters in educational_VMEC. The grid and mode-number counts
// are taken from the fixaray debugging output, which is where the reference
// computes them; the rest are properties of Sizes itself.
class SizesTest : public TestWithParam<std::string> {
 protected:
  void SetUp() override { identifier_ = GetParam(); }
  std::string identifier_;
};

TEST_P(SizesTest, MatchesFortranReference) {
  const double tolerance = 1.0e-30;

  const absl::StatusOr<std::string> indata_json =
      ReadFile(absl::StrFormat("vmecpp/test_data/%s.json", identifier_));
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  const Sizes sizes(*vmec_indata);

  // fixaray is dumped once, before the first iteration, so the file name is
  // fixed apart from the case identifier.
  const std::string filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/fixaray/"
      "fixaray_00000_000001_01.%s.json",
      identifier_, identifier_);
  std::ifstream ifs_fixaray(filename);
  ASSERT_TRUE(ifs_fixaray.is_open()) << filename;
  const json fixaray = json::parse(ifs_fixaray);

  // grid sizes
  EXPECT_EQ(sizes.nZeta, fixaray["nzeta"]);
  EXPECT_EQ(sizes.nThetaEff, fixaray["ntheta3"]);
  EXPECT_EQ(sizes.nZnT, fixaray["nznt"]);

  // number of Fourier coefficients, on the geometry and the Nyquist grids
  EXPECT_EQ(sizes.mnsize, fixaray["mnsize"]);
  EXPECT_EQ(sizes.mnmax, fixaray["mnmax"]);
  EXPECT_EQ(sizes.mnyq, fixaray["mnyq"]);
  EXPECT_EQ(sizes.nnyq, fixaray["nnyq"]);
  EXPECT_EQ(sizes.mnmax_nyq, fixaray["mnmax_nyq"]);

  // The remaining members are either taken straight from the input or follow
  // from it, and the reference has no separate value to compare against.
  EXPECT_EQ(sizes.lasym, vmec_indata->lasym);
  EXPECT_EQ(sizes.nfp, vmec_indata->nfp);
  EXPECT_EQ(sizes.mpol, vmec_indata->mpol);
  EXPECT_EQ(sizes.ntor, vmec_indata->ntor);

  EXPECT_EQ(sizes.lthreed, sizes.ntor > 0);
  EXPECT_EQ(sizes.num_basis, sizes.lthreed ? 2 : 1);

  if (vmec_indata->ntheta == 0) {
    // not given in the input, so VMEC picks it from mpol
    EXPECT_EQ(sizes.ntheta, 2 * sizes.mpol + 6);
  } else {
    EXPECT_EQ(sizes.ntheta, vmec_indata->ntheta);
  }
  EXPECT_EQ(sizes.nThetaEven, 2 * (sizes.ntheta / 2));
  EXPECT_EQ(sizes.nThetaReduced, sizes.nThetaEven / 2 + 1);
  EXPECT_EQ(sizes.nThetaEff,
            sizes.lasym ? sizes.nThetaEven : sizes.nThetaReduced);

  // The integration weights are uniform over the effective poloidal grid, at
  // half weight on the endpoints of a stellarator-symmetric half period, and
  // together with the toroidal grid they integrate to one.
  ASSERT_EQ(sizes.wInt.size(), sizes.nThetaEff);
  const double interior_weight = 1.0 / (sizes.nThetaEven * sizes.nZeta);
  double total_weight = 0.0;
  for (int l = 0; l < sizes.nThetaEff; ++l) {
    const bool is_endpoint =
        !sizes.lasym && (l == 0 || l == sizes.nThetaReduced - 1);
    EXPECT_TRUE(
        IsCloseRelAbs(is_endpoint ? interior_weight : 2 * interior_weight,
                      sizes.wInt[l], tolerance))
        << "l = " << l;
    total_weight += sizes.wInt[l];
  }
  EXPECT_TRUE(IsCloseRelAbs(1.0, total_weight * sizes.nZeta, 1.0e-14));
}  // MatchesFortranReference

INSTANTIATE_TEST_SUITE_P(TestSizes, SizesTest,
                         Values("solovev", "solovev_analytical",
                                "solovev_no_axis", "cth_like_fixed_bdy", "cma",
                                "cth_like_free_bdy"));

}  // namespace vmecpp
