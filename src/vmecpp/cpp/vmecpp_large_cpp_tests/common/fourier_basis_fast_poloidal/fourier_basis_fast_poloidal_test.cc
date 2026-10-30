// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"

#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

namespace {
using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;
}  // namespace

struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
};

class FourierBasisFastPoloidalTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourierBasisFastPoloidalTest, CheckFourierBasisFastPoloidal) {
  const double kTolerance = data_source_.tolerance;

  std::string filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/fixaray/"
      "fixaray_00000_000001_01.%s.json",
      data_source_.identifier, data_source_.identifier);
  std::ifstream ifs(filename);
  ASSERT_TRUE(ifs.is_open());
  json fixaray = json::parse(ifs);

  filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Sizes sizes(*vmec_indata);

  EXPECT_EQ(fixaray["ntheta3"], sizes.nThetaEff);
  EXPECT_EQ(fixaray["mnyq"], sizes.mnyq);
  EXPECT_EQ(fixaray["nzeta"], sizes.nZeta);
  EXPECT_EQ(fixaray["nnyq"], sizes.nnyq);
  EXPECT_EQ(fixaray["nznt"], sizes.nZnT);
  EXPECT_EQ(fixaray["mnmax"], sizes.mnmax);
  EXPECT_EQ(fixaray["mnsize"], sizes.mnsize);
  EXPECT_EQ(fixaray["mnmax_nyq"], sizes.mnmax_nyq);

  FourierBasisFastPoloidal fourier_basis(&sizes);

  for (int m = 0; m < sizes.mnyq2 + 1; ++m) {
    EXPECT_TRUE(IsCloseRelAbs(fixaray["mscale"][m], fourier_basis.mscale[m],
                              kTolerance));

    for (int l = 0; l < sizes.nThetaReduced; ++l) {
      int idx_ml = m * sizes.nThetaReduced + l;

      EXPECT_TRUE(IsCloseRelAbs(fixaray["cosmu"][l][m],
                                fourier_basis.cosmu[idx_ml], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(fixaray["sinmu"][l][m],
                                fourier_basis.sinmu[idx_ml], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(fixaray["cosmui"][l][m],
                                fourier_basis.cosmui[idx_ml], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(fixaray["sinmui"][l][m],
                                fourier_basis.sinmui[idx_ml], kTolerance));

      EXPECT_TRUE(IsCloseRelAbs(fixaray["cosmum"][l][m],
                                fourier_basis.cosmum[idx_ml], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(fixaray["sinmum"][l][m],
                                fourier_basis.sinmum[idx_ml], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(fixaray["cosmumi"][l][m],
                                fourier_basis.cosmumi[idx_ml], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(fixaray["sinmumi"][l][m],
                                fourier_basis.sinmumi[idx_ml], kTolerance));
    }  // l
  }  // m

  for (int n = 0; n < sizes.nnyq2 + 1; ++n) {
    EXPECT_TRUE(IsCloseRelAbs(fixaray["nscale"][n], fourier_basis.nscale[n],
                              kTolerance));
  }

  for (int k = 0; k < sizes.nZeta; ++k) {
    for (int n = 0; n < sizes.nnyq2 + 1; ++n) {
      int idx_kn = k * (sizes.nnyq2 + 1) + n;

      EXPECT_TRUE(IsCloseRelAbs(fixaray["cosnv"][k][n],
                                fourier_basis.cosnv[idx_kn], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(fixaray["sinnv"][k][n],
                                fourier_basis.sinnv[idx_kn], kTolerance));

      EXPECT_TRUE(IsCloseRelAbs(fixaray["cosnvn"][k][n],
                                fourier_basis.cosnvn[idx_kn], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(fixaray["sinnvn"][k][n],
                                fourier_basis.sinnvn[idx_kn], kTolerance));
    }  // k
  }  // n

  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    EXPECT_EQ(fixaray["xm"][mn], fourier_basis.xm[mn]);
    EXPECT_EQ(fixaray["xn"][mn], fourier_basis.xn[mn]);
  }

  for (int mn_nyq = 0; mn_nyq < sizes.mnmax_nyq; ++mn_nyq) {
    EXPECT_EQ(fixaray["xm_nyq"][mn_nyq], fourier_basis.xm_nyq[mn_nyq]);
    EXPECT_EQ(fixaray["xn_nyq"][mn_nyq], fourier_basis.xn_nyq[mn_nyq]);
  }
}  // CheckFourierBasisFastPoloidal

INSTANTIATE_TEST_SUITE_P(
    TestFourierBasisFastPoloidal, FourierBasisFastPoloidalTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-30},
           DataSource{.identifier = "solovev_analytical", .tolerance = 1.0e-30},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-30},
           DataSource{.identifier = "cma", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-30}));

}  // namespace vmecpp
