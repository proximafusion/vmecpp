#include "vmecpp/common/sizes/sizes.h"

#include <string>

#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"

namespace vmecpp {

namespace {
using file_io::ReadFile;
using testing::IsCloseRelAbs;
}  // namespace

// The tests below check that the Sizes setup from the JSON input file is
// consistent with the corresponding parameters in the Reference Fortran VMEC.
// For now, these are written manually here, since at the time of implementing
// these tests, the corresponding outputs were not written yet to debugging
// output files.
// TODO(jons): write a debugging output from educational_VMEC, which contains
// all these reference values, and test against those with a parameterized test
// for all cases

TEST(TestSizes, CheckSolovev) {
  double tolerance = 1.0e-30;

  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev.json");
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Sizes sizes(*vmec_indata);

  EXPECT_EQ(sizes.lasym, false);
  EXPECT_EQ(sizes.nfp, 1);
  EXPECT_EQ(sizes.mpol, 6);
  EXPECT_EQ(sizes.ntor, 0);

  // ntheta was 0 in the input file, so compute it here
  EXPECT_EQ(sizes.ntheta, 2 * sizes.mpol + 6);
  EXPECT_EQ(sizes.nZeta, 1);

  EXPECT_EQ(sizes.lthreed, false);
  EXPECT_EQ(sizes.num_basis, 1);

  EXPECT_EQ(sizes.nThetaEven, 18);
  EXPECT_EQ(sizes.nThetaReduced, 10);
  EXPECT_EQ(sizes.nThetaEff, 10);

  EXPECT_EQ(sizes.nZnT, 10);

  for (int l = 0; l < sizes.nThetaEff; ++l) {
    if (l == 0 || l == sizes.nThetaEff - 1) {
      EXPECT_TRUE(IsCloseRelAbs(1.0 / 18, sizes.wInt[l], tolerance));
    } else {
      EXPECT_TRUE(IsCloseRelAbs(2.0 / 18, sizes.wInt[l], tolerance));
    }
  }

  EXPECT_EQ(sizes.mnsize, 6);

  EXPECT_EQ(sizes.mnmax, 6);
}  // CheckSolovev

TEST(TestSizes, CheckSolovevAnalytical) {
  double tolerance = 1.0e-30;

  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev_analytical.json");
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Sizes sizes(*vmec_indata);

  EXPECT_EQ(sizes.lasym, false);
  EXPECT_EQ(sizes.nfp, 1);
  EXPECT_EQ(sizes.mpol, 13);
  EXPECT_EQ(sizes.ntor, 0);

  // ntheta was 0 in the input file, so compute it here
  EXPECT_EQ(sizes.ntheta, 2 * sizes.mpol + 6);
  EXPECT_EQ(sizes.nZeta, 1);

  EXPECT_EQ(sizes.lthreed, false);
  EXPECT_EQ(sizes.num_basis, 1);

  EXPECT_EQ(sizes.nThetaEven, 32);
  EXPECT_EQ(sizes.nThetaReduced, 17);
  EXPECT_EQ(sizes.nThetaEff, 17);

  EXPECT_EQ(sizes.nZnT, 17);

  for (int l = 0; l < sizes.nThetaEff; ++l) {
    if (l == 0 || l == sizes.nThetaEff - 1) {
      EXPECT_TRUE(IsCloseRelAbs(1.0 / 32, sizes.wInt[l], tolerance));
    } else {
      EXPECT_TRUE(IsCloseRelAbs(2.0 / 32, sizes.wInt[l], tolerance));
    }
  }

  EXPECT_EQ(sizes.mnsize, 13);

  EXPECT_EQ(sizes.mnmax, 13);
}  // CheckSolovevAnalytical

TEST(TestSizes, CheckSolovevNoAxis) {
  double tolerance = 1.0e-30;

  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev_no_axis.json");
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Sizes sizes(*vmec_indata);

  EXPECT_EQ(sizes.lasym, false);
  EXPECT_EQ(sizes.nfp, 1);
  EXPECT_EQ(sizes.mpol, 6);
  EXPECT_EQ(sizes.ntor, 0);

  // ntheta was 0 in the input file, so compute it here
  EXPECT_EQ(sizes.ntheta, 2 * sizes.mpol + 6);
  EXPECT_EQ(sizes.nZeta, 1);

  EXPECT_EQ(sizes.lthreed, false);
  EXPECT_EQ(sizes.num_basis, 1);

  EXPECT_EQ(sizes.nThetaEven, 18);
  EXPECT_EQ(sizes.nThetaReduced, 10);
  EXPECT_EQ(sizes.nThetaEff, 10);

  EXPECT_EQ(sizes.nZnT, 10);

  for (int l = 0; l < sizes.nThetaEff; ++l) {
    if (l == 0 || l == sizes.nThetaEff - 1) {
      EXPECT_TRUE(IsCloseRelAbs(1.0 / 18, sizes.wInt[l], tolerance));
    } else {
      EXPECT_TRUE(IsCloseRelAbs(2.0 / 18, sizes.wInt[l], tolerance));
    }
  }

  EXPECT_EQ(sizes.mnsize, 6);

  EXPECT_EQ(sizes.mnmax, 6);
}  // CheckSolovevNoAxis

TEST(TestSizes, CheckCthLikeFixedBoundary) {
  double tolerance = 1.0e-30;

  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_fixed_bdy.json");
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Sizes sizes(*vmec_indata);

  EXPECT_EQ(sizes.lasym, false);
  EXPECT_EQ(sizes.nfp, 5);
  EXPECT_EQ(sizes.mpol, 5);
  EXPECT_EQ(sizes.ntor, 4);

  // ntheta was 0 in the input file, so compute it here
  EXPECT_EQ(sizes.ntheta, 2 * sizes.mpol + 6);
  EXPECT_EQ(sizes.nZeta, 36);

  EXPECT_EQ(sizes.lthreed, true);
  EXPECT_EQ(sizes.num_basis, 2);

  EXPECT_EQ(sizes.nThetaEven, 16);
  EXPECT_EQ(sizes.nThetaReduced, 9);
  EXPECT_EQ(sizes.nThetaEff, 9);

  EXPECT_EQ(sizes.nZnT, 324);

  for (int l = 0; l < sizes.nThetaEff; ++l) {
    if (l == 0 || l == sizes.nThetaEff - 1) {
      EXPECT_TRUE(IsCloseRelAbs(1.0 / (36 * 16), sizes.wInt[l], tolerance));
    } else {
      EXPECT_TRUE(IsCloseRelAbs(2.0 / (36 * 16), sizes.wInt[l], tolerance));
    }
  }

  EXPECT_EQ(sizes.mnsize, 25);

  EXPECT_EQ(sizes.mnmax, 41);
}  // CheckCthLikeFixedBoundary

TEST(TestSizes, CheckCma) {
  double tolerance = 1.0e-30;

  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cma.json");
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Sizes sizes(*vmec_indata);

  EXPECT_EQ(sizes.lasym, false);
  EXPECT_EQ(sizes.nfp, 2);
  EXPECT_EQ(sizes.mpol, 5);
  EXPECT_EQ(sizes.ntor, 6);

  // ntheta and nzeta were not specified (=default to 0) in the input file, so
  // compute it here
  EXPECT_EQ(sizes.ntheta, 2 * sizes.mpol + 6);
  EXPECT_EQ(sizes.nZeta, 2 * sizes.ntor + 4);

  EXPECT_EQ(sizes.lthreed, true);
  EXPECT_EQ(sizes.num_basis, 2);

  EXPECT_EQ(sizes.nThetaEven, 16);
  EXPECT_EQ(sizes.nThetaReduced, 9);
  EXPECT_EQ(sizes.nThetaEff, 9);

  EXPECT_EQ(sizes.nZnT, 144);

  for (int l = 0; l < sizes.nThetaEff; ++l) {
    if (l == 0 || l == sizes.nThetaEff - 1) {
      EXPECT_TRUE(IsCloseRelAbs(1.0 / (16 * 16), sizes.wInt[l], tolerance));
    } else {
      EXPECT_TRUE(IsCloseRelAbs(2.0 / (16 * 16), sizes.wInt[l], tolerance));
    }
  }

  EXPECT_EQ(sizes.mnsize, 35);

  EXPECT_EQ(sizes.mnmax, 59);
}  // CheckCma

TEST(TestSizes, CheckCthLikeFreeBoundary) {
  double tolerance = 1.0e-30;

  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Sizes sizes(*vmec_indata);

  EXPECT_EQ(sizes.lasym, false);
  EXPECT_EQ(sizes.nfp, 5);
  EXPECT_EQ(sizes.mpol, 5);
  EXPECT_EQ(sizes.ntor, 4);

  // ntheta was 0 in the input file, so compute it here
  EXPECT_EQ(sizes.ntheta, 2 * sizes.mpol + 6);
  EXPECT_EQ(sizes.nZeta, 36);

  EXPECT_EQ(sizes.lthreed, true);
  EXPECT_EQ(sizes.num_basis, 2);

  EXPECT_EQ(sizes.nThetaEven, 16);
  EXPECT_EQ(sizes.nThetaReduced, 9);
  EXPECT_EQ(sizes.nThetaEff, 9);

  EXPECT_EQ(sizes.nZnT, 324);

  for (int l = 0; l < sizes.nThetaEff; ++l) {
    if (l == 0 || l == sizes.nThetaEff - 1) {
      EXPECT_TRUE(IsCloseRelAbs(1.0 / (36 * 16), sizes.wInt[l], tolerance));
    } else {
      EXPECT_TRUE(IsCloseRelAbs(2.0 / (36 * 16), sizes.wInt[l], tolerance));
    }
  }

  EXPECT_EQ(sizes.mnsize, 25);

  EXPECT_EQ(sizes.mnmax, 41);
}  // CheckCthLikeFreeBoundary

}  // namespace vmecpp
