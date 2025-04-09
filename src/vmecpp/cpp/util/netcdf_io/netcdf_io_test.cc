// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/netcdf_io/netcdf_io.h"

#include <netcdf.h>

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace netcdf_io {

using ::testing::ElementsAreArray;

TEST(TestNetcdfIO, CheckReadBool) {
  const std::string example_netcdf = "util/netcdf_io/example_netcdf.nc";

  int ncid = 0;
  ASSERT_EQ(nc_open(example_netcdf.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  const bool lasym = NetcdfReadBool(ncid, "lasym");

  EXPECT_FALSE(lasym);

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckReadBool

TEST(TestNetcdfIO, CheckReadChar) {
  const std::string example_netcdf = "util/netcdf_io/example_netcdf.nc";

  int ncid = 0;
  ASSERT_EQ(nc_open(example_netcdf.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  const char mgrid_mode = NetcdfReadChar(ncid, "mgrid_mode");

  EXPECT_EQ(mgrid_mode, 'R');

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckReadChar

TEST(TestNetcdfIO, CheckReadInt) {
  const std::string example_netcdf = "util/netcdf_io/example_netcdf.nc";

  int ncid = 0;
  ASSERT_EQ(nc_open(example_netcdf.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  const int nfp = NetcdfReadInt(ncid, "nfp");

  EXPECT_EQ(nfp, 5);

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckReadInt

TEST(TestNetcdfIO, CheckReadDouble) {
  const std::string example_netcdf = "util/netcdf_io/example_netcdf.nc";

  int ncid = 0;
  ASSERT_EQ(nc_open(example_netcdf.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  const double ftolv = NetcdfReadDouble(ncid, "ftolv");

  EXPECT_EQ(ftolv, 1.0e-10);

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckReadDouble

TEST(TestNetcdfIO, CheckReadString) {
  const std::string example_netcdf = "util/netcdf_io/example_netcdf.nc";

  int ncid = 0;
  ASSERT_EQ(nc_open(example_netcdf.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  const std::string mgrid_file = NetcdfReadString(ncid, "mgrid_file");

  EXPECT_EQ(mgrid_file, "mgrid_cth_like.nc");

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckReadString

TEST(TestNetcdfIO, CheckReadArray1D) {
  const std::string example_netcdf = "util/netcdf_io/example_netcdf.nc";

  int ncid = 0;
  ASSERT_EQ(nc_open(example_netcdf.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  std::vector<double> am = NetcdfReadArray1D(ncid, "am");

  // `am` is stored in the wout file with its default (maximum) length
  // and only the first few (relevant) entries are actually populated.
  std::vector<double> reference_am(21, 0.0);
  reference_am[0] = 1.0;
  reference_am[1] = 5.0;
  reference_am[2] = 10.0;

  EXPECT_THAT(am, ElementsAreArray(reference_am));

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckReadArray1D

TEST(TestNetcdfIO, CheckReadArray2D) {
  const std::string example_netcdf = "util/netcdf_io/example_netcdf.nc";

  int ncid = 0;
  ASSERT_EQ(nc_open(example_netcdf.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  std::vector<std::vector<double> > rmnc = NetcdfReadArray2D(ncid, "rmnc");

  std::vector<std::vector<double> > reference_rmnc = {{0.0, 1.0, 2.0},
                                                      {0.1, 1.1, 2.1}};

  ASSERT_EQ(rmnc.size(), reference_rmnc.size());
  for (size_t i = 0; i < reference_rmnc.size(); ++i) {
    EXPECT_THAT(rmnc[i], ElementsAreArray(reference_rmnc[i]));
  }

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckReadArray2D

TEST(TestNetcdfIO, CheckReadArray3D) {
  const std::string example_netcdf = "util/netcdf_io/example_netcdf.nc";

  int ncid = 0;
  ASSERT_EQ(nc_open(example_netcdf.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  std::vector<std::vector<std::vector<double> > > br_001 =
      NetcdfReadArray3D(ncid, "br_001");

  std::vector<std::vector<std::vector<double> > > reference_br_001 = {
      {{0.00, 0.01, 0.02, 0.03},
       {0.10, 0.11, 0.12, 0.13},
       {0.20, 0.21, 0.22, 0.23}},
      {{1.00, 1.01, 1.02, 1.03},
       {1.10, 1.11, 1.12, 1.13},
       {1.20, 1.21, 1.22, 1.23}}};

  ASSERT_EQ(br_001.size(), reference_br_001.size());
  for (size_t i = 0; i < reference_br_001.size(); ++i) {
    ASSERT_EQ(br_001[i].size(), reference_br_001[i].size());
    for (size_t j = 0; j < reference_br_001[i].size(); ++j) {
      EXPECT_THAT(br_001[i][j], ElementsAreArray(reference_br_001[i][j]));
    }
  }

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckReadArray3D

}  // namespace netcdf_io
