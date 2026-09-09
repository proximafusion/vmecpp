// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/hdf5_io/hdf5_io.h"

#include <string>
#include <vector>

#include "H5Cpp.h"
#include "gtest/gtest.h"

namespace {

TEST(Hdf5Io, StringVectorRoundTrip) {
  const std::vector<std::string> written = {
      "circuit_0", "circuit_1",
      // longer than the small-string buffer, so the characters live on the heap
      "a coil group name well past the small string optimization threshold"};

  const std::string fname = "test_hdf5_io_string_vector.h5";
  {
    H5::H5File file(fname, H5F_ACC_TRUNC);
    hdf5_io::WriteH5Dataset(written, "names", file);
  }

  std::vector<std::string> read;
  {
    H5::H5File file(fname, H5F_ACC_RDONLY);
    hdf5_io::ReadH5Dataset(read, "names", file);
  }

  EXPECT_EQ(read, written);
}

TEST(Hdf5Io, EmptyStringVectorRoundTrip) {
  const std::vector<std::string> written;

  const std::string fname = "test_hdf5_io_empty_string_vector.h5";
  {
    H5::H5File file(fname, H5F_ACC_TRUNC);
    hdf5_io::WriteH5Dataset(written, "names", file);
  }

  std::vector<std::string> read = {"leftover"};
  {
    H5::H5File file(fname, H5F_ACC_RDONLY);
    hdf5_io::ReadH5Dataset(read, "names", file);
  }

  EXPECT_TRUE(read.empty());
}

}  // namespace
