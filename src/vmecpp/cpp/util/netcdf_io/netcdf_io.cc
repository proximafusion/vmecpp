// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/netcdf_io/netcdf_io.h"

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "netcdf.h"

namespace netcdf_io {

bool NetcdfReadBool(int ncid, const std::string& variable_name) {
  // VMEC uses `int` to store booleans: 0 means false, otherwise true.
  // Also, the actual variable name is `<variable_name>__logical__`.
  // AFAIK this is because NetCDF3 did not have a `boolean` data type.

  // find variable ID for given variable name
  int variable_id = 0;
  CHECK_EQ(
      nc_inq_varid(ncid, (variable_name + "__logical__").c_str(), &variable_id),
      NC_NOERR)
      << "variable '" << variable_name << "' not found";

  // figure out rank of data, i.e., how many dimensions does it have
  int rank = 0;
  CHECK_EQ(nc_inq_varndims(ncid, variable_id, &rank), NC_NOERR);

  // only accept zero-dimensional array for scalar
  CHECK_EQ(rank, 0) << "Not a rank-0 array: " << variable_name;

  // actually read data
  int variable_data = 0;
  CHECK_EQ(nc_get_var_int(ncid, variable_id, &variable_data), NC_NOERR);

  return (variable_data != 0);
}  // NetcdfReadBool

char NetcdfReadChar(int ncid, const std::string& variable_name) {
  // find variable ID for given variable name
  int variable_id = 0;
  CHECK_EQ(nc_inq_varid(ncid, variable_name.c_str(), &variable_id), NC_NOERR)
      << "variable '" << variable_name << "' not found";

  // figure out rank of data, i.e., how many dimensions does it have
  int rank = 1;
  CHECK_EQ(nc_inq_varndims(ncid, variable_id, &rank), NC_NOERR);

  // only accept zero-dimensional array for scalar
  CHECK_EQ(rank, 1) << "Not a rank-1 array: " << variable_name;

  // figure out the dimension IDs
  std::vector<int> dimension_ids(rank, 0);
  CHECK_EQ(nc_inq_vardimid(ncid, variable_id, dimension_ids.data()), NC_NOERR);

  // figure out dimension of data, i.e., length of string
  std::vector<size_t> dimensions(rank, 0);
  size_t total_element_count = 1;
  for (int i = 0; i < rank; ++i) {
    size_t dimension = 0;
    CHECK_EQ(nc_inq_dimlen(ncid, dimension_ids[i], &dimension), NC_NOERR);
    dimensions[i] = dimension;
    total_element_count *= dimension;
  }

  // for a single char, make sure that the array dimension is 1
  CHECK_EQ(dimensions[0], (size_t)1)
      << "Not a length-1 array: " << variable_name;

  // actually read data
  std::vector<size_t> read_start_indices(rank, 0);
  std::vector<char> variable_data(total_element_count, 0);
  CHECK_EQ(nc_get_vara(ncid, variable_id, read_start_indices.data(),
                       dimensions.data(), variable_data.data()),
           NC_NOERR);

  return variable_data[0];
}  // NetcdfReadChar

int NetcdfReadInt(int ncid, const std::string& variable_name) {
  // find variable ID for given variable name
  int variable_id = 0;
  CHECK_EQ(nc_inq_varid(ncid, variable_name.c_str(), &variable_id), NC_NOERR)
      << "variable '" << variable_name << "' not found";

  // figure out rank of data, i.e., how many dimensions does it have
  int rank = 0;
  CHECK_EQ(nc_inq_varndims(ncid, variable_id, &rank), NC_NOERR);

  // only accept zero-dimensional array for scalar
  CHECK_EQ(rank, 0) << "Not a rank-0 array: " << variable_name;

  // actually read data
  int variable_data = 0;
  CHECK_EQ(nc_get_var_int(ncid, variable_id, &variable_data), NC_NOERR);

  return variable_data;
}  // NetcdfReadInt

double NetcdfReadDouble(int ncid, const std::string& variable_name) {
  // find variable ID for given variable name
  int variable_id = 0;
  CHECK_EQ(nc_inq_varid(ncid, variable_name.c_str(), &variable_id), NC_NOERR)
      << "variable '" << variable_name << "' not found";

  // figure out rank of data, i.e., how many dimensions does it have
  int rank = 0;
  CHECK_EQ(nc_inq_varndims(ncid, variable_id, &rank), NC_NOERR);

  // only accept zero-dimensional array for scalar
  CHECK_EQ(rank, 0) << "Not a rank-0 array: " << variable_name;

  // actually read data
  double variable_data = 0;
  CHECK_EQ(nc_get_var_double(ncid, variable_id, &variable_data), NC_NOERR);

  return variable_data;
}  // NetcdfReadDouble

std::string NetcdfReadString(int ncid, const std::string& variable_name) {
  // find variable ID for given variable name
  int varid = 0;
  CHECK_EQ(nc_inq_varid(ncid, variable_name.c_str(), &varid), NC_NOERR)
      << "variable '" << variable_name << "' not found";

  // figure out rank of data, i.e., how many dimensions does it have
  int rank = 0;
  CHECK_EQ(nc_inq_varndims(ncid, varid, &rank), NC_NOERR);

  // only accept one-dimensional array of CHAR for strings
  CHECK_EQ(rank, 1) << "Not a rank-1 array: " << variable_name;

  // figure out the dimension IDs
  std::vector<int> dimension_ids(rank, 0);
  CHECK_EQ(nc_inq_vardimid(ncid, varid, dimension_ids.data()), NC_NOERR);

  // figure out dimension of data, i.e., length of string
  std::vector<size_t> dimensions(rank, 0);
  size_t total_element_count = 1;
  for (int i = 0; i < rank; ++i) {
    size_t dimension = 0;
    CHECK_EQ(nc_inq_dimlen(ncid, dimension_ids[i], &dimension), NC_NOERR);
    dimensions[i] = dimension;
    total_element_count *= dimension;
  }

  // actually read data
  std::vector<size_t> read_start_indices(rank, 0);
  // one extra element that stays at 0 in order to properly zero-terminate the
  // string
  std::vector<char> variable_data(total_element_count + 1, 0);
  CHECK_EQ(nc_get_vara(ncid, varid, read_start_indices.data(),
                       dimensions.data(), variable_data.data()),
           NC_NOERR);
  std::string string_from_char_array = std::string(variable_data.data());

  // Strings are usually whitespace-padded when coming from Fortran
  // to reach the specified length, so get rid of that whitespace again.
  return std::string(absl::StripAsciiWhitespace(string_from_char_array));
}  // NetcdfReadString

std::vector<double> NetcdfReadArray1D(int ncid,
                                      const std::string& variable_name) {
  // find variable ID for given variable name
  int variable_id = 0;
  CHECK_EQ(nc_inq_varid(ncid, variable_name.c_str(), &variable_id), NC_NOERR)
      << "variable '" << variable_name << "' not found";

  // figure out rank of data, i.e., how many dimensions does it have
  int rank = 1;
  CHECK_EQ(nc_inq_varndims(ncid, variable_id, &rank), NC_NOERR);

  // only accept zero-dimensional array for scalar
  CHECK_EQ(rank, 1) << "Not a rank-1 array: " << variable_name;

  // figure out the dimension IDs
  std::vector<int> dimension_ids(rank, 0);
  CHECK_EQ(nc_inq_vardimid(ncid, variable_id, dimension_ids.data()), NC_NOERR);

  // figure out dimension of data, i.e., length of string
  std::vector<size_t> dimensions(rank, 0);
  size_t total_element_count = 1;
  for (int i = 0; i < rank; ++i) {
    size_t dimension = 0;
    CHECK_EQ(nc_inq_dimlen(ncid, dimension_ids[i], &dimension), NC_NOERR);
    dimensions[i] = dimension;
    total_element_count *= dimension;
  }

  // actually read data
  std::vector<size_t> read_start_indices(rank, 0);
  std::vector<double> variable_data(total_element_count, 0.0);
  CHECK_EQ(nc_get_vara(ncid, variable_id, read_start_indices.data(),
                       dimensions.data(), variable_data.data()),
           NC_NOERR);

  return variable_data;
}  // NetcdfReadArray1D

std::vector<std::vector<double> > NetcdfReadArray2D(
    int ncid, const std::string& variable_name) {
  // find variable ID for given variable name
  int variable_id = 0;
  CHECK_EQ(nc_inq_varid(ncid, variable_name.c_str(), &variable_id), NC_NOERR)
      << "variable '" << variable_name << "' not found";

  // figure out rank of data, i.e., how many dimensions does it have
  int rank = 1;
  CHECK_EQ(nc_inq_varndims(ncid, variable_id, &rank), NC_NOERR);

  // only accept zero-dimensional array for scalar
  CHECK_EQ(rank, 2) << "Not a rank-2 array: " << variable_name;

  // figure out the dimension IDs
  std::vector<int> dimension_ids(rank, 0);
  CHECK_EQ(nc_inq_vardimid(ncid, variable_id, dimension_ids.data()), NC_NOERR);

  // figure out dimension of data, i.e., length of string
  std::vector<size_t> dimensions(rank, 0);
  size_t total_element_count = 1;
  for (int i = 0; i < rank; ++i) {
    size_t dimension = 0;
    CHECK_EQ(nc_inq_dimlen(ncid, dimension_ids[i], &dimension), NC_NOERR);
    dimensions[i] = dimension;
    total_element_count *= dimension;
  }

  // actually read data
  std::vector<size_t> read_start_indices(rank, 0);
  std::vector<double> variable_data(total_element_count, 0.0);
  CHECK_EQ(nc_get_vara(ncid, variable_id, read_start_indices.data(),
                       dimensions.data(), variable_data.data()),
           NC_NOERR);

  // copy from flattened vector into two-dimensional vector of vectors
  std::vector<std::vector<double> > two_dimensional_data(dimensions[0]);
  for (size_t i = 0; i < dimensions[0]; ++i) {
    two_dimensional_data[i].resize(dimensions[1], 0.0);
    for (size_t j = 0; j < dimensions[1]; ++j) {
      two_dimensional_data[i][j] = variable_data[i * dimensions[1] + j];
    }  // j
  }  // i

  return two_dimensional_data;
}  // NetcdfReadArray2D

std::vector<std::vector<std::vector<double> > > NetcdfReadArray3D(
    int ncid, const std::string& variable_name) {
  // find variable ID for given variable name
  int variable_id = 0;
  CHECK_EQ(nc_inq_varid(ncid, variable_name.c_str(), &variable_id), NC_NOERR)
      << "variable '" << variable_name << "' not found";

  // figure out rank of data, i.e., how many dimensions does it have
  int rank = 1;
  CHECK_EQ(nc_inq_varndims(ncid, variable_id, &rank), NC_NOERR);

  // only accept zero-dimensional array for scalar
  CHECK_EQ(rank, 3) << "Not a rank-3 array: " << variable_name;

  // figure out the dimension IDs
  std::vector<int> dimension_ids(rank, 0);
  CHECK_EQ(nc_inq_vardimid(ncid, variable_id, dimension_ids.data()), NC_NOERR);

  // figure out dimension of data, i.e., length of string
  std::vector<size_t> dimensions(rank, 0);
  size_t total_element_count = 1;
  for (int i = 0; i < rank; ++i) {
    size_t dimension = 0;
    CHECK_EQ(nc_inq_dimlen(ncid, dimension_ids[i], &dimension), NC_NOERR);
    dimensions[i] = dimension;
    total_element_count *= dimension;
  }

  // actually read data
  std::vector<size_t> read_start_indices(rank, 0);
  std::vector<double> variable_data(total_element_count, 0.0);
  CHECK_EQ(nc_get_vara(ncid, variable_id, read_start_indices.data(),
                       dimensions.data(), variable_data.data()),
           NC_NOERR);

  // copy from flattened vector into three-dimensional vector of vectors
  std::vector<std::vector<std::vector<double> > > three_dimensional_data(
      dimensions[0]);
  for (size_t i = 0; i < dimensions[0]; ++i) {
    three_dimensional_data[i].resize(dimensions[1]);
    for (size_t j = 0; j < dimensions[1]; ++j) {
      three_dimensional_data[i][j].resize(dimensions[2]);
      for (size_t k = 0; k < dimensions[2]; ++k) {
        three_dimensional_data[i][j][k] =
            variable_data[(i * dimensions[1] + j) * dimensions[2] + k];
      }  // k
    }  // j
  }  // i

  return three_dimensional_data;
}  // NetcdfReadArray3D

}  // namespace netcdf_io
