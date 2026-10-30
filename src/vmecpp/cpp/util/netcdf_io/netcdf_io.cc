// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/netcdf_io/netcdf_io.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "netcdf.h"

namespace netcdf_io {

namespace {

absl::StatusOr<int> FindVariableId(int ncid, const std::string& variable_name) {
  int variable_id = 0;
  if (nc_inq_varid(ncid, variable_name.c_str(), &variable_id) != NC_NOERR) {
    return absl::NotFoundError(
        absl::StrFormat("variable '%s' not found", variable_name));
  }
  return variable_id;
}

absl::StatusOr<int> GetVariableRank(int ncid, int variable_id,
                                    const std::string& variable_name) {
  int rank = 0;
  if (nc_inq_varndims(ncid, variable_id, &rank) != NC_NOERR) {
    return absl::InternalError(absl::StrFormat(
        "could not determine rank of variable '%s'", variable_name));
  }
  return rank;
}

absl::StatusOr<std::vector<size_t> > GetVariableDimensions(
    int ncid, int variable_id, int rank, const std::string& variable_name) {
  std::vector<int> dimension_ids(rank, 0);
  if (nc_inq_vardimid(ncid, variable_id, dimension_ids.data()) != NC_NOERR) {
    return absl::InternalError(absl::StrFormat(
        "could not determine dimension ids of variable '%s'", variable_name));
  }

  std::vector<size_t> dimensions(rank, 0);
  for (int i = 0; i < rank; ++i) {
    size_t dimension = 0;
    if (nc_inq_dimlen(ncid, dimension_ids[i], &dimension) != NC_NOERR) {
      return absl::InternalError(
          absl::StrFormat("could not determine dimension %d of variable '%s'",
                          i, variable_name));
    }
    dimensions[i] = dimension;
  }
  return dimensions;
}

}  // namespace

absl::StatusOr<bool> NetcdfReadBool(int ncid,
                                    const std::string& variable_name) {
  // VMEC uses `int` to store booleans: 0 means false, otherwise true.
  // Also, the actual variable name is `<variable_name>__logical__`.
  // AFAIK this is because NetCDF3 did not have a `boolean` data type.
  const std::string logical_variable_name = variable_name + "__logical__";

  absl::StatusOr<int> variable_id = FindVariableId(ncid, logical_variable_name);
  if (!variable_id.ok()) {
    return variable_id.status();
  }

  absl::StatusOr<int> rank =
      GetVariableRank(ncid, *variable_id, logical_variable_name);
  if (!rank.ok()) {
    return rank.status();
  }
  if (*rank != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a rank-0 array: %s", logical_variable_name));
  }

  int variable_data = 0;
  if (nc_get_var_int(ncid, *variable_id, &variable_data) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("could not read variable '%s'", logical_variable_name));
  }

  return variable_data != 0;
}  // NetcdfReadBool

absl::StatusOr<char> NetcdfReadChar(int ncid,
                                    const std::string& variable_name) {
  absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
  if (!variable_id.ok()) {
    return variable_id.status();
  }

  absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
  if (!rank.ok()) {
    return rank.status();
  }
  if (*rank != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a rank-1 array: %s", variable_name));
  }

  absl::StatusOr<std::vector<size_t> > dimensions =
      GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
  if (!dimensions.ok()) {
    return dimensions.status();
  }

  // for a single char, make sure that the array dimension is 1
  if ((*dimensions)[0] != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a length-1 array: %s", variable_name));
  }

  // actually read data
  std::vector<size_t> read_start_indices(*rank, 0);
  std::vector<char> variable_data(1, 0);
  if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
                  dimensions->data(), variable_data.data()) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("could not read variable '%s'", variable_name));
  }

  return variable_data[0];
}  // NetcdfReadChar

absl::StatusOr<int> NetcdfReadInt(int ncid, const std::string& variable_name) {
  absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
  if (!variable_id.ok()) {
    return variable_id.status();
  }

  absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
  if (!rank.ok()) {
    return rank.status();
  }
  if (*rank != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a rank-0 array: %s", variable_name));
  }

  int variable_data = 0;
  if (nc_get_var_int(ncid, *variable_id, &variable_data) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("could not read variable '%s'", variable_name));
  }

  return variable_data;
}  // NetcdfReadInt

absl::StatusOr<double> NetcdfReadDouble(int ncid,
                                        const std::string& variable_name) {
  absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
  if (!variable_id.ok()) {
    return variable_id.status();
  }

  absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
  if (!rank.ok()) {
    return rank.status();
  }
  if (*rank != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a rank-0 array: %s", variable_name));
  }

  double variable_data = 0;
  if (nc_get_var_double(ncid, *variable_id, &variable_data) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("could not read variable '%s'", variable_name));
  }

  return variable_data;
}  // NetcdfReadDouble

absl::StatusOr<std::string> NetcdfReadString(int ncid,
                                             const std::string& variable_name) {
  absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
  if (!variable_id.ok()) {
    return variable_id.status();
  }

  absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
  if (!rank.ok()) {
    return rank.status();
  }
  // only accept one-dimensional array of CHAR for strings
  if (*rank != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a rank-1 array: %s", variable_name));
  }

  absl::StatusOr<std::vector<size_t> > dimensions =
      GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
  if (!dimensions.ok()) {
    return dimensions.status();
  }

  size_t total_element_count = (*dimensions)[0];

  // actually read data
  std::vector<size_t> read_start_indices(*rank, 0);
  // one extra element that stays at 0 in order to properly zero-terminate the
  // string
  std::vector<char> variable_data(total_element_count + 1, 0);
  if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
                  dimensions->data(), variable_data.data()) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("could not read variable '%s'", variable_name));
  }
  std::string string_from_char_array = std::string(variable_data.data());

  // Strings are usually whitespace-padded when coming from Fortran
  // to reach the specified length, so get rid of that whitespace again.
  return std::string(absl::StripAsciiWhitespace(string_from_char_array));
}  // NetcdfReadString

absl::StatusOr<std::vector<double> > NetcdfReadArray1D(
    int ncid, const std::string& variable_name) {
  absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
  if (!variable_id.ok()) {
    return variable_id.status();
  }

  absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
  if (!rank.ok()) {
    return rank.status();
  }
  if (*rank != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a rank-1 array: %s", variable_name));
  }

  absl::StatusOr<std::vector<size_t> > dimensions =
      GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
  if (!dimensions.ok()) {
    return dimensions.status();
  }

  size_t total_element_count = (*dimensions)[0];

  std::vector<size_t> read_start_indices(*rank, 0);
  std::vector<double> variable_data(total_element_count, 0.0);
  if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
                  dimensions->data(), variable_data.data()) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("could not read variable '%s'", variable_name));
  }

  return variable_data;
}  // NetcdfReadArray1D

absl::StatusOr<std::vector<std::vector<double> > > NetcdfReadArray2D(
    int ncid, const std::string& variable_name) {
  absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
  if (!variable_id.ok()) {
    return variable_id.status();
  }

  absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
  if (!rank.ok()) {
    return rank.status();
  }
  if (*rank != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a rank-2 array: %s", variable_name));
  }

  absl::StatusOr<std::vector<size_t> > dimensions_or =
      GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
  if (!dimensions_or.ok()) {
    return dimensions_or.status();
  }
  auto dimensions = dimensions_or.value();

  size_t total_element_count = dimensions[0] * dimensions[1];

  std::vector<size_t> read_start_indices(*rank, 0);
  std::vector<double> variable_data(total_element_count, 0.0);
  if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
                  dimensions.data(), variable_data.data()) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("could not read variable '%s'", variable_name));
  }

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

absl::StatusOr<std::vector<std::vector<std::vector<double> > > >
NetcdfReadArray3D(int ncid, const std::string& variable_name) {
  absl::StatusOr<int> variable_id = FindVariableId(ncid, variable_name);
  if (!variable_id.ok()) {
    return variable_id.status();
  }

  absl::StatusOr<int> rank = GetVariableRank(ncid, *variable_id, variable_name);
  if (!rank.ok()) {
    return rank.status();
  }
  if (*rank != 3) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not a rank-3 array: %s", variable_name));
  }

  absl::StatusOr<std::vector<size_t> > dimensions_or =
      GetVariableDimensions(ncid, *variable_id, *rank, variable_name);
  if (!dimensions_or.ok()) {
    return dimensions_or.status();
  }
  auto dimensions = dimensions_or.value();

  size_t total_element_count = dimensions[0] * dimensions[1] * dimensions[2];
  std::vector<size_t> read_start_indices(*rank, 0);
  std::vector<double> variable_data(total_element_count, 0.0);
  if (nc_get_vara(ncid, *variable_id, read_start_indices.data(),
                  dimensions.data(), variable_data.data()) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("could not read variable '%s'", variable_name));
  }

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
