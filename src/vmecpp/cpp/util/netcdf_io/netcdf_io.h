// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef UTIL_NETCDF_IO_NETCDF_IO_H_
#define UTIL_NETCDF_IO_NETCDF_IO_H_

#include <string>
#include <vector>

namespace netcdf_io {

// Read a scalar `bool` variable in Fortran VMEC style from the (opened) NetCDF
// file identified by `ncid`. It is expected that the value is stored in a
// scalar `int` variable named `<variable_name>__logical__`.
bool NetcdfReadBool(int ncid, const std::string& variable_name);

// Read a scalar `char` variable from the (opened) NetCDF file identified by
// `ncid`. It is expected that the value is stored in a length-1 `char` array.
char NetcdfReadChar(int ncid, const std::string& variable_name);

// Read a scalar `int` variable  from the (opened) NetCDF file identified by
// `ncid`.
int NetcdfReadInt(int ncid, const std::string& variable_name);

// Read a scalar `double` variable  from the (opened) NetCDF file identified by
// `ncid`.
double NetcdfReadDouble(int ncid, const std::string& variable_name);

// Read a string from the (opened) NetCDF file identified by `ncid`.
// It is expected that the data is stored as a rank-1 `char` array.
// Whitespace at the start and end of the `char` array is stripped.
std::string NetcdfReadString(int ncid, const std::string& variable_name);

// Read a rank-1 `double` array from the (opened) NetCDF file identified by
// `ncid`.
std::vector<double> NetcdfReadArray1D(int ncid,
                                      const std::string& variable_name);

// Read a rank-2 `double` array from the (opened) NetCDF file identified by
// `ncid`.
std::vector<std::vector<double> > NetcdfReadArray2D(
    int ncid, const std::string& variable_name);

// Read a rank-3 `double` array from the (opened) NetCDF file identified by
// `ncid`.
std::vector<std::vector<std::vector<double> > > NetcdfReadArray3D(
    int ncid, const std::string& variable_name);

}  // namespace netcdf_io

#endif  // UTIL_NETCDF_IO_NETCDF_IO_H_
