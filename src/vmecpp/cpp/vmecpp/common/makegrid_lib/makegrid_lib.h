// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_MAKEGRID_LIB_MAKEGRID_LIB_H_
#define VMECPP_COMMON_MAKEGRID_LIB_MAKEGRID_LIB_H_

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/util/util.h"

namespace makegrid {

using magnetics::MagneticConfiguration;

using vmecpp::RowMatrixXd;
using RowMatrix3Xd = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>;

struct MakegridParameters {
  // If true, normalize the magnetic field to the coil currents and number of
  // windings.
  bool normalize_by_currents = false;

  // If true, compute the magnetic field and vector potential only on the
  // stellarator-symmetric half-period and mirror it into the other half-period.
  bool assume_stellarator_symmetry = false;

  // Number of toroidal field periods for assuming toroidal symmetry.
  int number_of_field_periods = 0;

  // radial coordinate of first grid point
  double r_grid_minimum = 0.0;

  // radial coordinate of last grid point
  double r_grid_maximum = 0.0;

  // number of radial grid points
  int number_of_r_grid_points = 0;

  // vertical coordinate of first grid point
  double z_grid_minimum = 0.0;

  // vertical coordiante of last grid point
  double z_grid_maximum = 0.0;

  // number of vertical grid points
  int number_of_z_grid_points = 0;

  // number of toroidal grid points per field period
  // The grid in phi spans over [0, 1, ..., (nzeta-1)] * 2 pi / (nfp * nzeta),
  // where nfp == `number_of_field_periods`
  // and nzeta == `number_of_phi_grid_points`.
  int number_of_phi_grid_points = 0;
};  // MakegridParameters

struct MagneticFieldResponseTable {
  // the makegrid parameters used to construct this object
  MakegridParameters parameters;

  // cylindrical R components of magnetic field
  // [number_of_serial_circuits][number_of_phi_grid_points *
  // number_of_z_grid_points * number_of_r_grid_points]
  RowMatrixXd b_r;

  // cylindrical phi components of magnetic field
  // [number_of_serial_circuits][number_of_phi_grid_points *
  // number_of_z_grid_points * number_of_r_grid_points]
  RowMatrixXd b_p;

  // cylindrical Z components of magnetic field
  // [number_of_serial_circuits][number_of_phi_grid_points *
  // number_of_z_grid_points * number_of_r_grid_points]
  RowMatrixXd b_z;
};  // MagneticFieldResponseTable

struct MakegridCachedVectorPotential {
  // the makegrid parameters used to construct this object
  MakegridParameters parameters;

  // cylindrical R components of vector potential
  // [number_of_serial_circuits][number_of_phi_grid_points *
  // number_of_z_grid_points * number_of_r_grid_points]
  RowMatrixXd a_r;

  // cylindrical phi components of vector potential
  // [number_of_serial_circuits][number_of_phi_grid_points *
  // number_of_z_grid_points * number_of_r_grid_points]
  RowMatrixXd a_p;

  // cylindrical Z components of vector potential
  // [number_of_serial_circuits][number_of_phi_grid_points *
  // number_of_z_grid_points * number_of_r_grid_points]
  RowMatrixXd a_z;
};  // MakegridCachedVectorPotential

// Check if the parameters in given MakegridParameters are valid.
absl::Status IsValidMakegridParameters(
    const MakegridParameters& makegrid_parameters);

// Import MakegridParameters from a given JSON string.
absl::StatusOr<MakegridParameters> ImportMakegridParametersFromJson(
    const std::string& makegrid_parameters_json);

// Import MakegridParameters from a given JSON file.
absl::StatusOr<MakegridParameters> ImportMakegridParametersFromFile(
    const std::filesystem::path& makegrid_parameters_file);

// Compute the Cartesian coordinates of the MAKEGRID cylindrical grid.
// [total_number_of_grid_points][3: x, y, z]
// where total_number_of_grid_points = number_of_phi_grid_points *
// number_of_z_grid_points * number_of_r_grid_points. The ordering of the
// dimensions is:
//   * number_of_phi_grid_points (slowest)
//   * number_of_z_grid_points
//   * number_of_r_grid_points (fastest)
absl::StatusOr<RowMatrix3Xd> MakeCylindricalGrid(
    const MakegridParameters& makegrid_parameters);

// Compute the (normalized) magnetic field components on the given grid
// and store it (overwriting) in the provided cache.
absl::StatusOr<MagneticFieldResponseTable> ComputeMagneticFieldResponseTable(
    const MakegridParameters& makegrid_parameters,
    const MagneticConfiguration& magnetic_configuration);

// Compute the (normalized) vector potential components on the given grid
// and store it (overwriting) in the provided cache.
absl::StatusOr<MakegridCachedVectorPotential> ComputeVectorPotentialCache(
    const MakegridParameters& makegrid_parameters,
    const MagneticConfiguration& magnetic_configuration);

// Write the given magnetic field cache (and optionally, the given vector
// potential cache) to a NetCDF file specified by the given filename.
absl::Status WriteMakegridNetCDFFile(
    const std::string& makegrid_filename,
    const MakegridParameters& makegrid_parameters,
    const std::vector<double>& circuit_currents,
    const MagneticFieldResponseTable& magnetic_response_table,
    const std::optional<MakegridCachedVectorPotential>& vector_potential_cache);

}  // namespace makegrid

#endif  // VMECPP_COMMON_MAKEGRID_LIB_MAKEGRID_LIB_H_
