// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "netcdf.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/json_io/json_io.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/magnetic_field_provider/magnetic_field_provider_lib.h"
#include "vmecpp/common/util/util.h"
namespace makegrid {

using nlohmann::json;

using json_io::JsonReadBool;
using json_io::JsonReadDouble;
using json_io::JsonReadInt;

using magnetics::GetCircuitCurrents;
using magnetics::MagneticField;
using magnetics::NumWindingsToCircuitCurrents;
using magnetics::SetCircuitCurrents;
using magnetics::VectorPotential;

// TODO(jons): implement stellarator-symmetric grid and follow-up flip-mirroring
// of magnetic quantities NOTE: For now, everything here is computed as
// non-stellarator-symmetric,
//       so there is a factor of ~2 speedup around the corner.

absl::Status IsValidMakegridParameters(
    const MakegridParameters& makegrid_parameters) {
  // number of field periods has to be at least 1
  if (makegrid_parameters.number_of_field_periods <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("number_of_field_periods must be > 0, but is %d",
                        makegrid_parameters.number_of_field_periods));
  }

  // grid maximum has to be larger than grid minimum
  if (makegrid_parameters.r_grid_maximum <=
      makegrid_parameters.r_grid_minimum) {
    return absl::InvalidArgumentError(
        absl::StrFormat("R grid extent must be positive, but is from "
                        "r_grid_minimum = % .3e to r_grid_maximum = % .3e",
                        makegrid_parameters.r_grid_minimum,
                        makegrid_parameters.r_grid_maximum));
  }

  // at least 2 grid points: at minimum and at maximum
  if (makegrid_parameters.number_of_r_grid_points < 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("number_of_r_grid_points must be > 1, but is %d",
                        makegrid_parameters.number_of_r_grid_points));
  }

  // grid maximum has to be larger than grid minimum
  if (makegrid_parameters.z_grid_maximum <=
      makegrid_parameters.z_grid_minimum) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Z grid extent must be positive, but is from "
                        "z_grid_minimum = % .3e to z_grid_maximum = % .3e",
                        makegrid_parameters.z_grid_minimum,
                        makegrid_parameters.z_grid_maximum));
  }

  // at least 2 grid points: at minimum and at maximum
  if (makegrid_parameters.number_of_z_grid_points < 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("number_of_z_grid_points must be > 1, but is %d",
                        makegrid_parameters.number_of_z_grid_points));
  }

  // at least a single point in phi direction (one plane)
  if (makegrid_parameters.number_of_phi_grid_points < 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("number_of_phi_grid_points must be > 0, but is %d",
                        makegrid_parameters.number_of_phi_grid_points));
  }

  return absl::OkStatus();
}  // IsValidMakegridParameters

absl::StatusOr<MakegridParameters> ImportMakegridParametersFromJson(
    const std::string& makegrid_parameters_json) {
  json j = json::parse(makegrid_parameters_json);

  MakegridParameters makegrid_parameters;

  // normalize_by_currents
  absl::StatusOr<std::optional<bool>> maybe_normalize_by_currents =
      JsonReadBool(j, "normalize_by_currents");
  if (!maybe_normalize_by_currents.ok()) {
    return maybe_normalize_by_currents.status();
  } else {
    if (maybe_normalize_by_currents->has_value()) {
      makegrid_parameters.normalize_by_currents =
          maybe_normalize_by_currents->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'normalize_by_currents':bool must be specified in "
          "MakegridParameters");
    }
  }

  // assume_stellarator_symmetry
  absl::StatusOr<std::optional<bool>> maybe_assume_stellarator_symmetry =
      JsonReadBool(j, "assume_stellarator_symmetry");
  if (!maybe_assume_stellarator_symmetry.ok()) {
    return maybe_assume_stellarator_symmetry.status();
  } else {
    if (maybe_assume_stellarator_symmetry->has_value()) {
      makegrid_parameters.assume_stellarator_symmetry =
          maybe_assume_stellarator_symmetry->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'assume_stellarator_symmetry':bool must be specified in "
          "MakegridParameters");
    }
  }

  // number_of_field_periods
  absl::StatusOr<std::optional<int>> maybe_number_of_field_periods =
      JsonReadInt(j, "number_of_field_periods");
  if (!maybe_number_of_field_periods.ok()) {
    return maybe_number_of_field_periods.status();
  } else {
    if (maybe_number_of_field_periods->has_value()) {
      makegrid_parameters.number_of_field_periods =
          maybe_number_of_field_periods->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'number_of_field_periods':int must be specified in "
          "MakegridParameters");
    }
  }

  // r_grid_minimum
  absl::StatusOr<std::optional<double>> maybe_r_grid_minimum =
      JsonReadDouble(j, "r_grid_minimum");
  if (!maybe_r_grid_minimum.ok()) {
    return maybe_r_grid_minimum.status();
  } else {
    if (maybe_r_grid_minimum->has_value()) {
      makegrid_parameters.r_grid_minimum = maybe_r_grid_minimum->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'r_grid_minimum':double must be specified in MakegridParameters");
    }
  }

  // r_grid_maximum
  absl::StatusOr<std::optional<double>> maybe_r_grid_maximum =
      JsonReadDouble(j, "r_grid_maximum");
  if (!maybe_r_grid_maximum.ok()) {
    return maybe_r_grid_maximum.status();
  } else {
    if (maybe_r_grid_maximum->has_value()) {
      makegrid_parameters.r_grid_maximum = maybe_r_grid_maximum->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'r_grid_maximum':double must be specified in MakegridParameters");
    }
  }

  // number_of_r_grid_points
  absl::StatusOr<std::optional<int>> maybe_number_of_r_grid_points =
      JsonReadInt(j, "number_of_r_grid_points");
  if (!maybe_number_of_r_grid_points.ok()) {
    return maybe_number_of_r_grid_points.status();
  } else {
    if (maybe_number_of_r_grid_points->has_value()) {
      makegrid_parameters.number_of_r_grid_points =
          maybe_number_of_r_grid_points->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'number_of_r_grid_points':int must be specified in "
          "MakegridParameters");
    }
  }

  // z_grid_minimum
  absl::StatusOr<std::optional<double>> maybe_z_grid_minimum =
      JsonReadDouble(j, "z_grid_minimum");
  if (!maybe_z_grid_minimum.ok()) {
    return maybe_z_grid_minimum.status();
  } else {
    if (maybe_z_grid_minimum->has_value()) {
      makegrid_parameters.z_grid_minimum = maybe_z_grid_minimum->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'z_grid_minimum':double must be specified in MakegridParameters");
    }
  }

  // z_grid_maximum
  absl::StatusOr<std::optional<double>> maybe_z_grid_maximum =
      JsonReadDouble(j, "z_grid_maximum");
  if (!maybe_z_grid_maximum.ok()) {
    return maybe_z_grid_maximum.status();
  } else {
    if (maybe_z_grid_maximum->has_value()) {
      makegrid_parameters.z_grid_maximum = maybe_z_grid_maximum->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'z_grid_maximum':double must be specified in MakegridParameters");
    }
  }

  // number_of_z_grid_points
  absl::StatusOr<std::optional<int>> maybe_number_of_z_grid_points =
      JsonReadInt(j, "number_of_z_grid_points");
  if (!maybe_number_of_z_grid_points.ok()) {
    return maybe_number_of_z_grid_points.status();
  } else {
    if (maybe_number_of_z_grid_points->has_value()) {
      makegrid_parameters.number_of_z_grid_points =
          maybe_number_of_z_grid_points->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'number_of_z_grid_points':int must be specified in "
          "MakegridParameters");
    }
  }

  // number_of_phi_grid_points
  absl::StatusOr<std::optional<int>> maybe_number_of_phi_grid_points =
      JsonReadInt(j, "number_of_phi_grid_points");
  if (!maybe_number_of_phi_grid_points.ok()) {
    return maybe_number_of_phi_grid_points.status();
  } else {
    if (maybe_number_of_phi_grid_points->has_value()) {
      makegrid_parameters.number_of_phi_grid_points =
          maybe_number_of_phi_grid_points->value();
    } else {
      // MakegridParameters must be fully populated
      return absl::NotFoundError(
          "'number_of_phi_grid_points':int must be specified in "
          "MakegridParameters");
    }
  }

  // after having parsed the individual parameters,
  // check that their values actually make sense
  absl::Status makegrid_parameters_status =
      IsValidMakegridParameters(makegrid_parameters);
  if (!makegrid_parameters_status.ok()) {
    return makegrid_parameters_status;
  }

  return makegrid_parameters;
}  // ImportMakegridParametersFromJson

absl::StatusOr<MakegridParameters> ImportMakegridParametersFromFile(
    const std::filesystem::path& makegrid_parameters_file) {
  const auto maybe_makegrid_params_json =
      file_io::ReadFile(makegrid_parameters_file);
  CHECK_OK(maybe_makegrid_params_json);
  const auto& makegrid_params_json = *maybe_makegrid_params_json;
  return ImportMakegridParametersFromJson(makegrid_params_json);
}  // ImportMakegridParametersFromFile

absl::StatusOr<RowMatrix3Xd> MakeCylindricalGrid(
    const MakegridParameters& makegrid_parameters) {
  absl::Status makegrid_parameters_status =
      IsValidMakegridParameters(makegrid_parameters);
  if (!makegrid_parameters_status.ok()) {
    return makegrid_parameters_status;
  }

  // shorthand variables for grid dimensions
  const int num_field_periods = makegrid_parameters.number_of_field_periods;
  const int num_phi = makegrid_parameters.number_of_phi_grid_points;
  const int num_z = makegrid_parameters.number_of_z_grid_points;
  const int num_r = makegrid_parameters.number_of_r_grid_points;

  // grid extents along R and Z
  const double min_r = makegrid_parameters.r_grid_minimum;
  const double max_r = makegrid_parameters.r_grid_maximum;
  const double min_z = makegrid_parameters.z_grid_minimum;
  const double max_z = makegrid_parameters.z_grid_maximum;

  // dimensions of grid cells in cylindrical coordinates
  const double delta_r = (max_r - min_r) / (num_r - 1.0);
  const double delta_z = (max_z - min_z) / (num_z - 1.0);
  const double delta_phi = 2.0 * M_PI / (num_field_periods * num_phi);

  int num_phi_effective = num_phi;
  if (makegrid_parameters.assume_stellarator_symmetry) {
    if (num_phi % 2 != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "number of toroidal grid points has to be even for being able to "
          "make use to stellarator symmetry in makegrid, but was num_phi=",
          num_phi));
    }
    num_phi_effective = num_phi / 2 + 1;
  }

  const int total_number_of_grid_points = num_phi_effective * num_z * num_r;

  RowMatrix3Xd cylindrical_grid{
      RowMatrix3Xd::Zero(3, total_number_of_grid_points)};

  for (int phi_index = 0; phi_index < num_phi_effective; ++phi_index) {
    const double phi = phi_index * delta_phi;
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);
    for (int z_index = 0; z_index < num_z; ++z_index) {
      const double z = min_z + z_index * delta_z;
      for (int r_index = 0; r_index < num_r; ++r_index) {
        const double r = min_r + r_index * delta_r;

        const double x = r * cos_phi;
        const double y = r * sin_phi;

        const int linear_index =
            (phi_index * num_z + z_index) * num_r + r_index;
        // TODO(jurasic) bad loop order
        cylindrical_grid(0, linear_index) = x;
        cylindrical_grid(1, linear_index) = y;
        cylindrical_grid(2, linear_index) = z;
      }  // r_index
    }  // z_index
  }  // phi_index

  return cylindrical_grid;
}  // MakeCylindricalGrid

// Convert the cartesian magnetic field response to cylindrical coordinates
// using the precomputed sin and cos terms
template <typename Derived3>
void CartesianToCylindricalField(
    const Eigen::VectorXd& cos_phi, const Eigen::VectorXd& sin_phi,
    const Eigen::DenseBase<Derived3>& magnetic_field, int num_z, int num_r,
    int number_of_evaluation_points, Eigen::Ref<Eigen::RowVectorXd> m_b_r,
    Eigen::Ref<Eigen::RowVectorXd> m_b_p,
    Eigen::Ref<Eigen::RowVectorXd> m_b_z) {
  const Eigen::Index num_phi = cos_phi.size();
  const Eigen::Index total_number_of_grid_points = m_b_r.size();
  // Sanity checks for dimensions
  CHECK_EQ(cos_phi.size(), sin_phi.size());
  CHECK_EQ(m_b_p.size(), m_b_r.size());
  CHECK_EQ(m_b_z.size(), m_b_r.size());
  CHECK_EQ(num_phi * num_r * num_z, total_number_of_grid_points);
  // May be less than total_number_of_grid_points due to symmetry
  CHECK_LE(number_of_evaluation_points, total_number_of_grid_points);

// ABSCAB computes the Cartesian components of the magnetic field,
// so we need to convert the x and y componets into r and phi
// (cylindrical) components for comparison against the cylindrical
// components in the mgrid file.
#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
  for (Eigen::Index linear_index = 0;
       linear_index < number_of_evaluation_points; ++linear_index) {
    const double b_x = magnetic_field(0, linear_index);
    const double b_y = magnetic_field(1, linear_index);

    const Eigen::Index index_phi = linear_index / (num_z * num_r);
    const double b_r = b_x * cos_phi[index_phi] + b_y * sin_phi[index_phi];
    const double b_p = b_y * cos_phi[index_phi] - b_x * sin_phi[index_phi];

    m_b_r[linear_index] = b_r;
    m_b_p[linear_index] = b_p;
  }  // linear_index
  //
  m_b_z.head(number_of_evaluation_points) = magnetic_field.row(2);

  // mirror into other stellarator-symmetric part of grid
  // if making use of stellarator symmetry
  if (number_of_evaluation_points < total_number_of_grid_points) {
#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
    for (int linear_index = number_of_evaluation_points;
         linear_index < total_number_of_grid_points; ++linear_index) {
      const int idx_phi = linear_index / (num_z * num_r);
      const int idx_z_r = linear_index % (num_z * num_r);
      const int idx_z = idx_z_r / num_r;
      const int idx_r = idx_z_r % num_r;

      const int idx_phi_reversed = num_phi - idx_phi;
      const int idx_z_reversed = num_z - 1 - idx_z;

      const int linear_index_reversed =
          (idx_phi_reversed * num_z + idx_z_reversed) * num_r + idx_r;

      m_b_r[linear_index] = -m_b_r[linear_index_reversed];
      m_b_p[linear_index] = m_b_p[linear_index_reversed];
      m_b_z[linear_index] = m_b_z[linear_index_reversed];
    }
  }
}

absl::StatusOr<MagneticFieldResponseTable> ComputeMagneticFieldResponseTable(
    const MakegridParameters& makegrid_parameters,
    const MagneticConfiguration& magnetic_configuration) {
  absl::StatusOr<RowMatrix3Xd> maybe_cylindrical_grid =
      MakeCylindricalGrid(makegrid_parameters);
  if (!maybe_cylindrical_grid.ok()) {
    return maybe_cylindrical_grid.status();
  }
  // `MakeCylindricalGrid` only computes the grid points for which to actually
  // evaluate the magnetic field. If stellarator symmetry is activated, this
  // might be less grid points that what gets stored in the mgrid file.
  const int number_of_evaluation_points =
      static_cast<int>(maybe_cylindrical_grid.value().cols());

  // shorthand variables for grid dimensions
  const int num_field_periods = makegrid_parameters.number_of_field_periods;
  const int num_phi = makegrid_parameters.number_of_phi_grid_points;
  const int num_z = makegrid_parameters.number_of_z_grid_points;
  const int num_r = makegrid_parameters.number_of_r_grid_points;

  // This is the total number of grid points that is stored in the mgrid file.
  const int total_number_of_grid_points = num_phi * num_z * num_r;

  // precompute toroidal trigonometry tables
  Eigen::VectorXd cos_phi{Eigen::VectorXd::Zero(num_phi)};
  Eigen::VectorXd sin_phi{Eigen::VectorXd::Zero(num_phi)};
  const double delta_phi = 2.0 * M_PI / (num_field_periods * num_phi);
  for (int index_phi = 0; index_phi < num_phi; ++index_phi) {
    const double phi = index_phi * delta_phi;
    cos_phi(index_phi) = std::cos(phi);
    sin_phi(index_phi) = std::sin(phi);
  }

  // Make a backup of the full vector of original circuit currents,
  // but only after num_windings has potentially been migrated into circuit
  // currents.
  absl::StatusOr<std::vector<double>> maybe_original_currents =
      GetCircuitCurrents(magnetic_configuration);
  if (!maybe_original_currents.ok()) {
    return maybe_original_currents.status();
  }
  const std::vector<double>& original_currents = *maybe_original_currents;

  const int number_of_serial_circuits =
      magnetic_configuration.serial_circuits_size();

  MagneticFieldResponseTable response_table_b;
  response_table_b.parameters = makegrid_parameters;

  // fully allocate result tables
  response_table_b.b_r.resize(number_of_serial_circuits,
                              total_number_of_grid_points);
  response_table_b.b_p.resize(number_of_serial_circuits,
                              total_number_of_grid_points);
  response_table_b.b_z.resize(number_of_serial_circuits,
                              total_number_of_grid_points);

  std::vector<absl::Status> status(number_of_serial_circuits);

  // Now compute the magnetic field for each SerialCircuit individually in the
  // MagneticConfiguration. This is conveniently done by setting the currents to
  // all zeros and only restore the current for the SerialCircuit under
  // consideration to its original value. This has no (significant) performance
  // penalty, since the magnetic field contributions are skipped anyway if the
  // current of a given SerialCircuit is identically zero.
  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    // make second internal copy for being able to set inidividually only one
    // circuit current to != 0
    MagneticConfiguration m_magnetic_configuration = magnetic_configuration;

    std::vector<double> currents_for_circuit(number_of_serial_circuits);
    if (makegrid_parameters.normalize_by_currents) {
      currents_for_circuit[circuit_index] = 1.0;
    } else {
      currents_for_circuit[circuit_index] = original_currents[circuit_index];
    }

    absl::Status set_currents_status =
        SetCircuitCurrents(currents_for_circuit, m_magnetic_configuration);
    if (!set_currents_status.ok()) {
      status[circuit_index] = set_currents_status;
      continue;
    }

    // Evaluation result B (n, 3) in cartesian coordinates
    // TODO(jurasic) Remove after Eigen refactor
    std::vector<std::vector<double>> magnetic_field_stl(
        number_of_evaluation_points);
    for (int i = 0; i < number_of_evaluation_points; ++i) {
      magnetic_field_stl[i].resize(3, 0.0);
    }

    // TODO(jurasic) Remove after Eigen refactor
    std::vector<std::vector<double>> cylindrical_grid_stl{
        static_cast<std::size_t>(maybe_cylindrical_grid.value().cols())};
    CHECK_EQ(static_cast<std::size_t>(number_of_evaluation_points),
             cylindrical_grid_stl.size());
    for (int i = 0; i < number_of_evaluation_points; ++i) {
      cylindrical_grid_stl[i].resize(3);
      for (int j = 0; j < 3; ++j) {
        cylindrical_grid_stl[i][j] = maybe_cylindrical_grid.value()(j, i);
      }
    }
    CHECK_EQ(magnetic_field_stl.size(), cylindrical_grid_stl.size());

    // We parallelize over linear index of evaluation locations, since that
    // allows us to use more CPUs and parallelize also for configurations with
    // low number of circuits. It is much more common in practice to have few
    // independent circuits and many evaluation locations, rather than many
    // independent circuits but few evaluation locations. This is done inside of
    // ABSCAB, which is used within this call to `MagneticField`.
    absl::Status magnetic_field_status =
        MagneticField(m_magnetic_configuration, cylindrical_grid_stl,
                      /*m_magnetic_field=*/magnetic_field_stl);
    if (!magnetic_field_status.ok()) {
      status[circuit_index] = magnetic_field_status;
      continue;
    }
    RowMatrix3Xd magnetic_field =
        RowMatrix3Xd::Zero(3, number_of_evaluation_points);
    magnetic_field = vmecpp::ToEigenMatrix(magnetic_field_stl).transpose();
    CartesianToCylindricalField(cos_phi, sin_phi, magnetic_field, num_z, num_r,
                                number_of_evaluation_points,
                                response_table_b.b_r.row(circuit_index),
                                response_table_b.b_p.row(circuit_index),
                                response_table_b.b_z.row(circuit_index));

    LOG(INFO) << absl::StrFormat("B %2d/%2d: done", circuit_index + 1,
                                 number_of_serial_circuits);
  }  // circuit_index

  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    if (!status[circuit_index].ok()) {
      return status[circuit_index];
    }
  }

  return response_table_b;
}  // ComputeMagneticFieldResponseTable

absl::StatusOr<MakegridCachedVectorPotential> ComputeVectorPotentialCache(
    const MakegridParameters& makegrid_parameters,
    const MagneticConfiguration& magnetic_configuration) {
  absl::StatusOr<RowMatrix3Xd> maybe_cylindrical_grid =
      MakeCylindricalGrid(makegrid_parameters);
  if (!maybe_cylindrical_grid.ok()) {
    return maybe_cylindrical_grid.status();
  }

  // `MakeCylindricalGrid` only computes the grid points for which to actually
  // evaluate the magnetic field. If stellarator symmetry is activated, this
  // might be less grid points that what gets stored in the mgrid file.
  const int number_of_evaluation_points =
      static_cast<int>(maybe_cylindrical_grid.value().cols());

  // shorthand variables for grid dimensions
  const int num_field_periods = makegrid_parameters.number_of_field_periods;
  const int num_phi = makegrid_parameters.number_of_phi_grid_points;
  const int num_z = makegrid_parameters.number_of_z_grid_points;
  const int num_r = makegrid_parameters.number_of_r_grid_points;

  // This is the total number of grid points that is stored in the mgrid file.
  const int total_number_of_grid_points = num_phi * num_z * num_r;

  // precompute toroidal trigonometry tables
  std::vector<double> cos_phi(num_phi);
  std::vector<double> sin_phi(num_phi);
  const double delta_phi = 2.0 * M_PI / (num_field_periods * num_phi);
  for (int index_phi = 0; index_phi < num_phi; ++index_phi) {
    const double phi = index_phi * delta_phi;
    cos_phi[index_phi] = std::cos(phi);
    sin_phi[index_phi] = std::sin(phi);
  }

  // Make a backup of the full vector of original circuit currents,
  // but only after num_windings has potentially been migrated into circuit
  // currents.
  absl::StatusOr<std::vector<double>> maybe_original_currents =
      GetCircuitCurrents(magnetic_configuration);
  if (!maybe_original_currents.ok()) {
    return maybe_original_currents.status();
  }
  const std::vector<double>& original_currents = *maybe_original_currents;

  const int number_of_serial_circuits =
      magnetic_configuration.serial_circuits_size();

  MakegridCachedVectorPotential response_table_a;
  response_table_a.parameters = makegrid_parameters;

  // fully allocate result tables
  response_table_a.a_r.resize(number_of_serial_circuits,
                              total_number_of_grid_points);
  response_table_a.a_p.resize(number_of_serial_circuits,
                              total_number_of_grid_points);
  response_table_a.a_z.resize(number_of_serial_circuits,
                              total_number_of_grid_points);

  std::vector<absl::Status> status(number_of_serial_circuits);

  // Now compute the vector potential for each SerialCircuit individually in the
  // MagneticConfiguration. This is conveniently done by setting the currents to
  // all zeros and only restore the current for the SerialCircuit under
  // consideration to its original value. This has no (significant) performance
  // penalty, since the vector potential contributions are skipped anyway if the
  // current of a given SerialCircuit is identically zero.
  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    // make second internal copy for being able to set inidividually only one
    // circuit current to != 0
    MagneticConfiguration m_magnetic_configuration = magnetic_configuration;

    std::vector<double> currents_for_circuit(number_of_serial_circuits);
    if (makegrid_parameters.normalize_by_currents) {
      currents_for_circuit[circuit_index] = 1.0;
    } else {
      currents_for_circuit[circuit_index] = original_currents[circuit_index];
    }

    absl::Status set_currents_status =
        SetCircuitCurrents(currents_for_circuit, m_magnetic_configuration);
    if (!set_currents_status.ok()) {
      status[circuit_index] = set_currents_status;
      continue;
    }

    // TODO(jurasic) Remove after Eigen refactor
    std::vector<std::vector<double>> vector_potential_stl(
        number_of_evaluation_points);
    for (int i = 0; i < number_of_evaluation_points; ++i) {
      vector_potential_stl[i].resize(3, 0.0);
    }

    // TODO(jurasic) Remove after Eigen refactor
    std::vector<std::vector<double>> cylindrical_grid_stl{
        static_cast<std::size_t>(maybe_cylindrical_grid.value().cols())};
    for (int i = 0; i < number_of_evaluation_points; ++i) {
      cylindrical_grid_stl[i].resize(3);
      for (int j = 0; j < 3; ++j) {
        cylindrical_grid_stl[i][j] = maybe_cylindrical_grid.value()(j, i);
      }
    }

    // We parallelize over linear index of evaluation locations, since that
    // allows us to use more CPUs and parallelize also for configurations with
    // low number of circuits. It is much more common in practice to have few
    // independent circuits and many evaluation locations, rather than many
    // independent circuits but few evaluation locations. This is done inside of
    // ABSCAB, which is used within this call to `VectorPotential`.
    absl::Status vector_potential_status =
        VectorPotential(m_magnetic_configuration, cylindrical_grid_stl,
                        /*m_vector_potential=*/vector_potential_stl);
    if (!vector_potential_status.ok()) {
      status[circuit_index] = vector_potential_status;
      continue;
    }
    RowMatrix3Xd vector_potential =
        RowMatrix3Xd::Zero(3, number_of_evaluation_points);
    vector_potential = vmecpp::ToEigenMatrix(vector_potential_stl).transpose();

    // ABSCAB computes the Cartesian components of the vector potential,
    // so we need to convert the x and y componets into r and phi
    // (cylindrical) components for comparison against the cylindrical
    // components in the mgrid file.
#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
    for (int linear_index = 0; linear_index < number_of_evaluation_points;
         ++linear_index) {
      const double a_x = vector_potential(0, linear_index);
      const double a_y = vector_potential(1, linear_index);
      const double a_z = vector_potential(2, linear_index);

      const size_t index_phi = linear_index / (num_z * num_r);
      const double a_r = a_x * cos_phi[index_phi] + a_y * sin_phi[index_phi];
      const double a_p = a_y * cos_phi[index_phi] - a_x * sin_phi[index_phi];

      response_table_a.a_r(circuit_index, linear_index) = a_r;
      response_table_a.a_p(circuit_index, linear_index) = a_p;
      response_table_a.a_z(circuit_index, linear_index) = a_z;
    }  // linear_index

    // mirror into other stellarator-symmetric part of grid
    // if making use of stellarator symmetry
    if (makegrid_parameters.assume_stellarator_symmetry) {
#ifdef _OPENMP
#pragma omp parallel for
#endif  // _OPENMP
      for (int linear_index = number_of_evaluation_points;
           linear_index < total_number_of_grid_points; ++linear_index) {
        const int idx_phi = linear_index / (num_z * num_r);
        const int idx_z_r = linear_index % (num_z * num_r);
        const int idx_z = idx_z_r / num_r;
        const int idx_r = idx_z_r % num_r;

        const int idx_phi_reversed = num_phi - idx_phi;
        const int idx_z_reversed = num_z - 1 - idx_z;

        const int linear_index_reversed =
            (idx_phi_reversed * num_z + idx_z_reversed) * num_r + idx_r;

        response_table_a.a_r(circuit_index, linear_index) =
            -response_table_a.a_r(circuit_index, linear_index_reversed);
        response_table_a.a_p(circuit_index, linear_index) =
            response_table_a.a_p(circuit_index, linear_index_reversed);
        response_table_a.a_z(circuit_index, linear_index) =
            response_table_a.a_z(circuit_index, linear_index_reversed);
      }
    }

    LOG(INFO) << absl::StrFormat("A %2d/%2d: done", circuit_index + 1,
                                 number_of_serial_circuits);
  }  // circuit_index

  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    if (!status[circuit_index].ok()) {
      return status[circuit_index];
    }
  }

  return response_table_a;
}  // ComputeVectorPotentialCache

absl::Status WriteMakegridNetCDFFile(
    const std::string& makegrid_filename,
    const MakegridParameters& makegrid_parameters,
    const std::vector<double>& circuit_currents,
    const MagneticFieldResponseTable& response_table_b,
    const std::optional<MakegridCachedVectorPotential>& response_table_a) {
  static constexpr int kStringSize = 30;
  static constexpr char kRawCurrents = 'R';
  static constexpr char kNormalizeByCurrents = 'S';

  // number of response tables in this mgrid file
  const int number_of_serial_circuits =
      static_cast<int>(response_table_b.b_r.size());
  CHECK_GT(number_of_serial_circuits, 0)
      << "No magnetic field cache present to be written.";

  int ncid = 0;
  CHECK_EQ(nc_create(makegrid_filename.c_str(), NC_CLOBBER, &ncid), NC_NOERR);

  // create dimensions
  int id_dimension_stringsize = 0;
  CHECK_EQ(
      nc_def_dim(ncid, "stringsize", kStringSize, &id_dimension_stringsize),
      NC_NOERR);

  int id_dimension_external_coil_groups = 0;
  CHECK_EQ(nc_def_dim(ncid, "external_coil_groups", number_of_serial_circuits,
                      &id_dimension_external_coil_groups),
           NC_NOERR);

  int id_dimension_dim_00001 = 0;
  CHECK_EQ(nc_def_dim(ncid, "dim_00001", 1, &id_dimension_dim_00001), NC_NOERR);

  int id_dimension_external_coils = 0;
  CHECK_EQ(nc_def_dim(ncid, "external_coils", number_of_serial_circuits,
                      &id_dimension_external_coils),
           NC_NOERR);

  int id_dimension_rad = 0;
  CHECK_EQ(nc_def_dim(ncid, "rad", makegrid_parameters.number_of_r_grid_points,
                      &id_dimension_rad),
           NC_NOERR);

  int id_dimension_zee = 0;
  CHECK_EQ(nc_def_dim(ncid, "zee", makegrid_parameters.number_of_z_grid_points,
                      &id_dimension_zee),
           NC_NOERR);

  int id_dimension_phi = 0;
  CHECK_EQ(
      nc_def_dim(ncid, "phi", makegrid_parameters.number_of_phi_grid_points,
                 &id_dimension_phi),
      NC_NOERR);

  // create variables
  int id_variable_ir = 0;
  CHECK_EQ(nc_def_var(ncid, "ir", NC_INT, 0, nullptr, &id_variable_ir),
           NC_NOERR);

  int id_variable_jz = 0;
  CHECK_EQ(nc_def_var(ncid, "jz", NC_INT, 0, nullptr, &id_variable_jz),
           NC_NOERR);

  int id_variable_kp = 0;
  CHECK_EQ(nc_def_var(ncid, "kp", NC_INT, 0, nullptr, &id_variable_kp),
           NC_NOERR);

  int id_variable_nfp = 0;
  CHECK_EQ(nc_def_var(ncid, "nfp", NC_INT, 0, nullptr, &id_variable_nfp),
           NC_NOERR);

  int id_variable_nextcur = 0;
  CHECK_EQ(
      nc_def_var(ncid, "nextcur", NC_INT, 0, nullptr, &id_variable_nextcur),
      NC_NOERR);

  int id_variable_rmin = 0;
  CHECK_EQ(nc_def_var(ncid, "rmin", NC_DOUBLE, 0, nullptr, &id_variable_rmin),
           NC_NOERR);

  int id_variable_rmax = 0;
  CHECK_EQ(nc_def_var(ncid, "rmax", NC_DOUBLE, 0, nullptr, &id_variable_rmax),
           NC_NOERR);

  int id_variable_zmin = 0;
  CHECK_EQ(nc_def_var(ncid, "zmin", NC_DOUBLE, 0, nullptr, &id_variable_zmin),
           NC_NOERR);

  int id_variable_zmax = 0;
  CHECK_EQ(nc_def_var(ncid, "zmax", NC_DOUBLE, 0, nullptr, &id_variable_zmax),
           NC_NOERR);

  int id_variable_coil_group = 0;
  std::array<int, 2> coil_group_dimensions = {id_dimension_external_coil_groups,
                                              id_dimension_stringsize};
  CHECK_EQ(nc_def_var(ncid, "coil_group", NC_CHAR, 2,
                      coil_group_dimensions.data(), &id_variable_coil_group),
           NC_NOERR);

  int id_variable_mgrid_mode = 0;
  CHECK_EQ(nc_def_var(ncid, "mgrid_mode", NC_CHAR, 1, &id_dimension_dim_00001,
                      &id_variable_mgrid_mode),
           NC_NOERR);

  int id_variable_raw_coil_cur = 0;
  CHECK_EQ(nc_def_var(ncid, "raw_coil_cur", NC_DOUBLE, 1,
                      &id_dimension_external_coils, &id_variable_raw_coil_cur),
           NC_NOERR);

  std::vector<int> ids_variable_br(number_of_serial_circuits, 0);
  std::vector<int> ids_variable_bp(number_of_serial_circuits, 0);
  std::vector<int> ids_variable_bz(number_of_serial_circuits, 0);
  std::vector<int> ids_variable_ar(number_of_serial_circuits, 0);
  std::vector<int> ids_variable_ap(number_of_serial_circuits, 0);
  std::vector<int> ids_variable_az(number_of_serial_circuits, 0);
  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    std::array<int, 3> grid_dimension = {id_dimension_phi, id_dimension_zee,
                                         id_dimension_rad};

    std::string br_name = absl::StrFormat("br_%03d", circuit_index + 1);
    int id_variable_br = 0;
    CHECK_EQ(nc_def_var(ncid, br_name.c_str(), NC_DOUBLE, 3,
                        grid_dimension.data(), &id_variable_br),
             NC_NOERR);
    ids_variable_br[circuit_index] = id_variable_br;

    std::string bp_name = absl::StrFormat("bp_%03d", circuit_index + 1);
    int id_variable_bp = 0;
    CHECK_EQ(nc_def_var(ncid, bp_name.c_str(), NC_DOUBLE, 3,
                        grid_dimension.data(), &id_variable_bp),
             NC_NOERR);
    ids_variable_bp[circuit_index] = id_variable_bp;

    std::string bz_name = absl::StrFormat("bz_%03d", circuit_index + 1);
    int id_variable_bz = 0;
    CHECK_EQ(nc_def_var(ncid, bz_name.c_str(), NC_DOUBLE, 3,
                        grid_dimension.data(), &id_variable_bz),
             NC_NOERR);
    ids_variable_bz[circuit_index] = id_variable_bz;

    if (response_table_a.has_value()) {
      std::string ar_name = absl::StrFormat("ar_%03d", circuit_index + 1);
      int id_variable_ar = 0;
      CHECK_EQ(nc_def_var(ncid, ar_name.c_str(), NC_DOUBLE, 3,
                          grid_dimension.data(), &id_variable_ar),
               NC_NOERR);
      ids_variable_ar[circuit_index] = id_variable_ar;

      std::string ap_name = absl::StrFormat("ap_%03d", circuit_index + 1);
      int id_variable_ap = 0;
      CHECK_EQ(nc_def_var(ncid, ap_name.c_str(), NC_DOUBLE, 3,
                          grid_dimension.data(), &id_variable_ap),
               NC_NOERR);
      ids_variable_ap[circuit_index] = id_variable_ap;

      std::string az_name = absl::StrFormat("az_%03d", circuit_index + 1);
      int id_variable_az = 0;
      CHECK_EQ(nc_def_var(ncid, az_name.c_str(), NC_DOUBLE, 3,
                          grid_dimension.data(), &id_variable_az),
               NC_NOERR);
      ids_variable_az[circuit_index] = id_variable_az;
    }
  }  // number_of_serial_circuits

  // explicitly end "define mode" and switch over to "data writing mode"
  CHECK_EQ(nc_enddef(ncid), NC_NOERR);

  // write actual data
  CHECK_EQ(nc_put_var(ncid, id_variable_ir,
                      &(makegrid_parameters.number_of_r_grid_points)),
           NC_NOERR);

  CHECK_EQ(nc_put_var(ncid, id_variable_jz,
                      &(makegrid_parameters.number_of_z_grid_points)),
           NC_NOERR);

  CHECK_EQ(nc_put_var(ncid, id_variable_kp,
                      &(makegrid_parameters.number_of_phi_grid_points)),
           NC_NOERR);

  CHECK_EQ(nc_put_var(ncid, id_variable_nfp,
                      &(makegrid_parameters.number_of_field_periods)),
           NC_NOERR);

  CHECK_EQ(nc_put_var(ncid, id_variable_nextcur, &number_of_serial_circuits),
           NC_NOERR);

  CHECK_EQ(
      nc_put_var(ncid, id_variable_rmin, &(makegrid_parameters.r_grid_minimum)),
      NC_NOERR);

  CHECK_EQ(
      nc_put_var(ncid, id_variable_rmax, &(makegrid_parameters.r_grid_maximum)),
      NC_NOERR);

  CHECK_EQ(
      nc_put_var(ncid, id_variable_zmin, &(makegrid_parameters.z_grid_minimum)),
      NC_NOERR);

  CHECK_EQ(
      nc_put_var(ncid, id_variable_zmax, &(makegrid_parameters.z_grid_maximum)),
      NC_NOERR);

  // This is a flat storage of all coil group names.
  // The NetCDF writing routines will interpret this as a two-dimensional array
  // of strings, but since all entries have a fixed length of kStringSize (=30),
  // it is ok to have a one-dimensional storage here.
  std::string coil_group_names;

  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    // TODO(jons): Figure out how the name of the coil groups is determined in
    // MAKEGRID. The challenge is that many coils (with individual names) can
    // belong to one coil group, but only a single name for a coil group is
    // written to the mgrid file.
    std::string coil_group_name = absl::StrFormat("circuit_%d", circuit_index);

    // NOTE: 30 has to be consistent with the value of kStringSize.
    absl::StrAppend(&coil_group_names,
                    absl::StrFormat("%-30s", coil_group_name));
  }  // number_of_serial_circuits
  CHECK_EQ(nc_put_var(ncid, id_variable_coil_group, coil_group_names.c_str()),
           NC_NOERR);

  if (makegrid_parameters.normalize_by_currents) {
    CHECK_EQ(nc_put_var(ncid, id_variable_mgrid_mode, &kNormalizeByCurrents),
             NC_NOERR);
  } else {
    CHECK_EQ(nc_put_var(ncid, id_variable_mgrid_mode, &kRawCurrents), NC_NOERR);
  }

  CHECK_EQ(nc_put_var(ncid, id_variable_raw_coil_cur, circuit_currents.data()),
           NC_NOERR);

  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    CHECK_EQ(nc_put_var(ncid, ids_variable_br[circuit_index],
                        response_table_b.b_r.row(circuit_index).data()),
             NC_NOERR);
    CHECK_EQ(nc_put_var(ncid, ids_variable_bp[circuit_index],
                        response_table_b.b_p.row(circuit_index).data()),
             NC_NOERR);
    CHECK_EQ(nc_put_var(ncid, ids_variable_bz[circuit_index],
                        response_table_b.b_z.row(circuit_index).data()),
             NC_NOERR);

    if (response_table_a.has_value()) {
      CHECK_EQ(nc_put_var(ncid, ids_variable_ar[circuit_index],
                          response_table_a->a_r.row(circuit_index).data()),
               NC_NOERR);
      CHECK_EQ(nc_put_var(ncid, ids_variable_ap[circuit_index],
                          response_table_a->a_p.row(circuit_index).data()),
               NC_NOERR);
      CHECK_EQ(nc_put_var(ncid, ids_variable_az[circuit_index],
                          response_table_a->a_z.row(circuit_index).data()),
               NC_NOERR);
    }
  }  // number_of_serial_circuits

  CHECK_EQ(nc_close(ncid), NC_NOERR);

  return absl::OkStatus();
}  // NOLINT(readability/fn_size)

}  // namespace makegrid
