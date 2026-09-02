// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/magnetic_field_provider/magnetic_field_provider_lib.h"

#include <algorithm>  // fill, max
#include <cstddef>
#include <span>
#include <sstream>
#include <vector>

#include "abscab/abscab.hh"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "vmecpp/common/composed_types_definition/composed_types.h"
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"

namespace magnetics {

using composed_types::CurveRZFourier;
using composed_types::FourierCoefficient1D;
using composed_types::Normalize;
using composed_types::Vector3d;

namespace {

// ABSCAB takes point sets as flat arrays in array-of-structs order
// (x0, y0, z0, x1, y1, z1, ...), and declares its input pointers non-const
// without writing through them, hence the mutable spans below. Converting into
// and out of that order is done once per MagneticConfiguration rather than once
// per current carrier.

std::vector<double> ToAbscabOrder(
    const std::vector<std::vector<double>>& positions) {
  const std::size_t number_of_positions = positions.size();
  std::vector<double> flat(number_of_positions * 3);
  for (std::size_t i = 0; i < number_of_positions; ++i) {
    flat[i * 3 + 0] = positions[i][0];
    flat[i * 3 + 1] = positions[i][1];
    flat[i * 3 + 2] = positions[i][2];
  }
  return flat;
}

std::vector<double> ToAbscabOrder(const RowMatrix3Xd& positions) {
  const auto number_of_positions = static_cast<std::size_t>(positions.cols());
  std::vector<double> flat(number_of_positions * 3);
  for (std::size_t i = 0; i < number_of_positions; ++i) {
    const auto column = static_cast<Eigen::Index>(i);
    flat[i * 3 + 0] = positions(0, column);
    flat[i * 3 + 1] = positions(1, column);
    flat[i * 3 + 2] = positions(2, column);
  }
  return flat;
}

std::vector<double> VerticesInAbscabOrder(
    const PolygonFilament& polygon_filament) {
  const int number_of_vertices = polygon_filament.vertices_size();
  std::vector<double> flat(static_cast<std::size_t>(number_of_vertices) * 3);
  for (int i = 0; i < number_of_vertices; ++i) {
    const Vector3d& vertex = polygon_filament.vertices(i);
    flat[i * 3 + 0] = vertex.x();
    flat[i * 3 + 1] = vertex.y();
    flat[i * 3 + 2] = vertex.z();
  }
  return flat;
}

void AddFromAbscabOrder(const std::vector<double>& contribution,
                        std::vector<std::vector<double>>& m_target) {
  for (std::size_t i = 0; i < m_target.size(); ++i) {
    m_target[i][0] += contribution[i * 3 + 0];
    m_target[i][1] += contribution[i * 3 + 1];
    m_target[i][2] += contribution[i * 3 + 2];
  }
}

void AddFromAbscabOrder(const std::vector<double>& contribution,
                        RowMatrix3Xd& m_target) {
  for (Eigen::Index column = 0; column < m_target.cols(); ++column) {
    const auto i = static_cast<std::size_t>(column);
    m_target(0, column) += contribution[i * 3 + 0];
    m_target(1, column) += contribution[i * 3 + 1];
    m_target(2, column) += contribution[i * 3 + 2];
  }
}

// The contribution of a single current carrier, added into m_contribution in
// ABSCAB order. The current carrier is taken to have been checked already.

void AddMagneticField(
    const InfiniteStraightFilament& infinite_straight_filament, double current,
    std::span<double> evaluation_positions, std::span<double> m_contribution) {
  if (current == 0.0) {
    // no current -> no contribution
    return;
  }
  const double magnetic_field_scale = abscab::MU_0 * current / (2.0 * M_PI);

  const Vector3d& direction = infinite_straight_filament.direction();
  const double direction_x = direction.x();
  const double direction_y = direction.y();
  const double direction_z = direction.z();
  const double direction_length =
      std::hypot(direction_x, direction_y, direction_z);

  // Make sure that we were given a well-defined direction vector.
  CHECK_GT(direction_length, 0.0);

  // unit vector in direction of filament
  const double normalized_direction_x = direction_x / direction_length;
  const double normalized_direction_y = direction_y / direction_length;
  const double normalized_direction_z = direction_z / direction_length;

  const Vector3d& origin = infinite_straight_filament.origin();
  const double origin_x = origin.x();
  const double origin_y = origin.y();
  const double origin_z = origin.z();

  const std::size_t num_evaluation_locations = evaluation_positions.size() / 3;
  for (std::size_t i = 0; i < num_evaluation_locations; ++i) {
    const double evaluation_position_x = evaluation_positions[i * 3 + 0];
    const double evaluation_position_y = evaluation_positions[i * 3 + 1];
    const double evaluation_position_z = evaluation_positions[i * 3 + 2];

    // connection vector from evaluation position to origin on filament
    const double delta_eval_origin_x = origin_x - evaluation_position_x;
    const double delta_eval_origin_y = origin_y - evaluation_position_y;
    const double delta_eval_origin_z = origin_z - evaluation_position_z;

    // distance between evaluation position and origin on filament
    // parallel to filament direction
    const double parallel_distance =
        (delta_eval_origin_x * normalized_direction_x +
         delta_eval_origin_y * normalized_direction_y +
         delta_eval_origin_z * normalized_direction_z);

    // connector vector, projected onto the filament direction
    const double delta_parallel_x = normalized_direction_x * parallel_distance;
    const double delta_parallel_y = normalized_direction_y * parallel_distance;
    const double delta_parallel_z = normalized_direction_z * parallel_distance;

    // vector from evaluation position to filament,
    // perpendicular to filament
    const double delta_perpendicular_x = delta_eval_origin_x - delta_parallel_x;
    const double delta_perpendicular_y = delta_eval_origin_y - delta_parallel_y;
    const double delta_perpendicular_z = delta_eval_origin_z - delta_parallel_z;

    // radial distance from filament to evaluation position
    const double evaluation_position_radius = std::hypot(
        delta_perpendicular_x, delta_perpendicular_y, delta_perpendicular_z);

    // The magnetic field is not defined on the filament,
    // so must check that radius is > 0.
    CHECK_GT(evaluation_position_radius, 0.0);

    // Magnetic field strength of infinite straight filament,
    // cylindrical phi component in coordinate system of filament.
    const double magnetic_field_strength =
        magnetic_field_scale / evaluation_position_radius;

    // radial unit vector at evaluation location,
    // in coordinate system of filament
    const double radial_unit_vector_x =
        delta_perpendicular_x / evaluation_position_radius;
    const double radial_unit_vector_y =
        delta_perpendicular_y / evaluation_position_radius;
    const double radial_unit_vector_z =
        delta_perpendicular_z / evaluation_position_radius;

    // e_phi: unit vector in direction of magnetic field at evaluation location
    // Assume that radial_unit_vector and normalized_direction are unit vectors.
    // --> Can omit check/rescaling to ensure that toroidal_unit_vector has unit
    // length.
    const double toroidal_unit_vector_x =
        radial_unit_vector_y * normalized_direction_z -
        radial_unit_vector_z * normalized_direction_y;
    const double toroidal_unit_vector_y =
        radial_unit_vector_z * normalized_direction_x -
        radial_unit_vector_x * normalized_direction_z;
    const double toroidal_unit_vector_z =
        radial_unit_vector_x * normalized_direction_y -
        radial_unit_vector_y * normalized_direction_x;

    // compute magnetic field vector by scaling correct unit vector to correct
    // length
    const double magnetic_field_vector_x =
        toroidal_unit_vector_x * magnetic_field_strength;
    const double magnetic_field_vector_y =
        toroidal_unit_vector_y * magnetic_field_strength;
    const double magnetic_field_vector_z =
        toroidal_unit_vector_z * magnetic_field_strength;

    // add to target storage
    m_contribution[i * 3 + 0] += magnetic_field_vector_x;
    m_contribution[i * 3 + 1] += magnetic_field_vector_y;
    m_contribution[i * 3 + 2] += magnetic_field_vector_z;
  }
}  // AddMagneticField for InfiniteStraightFilament

void AddMagneticField(const CircularFilament& circular_filament, double current,
                      std::span<double> evaluation_positions,
                      std::span<double> m_contribution) {
  const Vector3d& center_vector = circular_filament.center();
  std::vector<double> center = {
      center_vector.x(),
      center_vector.y(),
      center_vector.z(),
  };

  const Vector3d& normal_vector = circular_filament.normal();
  std::vector<double> normal = {
      normal_vector.x(),
      normal_vector.y(),
      normal_vector.z(),
  };

  const double radius = circular_filament.radius();

  abscab::magneticFieldCircularFilament(
      center.data(), normal.data(), radius, current,
      static_cast<int>(evaluation_positions.size() / 3),
      evaluation_positions.data(), m_contribution.data());
}  // AddMagneticField for CircularFilament

void AddMagneticField(const PolygonFilament& polygon_filament, double current,
                      std::span<double> evaluation_positions,
                      std::span<double> m_contribution) {
  std::vector<double> vertices = VerticesInAbscabOrder(polygon_filament);

  abscab::magneticFieldPolygonFilament(
      polygon_filament.vertices_size(), vertices.data(), current,
      static_cast<int>(evaluation_positions.size() / 3),
      evaluation_positions.data(), m_contribution.data());
}  // AddMagneticField for PolygonFilament

void AddVectorPotential(const CircularFilament& circular_filament,
                        double current, std::span<double> evaluation_positions,
                        std::span<double> m_contribution) {
  const Vector3d& center_vector = circular_filament.center();
  std::vector<double> center = {
      center_vector.x(),
      center_vector.y(),
      center_vector.z(),
  };

  const Vector3d& normal_vector = circular_filament.normal();
  std::vector<double> normal = {
      normal_vector.x(),
      normal_vector.y(),
      normal_vector.z(),
  };

  const double radius = circular_filament.radius();

  // Negated because abscab's circular-filament vector potential has the
  // opposite sign convention to its polygon-filament one and to both of its
  // magnetic-field routines. With the negation, A is parallel to the current
  // that produces it, which agrees with MAKEGRID and with the closed form
  // checked in VectorPotential.CheckCircularFilament.
  abscab::vectorPotentialCircularFilament(
      center.data(), normal.data(), radius, -current,
      static_cast<int>(evaluation_positions.size() / 3),
      evaluation_positions.data(), m_contribution.data());
}  // AddVectorPotential for CircularFilament

void AddVectorPotential(const PolygonFilament& polygon_filament, double current,
                        std::span<double> evaluation_positions,
                        std::span<double> m_contribution) {
  std::vector<double> vertices = VerticesInAbscabOrder(polygon_filament);

  abscab::vectorPotentialPolygonFilament(
      polygon_filament.vertices_size(), vertices.data(), current,
      static_cast<int>(evaluation_positions.size() / 3),
      evaluation_positions.data(), m_contribution.data());
}  // AddVectorPotential for PolygonFilament

// The current of a coil within a serial circuit.
// NOTE: Re-compute the circuit current "from scratch" in every iteration.
// Otherwise, the number of winding of the different coils
// all get multiplied on top of each other for each successive coil!
double CoilCurrent(const SerialCircuit& serial_circuit, const Coil& coil) {
  if (coil.has_num_windings()) {
    return serial_circuit.current() * coil.num_windings();
  }
  // assume num_windings = 1, if not provided
  return serial_circuit.current();
}

absl::Status UnsupportedCurrentCarrier(const CurrentCarrier& current_carrier) {
  std::stringstream error_message;
  error_message << "current carrier type ";
  error_message << current_carrier.type_case();
  error_message << " not implemented yet.";
  return absl::InvalidArgumentError(error_message.str());
}

// Walk the current carriers of a MagneticConfiguration and add each
// contribution into m_target. Every carrier gets a freshly zeroed scratch
// buffer that is added on afterwards, so contributions are summed in the same
// order and with the same rounding as when each carrier is evaluated on its
// own.
template <typename Target>
absl::Status AccumulateMagneticField(
    const MagneticConfiguration& magnetic_configuration,
    std::span<double> evaluation_positions, Target& m_target) {
  std::vector<double> contribution(evaluation_positions.size(), 0.0);

  for (const SerialCircuit& serial_circuit :
       magnetic_configuration.serial_circuits()) {
    if (!serial_circuit.has_current() || serial_circuit.current() == 0.0) {
      // skip contributions with assumed zero current
      continue;
    }

    for (const Coil& coil : serial_circuit.coils()) {
      const double current = CoilCurrent(serial_circuit, coil);

      for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
        std::fill(contribution.begin(), contribution.end(), 0.0);
        switch (current_carrier.type_case()) {
          case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
            AddMagneticField(current_carrier.infinite_straight_filament(),
                             current, evaluation_positions, contribution);
            break;
          case CurrentCarrier::TypeCase::kCircularFilament:
            AddMagneticField(current_carrier.circular_filament(), current,
                             evaluation_positions, contribution);
            break;
          case CurrentCarrier::TypeCase::kPolygonFilament:
            AddMagneticField(current_carrier.polygon_filament(), current,
                             evaluation_positions, contribution);
            break;
          case CurrentCarrier::TypeCase::kTypeNotSet:
            // consider as empty CurrentCarrier -> ignore
            continue;
          default:
            return UnsupportedCurrentCarrier(current_carrier);
        }
        AddFromAbscabOrder(contribution, m_target);
      }  // CurrentCarrier
    }  // Coil
  }  // SerialCircuit

  return absl::OkStatus();
}  // AccumulateMagneticField

template <typename Target>
absl::Status AccumulateVectorPotential(
    const MagneticConfiguration& magnetic_configuration,
    std::span<double> evaluation_positions, Target& m_target) {
  std::vector<double> contribution(evaluation_positions.size(), 0.0);

  for (const SerialCircuit& serial_circuit :
       magnetic_configuration.serial_circuits()) {
    if (!serial_circuit.has_current() || serial_circuit.current() == 0.0) {
      // skip contributions with assumed zero current
      continue;
    }

    for (const Coil& coil : serial_circuit.coils()) {
      const double current = CoilCurrent(serial_circuit, coil);

      for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
        std::fill(contribution.begin(), contribution.end(), 0.0);
        switch (current_carrier.type_case()) {
          case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
            // The magnetic vector potential diverges for an infinite straight
            // filament, so do not compute a contribution from it here. This
            // should have been checked for alreay above, but programmers look
            // both ways in a one-way street...
            LOG(FATAL) << "Cannot compute the magnetic vector potential of an "
                          "infinite straight filament.";
            break;
          case CurrentCarrier::TypeCase::kCircularFilament:
            AddVectorPotential(current_carrier.circular_filament(), current,
                               evaluation_positions, contribution);
            break;
          case CurrentCarrier::TypeCase::kPolygonFilament:
            AddVectorPotential(current_carrier.polygon_filament(), current,
                               evaluation_positions, contribution);
            break;
          case CurrentCarrier::TypeCase::kTypeNotSet:
            // consider as empty CurrentCarrier -> ignore
            continue;
          default:
            return UnsupportedCurrentCarrier(current_carrier);
        }
        AddFromAbscabOrder(contribution, m_target);
      }  // CurrentCarrier
    }  // Coil
  }  // SerialCircuit

  return absl::OkStatus();
}  // AccumulateVectorPotential

// The magnetic vector potential diverges for an infinite straight filament, so
// a MagneticConfiguration carrying one has no vector potential to report.
absl::Status CheckFreeOfInfiniteStraightFilaments(
    const MagneticConfiguration& magnetic_configuration) {
  for (const SerialCircuit& serial_circuit :
       magnetic_configuration.serial_circuits()) {
    for (const Coil& coil : serial_circuit.coils()) {
      for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
        if (current_carrier.has_infinite_straight_filament()) {
          return absl::InvalidArgumentError(
              "Cannot compute the magnetic vector potential of an infinite "
              "straight filament.");
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status MagneticField(
    const InfiniteStraightFilament& infinite_straight_filament, double current,
    const std::vector<std::vector<double>>& evaluation_positions,
    std::vector<std::vector<double>>& m_magnetic_field,
    bool check_current_carrier) {
  if (check_current_carrier) {
    absl::Status status =
        IsInfiniteStraightFilamentFullyPopulated(infinite_straight_filament);
    if (!status.ok()) {
      // Do not modify m_magnetic_field if the current carrier is not
      // well-defined.
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

  AddMagneticField(infinite_straight_filament, current,
                   evaluation_positions_flat, contribution);
  AddFromAbscabOrder(contribution, m_magnetic_field);

  return absl::OkStatus();
}  // MagneticField for InfiniteStraightFilament

absl::Status MagneticField(
    const CircularFilament& circular_filament, double current,
    const std::vector<std::vector<double>>& evaluation_positions,
    std::vector<std::vector<double>>& m_magnetic_field,
    bool check_current_carrier) {
  if (check_current_carrier) {
    absl::Status status = IsCircularFilamentFullyPopulated(circular_filament);
    if (!status.ok()) {
      // Do not modify m_magnetic_field if the current carrier is not
      // well-defined.
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

  AddMagneticField(circular_filament, current, evaluation_positions_flat,
                   contribution);
  AddFromAbscabOrder(contribution, m_magnetic_field);

  return absl::OkStatus();
}  // MagneticField for CircularFilament

absl::Status MagneticField(
    const PolygonFilament& polygon_filament, double current,
    const std::vector<std::vector<double>>& evaluation_positions,
    std::vector<std::vector<double>>& m_magnetic_field,
    bool check_current_carrier) {
  if (check_current_carrier) {
    absl::Status status = IsPolygonFilamentFullyPopulated(polygon_filament);
    if (!status.ok()) {
      // Do not modify m_magnetic_field if the current carrier is not
      // well-defined.
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

  AddMagneticField(polygon_filament, current, evaluation_positions_flat,
                   contribution);
  AddFromAbscabOrder(contribution, m_magnetic_field);

  return absl::OkStatus();
}  // MagneticField for PolygonFilament

absl::Status MagneticField(
    const MagneticConfiguration& magnetic_configuration,
    const std::vector<std::vector<double>>& evaluation_positions,
    std::vector<std::vector<double>>& m_magnetic_field,
    bool check_current_carrier) {
  if (check_current_carrier) {
    absl::Status status =
        IsMagneticConfigurationFullyPopulated(magnetic_configuration);
    if (!status.ok()) {
      // Do not modify m_magnetic_field if the current carrier is not
      // well-defined.
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  return AccumulateMagneticField(magnetic_configuration,
                                 evaluation_positions_flat, m_magnetic_field);
}  // MagneticField for MagneticConfiguration

absl::Status MagneticField(const MagneticConfiguration& magnetic_configuration,
                           const RowMatrix3Xd& evaluation_positions,
                           RowMatrix3Xd& m_magnetic_field,
                           bool check_current_carrier) {
  CHECK_EQ(m_magnetic_field.cols(), evaluation_positions.cols())
      << "one magnetic field vector per evaluation position is required";

  if (check_current_carrier) {
    absl::Status status =
        IsMagneticConfigurationFullyPopulated(magnetic_configuration);
    if (!status.ok()) {
      // Do not modify m_magnetic_field if the current carrier is not
      // well-defined.
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  return AccumulateMagneticField(magnetic_configuration,
                                 evaluation_positions_flat, m_magnetic_field);
}  // MagneticField for MagneticConfiguration, Eigen layout

// ----------------

absl::Status VectorPotential(
    const CircularFilament& circular_filament, double current,
    const std::vector<std::vector<double>>& evaluation_positions,
    std::vector<std::vector<double>>& m_vector_potential,
    bool check_current_carrier) {
  if (check_current_carrier) {
    absl::Status status = IsCircularFilamentFullyPopulated(circular_filament);
    if (!status.ok()) {
      // Do not modify m_vector_potential if the current carrier is not
      // well-defined.
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

  AddVectorPotential(circular_filament, current, evaluation_positions_flat,
                     contribution);
  AddFromAbscabOrder(contribution, m_vector_potential);

  return absl::OkStatus();
}  // VectorPotential for CircularFilament

absl::Status VectorPotential(
    const PolygonFilament& polygon_filament, double current,
    const std::vector<std::vector<double>>& evaluation_positions,
    std::vector<std::vector<double>>& m_vector_potential,
    bool check_current_carrier) {
  if (check_current_carrier) {
    absl::Status status = IsPolygonFilamentFullyPopulated(polygon_filament);
    if (!status.ok()) {
      // Do not modify m_vector_potential if the current carrier is not
      // well-defined.
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  std::vector<double> contribution(evaluation_positions_flat.size(), 0.0);

  AddVectorPotential(polygon_filament, current, evaluation_positions_flat,
                     contribution);
  AddFromAbscabOrder(contribution, m_vector_potential);

  return absl::OkStatus();
}  // VectorPotential for PolygonFilament

absl::Status VectorPotential(
    const MagneticConfiguration& magnetic_configuration,
    const std::vector<std::vector<double>>& evaluation_positions,
    std::vector<std::vector<double>>& m_vector_potential,
    bool check_current_carrier) {
  if (check_current_carrier) {
    absl::Status status =
        IsMagneticConfigurationFullyPopulated(magnetic_configuration);
    if (!status.ok()) {
      // Do not modify m_vector_potential if the current carrier is not
      // well-defined.
      return status;
    }

    status = CheckFreeOfInfiniteStraightFilaments(magnetic_configuration);
    if (!status.ok()) {
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  return AccumulateVectorPotential(
      magnetic_configuration, evaluation_positions_flat, m_vector_potential);
}  // VectorPotential for MagneticConfiguration

absl::Status VectorPotential(
    const MagneticConfiguration& magnetic_configuration,
    const RowMatrix3Xd& evaluation_positions, RowMatrix3Xd& m_vector_potential,
    bool check_current_carrier) {
  CHECK_EQ(m_vector_potential.cols(), evaluation_positions.cols())
      << "one vector potential per evaluation position is required";

  if (check_current_carrier) {
    absl::Status status =
        IsMagneticConfigurationFullyPopulated(magnetic_configuration);
    if (!status.ok()) {
      // Do not modify m_vector_potential if the current carrier is not
      // well-defined.
      return status;
    }

    status = CheckFreeOfInfiniteStraightFilaments(magnetic_configuration);
    if (!status.ok()) {
      return status;
    }
  }

  std::vector<double> evaluation_positions_flat =
      ToAbscabOrder(evaluation_positions);
  return AccumulateVectorPotential(
      magnetic_configuration, evaluation_positions_flat, m_vector_potential);
}  // VectorPotential for MagneticConfiguration, Eigen layout

absl::StatusOr<double> LinkingCurrent(
    const MagneticConfiguration& magnetic_configuration,
    const CurveRZFourier& axis_coefficients) {
  static constexpr double kMu0 = 4.0e-7 * M_PI;

  // check that axis geometry is fully populated
  // and same number of coefficients is present for R and Z
  absl::Status status = IsCurveRZFourierFullyPopulated(axis_coefficients);
  if (!status.ok()) {
    return status;
  }
  const int num_coefficients = axis_coefficients.r_size();

  // Find maximum Fourier mode number in order to then choose number of toroidal
  // grid points along axis accordingly.
  int maximum_mode_number = 0;
  for (int coefficient_index = 0; coefficient_index < num_coefficients;
       ++coefficient_index) {
    maximum_mode_number =
        std::max(maximum_mode_number,
                 axis_coefficients.r(coefficient_index).mode_number());
    maximum_mode_number =
        std::max(maximum_mode_number,
                 axis_coefficients.z(coefficient_index).mode_number());
  }

  // number of toroidal grid points along axis:
  // two times above the Nyquist limit - should be safe,
  // but still fast enough for practical applications.
  const int num_axis_points = 2 * (2 * maximum_mode_number + 1) *
                              magnetic_configuration.num_field_periods();

  // Compute the axis geometry in realspace
  // from the Fourier coefficients in the axis CSV file.
  std::vector<std::vector<double>> axis_points(num_axis_points);
  std::vector<std::vector<double>> axis_tangent(num_axis_points);
  const double delta_phi = 2.0 * M_PI / num_axis_points;
  for (int k = 0; k < num_axis_points; ++k) {
    const double cos_phi = std::cos(k * delta_phi);
    const double sin_phi = std::sin(k * delta_phi);

    double axis_point_r = 0.0;
    double axis_point_z = 0.0;
    double axis_tangent_r = 0.0;
    double axis_tangent_z = 0.0;
    for (int coefficient_index = 0; coefficient_index < num_coefficients;
         ++coefficient_index) {
      const FourierCoefficient1D& coeff_r =
          axis_coefficients.r(coefficient_index);
      const FourierCoefficient1D& coeff_z =
          axis_coefficients.z(coefficient_index);

      // mode numbers have been checked to be the same for R and Z in
      // `IsCurveRZFourierFullyPopulated`
      const int mode_number = coeff_r.mode_number();

      const double kernel = k * mode_number * delta_phi;
      const double cos_kernel = std::cos(kernel);
      const double sin_kernel = std::sin(kernel);

      if (coeff_r.has_fc_cos()) {
        const double coeff = coeff_r.fc_cos();
        axis_point_r += coeff * cos_kernel;
        axis_tangent_r += coeff * mode_number * (-sin_kernel);
      }
      if (coeff_r.has_fc_sin()) {
        const double coeff = coeff_r.fc_sin();
        axis_point_r += coeff * sin_kernel;
        axis_tangent_r += coeff * mode_number * cos_kernel;
      }

      if (coeff_z.has_fc_cos()) {
        const double coeff = coeff_z.fc_cos();
        axis_point_z += coeff * cos_kernel;
        axis_tangent_z += coeff * mode_number * (-sin_kernel);
      }
      if (coeff_z.has_fc_sin()) {
        const double coeff = coeff_z.fc_sin();
        axis_point_z += coeff * sin_kernel;
        axis_tangent_z += coeff * mode_number * cos_kernel;
      }
    }

    const double axis_point_x = axis_point_r * cos_phi;
    const double axis_point_y = axis_point_r * sin_phi;
    axis_points[k] = {axis_point_x, axis_point_y, axis_point_z};

    const double axis_tangent_x =
        axis_tangent_r * cos_phi - axis_point_r * sin_phi;
    const double axis_tangent_y =
        axis_tangent_r * sin_phi + axis_point_r * cos_phi;
    axis_tangent[k] = {axis_tangent_x, axis_tangent_y, axis_tangent_z};
  }

  // for all points along the axis, evaluate the total magnetic field from all
  // coils, weighted by circuit currents
  std::vector<std::vector<double>> magnetic_field(num_axis_points,
                                                  std::vector<double>(3));
  status = MagneticField(magnetic_configuration, axis_points,
                         /*m_magnetic_field=*/magnetic_field);
  if (!status.ok()) {
    return status;
  }

  // Compute the line integral of (B \cdot tangent) along the axis:
  // \oint B . dl == \oint B . d(x)/d(phi) * d(phi)
  // and axis_tangent == d(x)/d(phi)
  double linking_current = 0.0;
  for (int k = 0; k < num_axis_points; ++k) {
    const double b_dot_tangent = magnetic_field[k][0] * axis_tangent[k][0] +
                                 magnetic_field[k][1] * axis_tangent[k][1] +
                                 magnetic_field[k][2] * axis_tangent[k][2];
    linking_current += b_dot_tangent;
  }
  // d(phi) == (2 pi) / num_axis_points is the differential of the loop integral
  linking_current *= 2.0 * M_PI / num_axis_points;

  // convert \oint B.dl into units of Amperes
  linking_current /= kMu0;

  return linking_current;
}  // LinkingCurrent

}  // namespace magnetics
