// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"

#include <netcdf.h>

#include <cmath>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "util/netcdf_io/netcdf_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"
#define ASSERT_OK(quantity) ASSERT_TRUE((quantity).ok()) << (quantity).status();

namespace makegrid {

using composed_types::DotProduct;
using composed_types::Length;
using composed_types::Normalize;
using composed_types::ScaleTo;
using composed_types::Subtract;
using composed_types::Vector3d;

using file_io::ReadFile;

using netcdf_io::NetcdfReadArray3D;
using netcdf_io::NetcdfReadBool;
using netcdf_io::NetcdfReadDouble;
using netcdf_io::NetcdfReadInt;

using magnetics::ImportMagneticConfigurationFromCoilsFile;

using magnetics::CircularFilament;
using magnetics::Coil;
using magnetics::CurrentCarrier;
using magnetics::InfiniteStraightFilament;
using magnetics::MagneticConfiguration;
using magnetics::PolygonFilament;
using magnetics::SerialCircuit;

using testing::IsCloseRelAbs;

template <typename Derived>
std::vector<std::vector<double>> EigenToStl(
    const Eigen::DenseBase<Derived>& matrix) {
  std::vector<std::vector<double>> stl_matrix(matrix.rows());

  for (int i = 0; i < matrix.rows(); ++i) {
    stl_matrix[i].resize(matrix.cols());
    for (int j = 0; j < matrix.cols(); ++j) {
      stl_matrix[i][j] = matrix(i, j);
    }
  }

  return stl_matrix;
}

TEST(TestMakegridLib, CheckMakeCylindricalGridSanityChecks) {
  // Knudge each of these parameters outside their allowed ranges, one at a
  // time, and test if MakeCylindricalGrid is able to detect the error.

  MakegridParameters makegrid_parameters = {
      // corresponding to mgrid_mode = 'R'
      .normalize_by_currents = false, .assume_stellarator_symmetry = false,
      .number_of_field_periods = 5,   .r_grid_minimum = 1.0,
      .r_grid_maximum = 2.0,          .number_of_r_grid_points = 11,
      .z_grid_minimum = -0.6,         .z_grid_maximum = 0.6,
      .number_of_z_grid_points = 13,  .number_of_phi_grid_points = 18};

  // both Boolean options are ok for normalize_by_currents

  // both Boolean options are ok for assume_stellarator_symmetry

  MakegridParameters makegrid_parameters_nfp = makegrid_parameters;
  makegrid_parameters_nfp.number_of_field_periods = 0;
  auto cylindrical_grid_nfp = MakeCylindricalGrid(makegrid_parameters_nfp);
  ASSERT_FALSE(cylindrical_grid_nfp.ok());

  MakegridParameters makegrid_parameters_rmin = makegrid_parameters;
  makegrid_parameters_rmin.r_grid_minimum = 3.0;
  auto cylindrical_grid_rmin = MakeCylindricalGrid(makegrid_parameters_rmin);
  ASSERT_FALSE(cylindrical_grid_rmin.ok());

  MakegridParameters makegrid_parameters_numr = makegrid_parameters;
  makegrid_parameters_numr.number_of_r_grid_points = 1;
  auto cylindrical_grid_numr = MakeCylindricalGrid(makegrid_parameters_numr);
  ASSERT_FALSE(cylindrical_grid_numr.ok());

  MakegridParameters makegrid_parameters_zmin = makegrid_parameters;
  makegrid_parameters_zmin.z_grid_maximum = -1.0;
  auto cylindrical_grid_zmin = MakeCylindricalGrid(makegrid_parameters_zmin);
  ASSERT_FALSE(cylindrical_grid_zmin.ok());

  MakegridParameters makegrid_parameters_numz = makegrid_parameters;
  makegrid_parameters_numz.number_of_z_grid_points = 1;
  auto cylindrical_grid_numz = MakeCylindricalGrid(makegrid_parameters_numz);
  ASSERT_FALSE(cylindrical_grid_numz.ok());

  MakegridParameters makegrid_parameters_numphi = makegrid_parameters;
  makegrid_parameters_numphi.number_of_phi_grid_points = 0;
  auto cylindrical_grid_numphi =
      MakeCylindricalGrid(makegrid_parameters_numphi);
  ASSERT_FALSE(cylindrical_grid_numphi.ok());
}  // CheckMakeCylindricalGridSanityChecks

TEST(TestMakegridLib, CheckMakeCylindricalGrid) {
  static constexpr double kTolerance = 1.0e-15;

  MakegridParameters makegrid_parameters = {
      // corresponding to mgrid_mode = 'R'
      .normalize_by_currents = false, .assume_stellarator_symmetry = true,
      .number_of_field_periods = 5,   .r_grid_minimum = 1.0,
      .r_grid_maximum = 2.0,          .number_of_r_grid_points = 11,
      .z_grid_minimum = -0.6,         .z_grid_maximum = 0.6,
      .number_of_z_grid_points = 13,  .number_of_phi_grid_points = 18};

  // for now, make sure that the struct initialization above correctly
  // identified the members
  ASSERT_FALSE(makegrid_parameters.normalize_by_currents);
  ASSERT_TRUE(makegrid_parameters.assume_stellarator_symmetry);
  ASSERT_EQ(makegrid_parameters.number_of_field_periods, 5);
  ASSERT_EQ(makegrid_parameters.r_grid_minimum, 1.0);
  ASSERT_EQ(makegrid_parameters.r_grid_maximum, 2.0);
  ASSERT_EQ(makegrid_parameters.number_of_r_grid_points, 11);
  ASSERT_EQ(makegrid_parameters.z_grid_minimum, -0.6);
  ASSERT_EQ(makegrid_parameters.z_grid_maximum, 0.6);
  ASSERT_EQ(makegrid_parameters.number_of_z_grid_points, 13);
  ASSERT_EQ(makegrid_parameters.number_of_phi_grid_points, 18);

  absl::StatusOr<RowMatrix3Xd> cylindrical_grid_eigen =
      MakeCylindricalGrid(makegrid_parameters);
  ASSERT_OK(cylindrical_grid_eigen);

  // MakeCylindricalGrid() returns a 3xN matrix, instead of Nx3, so we tranpose
  // to keep the test the same:
  absl::StatusOr<std::vector<std::vector<double>>> cylindrical_grid =
      EigenToStl(cylindrical_grid_eigen.value().transpose());

  int num_phi_effective = makegrid_parameters.number_of_phi_grid_points;
  if (makegrid_parameters.assume_stellarator_symmetry) {
    ASSERT_EQ(makegrid_parameters.number_of_phi_grid_points % 2, 0);
    num_phi_effective = makegrid_parameters.number_of_phi_grid_points / 2 + 1;
  }

  const int expected_total_number_of_grid_points =
      num_phi_effective * makegrid_parameters.number_of_z_grid_points *
      makegrid_parameters.number_of_r_grid_points;

  const int number_of_rz_grid_points =
      makegrid_parameters.number_of_z_grid_points *
      makegrid_parameters.number_of_r_grid_points;

  const double r_grid_increment =
      (makegrid_parameters.r_grid_maximum -
       makegrid_parameters.r_grid_minimum) /
      (makegrid_parameters.number_of_r_grid_points - 1.0);

  const double z_grid_increment =
      (makegrid_parameters.z_grid_maximum -
       makegrid_parameters.z_grid_minimum) /
      (makegrid_parameters.number_of_z_grid_points - 1.0);

  const double phi_grid_increment =
      2.0 * M_PI /
      (makegrid_parameters.number_of_field_periods *
       makegrid_parameters.number_of_phi_grid_points);

  ASSERT_EQ(cylindrical_grid->size(), expected_total_number_of_grid_points);
  for (int i = 0; i < expected_total_number_of_grid_points; ++i) {
    ASSERT_EQ((*cylindrical_grid)[i].size(), 3);

    const int phi_index = i / number_of_rz_grid_points;
    const int rz_index = i % number_of_rz_grid_points;
    const int z_index = rz_index / makegrid_parameters.number_of_r_grid_points;
    const int r_index = rz_index % makegrid_parameters.number_of_r_grid_points;

    const double r =
        makegrid_parameters.r_grid_minimum + r_index * r_grid_increment;
    const double phi = phi_index * phi_grid_increment;
    const double z =
        makegrid_parameters.z_grid_minimum + z_index * z_grid_increment;

    const double x = r * std::cos(phi);
    const double y = r * std::sin(phi);

    EXPECT_TRUE(IsCloseRelAbs((*cylindrical_grid)[i][0], x, kTolerance));
    EXPECT_TRUE(IsCloseRelAbs((*cylindrical_grid)[i][1], y, kTolerance));
    EXPECT_TRUE(IsCloseRelAbs((*cylindrical_grid)[i][2], z, kTolerance));
  }  // i
}  // CheckMakeCylindricalGrid

// For a CircularFilament, i.e., a circular wire loop, and an evaluation point
// at rho' = rho / a, z' = z / a, where a is the radius of the wire loop and
// rho, z are the cylindrical coordinates in the coordinate system of the
// straight wire segment, exclude an evaluation point
// * for A_phi if (rho' < 1e-15 or (z' < 1 and 0.5 < rho' < 2)) --> based on
// slide 46
// * for B_rho if (rho' < 1e-15 or (z' < 1 and 0.5 < rho' < 2)) --> based on
// slide 53
// * for B_z   if (z' < 1 and 0.5 < rho' < 2)) --> based on slide 60
// Since the magnetic field is always composed of the B_rho and B_z components
// and since the criteria are the same for A_phi and B_rho,
// we always test for the same criterion, no matter if we compare
// the magnetic field or the vector potential.
absl::Status DetermineIfTooCloseToCurrentCarrierForComparison(
    const CircularFilament& circular_filament,
    const std::vector<std::vector<double>>& evaluation_locations,
    std::vector<bool>& m_exclude_from_comparison) {
  static constexpr double kRhoMin = 1.0e-15;

  static constexpr double kTooCloseDistance = 1.0e-2;
  static constexpr double kZMax = kTooCloseDistance;
  static constexpr double kRhoZMin = 1.0 - kTooCloseDistance;
  static constexpr double kRhoZMax = 1.0 + kTooCloseDistance;

  const std::size_t number_of_evaluation_locations =
      evaluation_locations.size();
  if (number_of_evaluation_locations == 0) {
    return absl::InvalidArgumentError(
        "An empty vector of evaluation locations was provided.");
  }

  Vector3d normalized_normal = Normalize(circular_filament.normal());
  const double radius = circular_filament.radius();

  for (std::size_t i = 0; i < number_of_evaluation_locations; ++i) {
    Vector3d evaluation_location;
    evaluation_location.set_x(evaluation_locations[i][0]);
    evaluation_location.set_y(evaluation_locations[i][1]);
    evaluation_location.set_z(evaluation_locations[i][2]);

    // connection vector from center of loop to evaluation position
    Vector3d delta_eval_origin =
        Subtract(evaluation_location, circular_filament.center());

    // distance between evaluation position and center of loop, parallel to
    // filament direction
    // -> z of evaluation location in coordinate system of loop
    const double parallel_distance =
        DotProduct(delta_eval_origin, normalized_normal);

    // connector vector, projected onto the filament direction
    Vector3d delta_parallel = ScaleTo(normalized_normal, parallel_distance);

    // vector from evaluation position to filament, perpendicular to filament
    Vector3d delta_perpendicular = Subtract(delta_eval_origin, delta_parallel);

    // radial distance from filament to evaluation position
    // -> rho of evaluation location in coordinate system of loop
    const double evaluation_position_radius = Length(delta_perpendicular);

    const double normalized_z = parallel_distance / radius;
    const double normalized_rho = evaluation_position_radius / radius;

    if (normalized_rho < kRhoMin ||
        (std::abs(normalized_z) < kZMax && kRhoZMin < normalized_rho &&
         normalized_rho < kRhoZMax)) {
      m_exclude_from_comparison[i] = true;
    }
  }  // number_of_evaluation_locations

  return absl::OkStatus();
}  // DetermineIfTooCloseToCurrentCarrierForComparison

// A unit circular filament in the z = 0 plane, centered on the origin.
CircularFilament UnitCircularFilament() {
  CircularFilament circular_filament;
  circular_filament.set_radius(1.0);
  circular_filament.mutable_center()->set_x(0.0);
  circular_filament.mutable_center()->set_y(0.0);
  circular_filament.mutable_center()->set_z(0.0);
  circular_filament.mutable_normal()->set_x(0.0);
  circular_filament.mutable_normal()->set_y(0.0);
  circular_filament.mutable_normal()->set_z(1.0);
  return circular_filament;
}

TEST(TestMakegridLib, CheckTooCloseToCircularFilament) {
  // Excluded is the loop axis, where e_phi has no defined direction, and a box
  // of half-width kTooCloseDistance around the wire in both rho' and z'.
  const std::vector<std::vector<double>> evaluation_locations = {
      {0.0, 0.0, 0.3},     // on the axis
      {1.0, 0.0, 0.0},     // on the wire
      {1.0, 0.0, 0.005},   // just above the wire
      {1.0, 0.0, -0.005},  // just below the wire
      {1.0, 0.0, 0.5},     // above the wire, outside the box
      {1.0, 0.0, -0.5},    // below the wire, outside the box
      {1.5, 0.0, 0.0},     // outside the loop
      {0.5, 0.0, 0.0},     // inside the loop
  };
  const std::vector<bool> expected = {true,  true,  true,  true,
                                      false, false, false, false};

  std::vector<bool> exclude_from_comparison(evaluation_locations.size(), false);
  const absl::Status status = DetermineIfTooCloseToCurrentCarrierForComparison(
      UnitCircularFilament(), evaluation_locations,
      /*m_exclude_from_comparison=*/exclude_from_comparison);
  ASSERT_TRUE(status.ok()) << status;

  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(exclude_from_comparison[i], expected[i]) << "at location " << i;
  }
}

TEST(TestMakegridLib, CheckTooCloseToTiltedCircularFilament) {
  // Same criterion for a loop that is neither centered on the origin nor
  // aligned with a coordinate axis, and whose normal is not of unit length.
  CircularFilament circular_filament;
  circular_filament.set_radius(2.0);
  circular_filament.mutable_center()->set_x(1.0);
  circular_filament.mutable_center()->set_y(2.0);
  circular_filament.mutable_center()->set_z(3.0);
  circular_filament.mutable_normal()->set_x(0.0);
  circular_filament.mutable_normal()->set_y(3.0);
  circular_filament.mutable_normal()->set_z(0.0);

  const std::vector<std::vector<double>> evaluation_locations = {
      {3.0, 2.0, 3.0},  // on the wire
      {1.0, 5.0, 3.0},  // on the axis
      {3.0, 3.0, 3.0},  // at rho' = 1, half a radius off the loop plane
  };
  const std::vector<bool> expected = {true, true, false};

  std::vector<bool> exclude_from_comparison(evaluation_locations.size(), false);
  const absl::Status status = DetermineIfTooCloseToCurrentCarrierForComparison(
      circular_filament, evaluation_locations,
      /*m_exclude_from_comparison=*/exclude_from_comparison);
  ASSERT_TRUE(status.ok()) << status;

  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(exclude_from_comparison[i], expected[i]) << "at location " << i;
  }
}

// For a PolygonFilament made up of multiple straight wire segments,
// check for every segment and an evaluation point at rho' = rho / L, z' = z /
// L, where L is the length of the current wire segment and rho, z are
// cylindrical coordinates in the coordinate system of the straight wire
// segment, exclude an evaluation point
// * for A_z   if (rho' < 1 and -1 < z' < 2) --> based on slide 29
// * for B_phi if (rho' < 1 and -1 < z' < 2) --> based on slide 39
absl::Status DetermineIfTooCloseToCurrentCarrierForComparison(
    const PolygonFilament& polygon_filament,
    const std::vector<std::vector<double>>& evaluation_locations,
    std::vector<bool>& m_exclude_from_comparison) {
  static constexpr double kTooCloseDistance = 1.0e-2;
  static constexpr double kRhoMax = kTooCloseDistance;
  static constexpr double kZRhoMin = -kTooCloseDistance;
  static constexpr double kZRhoMax = 1.0 + kTooCloseDistance;

  const std::size_t number_of_evaluation_locations =
      evaluation_locations.size();
  if (number_of_evaluation_locations == 0) {
    return absl::InvalidArgumentError(
        "An empty vector of evaluation locations was provided.");
  }

  const int number_of_segments = polygon_filament.vertices_size() - 1;

  for (std::size_t i = 0; i < number_of_evaluation_locations; ++i) {
    Vector3d evaluation_location;
    evaluation_location.set_x(evaluation_locations[i][0]);
    evaluation_location.set_y(evaluation_locations[i][1]);
    evaluation_location.set_z(evaluation_locations[i][2]);

    for (int index_segment = 0; index_segment < number_of_segments;
         ++index_segment) {
      const Vector3d& origin = polygon_filament.vertices(index_segment);
      Vector3d segment =
          Subtract(polygon_filament.vertices(index_segment + 1), origin);
      const double length = Length(segment);
      Vector3d direction = Normalize(segment);

      // connection vector from start of segment to evaluation position
      Vector3d delta_eval_origin = Subtract(evaluation_location, origin);

      // distance between evaluation position and segment, parallel to filament
      // direction
      // -> z of evaluation location in coordinate system of loop
      const double parallel_distance = DotProduct(delta_eval_origin, direction);

      // connector vector, projected onto the filament direction
      Vector3d delta_parallel = ScaleTo(direction, parallel_distance);

      // vector from evaluation position to filament, perpendicular to filament
      Vector3d delta_perpendicular =
          Subtract(delta_eval_origin, delta_parallel);

      // radial distance from filament to evaluation position
      // -> rho of evaluation location in coordinate system of loop
      const double evaluation_position_radius = Length(delta_perpendicular);

      const double normalized_z = parallel_distance / length;
      const double normalized_rho = evaluation_position_radius / length;

      if (normalized_rho < kRhoMax && kZRhoMin < normalized_z &&
          normalized_z < kZRhoMax) {
        m_exclude_from_comparison[i] = true;

        // no need to check other segments, if one was too close already
        break;
      }
    }  // number_of_segments
  }  // number_of_evaluation_locations

  return absl::OkStatus();
}  // DetermineIfTooCloseToCurrentCarrierForComparison

// A single straight segment of unit length, from the origin along +x.
PolygonFilament UnitSegmentFilament() {
  PolygonFilament polygon_filament;
  Vector3d* start = polygon_filament.add_vertices();
  start->set_x(0.0);
  start->set_y(0.0);
  start->set_z(0.0);
  Vector3d* end = polygon_filament.add_vertices();
  end->set_x(1.0);
  end->set_y(0.0);
  end->set_z(0.0);
  return polygon_filament;
}

TEST(TestMakegridLib, CheckTooCloseToPolygonFilament) {
  // Excluded is a tube of radius kTooCloseDistance around the segment, running
  // from z' = -kTooCloseDistance to z' = 1 + kTooCloseDistance.
  const std::vector<std::vector<double>> evaluation_locations = {
      {0.5, 0.0, 0.0},     // on the segment
      {0.5, 0.005, 0.0},   // just off the segment
      {-0.005, 0.0, 0.0},  // just before the start
      {1.005, 0.0, 0.0},   // just past the end
      {0.5, 0.5, 0.0},     // off the segment in rho'
      {-0.5, 0.0, 0.0},    // on the segment's line, before the start
      {1.5, 0.0, 0.0},     // on the segment's line, past the end
  };
  const std::vector<bool> expected = {true,  true,  true, true,
                                      false, false, false};

  std::vector<bool> exclude_from_comparison(evaluation_locations.size(), false);
  const absl::Status status = DetermineIfTooCloseToCurrentCarrierForComparison(
      UnitSegmentFilament(), evaluation_locations,
      /*m_exclude_from_comparison=*/exclude_from_comparison);
  ASSERT_TRUE(status.ok()) << status;

  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(exclude_from_comparison[i], expected[i]) << "at location " << i;
  }
}

TEST(TestMakegridLib, CheckTooCloseToPolygonFilamentOverAllSegments) {
  // Being too close to any one segment is enough, and flags already set by an
  // earlier current carrier are left alone.
  PolygonFilament polygon_filament;
  const std::vector<std::vector<double>> vertices = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
  for (const std::vector<double>& vertex : vertices) {
    Vector3d* v = polygon_filament.add_vertices();
    v->set_x(vertex[0]);
    v->set_y(vertex[1]);
    v->set_z(vertex[2]);
  }

  const std::vector<std::vector<double>> evaluation_locations = {
      {0.5, 0.0, 0.0},  // on the first segment
      {1.0, 0.5, 0.0},  // on the second segment
      {0.5, 0.5, 0.0},  // close to neither
      {5.0, 5.0, 5.0},  // far away, but already excluded on entry
  };
  const std::vector<bool> expected = {true, true, false, true};

  std::vector<bool> exclude_from_comparison = {false, false, false, true};
  const absl::Status status = DetermineIfTooCloseToCurrentCarrierForComparison(
      polygon_filament, evaluation_locations,
      /*m_exclude_from_comparison=*/exclude_from_comparison);
  ASSERT_TRUE(status.ok()) << status;

  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(exclude_from_comparison[i], expected[i]) << "at location " << i;
  }
}

TEST(TestMakegridLib, CheckTooCloseRejectsEmptyEvaluationLocations) {
  std::vector<bool> exclude_from_comparison;

  EXPECT_EQ(DetermineIfTooCloseToCurrentCarrierForComparison(
                UnitCircularFilament(), {},
                /*m_exclude_from_comparison=*/exclude_from_comparison)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(DetermineIfTooCloseToCurrentCarrierForComparison(
                UnitSegmentFilament(), {},
                /*m_exclude_from_comparison=*/exclude_from_comparison)
                .code(),
            absl::StatusCode::kInvalidArgument);
}

// We need to exclude points which are too close to the current carrier
// filaments, as the Biot-Savart routines used in MAKEGRID do not feature the
// correct asymptotic behavior. This can be seen on slides:
// * 29 for A_z of a straight wire segment -> one segment of a PolygonFilament,
// * 39 for B_phi of a straight wire segment -> one segment of a
// PolygonFilament,
// * 46 for A_phi of a circular wire loop -> CircularFilament,
// * 53 for B_rho of a circular wire loop -> CircularFilament, and
// * 60 for B_z of a circular wire loop -> CircularFilament
// of this set of slides:
// https://github.com/jonathanschilling/abscab/blob/master/2022_08_24_Schilling_ABSCAB_talk.pdf
// These checks are implemented in the two
// `DetermineIfTooCloseToCurrentCarrierForComparison` methods above. The
// `evaluation_locations` are expected to be supplied as
// [number_of_evaluation_locations][3: x, y, z] and the too-close flags will be
// returned as [number_of_evaluation_locations].
absl::StatusOr<std::vector<bool>> IsTooCloseToCurrentCarrierForComparison(
    const SerialCircuit& serial_circuit,
    const std::vector<std::vector<double>>& evaluation_locations) {
  const std::size_t number_of_evaluation_locations =
      evaluation_locations.size();
  if (number_of_evaluation_locations == 0) {
    return absl::InvalidArgumentError(
        "An empty vector of evaluation locations was provided.");
  }

  // by default, do not exclude any evaluation location
  std::vector<bool> exclude_from_comparison(number_of_evaluation_locations,
                                            false);

  if (!serial_circuit.has_current() || serial_circuit.current() == 0.0) {
    // If the SerialCircuit does not contribute to the magnetic field, because
    // the current is zero, there is also no need to exclude any evaluation
    // points from a comparison, because all implementations should agree that
    // there is zero magnetic field from this SerialCircuit.

    return exclude_from_comparison;
  }

  for (const Coil& coil : serial_circuit.coils()) {
    for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
      switch (current_carrier.type_case()) {
        // TODO(jons): implement case for InfiniteStraightFilament
        // case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
        // break;
        case CurrentCarrier::TypeCase::kCircularFilament: {
          absl::Status circular_filament_status =
              DetermineIfTooCloseToCurrentCarrierForComparison(
                  current_carrier.circular_filament(), evaluation_locations,
                  /*m_exclude_from_comparison=*/exclude_from_comparison);
          if (!circular_filament_status.ok()) {
            return circular_filament_status;
          }
        } break;
        case CurrentCarrier::TypeCase::kPolygonFilament: {
          absl::Status circular_filament_status =
              DetermineIfTooCloseToCurrentCarrierForComparison(
                  current_carrier.polygon_filament(), evaluation_locations,
                  /*m_exclude_from_comparison=*/exclude_from_comparison);
          if (!circular_filament_status.ok()) {
            return circular_filament_status;
          }
        } break;
        // TODO(jons): implement case for FourierFilament
        // case CurrentCarrier::TypeCase::kFourierFilament:
        // break;
        case CurrentCarrier::TypeCase::kTypeNotSet:
          // consider as empty CurrentCarrier -> ignore
          break;
        default:
          std::stringstream error_message;
          error_message << "The current carrier type ";
          error_message << current_carrier.type_case();
          error_message << " is not implemented yet.";
          LOG(FATAL) << error_message.str();
      }
    }  // CurrentCarrier
  }  // Coil

  return exclude_from_comparison;
}  // IsTooCloseToCurrentCarrierForComparison

// Parameters for the reference-comparison parameterized tests below.
// Each instance selects a normalize_by_currents mode and the corresponding
// Fortran MAKEGRID reference file.
struct MakegridReferenceTestParams {
  bool normalize_by_currents;
  std::string reference_nc_file;
  // Defaults describe coils.test_symmetric_even; the other cases override them
  // to match their own MGRID_NLI namelist.
  std::string coils_file =
      "vmecpp/common/makegrid_lib/test_data/coils.test_symmetric_even";
  bool assume_stellarator_symmetry = true;
  int number_of_phi_grid_points = 18;
};

// Shared setup used by both B-field and vector-potential parameterized suites.
// Loads coils.test_symmetric_even with stellarator symmetry and returns the
// cylindrical grid, magnetic configuration, and open NetCDF file id.
struct MakegridReferenceTestFixture
    : public ::testing::TestWithParam<MakegridReferenceTestParams> {
  // NOTE: These parameters have to be consistent with the MGRID_NLI namelist
  // in the `coils.test_*` input files.
  static MakegridParameters MakeParams(
      const MakegridReferenceTestParams& test_params) {
    return {
        .normalize_by_currents = test_params.normalize_by_currents,
        .assume_stellarator_symmetry = test_params.assume_stellarator_symmetry,
        .number_of_field_periods = 5,
        .r_grid_minimum = 1.0,
        .r_grid_maximum = 2.0,
        .number_of_r_grid_points = 11,
        .z_grid_minimum = -0.6,
        .z_grid_maximum = 0.6,
        .number_of_z_grid_points = 13,
        .number_of_phi_grid_points = test_params.number_of_phi_grid_points};
  }
};

// Parameterized test: B-field response table vs. Fortran MAKEGRID reference.
// Covers both mgrid_mode='R' (raw, normalize_by_currents=false) and
// mgrid_mode='S' (scaled, normalize_by_currents=true).
using CheckComputeMagneticFieldResponseTable = MakegridReferenceTestFixture;

TEST_P(CheckComputeMagneticFieldResponseTable, MatchesFortranReference) {
  static constexpr double kTolerance = 1.0e-6;

  const MakegridReferenceTestParams& p = GetParam();
  const MakegridParameters makegrid_parameters = MakeParams(p);

  ASSERT_EQ(makegrid_parameters.normalize_by_currents, p.normalize_by_currents);
  ASSERT_EQ(makegrid_parameters.assume_stellarator_symmetry,
            p.assume_stellarator_symmetry);
  ASSERT_EQ(makegrid_parameters.number_of_field_periods, 5);
  ASSERT_EQ(makegrid_parameters.r_grid_minimum, 1.0);
  ASSERT_EQ(makegrid_parameters.r_grid_maximum, 2.0);
  ASSERT_EQ(makegrid_parameters.number_of_r_grid_points, 11);
  ASSERT_EQ(makegrid_parameters.z_grid_minimum, -0.6);
  ASSERT_EQ(makegrid_parameters.z_grid_maximum, 0.6);
  ASSERT_EQ(makegrid_parameters.number_of_z_grid_points, 13);
  ASSERT_EQ(makegrid_parameters.number_of_phi_grid_points,
            p.number_of_phi_grid_points);

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromCoilsFile(p.coils_file);
  ASSERT_OK(magnetic_configuration);

  const int number_of_serial_circuits =
      magnetic_configuration->serial_circuits_size();

  // The response table and the reference file both cover the whole field
  // period, so the exclusion mask has to as well: with stellarator symmetry
  // MakeCylindricalGrid only emits the half it computes, and the mirrored half
  // would go uncompared.
  MakegridParameters full_period_parameters = makegrid_parameters;
  full_period_parameters.assume_stellarator_symmetry = false;
  absl::StatusOr<RowMatrix3Xd> cylindrical_grid_eigen =
      MakeCylindricalGrid(full_period_parameters);
  ASSERT_OK(cylindrical_grid_eigen);
  absl::StatusOr<std::vector<std::vector<double>>> cylindrical_grid =
      EigenToStl(cylindrical_grid_eigen.value().transpose());

  const std::size_t number_of_evaluation_locations = cylindrical_grid->size();

  // compute magnetic field cache
  absl::StatusOr<MagneticFieldResponseTable> magnetic_response_table =
      ComputeMagneticFieldResponseTable(makegrid_parameters,
                                        *magnetic_configuration);
  ASSERT_OK(magnetic_response_table);

  // Load NetCDF mgrid file and make sure dimensions are consistent.
  int ncid = 0;
  ASSERT_EQ(nc_open(p.reference_nc_file.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  EXPECT_EQ(NetcdfReadInt(ncid, "nfp").value(),
            makegrid_parameters.number_of_field_periods);
  EXPECT_EQ(NetcdfReadInt(ncid, "ir").value(),
            makegrid_parameters.number_of_r_grid_points);
  EXPECT_EQ(NetcdfReadDouble(ncid, "rmin").value(),
            makegrid_parameters.r_grid_minimum);
  EXPECT_EQ(NetcdfReadDouble(ncid, "rmax").value(),
            makegrid_parameters.r_grid_maximum);
  EXPECT_EQ(NetcdfReadInt(ncid, "jz").value(),
            makegrid_parameters.number_of_z_grid_points);
  EXPECT_EQ(NetcdfReadDouble(ncid, "zmin").value(),
            makegrid_parameters.z_grid_minimum);
  EXPECT_EQ(NetcdfReadDouble(ncid, "zmax").value(),
            makegrid_parameters.z_grid_maximum);
  EXPECT_EQ(NetcdfReadInt(ncid, "kp").value(),
            makegrid_parameters.number_of_phi_grid_points);
  EXPECT_EQ(NetcdfReadInt(ncid, "nextcur").value(), number_of_serial_circuits);

  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    // Go through this SerialCircuit and check for points that are too close to
    // the current carrier filaments (see above) and thus need to be excluded
    // from the magnetic field comparison below.
    const SerialCircuit& serial_circuit =
        magnetic_configuration->serial_circuits(circuit_index);
    absl::StatusOr<std::vector<bool>> exclude_from_comparison =
        IsTooCloseToCurrentCarrierForComparison(serial_circuit,
                                                *cylindrical_grid);
    ASSERT_OK(exclude_from_comparison);

    // load mgrid data from NetCDF file
    std::vector<std::vector<std::vector<double>>> b_r_contribution =
        NetcdfReadArray3D(ncid, absl::StrFormat("br_%03d", circuit_index + 1))
            .value();
    std::vector<std::vector<std::vector<double>>> b_p_contribution =
        NetcdfReadArray3D(ncid, absl::StrFormat("bp_%03d", circuit_index + 1))
            .value();
    std::vector<std::vector<std::vector<double>>> b_z_contribution =
        NetcdfReadArray3D(ncid, absl::StrFormat("bz_%03d", circuit_index + 1))
            .value();

    // perform comparison of points that are not explicitly excluded from the
    // comparison
    int number_of_tested_evaluation_locations = 0;
    for (int index_phi = 0;
         index_phi < makegrid_parameters.number_of_phi_grid_points;
         ++index_phi) {
      for (int index_z = 0;
           index_z < makegrid_parameters.number_of_z_grid_points; ++index_z) {
        for (int index_r = 0;
             index_r < makegrid_parameters.number_of_r_grid_points; ++index_r) {
          const int linear_index =
              (index_phi * makegrid_parameters.number_of_z_grid_points +
               index_z) *
                  makegrid_parameters.number_of_r_grid_points +
              index_r;

          if ((*exclude_from_comparison)[linear_index]) {
            // skip points that are too close to the current carrier filaments
            continue;
          }

          EXPECT_TRUE(IsCloseRelAbs(
              magnetic_response_table->b_r(circuit_index, linear_index),
              b_r_contribution[index_phi][index_z][index_r], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(
              magnetic_response_table->b_p(circuit_index, linear_index),
              b_p_contribution[index_phi][index_z][index_r], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(
              magnetic_response_table->b_z(circuit_index, linear_index),
              b_z_contribution[index_phi][index_z][index_r], kTolerance));
          number_of_tested_evaluation_locations++;
        }  // index_r
      }  // index_z
    }  // index_phi

    // make sure that at least 99% of the grid points are actually tested
    const double tested_fraction =
        number_of_tested_evaluation_locations /
        static_cast<double>(number_of_evaluation_locations);
    EXPECT_GT(tested_fraction, 0.99);
  }  // circuit_index

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckComputeMagneticFieldResponseTable/MatchesFortranReference

INSTANTIATE_TEST_SUITE_P(
    SymmetricEven, CheckComputeMagneticFieldResponseTable,
    ::testing::Values(
        MakegridReferenceTestParams{
            .normalize_by_currents = false,
            .reference_nc_file = "vmecpp/common/makegrid_lib/test_data/"
                                 "mgrid_test_symmetric_even.nc"},
        MakegridReferenceTestParams{
            .normalize_by_currents = true,
            .reference_nc_file = "vmecpp/common/makegrid_lib/test_data/"
                                 "mgrid_test_symmetric_even_scaled.nc"}),
    [](const ::testing::TestParamInfo<MakegridReferenceTestParams>& info) {
      return info.param.normalize_by_currents ? "Scaled" : "Raw";
    });

// The stellarator symmetry of the coil set is not exploited here, so the whole
// toroidal range is evaluated directly rather than mirrored from a half period.
INSTANTIATE_TEST_SUITE_P(
    NonSymmetric, CheckComputeMagneticFieldResponseTable,
    ::testing::Values(MakegridReferenceTestParams{
        .normalize_by_currents = false,
        .reference_nc_file = "vmecpp/common/makegrid_lib/test_data/"
                             "mgrid_test_non_symmetric.nc",
        .coils_file =
            "vmecpp/common/makegrid_lib/test_data/coils.test_non_symmetric",
        .assume_stellarator_symmetry = false,
        .number_of_phi_grid_points = 18}),
    [](const ::testing::TestParamInfo<MakegridReferenceTestParams>& info) {
      return info.param.normalize_by_currents ? "Scaled" : "Raw";
    });

// An odd toroidal grid puts no point on the half-period plane, so every
// mirrored index is distinct from every computed one.
INSTANTIATE_TEST_SUITE_P(
    SymmetricOdd, CheckComputeMagneticFieldResponseTable,
    ::testing::Values(MakegridReferenceTestParams{
        .normalize_by_currents = false,
        .reference_nc_file = "vmecpp/common/makegrid_lib/test_data/"
                             "mgrid_test_symmetric_odd.nc",
        .coils_file =
            "vmecpp/common/makegrid_lib/test_data/coils.test_symmetric_odd",
        .assume_stellarator_symmetry = true,
        .number_of_phi_grid_points = 19}),
    [](const ::testing::TestParamInfo<MakegridReferenceTestParams>& info) {
      return info.param.normalize_by_currents ? "Scaled" : "Raw";
    });

// Parameterized test: vector-potential cache vs. Fortran MAKEGRID reference.
// Covers both mgrid_mode='R' (raw, normalize_by_currents=false) and
// mgrid_mode='S' (scaled, normalize_by_currents=true).
using CheckComputeVectorPotentialCache = MakegridReferenceTestFixture;

TEST_P(CheckComputeVectorPotentialCache, MatchesFortranReference) {
  static constexpr double kTolerance = 1.0e-6;

  const MakegridReferenceTestParams& p = GetParam();
  const MakegridParameters makegrid_parameters = MakeParams(p);

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromCoilsFile(p.coils_file);
  ASSERT_OK(magnetic_configuration);

  const int number_of_serial_circuits =
      magnetic_configuration->serial_circuits_size();

  // The exclusion mask has to cover the whole field period; see the same
  // comment in CheckComputeMagneticFieldResponseTable above.
  MakegridParameters full_period_parameters = makegrid_parameters;
  full_period_parameters.assume_stellarator_symmetry = false;
  absl::StatusOr<RowMatrix3Xd> cylindrical_grid_eigen =
      MakeCylindricalGrid(full_period_parameters);
  ASSERT_OK(cylindrical_grid_eigen);

  // MakeCylindricalGrid() returns a 3xN matrix, instead of Nx3, so we tranpose
  // to keep the test the same:
  absl::StatusOr<std::vector<std::vector<double>>> cylindrical_grid =
      EigenToStl(cylindrical_grid_eigen.value().transpose());

  const std::size_t number_of_evaluation_locations = cylindrical_grid->size();

  // compute vector potential cache
  absl::StatusOr<MakegridCachedVectorPotential> vector_potential_cache =
      ComputeVectorPotentialCache(makegrid_parameters, *magnetic_configuration);
  ASSERT_OK(vector_potential_cache);

  // Load NetCDF mgrid file and make sure dimensions are consistent.
  int ncid = 0;
  ASSERT_EQ(nc_open(p.reference_nc_file.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  const int nfp = NetcdfReadInt(ncid, "nfp").value();
  EXPECT_EQ(nfp, makegrid_parameters.number_of_field_periods);

  const int numR = NetcdfReadInt(ncid, "ir").value();
  EXPECT_EQ(numR, makegrid_parameters.number_of_r_grid_points);

  const double minR = NetcdfReadDouble(ncid, "rmin").value();
  EXPECT_EQ(minR, makegrid_parameters.r_grid_minimum);

  const double maxR = NetcdfReadDouble(ncid, "rmax").value();
  EXPECT_EQ(maxR, makegrid_parameters.r_grid_maximum);

  const int numZ = NetcdfReadInt(ncid, "jz").value();
  EXPECT_EQ(numZ, makegrid_parameters.number_of_z_grid_points);

  const double minZ = NetcdfReadDouble(ncid, "zmin").value();
  EXPECT_EQ(minZ, makegrid_parameters.z_grid_minimum);

  const double maxZ = NetcdfReadDouble(ncid, "zmax").value();
  EXPECT_EQ(maxZ, makegrid_parameters.z_grid_maximum);

  const int numPhi = NetcdfReadInt(ncid, "kp").value();
  EXPECT_EQ(numPhi, makegrid_parameters.number_of_phi_grid_points);

  const int nextcur = NetcdfReadInt(ncid, "nextcur").value();
  EXPECT_EQ(nextcur, number_of_serial_circuits);

  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    // Go through this SerialCircuit and check for points that are too close to
    // the current carrier filaments (see above) and thus need to be excluded
    // from the magnetic field comparison below.
    const SerialCircuit& serial_circuit =
        magnetic_configuration->serial_circuits(circuit_index);
    absl::StatusOr<std::vector<bool>> exclude_from_comparison =
        IsTooCloseToCurrentCarrierForComparison(serial_circuit,
                                                *cylindrical_grid);
    ASSERT_OK(exclude_from_comparison);

    // load mgrid data from NetCDF file
    std::vector<std::vector<std::vector<double>>> a_r_contribution =
        NetcdfReadArray3D(ncid, absl::StrFormat("ar_%03d", circuit_index + 1))
            .value();
    std::vector<std::vector<std::vector<double>>> a_p_contribution =
        NetcdfReadArray3D(ncid, absl::StrFormat("ap_%03d", circuit_index + 1))
            .value();
    std::vector<std::vector<std::vector<double>>> a_z_contribution =
        NetcdfReadArray3D(ncid, absl::StrFormat("az_%03d", circuit_index + 1))
            .value();

    // perform comparison of points that are not explicitly excluded from the
    // comparison
    int number_of_tested_evaluation_locations = 0;
    for (int index_phi = 0;
         index_phi < makegrid_parameters.number_of_phi_grid_points;
         ++index_phi) {
      for (int index_z = 0;
           index_z < makegrid_parameters.number_of_z_grid_points; ++index_z) {
        for (int index_r = 0;
             index_r < makegrid_parameters.number_of_r_grid_points; ++index_r) {
          const int linear_index =
              (index_phi * makegrid_parameters.number_of_z_grid_points +
               index_z) *
                  makegrid_parameters.number_of_r_grid_points +
              index_r;

          if ((*exclude_from_comparison)[linear_index]) {
            // skip points that are too close to the current carrier filaments
            continue;
          }

          EXPECT_TRUE(IsCloseRelAbs(
              vector_potential_cache->a_r(circuit_index, linear_index),
              a_r_contribution[index_phi][index_z][index_r], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(
              vector_potential_cache->a_p(circuit_index, linear_index),
              a_p_contribution[index_phi][index_z][index_r], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(
              vector_potential_cache->a_z(circuit_index, linear_index),
              a_z_contribution[index_phi][index_z][index_r], kTolerance));
          number_of_tested_evaluation_locations++;
        }  // index_r
      }  // index_z
    }  // index_phi

    // make sure that at least 99% of the grid points are actually tested
    const double tested_fraction =
        number_of_tested_evaluation_locations /
        static_cast<double>(number_of_evaluation_locations);
    EXPECT_GT(tested_fraction, 0.99);
  }  // circuit_index

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckComputeVectorPotentialCache/MatchesFortranReference

INSTANTIATE_TEST_SUITE_P(
    SymmetricEven, CheckComputeVectorPotentialCache,
    ::testing::Values(
        MakegridReferenceTestParams{
            .normalize_by_currents = false,
            .reference_nc_file = "vmecpp/common/makegrid_lib/test_data/"
                                 "mgrid_test_symmetric_even.nc"},
        MakegridReferenceTestParams{
            .normalize_by_currents = true,
            .reference_nc_file = "vmecpp/common/makegrid_lib/test_data/"
                                 "mgrid_test_symmetric_even_scaled.nc"}),
    [](const ::testing::TestParamInfo<MakegridReferenceTestParams>& info) {
      return info.param.normalize_by_currents ? "Scaled" : "Raw";
    });

INSTANTIATE_TEST_SUITE_P(
    NonSymmetric, CheckComputeVectorPotentialCache,
    ::testing::Values(MakegridReferenceTestParams{
        .normalize_by_currents = false,
        .reference_nc_file = "vmecpp/common/makegrid_lib/test_data/"
                             "mgrid_test_non_symmetric.nc",
        .coils_file =
            "vmecpp/common/makegrid_lib/test_data/coils.test_non_symmetric",
        .assume_stellarator_symmetry = false,
        .number_of_phi_grid_points = 18}),
    [](const ::testing::TestParamInfo<MakegridReferenceTestParams>& info) {
      return info.param.normalize_by_currents ? "Scaled" : "Raw";
    });

INSTANTIATE_TEST_SUITE_P(
    SymmetricOdd, CheckComputeVectorPotentialCache,
    ::testing::Values(MakegridReferenceTestParams{
        .normalize_by_currents = false,
        .reference_nc_file = "vmecpp/common/makegrid_lib/test_data/"
                             "mgrid_test_symmetric_odd.nc",
        .coils_file =
            "vmecpp/common/makegrid_lib/test_data/coils.test_symmetric_odd",
        .assume_stellarator_symmetry = true,
        .number_of_phi_grid_points = 19}),
    [](const ::testing::TestParamInfo<MakegridReferenceTestParams>& info) {
      return info.param.normalize_by_currents ? "Scaled" : "Raw";
    });

// TODO(jons): add test of WriteMakegridNetCDFFile
// -> in particular, make sure that the consistency of number of serial circuits
// in the response table and the number of circuit currents is properly checked

TEST(TestMakegridLib,
     CheckNormalizeByCurrentsScalesMagneticFieldResponseTable) {
  static constexpr double kTolerance = 1.0e-6;

  // Use a small grid (no stellarator symmetry) to keep the test fast while
  // still exercising all grid points.
  MakegridParameters makegrid_parameters = {
      .normalize_by_currents = false,
      .assume_stellarator_symmetry = false,
      .number_of_field_periods = 1,
      .r_grid_minimum = 1.0,
      .r_grid_maximum = 2.0,
      .number_of_r_grid_points = 3,
      .z_grid_minimum = -0.5,
      .z_grid_maximum = 0.5,
      .number_of_z_grid_points = 3,
      .number_of_phi_grid_points = 4};

  // Build a simple MagneticConfiguration with a single circuit, single coil,
  // and a single circular filament. Using non-trivial current and num_windings
  // ensures that the test is sensitive to the normalization factor.
  static constexpr double kCurrent = 5.0;
  static constexpr double kNumWindings = 7.0;

  MagneticConfiguration magnetic_configuration;
  SerialCircuit* serial_circuit = magnetic_configuration.add_serial_circuits();
  serial_circuit->set_current(kCurrent);

  Coil* coil = serial_circuit->add_coils();
  coil->set_num_windings(kNumWindings);

  CurrentCarrier* current_carrier = coil->add_current_carriers();
  CircularFilament* circular_filament =
      current_carrier->mutable_circular_filament();
  circular_filament->set_radius(1.5);

  Vector3d* center = circular_filament->mutable_center();
  center->set_x(0.0);
  center->set_y(0.0);
  center->set_z(0.0);

  Vector3d* normal = circular_filament->mutable_normal();
  normal->set_x(0.0);
  normal->set_y(0.0);
  normal->set_z(1.0);

  // The raw field must equal the normalized field scaled by
  // current * num_windings at every grid point.
  const double expected_factor = kCurrent * kNumWindings;

  // Compute the raw (unnormalized) response table.
  absl::StatusOr<MagneticFieldResponseTable> raw_response_table =
      ComputeMagneticFieldResponseTable(makegrid_parameters,
                                        magnetic_configuration);
  ASSERT_OK(raw_response_table);

  // Compute the normalized response table (field per unit current-turn).
  MakegridParameters normalized_params = makegrid_parameters;
  normalized_params.normalize_by_currents = true;
  absl::StatusOr<MagneticFieldResponseTable> normalized_response_table =
      ComputeMagneticFieldResponseTable(normalized_params,
                                        magnetic_configuration);
  ASSERT_OK(normalized_response_table);

  const int number_of_serial_circuits =
      magnetic_configuration.serial_circuits_size();
  const int total_number_of_grid_points =
      makegrid_parameters.number_of_phi_grid_points *
      makegrid_parameters.number_of_z_grid_points *
      makegrid_parameters.number_of_r_grid_points;

  // The raw field must equal the normalized field scaled by current *
  // num_windings at every grid point and for every circuit.
  for (int circuit_index = 0; circuit_index < number_of_serial_circuits;
       ++circuit_index) {
    for (int grid_index = 0; grid_index < total_number_of_grid_points;
         ++grid_index) {
      EXPECT_TRUE(IsCloseRelAbs(
          raw_response_table->b_r(circuit_index, grid_index),
          normalized_response_table->b_r(circuit_index, grid_index) *
              expected_factor,
          kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(
          raw_response_table->b_p(circuit_index, grid_index),
          normalized_response_table->b_p(circuit_index, grid_index) *
              expected_factor,
          kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(
          raw_response_table->b_z(circuit_index, grid_index),
          normalized_response_table->b_z(circuit_index, grid_index) *
              expected_factor,
          kTolerance));
    }  // grid_index
  }  // circuit_index
}  // CheckNormalizeByCurrentsScalesMagneticFieldResponseTable

}  // namespace makegrid
