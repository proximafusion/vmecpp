// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/magnetic_field_provider/magnetic_field_provider_lib.h"

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include "abscab/abscab.hh"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/composed_types_definition/composed_types.h"
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"
#include "vmecpp/common/magnetic_configuration_definition/magnetic_configuration.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"

namespace magnetics {

using composed_types::CurveRZFourier;
using composed_types::CurveRZFourierFromCsv;
using composed_types::Vector3d;
using file_io::ReadFile;

using ::testing::IsCloseRelAbs;

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Test;
using ::testing::TestWithParam;
using ::testing::Values;

// Compute the Cartesian coordinate of a circle, which has its center point at
// the origin (optionally offset by `z_offset` along the `z` axis) and the
// normal direction pointing along the z axis. num_phi specifies how many
// sampling points are returned. The provided PolygonFilament will be filled as
// [num_phi][3: x,y,z]. It must be empty on entry; otherwise, an
// InvalidArumentError is returned.
absl::Status PolygonCirclePopulate(PolygonFilament &m_polygon_filament,
                                   const double radius, const int num_phi,
                                   const double z_offset = 0.0) {
  if (m_polygon_filament.vertices_size() != 0) {
    return absl::InvalidArgumentError("non-empty PolygonFilament provided");
  }
  const double omega = 2.0 * M_PI / (num_phi - 1.0);
  for (int i = 0; i < num_phi; ++i) {
    const double phi = omega * i;
    Vector3d *vertex = m_polygon_filament.add_vertices();
    vertex->set_x(radius * cos(phi));
    vertex->set_y(radius * sin(phi));
    vertex->set_z(z_offset);
  }
  return absl::OkStatus();
}  // PolygonCirclePopulate

TEST(TestPolygonCircle, CheckPolygonCirclePopulate) {
  constexpr double kTolerance = 1.0e-15;

  constexpr double kRadius = 2.71;
  constexpr int kNumPhi = 360;

  PolygonFilament circle_geometry_around_origin;
  ASSERT_TRUE(PolygonCirclePopulate(
                  /*m_polygon_filament=*/circle_geometry_around_origin, kRadius,
                  kNumPhi)
                  .ok());

  for (int i = 0; i < kNumPhi; ++i) {
    const Vector3d &vertex = circle_geometry_around_origin.vertices(i);
    const double radius = hypot(vertex.x(), vertex.y());
    EXPECT_TRUE(IsCloseRelAbs(kRadius, radius, kTolerance));
    EXPECT_EQ(0.0, vertex.z());  // z should always be identically zero if
                                 // z_offset is not specified
  }

  // now check the case that z_offset is specified
  for (double z_offset : {0.0, 10.0}) {
    PolygonFilament circle_geometry;
    ASSERT_TRUE(PolygonCirclePopulate(/*m_polygon_filament=*/circle_geometry,
                                      kRadius, kNumPhi, z_offset)
                    .ok());

    for (int i = 0; i < kNumPhi; ++i) {
      const Vector3d &vertex = circle_geometry.vertices(i);
      const double radius = hypot(vertex.x(), vertex.y());
      EXPECT_TRUE(IsCloseRelAbs(kRadius, radius, kTolerance));
      EXPECT_EQ(z_offset, vertex.z());
    }
  }
}  // CheckPolygonCirclePopulate

// Compute the Cartesian coordinate of a circle, which has its center point at
// the origin (optionally offset by `z_offset` along the `z` axis) and the
// normal direction pointing along the z axis. num_phi specifies how many
// sampling points are returned. The returned geometry has dimensions
// [num_phi][3: x,y,z].
PolygonFilament PolygonCircle(const double radius, const int num_phi,
                              const double z_offset = 0.0) {
  PolygonFilament polygon_filament;
  CHECK_OK(PolygonCirclePopulate(/*m_polygon_filament=*/polygon_filament,
                                 radius, num_phi, z_offset));
  return polygon_filament;
}  // PolygonCircle

TEST(TestPolygonCircle, CheckPolygonCircle) {
  constexpr double kTolerance = 1.0e-15;

  constexpr double kRadius = 2.71;
  constexpr int kNumPhi = 360;

  PolygonFilament circle_geometry_around_origin =
      PolygonCircle(kRadius, kNumPhi);

  for (int i = 0; i < kNumPhi; ++i) {
    const Vector3d &vertex = circle_geometry_around_origin.vertices(i);
    const double radius = hypot(vertex.x(), vertex.y());
    EXPECT_TRUE(IsCloseRelAbs(kRadius, radius, kTolerance));
    EXPECT_EQ(0.0, vertex.z());  // z should always be identically zero if
                                 // z_offset is not specified
  }

  // now check the case that z_offset is specified
  for (double z_offset : {0.0, 10.0}) {
    PolygonFilament circle_geometry = PolygonCircle(kRadius, kNumPhi, z_offset);

    for (int i = 0; i < kNumPhi; ++i) {
      const Vector3d &vertex = circle_geometry.vertices(i);
      const double radius = hypot(vertex.x(), vertex.y());
      EXPECT_TRUE(IsCloseRelAbs(kRadius, radius, kTolerance));
      EXPECT_EQ(z_offset, vertex.z());
    }
  }
}  // CheckPolygonCircle

// --------------------

InfiniteStraightFilament MakeZAxisInfiniteStraightFilament() {
  InfiniteStraightFilament infinite_straight_filament;

  // centered at the origin
  Vector3d *origin = infinite_straight_filament.mutable_origin();
  origin->set_x(0.0);
  origin->set_y(0.0);
  origin->set_z(0.0);

  // direction along z axis
  Vector3d *direction = infinite_straight_filament.mutable_direction();
  direction->set_x(0.0);
  direction->set_y(0.0);
  direction->set_z(1.0);

  return infinite_straight_filament;
}  // MakeZAxisInfiniteStraightFilament

TEST(TestMagneticField, CheckInfiniteStraightFilamentAtSymmetryPoint) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 3.14;
  static constexpr double kRadiusEvaluationLocation = 1.23;

  InfiniteStraightFilament infinite_straight_filament =
      MakeZAxisInfiniteStraightFilament();

  std::vector<std::vector<double> > evaluation_positions = {
      {kRadiusEvaluationLocation, 0.0, 0.0}};

  std::vector<std::vector<double> > magnetic_field = {{0.0, 0.0, 0.0}};

  absl::Status status = MagneticField(infinite_straight_filament, kCurrent,
                                      evaluation_positions, magnetic_field);
  EXPECT_TRUE(status.ok());

  // The magnetic field should point purely in the y direction.
  // The magnitude should be given by:
  const double magnetic_field_reference =
      abscab::MU_0 * kCurrent / (2.0 * M_PI * kRadiusEvaluationLocation);

  EXPECT_TRUE(IsCloseRelAbs(0.0, magnetic_field[0][0], kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference, magnetic_field[0][1],
                            kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, magnetic_field[0][2], kTolerance));
}  // MagneticField: CheckInfiniteStraightFilamentAtSymmetryPoint

TEST(TestMagneticField, CheckInfiniteStraightFilamentAtManyPoints) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 3.14;
  static constexpr int kNumEvaluationLocations = 30;

  InfiniteStraightFilament infinite_straight_filament =
      MakeZAxisInfiniteStraightFilament();

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 0.5 + 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3, 0.0);
  }

  absl::Status status = MagneticField(infinite_straight_filament, kCurrent,
                                      evaluation_positions, magnetic_field);
  EXPECT_TRUE(status.ok());

  std::vector<std::vector<double> > magnetic_field_reference(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field_reference[i].resize(3, 0.0);
  }

  Vector3d normalized_direction =
      Normalize(infinite_straight_filament.direction());

  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    Vector3d evaluation_position;
    evaluation_position.set_x(evaluation_positions[i][0]);
    evaluation_position.set_y(evaluation_positions[i][1]);
    evaluation_position.set_z(evaluation_positions[i][2]);

    // connection vector from evaluation position to origin on filament
    Vector3d delta_eval_origin =
        Subtract(infinite_straight_filament.origin(), evaluation_position);

    // distance between evaluation position and origin on filament
    // parallel to filament direction
    const double parallel_distance =
        DotProduct(delta_eval_origin, normalized_direction);

    // connector vector, projected onto the filament direction
    Vector3d delta_parallel = ScaleTo(normalized_direction, parallel_distance);

    // vector from evaluation position to filament,
    // perpendicular to filament
    Vector3d delta_perpendicular = Subtract(delta_eval_origin, delta_parallel);

    // radial distance from filament to evaluation position
    const double evaluation_position_radius = Length(delta_perpendicular);

    // The magnetic field is not defined on the filament,
    // so must check that radius is > 0.
    ASSERT_GT(evaluation_position_radius, 0.0);

    // Magnetic field strength of infinite straight filament,
    // cylindrical phi component in coordinate system of filament.
    const double magnetic_field_strength =
        abscab::MU_0 * kCurrent / (2.0 * M_PI * evaluation_position_radius);

    // radial unit vector at evaluation location,
    // in coordinate system of filament
    Vector3d radial_unit_vector = Normalize(delta_perpendicular);

    // e_phi: unit vector in direction of magnetic field at evaluation location
    Vector3d toroidal_unit_vector =
        CrossProduct(radial_unit_vector, normalized_direction);

    // compute magnetic field vector by scaling correct unit vector to correct
    // length
    Vector3d magnetic_field_vector =
        ScaleTo(toroidal_unit_vector, magnetic_field_strength);
    magnetic_field_reference[i][0] = magnetic_field_vector.x();
    magnetic_field_reference[i][1] = magnetic_field_vector.y();
    magnetic_field_reference[i][2] = magnetic_field_vector.z();
  }

  // now check the calculation against the reference
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i][0],
                              magnetic_field[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i][1],
                              magnetic_field[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i][2],
                              magnetic_field[i][2], kTolerance));
  }
}  // MagneticField: CheckInfiniteStraightFilamentAtManyPoints

TEST(TestMagneticField, CheckInfiniteStraightFilamentAgainstPolygonFilament) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 3.14;
  static constexpr double kSufficientLengthToBeConsideredInfinite = 1.0e8;
  static constexpr int kNumEvaluationLocations = 30;

  InfiniteStraightFilament infinite_straight_filament =
      MakeZAxisInfiniteStraightFilament();

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 0.5 + 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3, 0.0);
  }

  absl::Status status = MagneticField(infinite_straight_filament, kCurrent,
                                      evaluation_positions, magnetic_field);
  EXPECT_TRUE(status.ok());

  std::vector<std::vector<double> > magnetic_field_reference(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field_reference[i].resize(3, 0.0);
  }

  // start of semi-infinite wire segment
  Vector3d start_of_segment =
      Subtract(infinite_straight_filament.origin(),
               ScaleTo(infinite_straight_filament.direction(),
                       kSufficientLengthToBeConsideredInfinite));

  // end of semi-infinite wire segment
  Vector3d end_of_segment =
      Add(infinite_straight_filament.origin(),
          ScaleTo(infinite_straight_filament.direction(),
                  kSufficientLengthToBeConsideredInfinite));

  // create PolygonFilament with a single wire segment to mimic
  // InfiniteStraightSegment
  PolygonFilament polygon_filament;
  Vector3d *vertex_1 = polygon_filament.add_vertices();
  vertex_1->CopyFrom(start_of_segment);
  Vector3d *vertex_2 = polygon_filament.add_vertices();
  vertex_2->CopyFrom(end_of_segment);

  status = MagneticField(polygon_filament, kCurrent, evaluation_positions,
                         magnetic_field_reference);
  EXPECT_TRUE(status.ok());

  // now check the calculation against the reference
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i][0],
                              magnetic_field[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i][1],
                              magnetic_field[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i][2],
                              magnetic_field[i][2], kTolerance));
  }
}  // MagneticField: CheckInfiniteStraightFilamentAgainstPolygonFilament

TEST(TestMagneticField, CheckCircularFilament) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 3.14;
  static constexpr double kRadius = 2.71;
  static constexpr int kNumEvaluationLocations = 30;

  // centered at the origin, normal along z axis
  std::vector<double> center = {
      0.0, 0.0, 0.0};  // TODO(jons): make const once ABSCAB supports this
  std::vector<double> normal = {
      0.0, 0.0, 1.0};  // TODO(jons): make const once ABSCAB supports this

  CircularFilament circular_filament;

  Vector3d *center_vector = circular_filament.mutable_center();
  center_vector->set_x(center[0]);
  center_vector->set_y(center[1]);
  center_vector->set_z(center[2]);

  Vector3d *normal_vector = circular_filament.mutable_normal();
  normal_vector->set_x(normal[0]);
  normal_vector->set_y(normal[1]);
  normal_vector->set_z(normal[2]);

  circular_filament.set_radius(kRadius);

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_z = 2.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    // point along the axis of the loop
    evaluation_positions[i][0] = 0.0;
    evaluation_positions[i][1] = 0.0;
    evaluation_positions[i][2] = -1.0 + i * delta_z;
  }

  // setup target array for magnetic field to be tested
  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3, 0.0);
  }

  // setup one-dimensional vector of evaluation location coordinates for ABSCAB
  std::vector<double> evaluation_positions_reference(kNumEvaluationLocations *
                                                     3);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions_reference[i * 3 + 0] = evaluation_positions[i][0];
    evaluation_positions_reference[i * 3 + 1] = evaluation_positions[i][1];
    evaluation_positions_reference[i * 3 + 2] = evaluation_positions[i][2];
  }

  // setup one-dimensional vector of reference magnetic field values, similar to
  // test using ABSCAB
  std::vector<double> magnetic_field_reference(kNumEvaluationLocations * 3);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    const double z = evaluation_positions_reference[i * 3 + 2];

    // use simplified analytical expression for axial component along axis of
    // circular filament from Eqn. (30) in Simpson et al, "Simple Analytical
    // Expressions for the Magnetic Field of a Circular Current Loop" (2001)
    // NASA 20010038494
    const double b_z = abscab::MU_0 * kCurrent * kRadius * kRadius /
                       (2.0 * pow(kRadius * kRadius + z * z, 3.0 / 2.0));

    magnetic_field_reference[i * 3 + 0] = 0.0;
    magnetic_field_reference[i * 3 + 1] = 0.0;
    magnetic_field_reference[i * 3 + 2] = b_z;
  }

  absl::Status status = MagneticField(circular_filament, kCurrent,
                                      evaluation_positions, magnetic_field);
  EXPECT_EQ(status, absl::OkStatus());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 0],
                              magnetic_field[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 1],
                              magnetic_field[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 2],
                              magnetic_field[i][2], kTolerance));
  }
}  // MagneticField: CheckCircularFilament

TEST(TestMagneticField, CheckCircularFilamentAgainstDirectAbscab) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 3.14;
  static constexpr double kRadius = 2.71;
  static constexpr int kNumEvaluationLocations = 3;

  std::vector<double> center = {
      1.23, 4.56, 7.89};  // TODO(jons): make const once ABSCAB supports this
  std::vector<double> normal = {
      9.87, 6.54, 3.21};  // TODO(jons): make const once ABSCAB supports this

  CircularFilament circular_filament;

  Vector3d *center_vector = circular_filament.mutable_center();
  center_vector->set_x(center[0]);
  center_vector->set_y(center[1]);
  center_vector->set_z(center[2]);

  Vector3d *normal_vector = circular_filament.mutable_normal();
  normal_vector->set_x(normal[0]);
  normal_vector->set_y(normal[1]);
  normal_vector->set_z(normal[2]);

  circular_filament.set_radius(kRadius);

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -(i + 1);
    evaluation_positions[i][1] = -(i + 1);
    evaluation_positions[i][2] = -(i + 1);
  }

  // setup target array for magnetic field to be tested
  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3, 0.0);
  }

  // setup one-dimensional vector of evaluation location coordinates for ABSCAB
  std::vector<double> evaluation_positions_reference(kNumEvaluationLocations *
                                                     3);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions_reference[i * 3 + 0] = evaluation_positions[i][0];
    evaluation_positions_reference[i * 3 + 1] = evaluation_positions[i][1];
    evaluation_positions_reference[i * 3 + 2] = evaluation_positions[i][2];
  }

  // setup one-dimensional vector of magnetic field values for ABSCAB
  std::vector<double> magnetic_field_reference(kNumEvaluationLocations * 3,
                                               0.0);

  absl::Status status = MagneticField(circular_filament, kCurrent,
                                      evaluation_positions, magnetic_field);
  EXPECT_EQ(status, absl::OkStatus());

  // reference computation using ABSCAB directly
  abscab::magneticFieldCircularFilament(
      center.data(), normal.data(), kRadius, kCurrent, kNumEvaluationLocations,
      evaluation_positions_reference.data(), magnetic_field_reference.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 0],
                              magnetic_field[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 1],
                              magnetic_field[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 2],
                              magnetic_field[i][2], kTolerance));
  }
}  // MagneticField: CheckCircularFilamentAgainstDirectAbscab

// Mimic a circular wire loop (for which we have accurate methods) by a polygon
// circle with adjusted radius. The radius adjustment is described in the
// following reference: McGreivy et al, "Computation of the Biot-Savart line
// integral with higher-order convergence using straight segments" Physics of
// Plasmas 28, 082111 (2021) (https://doi.org/10.1063/5.0058014)
TEST(TestMagneticField, CheckPolygonFilament) {
  // A looser tolerance is allowed here in comparison with the other tests,
  // since a finite-element approximation is tested against an analytical
  // expression.
  constexpr double kTolerance = 1.0e-10;

  constexpr double kCurrent = 17.0;
  constexpr double kRadius = 1.23;
  static constexpr int kNumEvaluationLocations = 30;
  constexpr double kNumPhi = 100;

  // McGreivy radius correction
  const double delta_phi =
      2.0 * M_PI / (kNumPhi - 1);  // spacing between points

  // TODO(jons): understand derivation of alpha for special case of closed
  // circle |dr/ds| = 2*pi
  // --> alpha = 1/R * (dr)^2 / 12
  // == 4 pi^2 / (12 R)
  const double adjusted_radius = kRadius * (1.0 + delta_phi * delta_phi / 12.0);

  PolygonFilament polygon_filament = PolygonCircle(adjusted_radius, kNumPhi);

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  // setup one-dimensional vector of evaluation location coordinates for ABSCAB
  std::vector<double> evaluation_positions_reference(kNumEvaluationLocations *
                                                     3);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions_reference[i * 3 + 0] = evaluation_positions[i][0];
    evaluation_positions_reference[i * 3 + 1] = evaluation_positions[i][1];
    evaluation_positions_reference[i * 3 + 2] = evaluation_positions[i][2];
  }

  // compute reference using analytical expression for circular filament from
  // ABSCAB
  std::vector<double> magnetic_field_reference(kNumEvaluationLocations * 3,
                                               0.0);
  std::vector<double> center = {0.0, 0.0, 0.0};
  std::vector<double> normal = {0.0, 0.0, 1.0};
  abscab::magneticFieldCircularFilament(
      center.data(), normal.data(), kRadius, kCurrent, kNumEvaluationLocations,
      evaluation_positions_reference.data(), magnetic_field_reference.data());

  // target array for method to test
  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3, 0.0);
  }

  // method under test here
  absl::Status status = MagneticField(polygon_filament, kCurrent,
                                      evaluation_positions, magnetic_field);
  EXPECT_EQ(status, absl::OkStatus());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 0],
                              magnetic_field[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 1],
                              magnetic_field[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 2],
                              magnetic_field[i][2], kTolerance));
  }
}  // MagneticField: CheckPolygonFilament

TEST(TestMagneticField, CheckPolygonFilamentAgainstDirectAbscab) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 3.14;
  static constexpr int kNumEvaluationLocations = 3;

  PolygonFilament polygon_filament;
  Vector3d *vertex_1 = polygon_filament.add_vertices();
  vertex_1->set_x(1.0);
  vertex_1->set_y(2.0);
  vertex_1->set_z(3.0);
  Vector3d *vertex_2 = polygon_filament.add_vertices();
  vertex_2->set_x(3.0);
  vertex_2->set_y(2.0);
  vertex_2->set_z(1.0);

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -(i + 1);
    evaluation_positions[i][1] = -(i + 1);
    evaluation_positions[i][2] = -(i + 1);
  }

  // setup target array for magnetic field to be tested
  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3, 0.0);
  }

  // setup one-dimensional vector of evaluation location coordinates for ABSCAB
  std::vector<double> evaluation_positions_reference(kNumEvaluationLocations *
                                                     3);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions_reference[i * 3 + 0] = evaluation_positions[i][0];
    evaluation_positions_reference[i * 3 + 1] = evaluation_positions[i][1];
    evaluation_positions_reference[i * 3 + 2] = evaluation_positions[i][2];
  }

  // setup one-dimensional vector for polygon geometry for ABSCAB
  const int num_vertices = polygon_filament.vertices_size();
  std::vector<double> vertices(num_vertices * 3);
  for (int i = 0; i < num_vertices; ++i) {
    vertices[i * 3 + 0] = polygon_filament.vertices(i).x();
    vertices[i * 3 + 1] = polygon_filament.vertices(i).y();
    vertices[i * 3 + 2] = polygon_filament.vertices(i).z();
  }

  // setup one-dimensional vector of magnetic field values for ABSCAB
  std::vector<double> magnetic_field_reference(kNumEvaluationLocations * 3,
                                               0.0);

  // reference computation using ABSCAB directly
  abscab::magneticFieldPolygonFilament(
      num_vertices, vertices.data(), kCurrent, kNumEvaluationLocations,
      evaluation_positions_reference.data(), magnetic_field_reference.data());

  // calculation under test here
  absl::Status status = MagneticField(polygon_filament, kCurrent,
                                      evaluation_positions, magnetic_field);
  EXPECT_EQ(status, absl::OkStatus());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 0],
                              magnetic_field[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 1],
                              magnetic_field[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference[i * 3 + 2],
                              magnetic_field[i][2], kTolerance));
  }
}  // MagneticField: CheckPolygonFilamentAgainstDirectAbscab

TEST(TestMagneticField, CheckFourierFilament) {
  // TODO(jons): implement
}  // MagneticField: CheckFourierFilament

// ---------------------

// No test for vector potential of InfiniteStraightFilament,
// because the vector potential of an infinite straight filament is not defined.

TEST(TestVectorPotential, CheckCircularFilament) {
  // TODO(jons): find and implement a meaningful reference calculation for the
  // vector potential of a circular filament
}  // VectorPotential: CheckCircularFilament

TEST(TestVectorPotential, CheckCircularFilamentAgainstDirectAbscab) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 3.14;
  static constexpr double kRadius = 2.71;
  static constexpr int kNumEvaluationLocations = 3;

  std::vector<double> center = {
      1.23, 4.56, 7.89};  // TODO(jons): make const once ABSCAB supports this
  std::vector<double> normal = {
      9.87, 6.54, 3.21};  // TODO(jons): make const once ABSCAB supports this

  CircularFilament circular_filament;

  Vector3d *center_vector = circular_filament.mutable_center();
  center_vector->set_x(center[0]);
  center_vector->set_y(center[1]);
  center_vector->set_z(center[2]);

  Vector3d *normal_vector = circular_filament.mutable_normal();
  normal_vector->set_x(normal[0]);
  normal_vector->set_y(normal[1]);
  normal_vector->set_z(normal[2]);

  circular_filament.set_radius(kRadius);

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -(i + 1);
    evaluation_positions[i][1] = -(i + 1);
    evaluation_positions[i][2] = -(i + 1);
  }

  // setup target array for magnetic field to be tested
  std::vector<std::vector<double> > vector_potential(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential[i].resize(3, 0.0);
  }

  // setup one-dimensional vector of evaluation location coordinates for ABSCAB
  std::vector<double> evaluation_positions_reference(kNumEvaluationLocations *
                                                     3);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions_reference[i * 3 + 0] = evaluation_positions[i][0];
    evaluation_positions_reference[i * 3 + 1] = evaluation_positions[i][1];
    evaluation_positions_reference[i * 3 + 2] = evaluation_positions[i][2];
  }

  // setup one-dimensional vector of magnetic field values for ABSCAB
  std::vector<double> vector_potential_reference(kNumEvaluationLocations * 3,
                                                 0.0);

  absl::Status status = VectorPotential(circular_filament, kCurrent,
                                        evaluation_positions, vector_potential);
  EXPECT_EQ(status, absl::OkStatus());

  // reference computation using ABSCAB directly
  // FIXME(jons): Figure out what the actual sign definition must be.
  // For now, adjusted to agree with MAKEGRID.
  abscab::vectorPotentialCircularFilament(
      center.data(), normal.data(), kRadius, -kCurrent, kNumEvaluationLocations,
      evaluation_positions_reference.data(), vector_potential_reference.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 0],
                              vector_potential[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 1],
                              vector_potential[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 2],
                              vector_potential[i][2], kTolerance));
  }
}  // VectorPotential: CheckCircularFilamentAgainstDirectAbscab

// Mimic a circular wire loop (for which we have accurate methods) by a polygon
// circle with adjusted radius. The radius adjustment is described in the
// following reference: McGreivy et al, "Computation of the Biot-Savart line
// integral with higher-order convergence using straight segments" Physics of
// Plasmas 28, 082111 (2021) (https://doi.org/10.1063/5.0058014)
TEST(TestVectorPotential, CheckPolygonFilament) {
  // A looser tolerance is allowed here in comparison with the other tests,
  // since a finite-element approximation is tested against an analytical
  // expression.
  constexpr double kTolerance = 1.0e-10;

  constexpr double kCurrent = 17.0;
  constexpr double kRadius = 1.23;
  static constexpr int kNumEvaluationLocations = 30;
  constexpr double kNumPhi = 100;

  // McGreivy radius correction
  const double delta_phi =
      2.0 * M_PI / (kNumPhi - 1);  // spacing between points

  // TODO(jons): understand derivation of alpha for special case of closed
  // circle |dr/ds| = 2*pi
  // --> alpha = 1/R * (dr)^2 / 12
  // == 4 pi^2 / (12 R)
  const double adjusted_radius = kRadius * (1.0 + delta_phi * delta_phi / 12.0);

  PolygonFilament polygon_filament = PolygonCircle(adjusted_radius, kNumPhi);

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  // setup one-dimensional vector of evaluation location coordinates for ABSCAB
  std::vector<double> evaluation_positions_reference(kNumEvaluationLocations *
                                                     3);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions_reference[i * 3 + 0] = evaluation_positions[i][0];
    evaluation_positions_reference[i * 3 + 1] = evaluation_positions[i][1];
    evaluation_positions_reference[i * 3 + 2] = evaluation_positions[i][2];
  }

  // compute reference using analytical expression for circular filament from
  // ABSCAB
  std::vector<double> vector_potential_reference(kNumEvaluationLocations * 3,
                                                 0.0);
  std::vector<double> center = {0.0, 0.0, 0.0};
  std::vector<double> normal = {0.0, 0.0, 1.0};

  // FIXME(jons): Figure out what the actual sign definition must be.
  // For now, adjusted to agree with MAKEGRID.
  abscab::vectorPotentialCircularFilament(
      center.data(), normal.data(), kRadius, -kCurrent, kNumEvaluationLocations,
      evaluation_positions_reference.data(), vector_potential_reference.data());

  // target array for method to test
  std::vector<std::vector<double> > vector_potential(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential[i].resize(3, 0.0);
  }

  // method under test here
  absl::Status status = VectorPotential(polygon_filament, kCurrent,
                                        evaluation_positions, vector_potential);
  EXPECT_EQ(status, absl::OkStatus());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 0],
                              vector_potential[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 1],
                              vector_potential[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 2],
                              vector_potential[i][2], kTolerance));
  }
}  // VectorPotential: CheckPolygonFilament

TEST(TestVectorPotential, CheckPolygonFilamentAgainstDirectAbscab) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 3.14;
  static constexpr int kNumEvaluationLocations = 3;

  PolygonFilament polygon_filament;
  Vector3d *vertex_1 = polygon_filament.add_vertices();
  vertex_1->set_x(1.0);
  vertex_1->set_y(2.0);
  vertex_1->set_z(3.0);
  Vector3d *vertex_2 = polygon_filament.add_vertices();
  vertex_2->set_x(3.0);
  vertex_2->set_y(2.0);
  vertex_2->set_z(1.0);

  // setup evaluation locations
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -(i + 1);
    evaluation_positions[i][1] = -(i + 1);
    evaluation_positions[i][2] = -(i + 1);
  }

  // setup target array for magnetic field to be tested
  std::vector<std::vector<double> > vector_potential(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential[i].resize(3, 0.0);
  }

  // setup one-dimensional vector of evaluation location coordinates for ABSCAB
  std::vector<double> evaluation_positions_reference(kNumEvaluationLocations *
                                                     3);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions_reference[i * 3 + 0] = evaluation_positions[i][0];
    evaluation_positions_reference[i * 3 + 1] = evaluation_positions[i][1];
    evaluation_positions_reference[i * 3 + 2] = evaluation_positions[i][2];
  }

  // setup one-dimensional vector for polygon geometry for ABSCAB
  const int num_vertices = polygon_filament.vertices_size();
  std::vector<double> vertices(num_vertices * 3);
  for (int i = 0; i < num_vertices; ++i) {
    vertices[i * 3 + 0] = polygon_filament.vertices(i).x();
    vertices[i * 3 + 1] = polygon_filament.vertices(i).y();
    vertices[i * 3 + 2] = polygon_filament.vertices(i).z();
  }

  // setup one-dimensional vector of magnetic field values for ABSCAB
  std::vector<double> vector_potential_reference(kNumEvaluationLocations * 3,
                                                 0.0);

  absl::Status status = VectorPotential(polygon_filament, kCurrent,
                                        evaluation_positions, vector_potential);
  EXPECT_EQ(status, absl::OkStatus());

  // reference computation using ABSCAB directly
  abscab::vectorPotentialPolygonFilament(
      num_vertices, vertices.data(), kCurrent, kNumEvaluationLocations,
      evaluation_positions_reference.data(), vector_potential_reference.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 0],
                              vector_potential[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 1],
                              vector_potential[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference[i * 3 + 2],
                              vector_potential[i][2], kTolerance));
  }
}  // VectorPotential: CheckPolygonFilamentAgainstDirectAbscab

TEST(TestVectorPotential, CheckFourierFilament) {
  // TODO(jons): implement
}  // VectorPotential: CheckFourierFilament

// ------------------

struct IntroducedErrorsInMagneticConfiguration {
  // bit field:
  // if bit 0 is true, omit the center's x component
  // if bit 1 is true, omit the center's y component
  // if bit 2 is true, omit the center's z component
  int circular_filament_center_omitted_components;

  // bit field:
  // if bit 0 is true, omit the normal's x component
  // if bit 1 is true, omit the normal's y component
  // if bit 2 is true, omit the normal's z component
  int circular_filament_normal_omitted_components;

  bool circular_filament_omit_radius;

  // ------------

  // bit field:
  // if bit 0 is true, omit the x component of the first vertex
  // if bit 1 is true, omit the y component of the first vertex
  // if bit 2 is true, omit the z component of the first vertex
  bool polygon_filament_vertex_1_omitted_components;

  // bit field:
  // if bit 0 is true, omit the x component of the second vertex
  // if bit 1 is true, omit the y component of the second vertex
  // if bit 2 is true, omit the z component of the second vertex
  bool polygon_filament_vertex_2_omitted_components;

  static IntroducedErrorsInMagneticConfiguration NoErrors() {
    IntroducedErrorsInMagneticConfiguration no_errors;

    no_errors.circular_filament_center_omitted_components = 0;
    no_errors.circular_filament_normal_omitted_components = 0;
    no_errors.circular_filament_omit_radius = false;

    no_errors.polygon_filament_vertex_1_omitted_components = 0;
    no_errors.polygon_filament_vertex_2_omitted_components = 0;

    return no_errors;
  }

  static bool HasAnyError(
      const IntroducedErrorsInMagneticConfiguration &introduced_errors) {
    if (introduced_errors.circular_filament_center_omitted_components != 0) {
      return true;
    }
    if (introduced_errors.circular_filament_normal_omitted_components != 0) {
      return true;
    }
    if (introduced_errors.circular_filament_omit_radius) {
      return true;
    }

    if (introduced_errors.polygon_filament_vertex_1_omitted_components != 0) {
      return true;
    }
    if (introduced_errors.polygon_filament_vertex_2_omitted_components != 0) {
      return true;
    }

    return false;
  }
};

MagneticConfiguration MakeMagneticConfiguration(
    const IntroducedErrorsInMagneticConfiguration &introduced_errors =
        IntroducedErrorsInMagneticConfiguration::NoErrors()) {
  MagneticConfiguration magnetic_configuration;

  // first current carrier: a CircularFilament
  static constexpr double kCurrent1 = 3.14;
  static constexpr double kRadius = 2.71;

  // centered at the origin, normal along z axis
  std::vector<double> center = {
      0.0, 0.0, 0.0};  // TODO(jons): make const once ABSCAB supports this
  std::vector<double> normal = {
      0.0, 0.0, 1.0};  // TODO(jons): make const once ABSCAB supports this

  SerialCircuit *serial_circuit_1 =
      magnetic_configuration.add_serial_circuits();
  serial_circuit_1->set_current(kCurrent1);
  Coil *coil_1 = serial_circuit_1->add_coils();
  CurrentCarrier *current_carrier_1 = coil_1->add_current_carriers();

  CircularFilament *circular_filament =
      current_carrier_1->mutable_circular_filament();

  Vector3d *center_vector = circular_filament->mutable_center();
  if (!(introduced_errors.circular_filament_center_omitted_components &
        (1 << 0))) {
    center_vector->set_x(center[0]);
  }
  if (!(introduced_errors.circular_filament_center_omitted_components &
        (1 << 1))) {
    center_vector->set_y(center[1]);
  }
  if (!(introduced_errors.circular_filament_center_omitted_components &
        (1 << 2))) {
    center_vector->set_z(center[2]);
  }

  Vector3d *normal_vector = circular_filament->mutable_normal();
  if (!(introduced_errors.circular_filament_normal_omitted_components &
        (1 << 0))) {
    normal_vector->set_x(normal[0]);
  }
  if (!(introduced_errors.circular_filament_normal_omitted_components &
        (1 << 1))) {
    normal_vector->set_y(normal[1]);
  }
  if (!(introduced_errors.circular_filament_normal_omitted_components &
        (1 << 2))) {
    normal_vector->set_z(normal[2]);
  }

  if (!introduced_errors.circular_filament_omit_radius) {
    circular_filament->set_radius(kRadius);
  }

  // second current carrier: a PolygonFilament
  static constexpr double kCurrent2 = 13.0;

  SerialCircuit *serial_circuit_2 =
      magnetic_configuration.add_serial_circuits();
  serial_circuit_2->set_current(kCurrent2);
  Coil *coil_2 = serial_circuit_2->add_coils();
  CurrentCarrier *current_carrier_2 = coil_2->add_current_carriers();

  PolygonFilament *polygon_filament =
      current_carrier_2->mutable_polygon_filament();

  Vector3d *vertex_1 = polygon_filament->add_vertices();
  if (!(introduced_errors.polygon_filament_vertex_1_omitted_components &
        (1 << 0))) {
    vertex_1->set_x(1.0);
  }
  if (!(introduced_errors.polygon_filament_vertex_1_omitted_components &
        (1 << 1))) {
    vertex_1->set_y(2.0);
  }
  if (!(introduced_errors.polygon_filament_vertex_1_omitted_components &
        (1 << 2))) {
    vertex_1->set_z(3.0);
  }

  Vector3d *vertex_2 = polygon_filament->add_vertices();
  if (!(introduced_errors.polygon_filament_vertex_2_omitted_components &
        (1 << 0))) {
    vertex_2->set_x(3.0);
  }
  if (!(introduced_errors.polygon_filament_vertex_2_omitted_components &
        (1 << 1))) {
    vertex_2->set_y(2.0);
  }
  if (!(introduced_errors.polygon_filament_vertex_2_omitted_components &
        (1 << 2))) {
    vertex_2->set_z(1.0);
  }

  return magnetic_configuration;
}  // MakeMagneticConfiguration

// Check that the magnetic field from the MagneticConfiguration made by
// MakeMagneticConfiguration() actually contains all contributions from the
// contained current carriers.
TEST(TestMagneticField, CheckFullyPopulatedMagneticConfiguration) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr int kNumEvaluationLocations = 30;

  MagneticConfiguration magnetic_configuration = MakeMagneticConfiguration();

  // first make sure that it is actually fully populated
  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration);
  EXPECT_TRUE(status.ok());

  // evaluation locations for testing
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  std::vector<std::vector<double> > magnetic_field_reference(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field_reference[i].resize(3, 0.0);
  }

  // now compute the magnetic field using the method for a full
  // MagneticConfiguration
  status = MagneticField(magnetic_configuration, evaluation_positions,
                         magnetic_field_reference);
  EXPECT_TRUE(status.ok());

  // This is the target array for testing, filled with some previous data that
  // should be preserved.
  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3);

    magnetic_field[i][0] = i * 3 + 0;
    magnetic_field[i][1] = i * 3 + 1;
    magnetic_field[i][2] = i * 3 + 2;
  }

  // now compute the contributions individually
  for (const SerialCircuit &serial_circuit :
       magnetic_configuration.serial_circuits()) {
    // This test only makes sense if a current is present.
    EXPECT_TRUE(serial_circuit.has_current());
    double current = serial_circuit.current();

    for (const Coil &coil : serial_circuit.coils()) {
      // num_windings is optional, so include it here if it was to be included
      // in a future version of this test
      if (coil.has_num_windings()) {
        current *= coil.num_windings();
      }

      for (const CurrentCarrier &current_carrier : coil.current_carriers()) {
        switch (current_carrier.type_case()) {
          case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
            CHECK_OK(MagneticField(current_carrier.infinite_straight_filament(),
                                   current, evaluation_positions,
                                   magnetic_field));
            break;
          case CurrentCarrier::TypeCase::kCircularFilament:
            CHECK_OK(MagneticField(current_carrier.circular_filament(), current,
                                   evaluation_positions, magnetic_field));
            break;
          case CurrentCarrier::TypeCase::kPolygonFilament:
            CHECK_OK(MagneticField(current_carrier.polygon_filament(), current,
                                   evaluation_positions, magnetic_field));
            break;
          case CurrentCarrier::TypeCase::kTypeNotSet:
            // consider as empty CurrentCarrier -> ignore
            break;
          default:
            std::stringstream error_message;
            error_message << "current carrier type ";
            error_message << current_carrier.type_case();
            error_message << " not implemented yet.";
            LOG(FATAL) << error_message.str();
        }
      }  // CurrentCarrier
    }  // Coil
  }  // SerialCircuit

  // now check that the correct magnetic field has been computed
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0 + magnetic_field_reference[i][0],
                              magnetic_field[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1 + magnetic_field_reference[i][1],
                              magnetic_field[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2 + magnetic_field_reference[i][2],
                              magnetic_field[i][2], kTolerance));
  }
}  // MagneticField: CheckFullyPopulatedMagneticConfiguration

// The first integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the CircularFilament's center vector are
// populated. Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
// The second integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the CircularFilament's normal vector are
// populated. Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
class NoMagneticFieldModificationIfErrorInCircularFilamentTest
    : public TestWithParam< ::std::tuple<int, int, bool> > {
 protected:
  void SetUp() override {
    std::tie(omitted_center_components_, omitted_normal_components_,
             omit_radius_) = GetParam();
  }
  int omitted_center_components_;
  int omitted_normal_components_;
  bool omit_radius_;
};

// Rely on CheckFullyPopulatedMagneticConfiguration working and then test
// if the contributions are omitted if an error is introduced
// in any of the parameters of the CircularFilament.
TEST_P(NoMagneticFieldModificationIfErrorInCircularFilamentTest,
       CheckNoModificationIfErrorInCircularFilament) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr int kNumEvaluationLocations = 30;

  // reference, without errors
  MagneticConfiguration magnetic_configuration = MakeMagneticConfiguration();

  // evaluation locations for testing
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  std::vector<std::vector<double> > magnetic_field_reference(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field_reference[i].resize(3, 0.0);
  }

  // now compute the magnetic field using the method for a full
  // MagneticConfiguration
  absl::Status status = MagneticField(
      magnetic_configuration, evaluation_positions, magnetic_field_reference);
  EXPECT_TRUE(status.ok());

  // This is the target array for testing, filled with some previous data that
  // should be preserved.
  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3);

    magnetic_field[i][0] = i * 3 + 0;
    magnetic_field[i][1] = i * 3 + 1;
    magnetic_field[i][2] = i * 3 + 2;
  }

  IntroducedErrorsInMagneticConfiguration introduced_errors =
      IntroducedErrorsInMagneticConfiguration::NoErrors();
  introduced_errors.circular_filament_center_omitted_components =
      omitted_center_components_;
  introduced_errors.circular_filament_normal_omitted_components =
      omitted_normal_components_;
  introduced_errors.circular_filament_omit_radius = omit_radius_;

  MagneticConfiguration magnetic_configuration_with_errors =
      MakeMagneticConfiguration(introduced_errors);

  absl::Status magnetic_field_status = MagneticField(
      magnetic_configuration_with_errors, evaluation_positions, magnetic_field);

  // make sure that the MagneticConfiguration is actually broken when it should
  // be
  absl::Status fully_populated_status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_with_errors);
  if (IntroducedErrorsInMagneticConfiguration::HasAnyError(introduced_errors)) {
    EXPECT_FALSE(magnetic_field_status.ok());
    EXPECT_FALSE(fully_populated_status.ok());

    // check that the previous array contents are not modified
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0, magnetic_field[i][0], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1, magnetic_field[i][1], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2, magnetic_field[i][2], kTolerance));
    }
  } else {
    EXPECT_TRUE(magnetic_field_status.ok());
    EXPECT_TRUE(fully_populated_status.ok());

    // check that correct magnetic field is computed and added to previous array
    // contents
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0 + magnetic_field_reference[i][0],
                                magnetic_field[i][0], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1 + magnetic_field_reference[i][1],
                                magnetic_field[i][1], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2 + magnetic_field_reference[i][2],
                                magnetic_field[i][2], kTolerance));
    }
  }
}  // CheckNoModificationIfErrorInCircularFilament

INSTANTIATE_TEST_SUITE_P(
    TestMagneticField, NoMagneticFieldModificationIfErrorInCircularFilamentTest,
    Combine(Values(0, 1, 2, 3, 4, 5, 6, 7), Values(0, 1, 2, 3, 4, 5, 6, 7),
            Bool()));

// The first integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the PolygonFilaments's first vertex are
// populated. Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
// The second integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the PolygonFilaments's second vertex are
// populated. Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
class NoMagneticFieldModificationIfErrorInPolygonFilamentTest
    : public TestWithParam< ::std::tuple<int, int> > {
 protected:
  void SetUp() override {
    std::tie(omitted_vertex_1_components_, omitted_vertex_2_components_) =
        GetParam();
  }
  int omitted_vertex_1_components_;
  int omitted_vertex_2_components_;
};

// Rely on CheckFullyPopulatedMagneticConfiguration working and then test
// if the contributions are omitted if an error is introduced
// in any of the parameters of the PolygonFilaments.
TEST_P(NoMagneticFieldModificationIfErrorInPolygonFilamentTest,
       CheckNoModificationIfErrorInPolygonFilament) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr int kNumEvaluationLocations = 30;

  // reference, without errors
  MagneticConfiguration magnetic_configuration = MakeMagneticConfiguration();

  // evaluation locations for testing
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  std::vector<std::vector<double> > magnetic_field_reference(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field_reference[i].resize(3, 0.0);
  }

  // now compute the magnetic field using the method for a full
  // MagneticConfiguration
  absl::Status status = MagneticField(
      magnetic_configuration, evaluation_positions, magnetic_field_reference);
  EXPECT_TRUE(status.ok());

  // This is the target array for testing, filled with some previous data that
  // should be preserved.
  std::vector<std::vector<double> > magnetic_field(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field[i].resize(3);

    magnetic_field[i][0] = i * 3 + 0;
    magnetic_field[i][1] = i * 3 + 1;
    magnetic_field[i][2] = i * 3 + 2;
  }

  IntroducedErrorsInMagneticConfiguration introduced_errors =
      IntroducedErrorsInMagneticConfiguration::NoErrors();
  introduced_errors.polygon_filament_vertex_1_omitted_components =
      omitted_vertex_1_components_;
  introduced_errors.polygon_filament_vertex_2_omitted_components =
      omitted_vertex_2_components_;

  MagneticConfiguration magnetic_configuration_with_errors =
      MakeMagneticConfiguration(introduced_errors);

  absl::Status magnetic_field_status = MagneticField(
      magnetic_configuration_with_errors, evaluation_positions, magnetic_field);

  // make sure that the MagneticConfiguration is actually broken when it should
  // be
  absl::Status fully_populated_status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_with_errors);
  if (IntroducedErrorsInMagneticConfiguration::HasAnyError(introduced_errors)) {
    EXPECT_FALSE(magnetic_field_status.ok());
    EXPECT_FALSE(fully_populated_status.ok());

    // check that the previous array contents are not modified
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0, magnetic_field[i][0], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1, magnetic_field[i][1], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2, magnetic_field[i][2], kTolerance));
    }
  } else {
    EXPECT_TRUE(magnetic_field_status.ok());
    EXPECT_TRUE(fully_populated_status.ok());

    // check that correct magnetic field is computed and added to previous array
    // contents
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0 + magnetic_field_reference[i][0],
                                magnetic_field[i][0], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1 + magnetic_field_reference[i][1],
                                magnetic_field[i][1], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2 + magnetic_field_reference[i][2],
                                magnetic_field[i][2], kTolerance));
    }
  }
}  // CheckNoModificationIfErrorInPolygonFilament

INSTANTIATE_TEST_SUITE_P(
    TestMagneticField, NoMagneticFieldModificationIfErrorInPolygonFilamentTest,
    Combine(Values(0, 1, 2, 3, 4, 5, 6, 7), Values(0, 1, 2, 3, 4, 5, 6, 7)));

// -------------------

// Check that the magnetic field from the MagneticConfiguration made by
// MakeFullyPopulatedMagneticConfiguration() actually contains
// all contributions from the contained current carriers.
TEST(TestVectorPotential, CheckFullyPopulatedMagneticConfiguration) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr int kNumEvaluationLocations = 30;

  MagneticConfiguration magnetic_configuration = MakeMagneticConfiguration();

  // first make sure that it is actually fully populated
  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration);
  EXPECT_TRUE(status.ok());

  // evaluation locations for testing
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  std::vector<std::vector<double> > vector_potential_reference(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential_reference[i].resize(3, 0.0);
  }

  // now compute the magnetic field using the method for a full
  // MagneticConfiguration
  status = VectorPotential(magnetic_configuration, evaluation_positions,
                           vector_potential_reference);
  EXPECT_TRUE(status.ok());

  // This is the target array for testing, filled with some previous data that
  // should be preserved.
  std::vector<std::vector<double> > vector_potential(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential[i].resize(3);

    vector_potential[i][0] = i * 3 + 0;
    vector_potential[i][1] = i * 3 + 1;
    vector_potential[i][2] = i * 3 + 2;
  }

  // now compute the contributions individually
  for (const SerialCircuit &serial_circuit :
       magnetic_configuration.serial_circuits()) {
    // This test only makes sense if a current has been specified.
    EXPECT_TRUE(serial_circuit.has_current());
    double current = serial_circuit.current();

    for (const Coil &coil : serial_circuit.coils()) {
      // num_windings is optional, so include it here if it was to be included
      // in a future version of this test
      if (coil.has_num_windings()) {
        current *= coil.num_windings();
      }

      for (const CurrentCarrier &current_carrier : coil.current_carriers()) {
        switch (current_carrier.type_case()) {
          case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
            LOG(FATAL) << "Cannot compute the magnetic vector potential of an "
                          "infinite straight filament.";
            break;
          case CurrentCarrier::TypeCase::kCircularFilament:
            CHECK_OK(VectorPotential(current_carrier.circular_filament(),
                                     current, evaluation_positions,
                                     vector_potential));
            break;
          case CurrentCarrier::TypeCase::kPolygonFilament:
            CHECK_OK(VectorPotential(current_carrier.polygon_filament(),
                                     current, evaluation_positions,
                                     vector_potential));
            break;
          case CurrentCarrier::TypeCase::kTypeNotSet:
            // consider as empty CurrentCarrier -> ignore
            break;
          default:
            std::stringstream error_message;
            error_message << "current carrier type ";
            error_message << current_carrier.type_case();
            error_message << " not implemented yet.";
            LOG(FATAL) << error_message.str();
        }
      }  // CurrentCarrier
    }  // Coil
  }  // SerialCircuit

  // now check that the correct vector potential has been computed
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0 + vector_potential_reference[i][0],
                              vector_potential[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1 + vector_potential_reference[i][1],
                              vector_potential[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2 + vector_potential_reference[i][2],
                              vector_potential[i][2], kTolerance));
  }
}  // VectorPotential: CheckFullyPopulatedMagneticConfiguration

// The first integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the CircularFilament's center vector are
// populated. Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
// The second integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the CircularFilament's normal vector are
// populated. Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
class NoVectorPotentialModificationIfErrorInCircularFilamentTest
    : public TestWithParam< ::std::tuple<int, int, bool> > {
 protected:
  void SetUp() override {
    std::tie(omitted_center_components_, omitted_normal_components_,
             omit_radius_) = GetParam();
  }
  int omitted_center_components_;
  int omitted_normal_components_;
  bool omit_radius_;
};

// Rely on CheckFullyPopulatedMagneticConfiguration working and then test
// if the contributions are omitted if an error is introduced
// in any of the parameters of the CircularFilament.
TEST_P(NoVectorPotentialModificationIfErrorInCircularFilamentTest,
       CheckNoModificationIfErrorInCircularFilament) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr int kNumEvaluationLocations = 30;

  // reference, without errors
  MagneticConfiguration magnetic_configuration = MakeMagneticConfiguration();

  // evaluation locations for testing
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  std::vector<std::vector<double> > vector_potential_reference(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential_reference[i].resize(3, 0.0);
  }

  // now compute the vector potential using the method for a full
  // MagneticConfiguration
  absl::Status status = VectorPotential(
      magnetic_configuration, evaluation_positions, vector_potential_reference);
  EXPECT_TRUE(status.ok());

  // This is the target array for testing, filled with some previous data that
  // should be preserved.
  std::vector<std::vector<double> > vector_potential(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential[i].resize(3);

    vector_potential[i][0] = i * 3 + 0;
    vector_potential[i][1] = i * 3 + 1;
    vector_potential[i][2] = i * 3 + 2;
  }

  IntroducedErrorsInMagneticConfiguration introduced_errors =
      IntroducedErrorsInMagneticConfiguration::NoErrors();
  introduced_errors.circular_filament_center_omitted_components =
      omitted_center_components_;
  introduced_errors.circular_filament_normal_omitted_components =
      omitted_normal_components_;
  introduced_errors.circular_filament_omit_radius = omit_radius_;

  MagneticConfiguration magnetic_configuration_with_errors =
      MakeMagneticConfiguration(introduced_errors);

  absl::Status vector_potential_status =
      VectorPotential(magnetic_configuration_with_errors, evaluation_positions,
                      vector_potential);

  // make sure that the MagneticConfiguration is actually broken when it should
  // be
  absl::Status fully_populated_status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_with_errors);
  if (IntroducedErrorsInMagneticConfiguration::HasAnyError(introduced_errors)) {
    EXPECT_FALSE(vector_potential_status.ok());
    EXPECT_FALSE(fully_populated_status.ok());

    // check that the previous array contents are not modified
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0, vector_potential[i][0], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1, vector_potential[i][1], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2, vector_potential[i][2], kTolerance));
    }
  } else {
    EXPECT_TRUE(vector_potential_status.ok());
    EXPECT_TRUE(fully_populated_status.ok());

    // check that correct vector potential is computed and added to previous
    // array contents
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0 + vector_potential_reference[i][0],
                                vector_potential[i][0], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1 + vector_potential_reference[i][1],
                                vector_potential[i][1], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2 + vector_potential_reference[i][2],
                                vector_potential[i][2], kTolerance));
    }
  }
}  // CheckNoModificationIfErrorInCircularFilament

INSTANTIATE_TEST_SUITE_P(
    TestVectorPotential,
    NoVectorPotentialModificationIfErrorInCircularFilamentTest,
    Combine(Values(0, 1, 2, 3, 4, 5, 6, 7), Values(0, 1, 2, 3, 4, 5, 6, 7),
            Bool()));

// The first integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the PolygonFilaments's first vertex are
// populated. Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
// The second integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the PolygonFilaments's second vertex are
// populated. Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
class NoVectorPotentialModificationIfErrorInPolygonFilamentTest
    : public TestWithParam< ::std::tuple<int, int> > {
 protected:
  void SetUp() override {
    std::tie(omitted_vertex_1_components_, omitted_vertex_2_components_) =
        GetParam();
  }
  int omitted_vertex_1_components_;
  int omitted_vertex_2_components_;
};

// Rely on CheckFullyPopulatedMagneticConfiguration working and then test
// if the contributions are omitted if an error is introduced
// in any of the parameters of the PolygonFilaments.
TEST_P(NoVectorPotentialModificationIfErrorInPolygonFilamentTest,
       CheckNoModificationIfErrorInPolygonFilament) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr int kNumEvaluationLocations = 30;

  // reference, without errors
  MagneticConfiguration magnetic_configuration = MakeMagneticConfiguration();

  // evaluation locations for testing
  std::vector<std::vector<double> > evaluation_positions(
      kNumEvaluationLocations);
  const double delta_r = 2.0 / (kNumEvaluationLocations - 1.0);
  const double delta_z = 5.0 / (kNumEvaluationLocations - 1.0);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    evaluation_positions[i].resize(3);

    evaluation_positions[i][0] = -1.0 + delta_r * i;
    evaluation_positions[i][1] = 1.0 - delta_r * i;
    evaluation_positions[i][2] = -2.5 + delta_z * i;
  }

  std::vector<std::vector<double> > vector_potential_reference(
      kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential_reference[i].resize(3, 0.0);
  }

  // now compute the vector potential using the method for a full
  // MagneticConfiguration
  absl::Status status = VectorPotential(
      magnetic_configuration, evaluation_positions, vector_potential_reference);
  EXPECT_TRUE(status.ok());

  // This is the target array for testing, filled with some previous data that
  // should be preserved.
  std::vector<std::vector<double> > vector_potential(kNumEvaluationLocations);
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential[i].resize(3);

    vector_potential[i][0] = i * 3 + 0;
    vector_potential[i][1] = i * 3 + 1;
    vector_potential[i][2] = i * 3 + 2;
  }

  IntroducedErrorsInMagneticConfiguration introduced_errors =
      IntroducedErrorsInMagneticConfiguration::NoErrors();
  introduced_errors.polygon_filament_vertex_1_omitted_components =
      omitted_vertex_1_components_;
  introduced_errors.polygon_filament_vertex_2_omitted_components =
      omitted_vertex_2_components_;

  MagneticConfiguration magnetic_configuration_with_errors =
      MakeMagneticConfiguration(introduced_errors);

  absl::Status vector_potential_status =
      VectorPotential(magnetic_configuration_with_errors, evaluation_positions,
                      vector_potential);

  // make sure that the MagneticConfiguration is actually broken when it should
  // be
  absl::Status fully_populated_status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_with_errors);
  if (IntroducedErrorsInMagneticConfiguration::HasAnyError(introduced_errors)) {
    EXPECT_FALSE(vector_potential_status.ok());
    EXPECT_FALSE(fully_populated_status.ok());

    // check that the previous array contents are not modified
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0, vector_potential[i][0], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1, vector_potential[i][1], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2, vector_potential[i][2], kTolerance));
    }
  } else {
    EXPECT_TRUE(vector_potential_status.ok());
    EXPECT_TRUE(fully_populated_status.ok());

    // check that correct vector potential is computed and added to previous
    // array contents
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 0 + vector_potential_reference[i][0],
                                vector_potential[i][0], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 1 + vector_potential_reference[i][1],
                                vector_potential[i][1], kTolerance));
      EXPECT_TRUE(IsCloseRelAbs(i * 3 + 2 + vector_potential_reference[i][2],
                                vector_potential[i][2], kTolerance));
    }
  }
}  // CheckNoModificationIfErrorInPolygonFilament

INSTANTIATE_TEST_SUITE_P(
    TestVectorPotential,
    NoVectorPotentialModificationIfErrorInPolygonFilamentTest,
    Combine(Values(0, 1, 2, 3, 4, 5, 6, 7), Values(0, 1, 2, 3, 4, 5, 6, 7)));

// ---------------------

class MagneticFieldNumWindingsTest : public Test {
 protected:
  static constexpr double kCurrent = 3.14;
  static constexpr double kRadius = 2.71;
  static constexpr int kNumEvaluationLocations = 3;
  void SetUp() override {
    // setup MagneticConfiguration
    SerialCircuit *serial_circuit =
        magnetic_configuration_.add_serial_circuits();
    serial_circuit->set_current(kCurrent);
    coil_ = serial_circuit->add_coils();
    CurrentCarrier *current_carrier = coil_->add_current_carriers();
    CircularFilament *circular_filament =
        current_carrier->mutable_circular_filament();
    Vector3d *center = circular_filament->mutable_center();
    center->set_x(center_[0]);
    center->set_y(center_[1]);
    center->set_z(center_[2]);
    Vector3d *normal = circular_filament->mutable_normal();
    normal->set_x(normal_[0]);
    normal->set_y(normal_[1]);
    normal->set_z(normal_[2]);
    circular_filament->set_radius(kRadius);

    // setup evaluation locations
    evaluation_positions_.resize(kNumEvaluationLocations);
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      evaluation_positions_[i].resize(3);

      evaluation_positions_[i][0] = -(i + 1);
      evaluation_positions_[i][1] = -(i + 1);
      evaluation_positions_[i][2] = -(i + 1);
    }

    // setup target array for magnetic field to be tested
    magnetic_field_.resize(kNumEvaluationLocations);
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      magnetic_field_[i].resize(3, 0.0);
    }

    // setup one-dimensional vector of evaluation location coordinates for
    // ABSCAB
    evaluation_positions_reference_.resize(kNumEvaluationLocations * 3);
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      evaluation_positions_reference_[i * 3 + 0] = evaluation_positions_[i][0];
      evaluation_positions_reference_[i * 3 + 1] = evaluation_positions_[i][1];
      evaluation_positions_reference_[i * 3 + 2] = evaluation_positions_[i][2];
    }

    // setup one-dimensional vector of magnetic field values for ABSCAB
    magnetic_field_reference_.resize(kNumEvaluationLocations * 3, 0.0);
  }
  void SetNumWindings(int num_windings) {
    coil_->set_num_windings(num_windings);
  }
  std::vector<double> center_ = {
      1.23, 4.56, 7.89};  // TODO(jons): make const once ABSCAB supports this
  std::vector<double> normal_ = {
      9.87, 6.54, 3.21};  // TODO(jons): make const once ABSCAB supports this
  MagneticConfiguration magnetic_configuration_;
  Coil *coil_;
  std::vector<std::vector<double> > evaluation_positions_;
  std::vector<std::vector<double> > magnetic_field_;
  std::vector<double> evaluation_positions_reference_;
  std::vector<double> magnetic_field_reference_;
};  // MagneticFieldNumWindingsTest

TEST_F(MagneticFieldNumWindingsTest, CheckNoNumWindingsSpecified) {
  static constexpr double kTolerance = 1.0e-15;

  // do not specify num_windings -> expect assumed num_windings = 1
  const double current = 1 * kCurrent;

  absl::Status status = MagneticField(magnetic_configuration_,
                                      evaluation_positions_, magnetic_field_);
  EXPECT_TRUE(status.ok());

  // reference computation using ABSCAB directly
  abscab::magneticFieldCircularFilament(
      center_.data(), normal_.data(), kRadius, current, kNumEvaluationLocations,
      evaluation_positions_reference_.data(), magnetic_field_reference_.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 0],
                              magnetic_field_[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 1],
                              magnetic_field_[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 2],
                              magnetic_field_[i][2], kTolerance));
  }
}  // MagneticField: CheckNoNumWindingsSpecified

TEST_F(MagneticFieldNumWindingsTest, CheckZeroNumWindingsSpecified) {
  static constexpr double kTolerance = 1.0e-15;

  // specify zero num_windings
  // -> expect to see that the previous array contents are preserved,
  //    as this implies that no magnetic field is produced.
  constexpr int kNumWindings = 0;
  SetNumWindings(kNumWindings);
  const double current = kNumWindings * kCurrent;

  // Put some previous contents in both magnetic fields
  // for testing that these contens are preserved.
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    magnetic_field_[i][0] += i * 3 + 0;
    magnetic_field_[i][1] += i * 3 + 1;
    magnetic_field_[i][2] += i * 3 + 2;

    magnetic_field_reference_[i * 3 + 0] += i * 3 + 0;
    magnetic_field_reference_[i * 3 + 1] += i * 3 + 1;
    magnetic_field_reference_[i * 3 + 2] += i * 3 + 2;
  }

  absl::Status status = MagneticField(magnetic_configuration_,
                                      evaluation_positions_, magnetic_field_);
  EXPECT_TRUE(status.ok());

  // reference computation using ABSCAB directly
  abscab::magneticFieldCircularFilament(
      center_.data(), normal_.data(), kRadius, current, kNumEvaluationLocations,
      evaluation_positions_reference_.data(), magnetic_field_reference_.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 0],
                              magnetic_field_[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 1],
                              magnetic_field_[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 2],
                              magnetic_field_[i][2], kTolerance));
  }
}  // MagneticField: CheckZeroNumWindingsSpecified

TEST_F(MagneticFieldNumWindingsTest, CheckNonZeroNumWindingsSpecified) {
  static constexpr double kTolerance = 1.0e-15;

  // specify non-zero num_windings -> expect to see this as a multiplier for the
  // current
  constexpr int kNumWindings = 13;
  SetNumWindings(kNumWindings);
  const double current = kNumWindings * kCurrent;

  absl::Status status = MagneticField(magnetic_configuration_,
                                      evaluation_positions_, magnetic_field_);
  EXPECT_TRUE(status.ok());

  // reference computation using ABSCAB directly
  abscab::magneticFieldCircularFilament(
      center_.data(), normal_.data(), kRadius, current, kNumEvaluationLocations,
      evaluation_positions_reference_.data(), magnetic_field_reference_.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 0],
                              magnetic_field_[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 1],
                              magnetic_field_[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(magnetic_field_reference_[i * 3 + 2],
                              magnetic_field_[i][2], kTolerance));
  }
}  // MagneticField: CheckNonZeroNumWindingsSpecified

TEST(TestMagneticField, CheckMultipleCoilsWithWindings) {
  // Make two different PolygonFilament current carriers.
  // Compute the sum of the individual magnetic fields.
  // Check that this is the same as the magnetic field of the composite
  // MagneticConfiguration.

  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 42.0;

  static constexpr double kRadius1 = 1.23;
  static constexpr int kNumberOfPhiGridPoints1 = 127;
  static constexpr double kZOffset1 = 0.45;
  static constexpr int kNumberOfWindings1 = 13;

  static constexpr double kRadius2 = 2.71;
  static constexpr int kNumberOfPhiGridPoints2 = 63;
  static constexpr double kZOffset2 = 0.13;
  static constexpr int kNumberOfWindings2 = 45;

  MagneticConfiguration magnetic_configuration_1;
  SerialCircuit *serial_circuit_1 =
      magnetic_configuration_1.add_serial_circuits();
  serial_circuit_1->set_current(kCurrent);
  Coil *coil_1 = serial_circuit_1->add_coils();
  coil_1->set_num_windings(kNumberOfWindings1);
  CurrentCarrier *current_carrier_1 = coil_1->add_current_carriers();
  PolygonFilament *circle_1 = current_carrier_1->mutable_polygon_filament();
  ASSERT_TRUE(PolygonCirclePopulate(*circle_1, kRadius1,
                                    kNumberOfPhiGridPoints1, kZOffset1)
                  .ok());

  MagneticConfiguration magnetic_configuration_2;
  SerialCircuit *serial_circuit_2 =
      magnetic_configuration_2.add_serial_circuits();
  serial_circuit_2->set_current(kCurrent);
  Coil *coil_2 = serial_circuit_2->add_coils();
  coil_2->set_num_windings(kNumberOfWindings2);
  CurrentCarrier *current_carrier_2 = coil_2->add_current_carriers();
  PolygonFilament *circle_2 = current_carrier_2->mutable_polygon_filament();
  ASSERT_TRUE(PolygonCirclePopulate(*circle_2, kRadius2,
                                    kNumberOfPhiGridPoints2, kZOffset2)
                  .ok());

  // evaluation locations [number_evaluation_locations][3: x, y, z]
  std::vector<std::vector<double> > evaluation_locations = {
      {0.0, 0.0, 0.0},
      {0.1, 0.2, 0.3},
      {1.0, 2.0, 3.0},
      {10.0, 20.0, 30.0},
  };

  // magnetic field [number_evaluation_locations][3: x, y, z] from only circle_1
  std::vector<std::vector<double> > magnetic_field_1 = {
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  // magnetic field [number_evaluation_locations][3: x, y, z] from only circle_2
  std::vector<std::vector<double> > magnetic_field_2 = {
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  // magnetic field [number_evaluation_locations][3: x, y, z] from both circle_1
  // and circle_2
  std::vector<std::vector<double> > magnetic_field = {
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  // compute magnetic fields individually
  absl::Status status_1 =
      MagneticField(magnetic_configuration_1, evaluation_locations,
                    /*m_magnetic_field=*/magnetic_field_1);
  ASSERT_TRUE(status_1.ok());

  absl::Status status_2 =
      MagneticField(magnetic_configuration_2, evaluation_locations,
                    /*m_magnetic_field=*/magnetic_field_2);
  ASSERT_TRUE(status_2.ok());

  // make a MagneticConfiguration which holds both circle_1 and circle_2
  MagneticConfiguration magnetic_configuration;
  SerialCircuit *serial_circuit = magnetic_configuration.add_serial_circuits();
  serial_circuit->set_current(kCurrent);

  Coil *coil_1_of_2 = serial_circuit->add_coils();
  coil_1_of_2->set_num_windings(kNumberOfWindings1);
  CurrentCarrier *current_carrier_1_of_2 = coil_1_of_2->add_current_carriers();
  PolygonFilament *circle_1_of_2 =
      current_carrier_1_of_2->mutable_polygon_filament();
  ASSERT_TRUE(PolygonCirclePopulate(*circle_1_of_2, kRadius1,
                                    kNumberOfPhiGridPoints1, kZOffset1)
                  .ok());

  Coil *coil_2_of_2 = serial_circuit->add_coils();
  coil_2_of_2->set_num_windings(kNumberOfWindings2);
  CurrentCarrier *current_carrier_2_of_2 = coil_2_of_2->add_current_carriers();
  PolygonFilament *circle_2_of_2 =
      current_carrier_2_of_2->mutable_polygon_filament();
  ASSERT_TRUE(PolygonCirclePopulate(*circle_2_of_2, kRadius2,
                                    kNumberOfPhiGridPoints2, kZOffset2)
                  .ok());

  // compute the magnetic field from both circle_1 and circle_2 at the same time
  absl::Status status =
      MagneticField(magnetic_configuration, evaluation_locations,
                    /*m_magnetic_field=*/magnetic_field);
  ASSERT_TRUE(status.ok());

  // compare the sum of the inidvidually-computed magnetic fields
  // with the magnetic field where both contributions were computed together
  const int number_of_evaluation_locations =
      static_cast<int>(evaluation_locations.size());
  for (int i = 0; i < number_of_evaluation_locations; ++i) {
    const double magnetic_field_x =
        magnetic_field_1[i][0] + magnetic_field_2[i][0];
    const double magnetic_field_y =
        magnetic_field_1[i][1] + magnetic_field_2[i][1];
    const double magnetic_field_z =
        magnetic_field_1[i][2] + magnetic_field_2[i][2];

    EXPECT_TRUE(
        IsCloseRelAbs(magnetic_field[i][0], magnetic_field_x, kTolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(magnetic_field[i][1], magnetic_field_y, kTolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(magnetic_field[i][2], magnetic_field_z, kTolerance));
  }
}  // MagneticField: CheckMultipleCoilsWithWindings

// -------------------

class VectorPotentialNumWindingsTest : public Test {
 protected:
  static constexpr double kCurrent = 3.14;
  static constexpr double kRadius = 2.71;
  static constexpr int kNumEvaluationLocations = 3;
  void SetUp() override {
    // setup MagneticConfiguration
    SerialCircuit *serial_circuit =
        magnetic_configuration_.add_serial_circuits();
    serial_circuit->set_current(kCurrent);
    coil_ = serial_circuit->add_coils();
    CurrentCarrier *current_carrier = coil_->add_current_carriers();
    CircularFilament *circular_filament =
        current_carrier->mutable_circular_filament();
    Vector3d *center = circular_filament->mutable_center();
    center->set_x(center_[0]);
    center->set_y(center_[1]);
    center->set_z(center_[2]);
    Vector3d *normal = circular_filament->mutable_normal();
    normal->set_x(normal_[0]);
    normal->set_y(normal_[1]);
    normal->set_z(normal_[2]);
    circular_filament->set_radius(kRadius);

    // setup evaluation locations
    evaluation_positions_.resize(kNumEvaluationLocations);
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      evaluation_positions_[i].resize(3);

      evaluation_positions_[i][0] = -(i + 1);
      evaluation_positions_[i][1] = -(i + 1);
      evaluation_positions_[i][2] = -(i + 1);
    }

    // setup target array for vector potential to be tested
    vector_potential_.resize(kNumEvaluationLocations);
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      vector_potential_[i].resize(3, 0.0);
    }

    // setup one-dimensional vector of evaluation location coordinates for
    // ABSCAB
    evaluation_positions_reference_.resize(kNumEvaluationLocations * 3);
    for (int i = 0; i < kNumEvaluationLocations; ++i) {
      evaluation_positions_reference_[i * 3 + 0] = evaluation_positions_[i][0];
      evaluation_positions_reference_[i * 3 + 1] = evaluation_positions_[i][1];
      evaluation_positions_reference_[i * 3 + 2] = evaluation_positions_[i][2];
    }

    // setup one-dimensional vector of magnetic field values for ABSCAB
    vector_potential_reference_.resize(kNumEvaluationLocations * 3, 0.0);
  }
  void SetNumWindings(int num_windings) {
    coil_->set_num_windings(num_windings);
  }
  std::vector<double> center_ = {
      1.23, 4.56, 7.89};  // TODO(jons): make const once ABSCAB supports this
  std::vector<double> normal_ = {
      9.87, 6.54, 3.21};  // TODO(jons): make const once ABSCAB supports this
  MagneticConfiguration magnetic_configuration_;
  Coil *coil_;
  std::vector<std::vector<double> > evaluation_positions_;
  std::vector<std::vector<double> > vector_potential_;
  std::vector<double> evaluation_positions_reference_;
  std::vector<double> vector_potential_reference_;
};  // VectorPotentialNumWindingsTest

TEST_F(VectorPotentialNumWindingsTest, CheckNoNumWindingsSpecified) {
  static constexpr double kTolerance = 1.0e-15;

  // do not specify num_windings -> expect assumed num_windings = 1
  const double current = 1 * kCurrent;

  absl::Status status = VectorPotential(
      magnetic_configuration_, evaluation_positions_, vector_potential_);
  EXPECT_TRUE(status.ok());

  // reference computation using ABSCAB directly
  // FIXME(jons): Figure out what the actual sign definition must be.
  // For now, adjusted to agree with MAKEGRID.
  abscab::vectorPotentialCircularFilament(
      center_.data(), normal_.data(), kRadius, -current,
      kNumEvaluationLocations, evaluation_positions_reference_.data(),
      vector_potential_reference_.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 0],
                              vector_potential_[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 1],
                              vector_potential_[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 2],
                              vector_potential_[i][2], kTolerance));
  }
}  // VectorPotential: CheckNoNumWindingsSpecified

TEST_F(VectorPotentialNumWindingsTest, CheckZeroNumWindingsSpecified) {
  static constexpr double kTolerance = 1.0e-15;

  // specify zero num_windings
  // -> expect to see that the previous array contents are preserved,
  //    as this implies that no magnetic field is produced.
  constexpr int kNumWindings = 0;
  SetNumWindings(kNumWindings);
  const double current = kNumWindings * kCurrent;

  // Put some previous contents in both magnetic fields
  // for testing that these contens are preserved.
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    vector_potential_[i][0] += i * 3 + 0;
    vector_potential_[i][1] += i * 3 + 1;
    vector_potential_[i][2] += i * 3 + 2;

    vector_potential_reference_[i * 3 + 0] += i * 3 + 0;
    vector_potential_reference_[i * 3 + 1] += i * 3 + 1;
    vector_potential_reference_[i * 3 + 2] += i * 3 + 2;
  }

  absl::Status status = VectorPotential(
      magnetic_configuration_, evaluation_positions_, vector_potential_);
  EXPECT_TRUE(status.ok());

  // reference computation using ABSCAB directly
  // FIXME(jons): Figure out what the actual sign definition must be.
  // For now, adjusted to agree with MAKEGRID.
  abscab::vectorPotentialCircularFilament(
      center_.data(), normal_.data(), kRadius, -current,
      kNumEvaluationLocations, evaluation_positions_reference_.data(),
      vector_potential_reference_.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 0],
                              vector_potential_[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 1],
                              vector_potential_[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 2],
                              vector_potential_[i][2], kTolerance));
  }
}  // VectorPotential: CheckZeroNumWindingsSpecified

TEST_F(VectorPotentialNumWindingsTest, CheckNonZeroNumWindingsSpecified) {
  static constexpr double kTolerance = 1.0e-15;

  // specify non-zero num_windings -> expect to see this as a multiplier for the
  // current
  constexpr int kNumWindings = 13;
  SetNumWindings(kNumWindings);
  const double current = kNumWindings * kCurrent;

  absl::Status status = VectorPotential(
      magnetic_configuration_, evaluation_positions_, vector_potential_);
  EXPECT_TRUE(status.ok());

  // reference computation using ABSCAB directly
  // FIXME(jons): Figure out what the actual sign definition must be.
  // For now, adjusted to agree with MAKEGRID.
  abscab::vectorPotentialCircularFilament(
      center_.data(), normal_.data(), kRadius, -current,
      kNumEvaluationLocations, evaluation_positions_reference_.data(),
      vector_potential_reference_.data());

  // perform comparison
  for (int i = 0; i < kNumEvaluationLocations; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 0],
                              vector_potential_[i][0], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 1],
                              vector_potential_[i][1], kTolerance));
    EXPECT_TRUE(IsCloseRelAbs(vector_potential_reference_[i * 3 + 2],
                              vector_potential_[i][2], kTolerance));
  }
}  // VectorPotential: CheckNonZeroNumWindingsSpecified

TEST(TestVectorPotential, CheckMultipleCoilsWithWindings) {
  // Make two different PolygonFilament current carriers.
  // Compute the sum of the individual vector potentials.
  // Check that this is the same as the vector potential of the composite
  // MagneticConfiguration.

  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kCurrent = 42.0;

  static constexpr double kRadius1 = 1.23;
  static constexpr int kNumberOfPhiGridPoints1 = 127;
  static constexpr double kZOffset1 = 0.45;
  static constexpr int kNumberOfWindings1 = 13;

  static constexpr double kRadius2 = 2.71;
  static constexpr int kNumberOfPhiGridPoints2 = 63;
  static constexpr double kZOffset2 = 0.13;
  static constexpr int kNumberOfWindings2 = 45;

  MagneticConfiguration magnetic_configuration_1;
  SerialCircuit *serial_circuit_1 =
      magnetic_configuration_1.add_serial_circuits();
  serial_circuit_1->set_current(kCurrent);
  Coil *coil_1 = serial_circuit_1->add_coils();
  coil_1->set_num_windings(kNumberOfWindings1);
  CurrentCarrier *current_carrier_1 = coil_1->add_current_carriers();
  PolygonFilament *circle_1 = current_carrier_1->mutable_polygon_filament();
  ASSERT_TRUE(PolygonCirclePopulate(*circle_1, kRadius1,
                                    kNumberOfPhiGridPoints1, kZOffset1)
                  .ok());

  MagneticConfiguration magnetic_configuration_2;
  SerialCircuit *serial_circuit_2 =
      magnetic_configuration_2.add_serial_circuits();
  serial_circuit_2->set_current(kCurrent);
  Coil *coil_2 = serial_circuit_2->add_coils();
  coil_2->set_num_windings(kNumberOfWindings2);
  CurrentCarrier *current_carrier_2 = coil_2->add_current_carriers();
  PolygonFilament *circle_2 = current_carrier_2->mutable_polygon_filament();
  ASSERT_TRUE(PolygonCirclePopulate(*circle_2, kRadius2,
                                    kNumberOfPhiGridPoints2, kZOffset2)
                  .ok());

  // evaluation locations [number_evaluation_locations][3: x, y, z]
  std::vector<std::vector<double> > evaluation_locations = {
      {0.0, 0.0, 0.0},
      {0.1, 0.2, 0.3},
      {1.0, 2.0, 3.0},
      {10.0, 20.0, 30.0},
  };

  // vector potential [number_evaluation_locations][3: x, y, z] from only
  // circle_1
  std::vector<std::vector<double> > vector_potential_1 = {
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  // vector potential [number_evaluation_locations][3: x, y, z] from only
  // circle_2
  std::vector<std::vector<double> > vector_potential_2 = {
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  // vector potential [number_evaluation_locations][3: x, y, z] from both
  // circle_1 and circle_2
  std::vector<std::vector<double> > vector_potential = {
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  // compute vector potentials individually
  absl::Status status_1 =
      VectorPotential(magnetic_configuration_1, evaluation_locations,
                      /*m_vector_potential=*/vector_potential_1);
  ASSERT_TRUE(status_1.ok());

  absl::Status status_2 =
      VectorPotential(magnetic_configuration_2, evaluation_locations,
                      /*m_vector_potential=*/vector_potential_2);
  ASSERT_TRUE(status_2.ok());

  // make a MagneticConfiguration which holds both circle_1 and circle_2
  MagneticConfiguration magnetic_configuration;
  SerialCircuit *serial_circuit = magnetic_configuration.add_serial_circuits();
  serial_circuit->set_current(kCurrent);

  Coil *coil_1_of_2 = serial_circuit->add_coils();
  coil_1_of_2->set_num_windings(kNumberOfWindings1);
  CurrentCarrier *current_carrier_1_of_2 = coil_1_of_2->add_current_carriers();
  PolygonFilament *circle_1_of_2 =
      current_carrier_1_of_2->mutable_polygon_filament();
  ASSERT_TRUE(PolygonCirclePopulate(*circle_1_of_2, kRadius1,
                                    kNumberOfPhiGridPoints1, kZOffset1)
                  .ok());

  Coil *coil_2_of_2 = serial_circuit->add_coils();
  coil_2_of_2->set_num_windings(kNumberOfWindings2);
  CurrentCarrier *current_carrier_2_of_2 = coil_2_of_2->add_current_carriers();
  PolygonFilament *circle_2_of_2 =
      current_carrier_2_of_2->mutable_polygon_filament();
  ASSERT_TRUE(PolygonCirclePopulate(*circle_2_of_2, kRadius2,
                                    kNumberOfPhiGridPoints2, kZOffset2)
                  .ok());

  // compute the vector potential from both circle_1 and circle_2 at the same
  // time
  absl::Status status =
      VectorPotential(magnetic_configuration, evaluation_locations,
                      /*m_vector_potential=*/vector_potential);
  ASSERT_TRUE(status.ok());

  // compare the sum of the inidvidually-computed vector potentials
  // with the vector potential where both contributions were computed together
  const int number_of_evaluation_locations =
      static_cast<int>(evaluation_locations.size());
  for (int i = 0; i < number_of_evaluation_locations; ++i) {
    const double vector_potential_x =
        vector_potential_1[i][0] + vector_potential_2[i][0];
    const double vector_potential_y =
        vector_potential_1[i][1] + vector_potential_2[i][1];
    const double vector_potential_z =
        vector_potential_1[i][2] + vector_potential_2[i][2];

    EXPECT_TRUE(
        IsCloseRelAbs(vector_potential[i][0], vector_potential_x, kTolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(vector_potential[i][1], vector_potential_y, kTolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(vector_potential[i][2], vector_potential_z, kTolerance));
  }
}  // VectorPotential: CheckMultipleCoilsWithWindings

}  // namespace magnetics
