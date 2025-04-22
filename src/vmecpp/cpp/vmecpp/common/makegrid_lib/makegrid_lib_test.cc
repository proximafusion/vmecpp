// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"

#include <netcdf.h>

#include <string>
#include <vector>

#include "absl/log/log.h"
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

    // connection vector from evaluation position to center of loop
    Vector3d delta_eval_origin =
        Subtract(circular_filament.center(), evaluation_location);

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
        (normalized_z < kZMax && kRhoZMin < normalized_rho &&
         normalized_rho < kRhoZMax)) {
      m_exclude_from_comparison[i] = true;
    }
  }  // number_of_evaluation_locations

  return absl::OkStatus();
}  // DetermineIfTooCloseToCurrentCarrierForComparison

// TODO(jons): write a test for
// DetermineIfTooCloseToCurrentCarrierForComparison(CircularFilament ...)

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

      // connection vector from evaluation position to start of segment
      Vector3d delta_eval_origin = Subtract(origin, evaluation_location);

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

// TODO(jons): write a test for
// DetermineIfTooCloseToCurrentCarrierForComparison(PolygonFilament ...)

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

TEST(TestMakegridLib, CheckComputeMagneticFieldResponseTable) {
  static constexpr double kTolerance = 1.0e-6;

  // NOTE: These parameters have to be consistent with the MGRID_NLI namelist
  // in the `coils.test_*` input files.
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

  // load the MagneticConfiguration from the MAKEGRID input file
  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromCoilsFile(
          "vmecpp/common/makegrid_lib/test_data/coils.test_symmetric_even");
  ASSERT_OK(magnetic_configuration);

  const int number_of_serial_circuits =
      magnetic_configuration->serial_circuits_size();

  absl::StatusOr<RowMatrix3Xd> cylindrical_grid_eigen =
      MakeCylindricalGrid(makegrid_parameters);
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
  ASSERT_EQ(
      nc_open(
          "vmecpp/common/makegrid_lib/test_data/mgrid_test_symmetric_even.nc",
          NC_NOWRITE, &ncid),
      NC_NOERR);

  const int nfp = NetcdfReadInt(ncid, "nfp");
  EXPECT_EQ(nfp, makegrid_parameters.number_of_field_periods);

  const int numR = NetcdfReadInt(ncid, "ir");
  EXPECT_EQ(numR, makegrid_parameters.number_of_r_grid_points);

  const double minR = NetcdfReadDouble(ncid, "rmin");
  EXPECT_EQ(minR, makegrid_parameters.r_grid_minimum);

  const double maxR = NetcdfReadDouble(ncid, "rmax");
  EXPECT_EQ(maxR, makegrid_parameters.r_grid_maximum);

  const int numZ = NetcdfReadInt(ncid, "jz");
  EXPECT_EQ(numZ, makegrid_parameters.number_of_z_grid_points);

  const double minZ = NetcdfReadDouble(ncid, "zmin");
  EXPECT_EQ(minZ, makegrid_parameters.z_grid_minimum);

  const double maxZ = NetcdfReadDouble(ncid, "zmax");
  EXPECT_EQ(maxZ, makegrid_parameters.z_grid_maximum);

  const int numPhi = NetcdfReadInt(ncid, "kp");
  EXPECT_EQ(numPhi, makegrid_parameters.number_of_phi_grid_points);

  const int nextcur = NetcdfReadInt(ncid, "nextcur");
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
    std::string br_variable = absl::StrFormat("br_%03d", circuit_index + 1);
    std::vector<std::vector<std::vector<double>>> b_r_contribution =
        NetcdfReadArray3D(ncid, br_variable);

    std::string bp_variable = absl::StrFormat("bp_%03d", circuit_index + 1);
    std::vector<std::vector<std::vector<double>>> b_p_contribution =
        NetcdfReadArray3D(ncid, bp_variable);

    std::string bz_variable = absl::StrFormat("bz_%03d", circuit_index + 1);
    std::vector<std::vector<std::vector<double>>> b_z_contribution =
        NetcdfReadArray3D(ncid, bz_variable);

    // perform comparison of points that are not explicitly excluded from the
    // comparison
    // FIXME(jons): allocate `exclude_from_comparison` on the whole field
    // period and actually check the whole field period.
    int number_of_tested_evaluation_locations = 0;
    int num_phi_effective = makegrid_parameters.number_of_phi_grid_points;
    if (makegrid_parameters.assume_stellarator_symmetry) {
      num_phi_effective = num_phi_effective / 2 + 1;
    }
    for (int index_phi = 0; index_phi < num_phi_effective; ++index_phi) {
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
}  // CheckComputeMagneticFieldResponseTable

TEST(TestMakegridLib, CheckComputeVectorPotentialCache) {
  static constexpr double kTolerance = 1.0e-6;

  // NOTE: These parameters have to be consistent with the MGRID_NLI namelist
  // in the `coils.test_*` input files.
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

  // load the MagneticConfiguration from the MAKEGRID input file
  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromCoilsFile(
          "vmecpp/common/makegrid_lib/test_data/coils.test_symmetric_even");
  ASSERT_OK(magnetic_configuration);

  const int number_of_serial_circuits =
      magnetic_configuration->serial_circuits_size();

  absl::StatusOr<RowMatrix3Xd> cylindrical_grid_eigen =
      MakeCylindricalGrid(makegrid_parameters);
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
  ASSERT_EQ(
      nc_open(
          "vmecpp/common/makegrid_lib/test_data/mgrid_test_symmetric_even.nc",
          NC_NOWRITE, &ncid),
      NC_NOERR);

  const int nfp = NetcdfReadInt(ncid, "nfp");
  EXPECT_EQ(nfp, makegrid_parameters.number_of_field_periods);

  const int numR = NetcdfReadInt(ncid, "ir");
  EXPECT_EQ(numR, makegrid_parameters.number_of_r_grid_points);

  const double minR = NetcdfReadDouble(ncid, "rmin");
  EXPECT_EQ(minR, makegrid_parameters.r_grid_minimum);

  const double maxR = NetcdfReadDouble(ncid, "rmax");
  EXPECT_EQ(maxR, makegrid_parameters.r_grid_maximum);

  const int numZ = NetcdfReadInt(ncid, "jz");
  EXPECT_EQ(numZ, makegrid_parameters.number_of_z_grid_points);

  const double minZ = NetcdfReadDouble(ncid, "zmin");
  EXPECT_EQ(minZ, makegrid_parameters.z_grid_minimum);

  const double maxZ = NetcdfReadDouble(ncid, "zmax");
  EXPECT_EQ(maxZ, makegrid_parameters.z_grid_maximum);

  const int numPhi = NetcdfReadInt(ncid, "kp");
  EXPECT_EQ(numPhi, makegrid_parameters.number_of_phi_grid_points);

  const int nextcur = NetcdfReadInt(ncid, "nextcur");
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
    std::string ar_variable = absl::StrFormat("ar_%03d", circuit_index + 1);
    std::vector<std::vector<std::vector<double>>> a_r_contribution =
        NetcdfReadArray3D(ncid, ar_variable);

    std::string ap_variable = absl::StrFormat("ap_%03d", circuit_index + 1);
    std::vector<std::vector<std::vector<double>>> a_p_contribution =
        NetcdfReadArray3D(ncid, ap_variable);

    std::string az_variable = absl::StrFormat("az_%03d", circuit_index + 1);
    std::vector<std::vector<std::vector<double>>> a_z_contribution =
        NetcdfReadArray3D(ncid, az_variable);

    // perform comparison of points that are not explicitly excluded from the
    // comparison
    // FIXME(jons): allocate `exclude_from_comparison` on the whole field
    // period and actually check the whole field period.
    int number_of_tested_evaluation_locations = 0;
    int num_phi_effective = makegrid_parameters.number_of_phi_grid_points;
    if (makegrid_parameters.assume_stellarator_symmetry) {
      num_phi_effective = num_phi_effective / 2 + 1;
    }
    for (int index_phi = 0; index_phi < num_phi_effective; ++index_phi) {
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
}  // CheckComputeVectorPotentialCache

// TODO(jons): implement test of non-stellarator-symmetric mgrid file
// TODO(jons): implement test of stellarator-symmetric mgrid file using
// symmetric C++ implementation

}  // namespace makegrid
