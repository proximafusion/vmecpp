// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/composed_types_definition/composed_types.h"

namespace magnetics {

using composed_types::FourierCoefficient1D;
using composed_types::Vector3d;

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Test;
using ::testing::TestWithParam;
using ::testing::Values;

using ::testing::ElementsAreArray;

using testing::IsCloseRelAbs;

TEST(TestMagneticConfigurationLib, SingleCircularFilament) {
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 0.0 2.0 3.0 1 circular_filament
end)";

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  EXPECT_FALSE(magnetic_configuration->name.has_value());
  ASSERT_TRUE(magnetic_configuration->num_field_periods_.has_value());
  EXPECT_EQ(magnetic_configuration->num_field_periods_.value(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.name.has_value());
  EXPECT_EQ(serial_circuit.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 1);

  const Coil &coil = serial_circuit.coils(0);
  EXPECT_FALSE(coil.name.has_value());
  ASSERT_TRUE(coil.num_windings_.has_value());
  EXPECT_EQ(coil.num_windings_.value(), 3.0);
  ASSERT_EQ(coil.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier = coil.current_carriers(0);
  ASSERT_TRUE(current_carrier.has_circular_filament());

  const CircularFilament &circular_filament =
      current_carrier.circular_filament();
  ASSERT_TRUE(circular_filament.name.has_value());
  EXPECT_EQ(circular_filament.name.value(), "circular_filament");
  EXPECT_EQ(circular_filament.radius, 1.0);

  const Vector3d &center = circular_filament.center;
  EXPECT_EQ(center.x(), 0.0);
  EXPECT_EQ(center.y(), 0.0);
  EXPECT_EQ(center.z(), 2.0);

  const Vector3d &normal = circular_filament.normal;
  EXPECT_EQ(normal.x(), 0.0);
  EXPECT_EQ(normal.y(), 0.0);
  EXPECT_EQ(normal.z(), 1.0);
}  // SingleCircularFilament

TEST(TestMagneticConfigurationLib, SingleCircularFilamentDifferentWhitespace) {
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
  1.0   0.0 	 2.0 3.0 1 circular_filament
end)";

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  EXPECT_FALSE(magnetic_configuration->name.has_value());
  ASSERT_TRUE(magnetic_configuration->num_field_periods_.has_value());
  EXPECT_EQ(magnetic_configuration->num_field_periods_.value(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.name.has_value());
  ASSERT_TRUE(serial_circuit.current_.has_value());
  EXPECT_EQ(serial_circuit.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 1);

  const Coil &coil = serial_circuit.coils(0);
  EXPECT_FALSE(coil.name.has_value());
  ASSERT_TRUE(coil.num_windings_.has_value());
  EXPECT_EQ(coil.num_windings_.value(), 3.0);
  ASSERT_EQ(coil.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier = coil.current_carriers(0);
  ASSERT_TRUE(current_carrier.has_circular_filament());

  const CircularFilament &circular_filament =
      current_carrier.circular_filament();
  ASSERT_TRUE(circular_filament.name.has_value());
  EXPECT_EQ(circular_filament.name.value(), "circular_filament");
  EXPECT_EQ(circular_filament.radius, 1.0);

  const Vector3d &center = circular_filament.center;
  EXPECT_EQ(center.x(), 0.0);
  EXPECT_EQ(center.y(), 0.0);
  EXPECT_EQ(center.z(), 2.0);

  const Vector3d &normal = circular_filament.normal;
  EXPECT_EQ(normal.x(), 0.0);
  EXPECT_EQ(normal.y(), 0.0);
  EXPECT_EQ(normal.z(), 1.0);
}  // SingleCircularFilamentDifferentWhitespace

TEST(TestMagneticConfigurationLib, TwoCircularFilamentsInSameCircuit) {
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 0.0 2.0 3.0 1 circular_filament_1a
4.0 0.0 5.0 6.0 1 circular_filament_1b
end)";

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  EXPECT_FALSE(magnetic_configuration->name.has_value());
  ASSERT_TRUE(magnetic_configuration->num_field_periods_.has_value());
  EXPECT_EQ(magnetic_configuration->num_field_periods_.value(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.name.has_value());
  ASSERT_TRUE(serial_circuit.current_.has_value());
  EXPECT_EQ(serial_circuit.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 2);

  // first circular filament

  const Coil &coil_1a = serial_circuit.coils(0);
  EXPECT_FALSE(coil_1a.name.has_value());
  ASSERT_TRUE(coil_1a.num_windings_.has_value());
  EXPECT_EQ(coil_1a.num_windings_.value(), 3.0);
  ASSERT_EQ(coil_1a.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1a = coil_1a.current_carriers(0);
  ASSERT_TRUE(current_carrier_1a.has_circular_filament());

  const CircularFilament &circular_filament_1a =
      current_carrier_1a.circular_filament();
  ASSERT_TRUE(circular_filament_1a.name.has_value());
  EXPECT_EQ(circular_filament_1a.name.value(), "circular_filament_1a");
  EXPECT_EQ(circular_filament_1a.radius, 1.0);

  const Vector3d &center_1a = circular_filament_1a.center;
  EXPECT_EQ(center_1a.x(), 0.0);
  EXPECT_EQ(center_1a.y(), 0.0);
  EXPECT_EQ(center_1a.z(), 2.0);

  const Vector3d &normal_1a = circular_filament_1a.normal;
  EXPECT_EQ(normal_1a.x(), 0.0);
  EXPECT_EQ(normal_1a.y(), 0.0);
  EXPECT_EQ(normal_1a.z(), 1.0);

  // second circular filament

  const Coil &coil_1b = serial_circuit.coils(1);
  EXPECT_FALSE(coil_1b.name.has_value());
  ASSERT_TRUE(coil_1b.num_windings_.has_value());
  EXPECT_EQ(coil_1b.num_windings_.value(), 6.0);
  EXPECT_EQ(coil_1b.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1b = coil_1b.current_carriers(0);
  EXPECT_TRUE(current_carrier_1b.has_circular_filament());

  const CircularFilament &circular_filament_1b =
      current_carrier_1b.circular_filament();
  EXPECT_TRUE(circular_filament_1b.name.has_value());
  EXPECT_EQ(circular_filament_1b.name.value(), "circular_filament_1b");
  EXPECT_EQ(circular_filament_1b.radius, 4.0);

  const Vector3d &center_1b = circular_filament_1b.center;
  EXPECT_EQ(center_1b.x(), 0.0);
  EXPECT_EQ(center_1b.y(), 0.0);
  EXPECT_EQ(center_1b.z(), 5.0);

  const Vector3d &normal_1b = circular_filament_1b.normal;
  EXPECT_EQ(normal_1b.x(), 0.0);
  EXPECT_EQ(normal_1b.y(), 0.0);
  EXPECT_EQ(normal_1b.z(), 1.0);
}  // TwoCircularFilamentsInSameCircuit

TEST(TestMagneticConfigurationLib, TwoCircularFilamentsInTwoCircuits) {
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 0.0 2.0 3.0 1 circular_filament_1
4.0 0.0 5.0 6.0 2 circular_filament_2
end)";

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  EXPECT_FALSE(magnetic_configuration->name.has_value());
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 2);

  // first circular filament

  SerialCircuit serial_circuit_1 = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit_1.name.has_value());
  ASSERT_TRUE(serial_circuit_1.current_.has_value());
  EXPECT_EQ(serial_circuit_1.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit_1.coils_size(), 1);

  const Coil &coil_1 = serial_circuit_1.coils(0);
  EXPECT_FALSE(coil_1.name.has_value());
  ASSERT_TRUE(coil_1.num_windings_.has_value());
  EXPECT_EQ(coil_1.num_windings_.value(), 3.0);
  ASSERT_EQ(coil_1.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1 = coil_1.current_carriers(0);
  ASSERT_TRUE(current_carrier_1.has_circular_filament());

  const CircularFilament &circular_filament_1 =
      current_carrier_1.circular_filament();
  ASSERT_TRUE(circular_filament_1.name.has_value());
  EXPECT_EQ(circular_filament_1.name.value(), "circular_filament_1");
  EXPECT_EQ(circular_filament_1.radius, 1.0);

  const Vector3d &center_1 = circular_filament_1.center;
  EXPECT_EQ(center_1.x(), 0.0);
  EXPECT_EQ(center_1.y(), 0.0);
  EXPECT_EQ(center_1.z(), 2.0);

  const Vector3d &normal_1 = circular_filament_1.normal;
  EXPECT_EQ(normal_1.x(), 0.0);
  EXPECT_EQ(normal_1.y(), 0.0);
  EXPECT_EQ(normal_1.z(), 1.0);

  // second circular filament

  SerialCircuit serial_circuit_2 = magnetic_configuration->serial_circuits(1);
  EXPECT_FALSE(serial_circuit_2.name.has_value());
  ASSERT_TRUE(serial_circuit_2.current_.has_value());
  EXPECT_EQ(serial_circuit_2.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit_2.coils_size(), 1);

  const Coil &coil_2 = serial_circuit_2.coils(0);
  EXPECT_FALSE(coil_2.name.has_value());
  ASSERT_TRUE(coil_2.num_windings_.has_value());
  EXPECT_EQ(coil_2.num_windings_.value(), 6.0);
  ASSERT_EQ(coil_2.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_2 = coil_2.current_carriers(0);
  ASSERT_TRUE(current_carrier_2.has_circular_filament());

  const CircularFilament &circular_filament_2 =
      current_carrier_2.circular_filament();
  ASSERT_TRUE(circular_filament_2.name.has_value());
  EXPECT_EQ(circular_filament_2.name.value(), "circular_filament_2");
  EXPECT_EQ(circular_filament_2.radius, 4.0);

  const Vector3d &center_2 = circular_filament_2.center;
  EXPECT_EQ(center_2.x(), 0.0);
  EXPECT_EQ(center_2.y(), 0.0);
  EXPECT_EQ(center_2.z(), 5.0);

  const Vector3d &normal_2 = circular_filament_2.normal;
  EXPECT_EQ(normal_2.x(), 0.0);
  EXPECT_EQ(normal_2.y(), 0.0);
  EXPECT_EQ(normal_2.z(), 1.0);
}  // TwoCircularFilamentsInTwoCircuits

TEST(TestMagneticConfigurationLib, SinglePolygonFilament) {
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 2.0 3.0 4.0
5.0 6.0 7.0 0.0 1 polygon_filament
end)";

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  EXPECT_FALSE(magnetic_configuration->name.has_value());
  ASSERT_TRUE(magnetic_configuration->num_field_periods_.has_value());
  EXPECT_EQ(magnetic_configuration->num_field_periods_.value(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.name.has_value());
  ASSERT_TRUE(serial_circuit.current_.has_value());
  EXPECT_EQ(serial_circuit.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 1);

  const Coil &coil = serial_circuit.coils(0);
  EXPECT_FALSE(coil.name.has_value());
  ASSERT_TRUE(coil.num_windings_.has_value());
  EXPECT_EQ(coil.num_windings_.value(), 4.0);
  ASSERT_EQ(coil.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier = coil.current_carriers(0);
  ASSERT_TRUE(current_carrier.has_polygon_filament());

  const PolygonFilament &polygon_filament = current_carrier.polygon_filament();
  ASSERT_TRUE(polygon_filament.name.has_value());
  EXPECT_EQ(polygon_filament.name.value(), "polygon_filament");
  ASSERT_EQ(polygon_filament.vertices.cols(), 2);

  EXPECT_EQ(polygon_filament.vertices(0, 0), 1.0);
  EXPECT_EQ(polygon_filament.vertices(1, 0), 2.0);
  EXPECT_EQ(polygon_filament.vertices(2, 0), 3.0);

  EXPECT_EQ(polygon_filament.vertices(0, 1), 5.0);
  EXPECT_EQ(polygon_filament.vertices(1, 1), 6.0);
  EXPECT_EQ(polygon_filament.vertices(2, 1), 7.0);
}  // SinglePolygonFilament

TEST(TestMagneticConfigurationLib, TwoPolygonFilamentsInSameCircuit) {
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 2.0 3.0 4.0
5.0 6.0 7.0 0.0 1 polygon_filament_1a
1.5 2.5 3.5 4.5
5.5 6.5 7.5 0.0 1 polygon_filament_1b
end)";

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  EXPECT_FALSE(magnetic_configuration->name.has_value());
  ASSERT_TRUE(magnetic_configuration->num_field_periods_.has_value());
  EXPECT_EQ(magnetic_configuration->num_field_periods_.value(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.name.has_value());
  ASSERT_TRUE(serial_circuit.current_.has_value());
  EXPECT_EQ(serial_circuit.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 2);

  // first polygon filament

  const Coil &coil_1a = serial_circuit.coils(0);
  EXPECT_FALSE(coil_1a.name.has_value());
  ASSERT_TRUE(coil_1a.num_windings_.has_value());
  EXPECT_EQ(coil_1a.num_windings_.value(), 4.0);
  ASSERT_EQ(coil_1a.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1a = coil_1a.current_carriers(0);
  ASSERT_TRUE(current_carrier_1a.has_polygon_filament());

  const PolygonFilament &polygon_filament_1a =
      current_carrier_1a.polygon_filament();
  ASSERT_TRUE(polygon_filament_1a.name.has_value());
  EXPECT_EQ(polygon_filament_1a.name.value(), "polygon_filament_1a");
  ASSERT_EQ(polygon_filament_1a.vertices.cols(), 2);

  EXPECT_EQ(polygon_filament_1a.vertices(0, 0), 1.0);
  EXPECT_EQ(polygon_filament_1a.vertices(1, 0), 2.0);
  EXPECT_EQ(polygon_filament_1a.vertices(2, 0), 3.0);

  EXPECT_EQ(polygon_filament_1a.vertices(0, 1), 5.0);
  EXPECT_EQ(polygon_filament_1a.vertices(1, 1), 6.0);
  EXPECT_EQ(polygon_filament_1a.vertices(2, 1), 7.0);

  // second polygon filament

  const Coil &coil_1b = serial_circuit.coils(1);
  EXPECT_FALSE(coil_1b.name.has_value());
  ASSERT_TRUE(coil_1b.num_windings_.has_value());
  EXPECT_EQ(coil_1b.num_windings_.value(), 4.5);
  ASSERT_EQ(coil_1b.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1b = coil_1b.current_carriers(0);
  ASSERT_TRUE(current_carrier_1b.has_polygon_filament());

  const PolygonFilament &polygon_filament_1b =
      current_carrier_1b.polygon_filament();
  ASSERT_TRUE(polygon_filament_1b.name.has_value());
  EXPECT_EQ(polygon_filament_1b.name.value(), "polygon_filament_1b");
  ASSERT_EQ(polygon_filament_1b.vertices.cols(), 2);

  EXPECT_EQ(polygon_filament_1b.vertices(0, 0), 1.5);
  EXPECT_EQ(polygon_filament_1b.vertices(1, 0), 2.5);
  EXPECT_EQ(polygon_filament_1b.vertices(2, 0), 3.5);

  EXPECT_EQ(polygon_filament_1b.vertices(0, 1), 5.5);
  EXPECT_EQ(polygon_filament_1b.vertices(1, 1), 6.5);
  EXPECT_EQ(polygon_filament_1b.vertices(2, 1), 7.5);
}  // TwoPolygonFilamentsInSameCircuit

TEST(TestMagneticConfigurationLib, TwoPolygonFilamentsInTwoCircuits) {
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 2.0 3.0 4.0
5.0 6.0 7.0 0.0 1 polygon_filament_1
1.5 2.5 3.5 4.5
5.5 6.5 7.5 0.0 2 polygon_filament_2
end)";

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  EXPECT_FALSE(magnetic_configuration->name.has_value());
  ASSERT_TRUE(magnetic_configuration->num_field_periods_.has_value());
  EXPECT_EQ(magnetic_configuration->num_field_periods_.value(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 2);

  // first polygon filament

  SerialCircuit serial_circuit_0 = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit_0.name.has_value());
  ASSERT_TRUE(serial_circuit_0.current_.has_value());
  EXPECT_EQ(serial_circuit_0.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit_0.coils_size(), 1);

  const Coil &coil_0 = serial_circuit_0.coils(0);
  EXPECT_FALSE(coil_0.name.has_value());
  ASSERT_TRUE(coil_0.num_windings_.has_value());
  EXPECT_EQ(coil_0.num_windings_.value(), 4.0);
  ASSERT_EQ(coil_0.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_0 = coil_0.current_carriers(0);
  ASSERT_TRUE(current_carrier_0.has_polygon_filament());

  const PolygonFilament &polygon_filament_1 =
      current_carrier_0.polygon_filament();
  ASSERT_TRUE(polygon_filament_1.name.has_value());
  EXPECT_EQ(polygon_filament_1.name.value(), "polygon_filament_1");
  ASSERT_EQ(polygon_filament_1.vertices.cols(), 2);

  EXPECT_EQ(polygon_filament_1.vertices(0, 0), 1.0);
  EXPECT_EQ(polygon_filament_1.vertices(1, 0), 2.0);
  EXPECT_EQ(polygon_filament_1.vertices(2, 0), 3.0);

  EXPECT_EQ(polygon_filament_1.vertices(0, 1), 5.0);
  EXPECT_EQ(polygon_filament_1.vertices(1, 1), 6.0);
  EXPECT_EQ(polygon_filament_1.vertices(2, 1), 7.0);

  // second polygon filament

  SerialCircuit serial_circuit_1 = magnetic_configuration->serial_circuits(1);
  EXPECT_FALSE(serial_circuit_1.name.has_value());
  ASSERT_TRUE(serial_circuit_1.current_.has_value());
  EXPECT_EQ(serial_circuit_1.current_.value(), 1.0);
  ASSERT_EQ(serial_circuit_1.coils_size(), 1);

  const Coil &coil_1 = serial_circuit_1.coils(0);
  EXPECT_FALSE(coil_1.name.has_value());
  ASSERT_TRUE(coil_1.num_windings_.has_value());
  EXPECT_EQ(coil_1.num_windings_.value(), 4.5);
  ASSERT_EQ(coil_1.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1 = coil_1.current_carriers(0);
  ASSERT_TRUE(current_carrier_1.has_polygon_filament());

  const PolygonFilament &polygon_filament_2 =
      current_carrier_1.polygon_filament();
  ASSERT_TRUE(polygon_filament_2.name.has_value());
  EXPECT_EQ(polygon_filament_2.name.value(), "polygon_filament_2");
  ASSERT_EQ(polygon_filament_2.vertices.cols(), 2);

  EXPECT_EQ(polygon_filament_2.vertices(0, 0), 1.5);
  EXPECT_EQ(polygon_filament_2.vertices(1, 0), 2.5);
  EXPECT_EQ(polygon_filament_2.vertices(2, 0), 3.5);

  EXPECT_EQ(polygon_filament_2.vertices(0, 1), 5.5);
  EXPECT_EQ(polygon_filament_2.vertices(1, 1), 6.5);
  EXPECT_EQ(polygon_filament_2.vertices(2, 1), 7.5);
}  // TwoPolygonFilamentsInTwoCircuits

TEST(TestMagneticConfigurationLib, CheckGetCircuitCurrents) {
  // two PolygonFilaments in two different circuits
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 2.0 3.0 4.0
5.0 6.0 7.0 0.0 1 polygon_filament_1
1.5 2.5 3.5 4.5
5.5 6.5 7.5 0.0 2 polygon_filament_2
end)";

  // The numbers in above makegrid_coils are parsed into the number of windings.
  // The currents are originally initialized to 1.0 for each SerialCircuit in
  // ImportMagneticConfigurationFromMakegrid.
  Eigen::VectorXd expected_currents(2);
  expected_currents << 1.0, 1.0;

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  absl::StatusOr<Eigen::VectorXd> circuit_currents =
      GetCircuitCurrents(*magnetic_configuration);
  ASSERT_TRUE(circuit_currents.ok());

  EXPECT_THAT(*circuit_currents, ElementsAreArray(expected_currents));
}  // CheckGetCircuitCurrents

TEST(TestMagneticConfigurationLib, CheckSetCircuitCurrents) {
  // two PolygonFilaments in two different circuits
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 2.0 3.0 4.0
5.0 6.0 7.0 0.0 1 polygon_filament_1
1.5 2.5 3.5 4.5
5.5 6.5 7.5 0.0 2 polygon_filament_2
end)";

  // The numbers in above makegrid_coils are parsed into the number of windings.
  // The currents are originally initialized to 1.0 for each SerialCircuit in
  // ImportMagneticConfigurationFromMakegrid.
  Eigen::VectorXd original_currents(2);
  original_currents << 1.0, 1.0;

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  // specifying only a single current should be rejected, since two circuits are
  // in the MagneticConfiguration
  Eigen::VectorXd one_current(1);
  one_current << 2.0;
  absl::Status status_one_current = SetCircuitCurrents(
      one_current, /*m_magnetic_configuration=*/*magnetic_configuration);
  EXPECT_FALSE(status_one_current.ok());

  // check that no change was made to the currents (assume that no other part in
  // the MagneticConfiguration was touched)
  absl::StatusOr<Eigen::VectorXd> currents_after_first_attempt =
      GetCircuitCurrents(*magnetic_configuration);
  ASSERT_TRUE(currents_after_first_attempt.ok());
  EXPECT_THAT(*currents_after_first_attempt,
              ElementsAreArray(original_currents));

  // specifying two currents should be accepted, since two circuits are in the
  // MagneticConfiguration
  Eigen::VectorXd two_currents(2);
  two_currents << 2.0, 3.0;
  absl::Status status_two_current = SetCircuitCurrents(
      two_currents, /*m_magnetic_configuration=*/*magnetic_configuration);
  EXPECT_TRUE(status_two_current.ok());

  // now check that the currents actually appeared in the MagneticConfiguration
  absl::StatusOr<Eigen::VectorXd> currents_after_second_attempt =
      GetCircuitCurrents(*magnetic_configuration);
  ASSERT_TRUE(currents_after_second_attempt.ok());
  EXPECT_THAT(*currents_after_second_attempt, ElementsAreArray(two_currents));
}  // CheckSetCircuitCurrents

TEST(TestMagneticConfigurationLib, CheckNumWindingsToCircuitCurrents) {
  // three PolygonFilaments in two different circuits
  std::string makegrid_coils = R"(periods 1
mirror NIL
begin filament
1.0 2.0 3.0 4.0
5.0 6.0 7.0 0.0 1 polygon_filament_1
1.5 2.5 3.5 4.5
5.5 6.5 7.5 0.0 2 polygon_filament_2a
7.5 6.5 5.5 4.5
3.5 2.5 1.5 0.0 2 polygon_filament_2b
end)";
  absl::StatusOr<MagneticConfiguration> maybe_magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(maybe_magnetic_configuration.ok());
  const MagneticConfiguration &magnetic_configuration =
      *maybe_magnetic_configuration;

  // by default, the 4th column is parsed into num_windings
  // and the currents are set to 1.0 -> check this first
  ASSERT_EQ(magnetic_configuration.serial_circuits_size(), 2);

  EXPECT_EQ(magnetic_configuration.serial_circuits(0).current_.value(), 1.0);
  ASSERT_EQ(magnetic_configuration.serial_circuits(0).coils_size(), 1);
  EXPECT_EQ(
      magnetic_configuration.serial_circuits(0).coils(0).num_windings_.value(),
      4.0);

  EXPECT_EQ(magnetic_configuration.serial_circuits(1).current_.value(), 1.0);
  ASSERT_EQ(magnetic_configuration.serial_circuits(1).coils_size(), 2);
  EXPECT_EQ(
      magnetic_configuration.serial_circuits(1).coils(0).num_windings_.value(),
      4.5);
  EXPECT_EQ(
      magnetic_configuration.serial_circuits(1).coils(1).num_windings_.value(),
      4.5);

  // now make a mutable copy of the MagneticConfiguration
  // and migrate `num_windings` into the circuit currents
  MagneticConfiguration m_magnetic_configuration = magnetic_configuration;

  // call under test
  absl::Status status = NumWindingsToCircuitCurrents(m_magnetic_configuration);
  ASSERT_TRUE(status.ok()) << status.message();

  // now check that currents actually have been migrated into circuit currents
  EXPECT_EQ(m_magnetic_configuration.serial_circuits(0).current_.value(), 4.0);
  ASSERT_EQ(m_magnetic_configuration.serial_circuits(0).coils_size(), 1);
  EXPECT_EQ(m_magnetic_configuration.serial_circuits(0)
                .coils(0)
                .num_windings_.value(),
            1.0);

  EXPECT_EQ(m_magnetic_configuration.serial_circuits(1).current_.value(), 4.5);
  ASSERT_EQ(m_magnetic_configuration.serial_circuits(1).coils_size(), 2);
  EXPECT_EQ(m_magnetic_configuration.serial_circuits(1)
                .coils(0)
                .num_windings_.value(),
            1.0);
  EXPECT_EQ(m_magnetic_configuration.serial_circuits(1)
                .coils(1)
                .num_windings_.value(),
            1.0);

  // now check also a case that should not work:
  // two filaments in the same circuit, but with different number of windings
  std::string makegrid_coils_2 = R"(periods 1
mirror NIL
begin filament
1.0 2.0 3.0 4.0
5.0 6.0 7.0 0.0 1 polygon_filament_1a
1.5 2.5 3.5 4.5
5.5 6.5 7.5 0.0 1 polygon_filament_1b
end)";
  absl::StatusOr<MagneticConfiguration> maybe_magnetic_configuration_2 =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils_2);
  ASSERT_TRUE(maybe_magnetic_configuration_2.ok());
  const MagneticConfiguration &magnetic_configuration_2 =
      *maybe_magnetic_configuration_2;

  // by default, the 4th column is parsed into num_windings
  // and the currents are set to 1.0 -> check this first
  ASSERT_EQ(magnetic_configuration_2.serial_circuits_size(), 1);

  EXPECT_EQ(magnetic_configuration_2.serial_circuits(0).current_.value(), 1.0);
  ASSERT_EQ(magnetic_configuration_2.serial_circuits(0).coils_size(), 2);
  EXPECT_EQ(magnetic_configuration_2.serial_circuits(0)
                .coils(0)
                .num_windings_.value(),
            4.0);
  EXPECT_EQ(magnetic_configuration_2.serial_circuits(0)
                .coils(1)
                .num_windings_.value(),
            4.5);

  // now make a mutable copy of the MagneticConfiguration
  // and migrate `num_windings` into the circuit currents
  MagneticConfiguration m_magnetic_configuration_2 = magnetic_configuration_2;

  // call under test
  absl::Status status_2 =
      NumWindingsToCircuitCurrents(m_magnetic_configuration_2);
  EXPECT_FALSE(status_2.ok());
}  // CheckNumWindingsToCircuitCurrents

// -------------------

// The two integer parameters are interpreted as bitfields that control
// which Cartesian components of the origin and direction vectors are populated.
// Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
class IsInfiniteStraightFilamentFullyPopulatedTest
    : public TestWithParam< ::std::tuple<bool, bool, int, bool, int> > {
 protected:
  void SetUp() override {
    std::tie(specify_name_, specify_origin_, origin_components_,
             specify_direction_, direction_components_) = GetParam();
  }
  bool specify_name_;
  bool specify_origin_;
  int origin_components_;
  bool specify_direction_;
  int direction_components_;
};

TEST_P(IsInfiniteStraightFilamentFullyPopulatedTest,
       CheckIsInfiniteStraightFilamentFullyPopulated) {
  InfiniteStraightFilament infinite_straight_filament;
  if (specify_name_) {
    infinite_straight_filament.name = "filament_1";
  }
  if (specify_origin_) {
    infinite_straight_filament.origin_.emplace();
    Vector3d &origin = *infinite_straight_filament.origin_;
    if (origin_components_ & (1 << 0)) {
      origin.set_x(1.23);
    }
    if (origin_components_ & (1 << 1)) {
      origin.set_y(4.56);
    }
    if (origin_components_ & (1 << 2)) {
      origin.set_z(7.89);
    }
  }
  if (specify_direction_) {
    infinite_straight_filament.direction_.emplace();
    Vector3d &direction = *infinite_straight_filament.direction_;
    if (direction_components_ & (1 << 0)) {
      direction.set_x(9.87);
    }
    if (direction_components_ & (1 << 1)) {
      direction.set_y(6.54);
    }
    if (direction_components_ & (1 << 2)) {
      direction.set_z(3.21);
    }
  }

  absl::Status status =
      IsInfiniteStraightFilamentFullyPopulated(infinite_straight_filament);
  if (specify_origin_ && origin_components_ == 7 && specify_direction_ &&
      direction_components_ == 7) {
    EXPECT_TRUE(status.ok());
  } else {
    EXPECT_FALSE(status.ok());
  }
}  // CheckIsInfiniteStraightFilamentFullyPopulated

INSTANTIATE_TEST_SUITE_P(TestMagneticConfigurationLib,
                         IsInfiniteStraightFilamentFullyPopulatedTest,
                         Combine(Bool(), Bool(), Values(0, 1, 2, 3, 4, 5, 6, 7),
                                 Bool(), Values(0, 1, 2, 3, 4, 5, 6, 7)));

// -------------------

// Since CircularFilament fields (center, normal, radius) are now non-optional,
// IsCircularFilamentFullyPopulated always succeeds.
TEST(TestMagneticConfigurationLib, IsCircularFilamentFullyPopulated) {
  CircularFilament circular_filament;
  // Even with default-constructed values, the check should pass
  absl::Status status = IsCircularFilamentFullyPopulated(circular_filament);
  EXPECT_TRUE(status.ok());

  // Also with explicit values
  circular_filament.name = "test_filament";
  circular_filament.center.set_x(1.23);
  circular_filament.center.set_y(4.56);
  circular_filament.center.set_z(7.89);
  circular_filament.normal.set_x(0.0);
  circular_filament.normal.set_y(0.0);
  circular_filament.normal.set_z(1.0);
  circular_filament.radius = 3.14;
  status = IsCircularFilamentFullyPopulated(circular_filament);
  EXPECT_TRUE(status.ok());
}

// -------------------

class IsPolygonFilamentFullyPopulatedTest
    : public TestWithParam< ::std::tuple<bool, int> > {
 protected:
  void SetUp() override {
    std::tie(specify_name_, number_of_vertices_) = GetParam();
  }
  bool specify_name_;
  int number_of_vertices_;
};

TEST_P(IsPolygonFilamentFullyPopulatedTest,
       CheckIsPolygonFilamentFullyPopulated) {
  PolygonFilament polygon_filament;
  if (specify_name_) {
    polygon_filament.name = "filament_3";
  }
  if (number_of_vertices_ > 0) {
    polygon_filament.vertices.resize(3, number_of_vertices_);
    for (int i = 0; i < number_of_vertices_; ++i) {
      polygon_filament.vertices(0, i) = 3.14;
      polygon_filament.vertices(1, i) = 2.71;
      polygon_filament.vertices(2, i) = 1.41;
    }
  }

  absl::Status status = IsPolygonFilamentFullyPopulated(polygon_filament);
  if (number_of_vertices_ > 1) {
    EXPECT_TRUE(status.ok());
  } else {
    EXPECT_FALSE(status.ok());
  }
}  // CheckIsPolygonFilamentFullyPopulated

INSTANTIATE_TEST_SUITE_P(TestMagneticConfigurationLib,
                         IsPolygonFilamentFullyPopulatedTest,
                         Combine(Bool(), Values(0, 1, 2, 3)));

// -------------------

class IsMagneticConfigurationFullyPopulatedTest : public Test {
 protected:
  void SetUp() override {
    magnetic_configuration_.num_field_periods_ = 1;
    SerialCircuit *serial_circuit =
        magnetic_configuration_.add_serial_circuits();
    serial_circuit->current_ = 1.0;
    Coil *coil = serial_circuit->add_coils();
    coil->num_windings_ = 1.0;
    current_carrier_ = coil->add_current_carriers();
  }
  MagneticConfiguration magnetic_configuration_;
  CurrentCarrier *current_carrier_;
};

TEST_F(IsMagneticConfigurationFullyPopulatedTest,
       CheckIsMagneticConfigurationFullyPopulatedWithNoCurrentCarrier) {
  // The current carrier has kTypeNotSet, which is now an error
  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_);

  EXPECT_FALSE(status.ok());
}  // CheckIsMagneticConfigurationFullyPopulatedWithNoCurrentCarrier

TEST_F(IsMagneticConfigurationFullyPopulatedTest,
       CheckIsMagneticConfigurationFullyPopulatedWithInfiniteStraightFilament) {
  InfiniteStraightFilament *infinite_straight_filament =
      current_carrier_->mutable_infinite_straight_filament();
  infinite_straight_filament->origin_.emplace();
  Vector3d &origin = *infinite_straight_filament->origin_;
  origin.set_x(1.23);
  origin.set_y(4.56);
  origin.set_z(7.89);
  infinite_straight_filament->direction_.emplace();
  Vector3d &direction = *infinite_straight_filament->direction_;
  direction.set_x(9.87);
  direction.set_y(6.54);
  direction.set_z(3.21);

  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_);

  EXPECT_TRUE(status.ok());
}  // CheckIsMagneticConfigurationFullyPopulatedWithInfiniteStraightFilament

TEST_F(IsMagneticConfigurationFullyPopulatedTest,
       CheckIsMagneticConfigurationFullyPopulatedWithCircularFilament) {
  CircularFilament *circular_filament =
      current_carrier_->mutable_circular_filament();
  circular_filament->center.set_x(1.23);
  circular_filament->center.set_y(4.56);
  circular_filament->center.set_z(7.89);
  circular_filament->normal.set_x(9.87);
  circular_filament->normal.set_y(6.54);
  circular_filament->normal.set_z(3.21);
  circular_filament->radius = 3.14;

  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_);

  EXPECT_TRUE(status.ok());
}  // CheckIsMagneticConfigurationFullyPopulatedWithCircularFilament

TEST_F(IsMagneticConfigurationFullyPopulatedTest,
       CheckIsMagneticConfigurationFullyPopulatedWithPolygonFilament) {
  PolygonFilament *polygon_filament =
      current_carrier_->mutable_polygon_filament();
  polygon_filament->vertices.resize(3, 2);
  polygon_filament->vertices.col(0) = Eigen::Vector3d{1.23, 4.56, 7.89};
  polygon_filament->vertices.col(1) = Eigen::Vector3d{9.87, 6.54, 3.21};

  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_);

  EXPECT_TRUE(status.ok());
}  // CheckIsMagneticConfigurationFullyPopulatedWithPolygonFilament

// -------------------

class PrintInfiniteStraightFilamentTest
    : public TestWithParam< ::std::tuple<bool, bool, bool> > {
 protected:
  void SetUp() override {
    std::tie(specify_name_, specify_origin_, specify_direction_) = GetParam();
  }
  bool specify_name_;
  bool specify_origin_;
  bool specify_direction_;
};

TEST_P(PrintInfiniteStraightFilamentTest, CheckPrintInfiniteStraightFilament) {
  InfiniteStraightFilament infinite_straight_filament;
  if (specify_name_) {
    infinite_straight_filament.name = "filament_1";
  }
  if (specify_origin_) {
    infinite_straight_filament.origin_.emplace();
    infinite_straight_filament.origin_->set_x(1.23);
    infinite_straight_filament.origin_->set_y(4.56);
    infinite_straight_filament.origin_->set_z(7.89);
  }
  if (specify_direction_) {
    infinite_straight_filament.direction_.emplace();
    infinite_straight_filament.direction_->set_x(9.87);
    infinite_straight_filament.direction_->set_y(6.54);
    infinite_straight_filament.direction_->set_z(3.21);
  }

  testing::internal::CaptureStdout();
  PrintInfiniteStraightFilament(infinite_straight_filament);
  std::string output = testing::internal::GetCapturedStdout();

  std::string expected_output = "InfiniteStraightFilament {\n";
  if (specify_name_) {
    expected_output += "  name: 'filament_1'\n";
  } else {
    expected_output += "  name: [not set]\n";
  }
  if (specify_origin_) {
    expected_output += "  origin: [1.23, 4.56, 7.89]\n";
  } else {
    expected_output += "  origin: [not set]\n";
  }
  if (specify_direction_) {
    expected_output += "  direction: [9.87, 6.54, 3.21]\n";
  } else {
    expected_output += "  direction: [not set]\n";
  }
  expected_output += "}\n";

  EXPECT_TRUE(output == expected_output);
}  // CheckPrintInfiniteStraightFilament

INSTANTIATE_TEST_SUITE_P(TestMagneticConfigurationLib,
                         PrintInfiniteStraightFilamentTest,
                         Combine(Bool(), Bool(), Bool()));

// -------------------

// Since center, normal, and radius are now non-optional, Print always outputs
// them. Only name is optional.
TEST(TestMagneticConfigurationLib, CheckPrintCircularFilament) {
  CircularFilament circular_filament;
  circular_filament.center.set_x(1.23);
  circular_filament.center.set_y(4.56);
  circular_filament.center.set_z(7.89);
  circular_filament.normal.set_x(9.87);
  circular_filament.normal.set_y(6.54);
  circular_filament.normal.set_z(3.21);
  circular_filament.radius = 3.14;

  // With name
  circular_filament.name = "filament_2";
  testing::internal::CaptureStdout();
  PrintCircularFilament(circular_filament);
  std::string output = testing::internal::GetCapturedStdout();

  std::string expected_output = "CircularFilament {\n";
  expected_output += "  name: 'filament_2'\n";
  expected_output += "  center: [1.23, 4.56, 7.89]\n";
  expected_output += "  normal: [9.87, 6.54, 3.21]\n";
  expected_output += "  radius: 3.14\n";
  expected_output += "}\n";
  EXPECT_EQ(output, expected_output);

  // Without name
  circular_filament.name.reset();
  testing::internal::CaptureStdout();
  PrintCircularFilament(circular_filament);
  output = testing::internal::GetCapturedStdout();

  expected_output = "CircularFilament {\n";
  expected_output += "  name: [not set]\n";
  expected_output += "  center: [1.23, 4.56, 7.89]\n";
  expected_output += "  normal: [9.87, 6.54, 3.21]\n";
  expected_output += "  radius: 3.14\n";
  expected_output += "}\n";
  EXPECT_EQ(output, expected_output);
}

// -------------------

class PrintPolygonFilamentTest
    : public TestWithParam< ::std::tuple<bool, bool> > {
 protected:
  void SetUp() override {
    std::tie(specify_name_, specify_vertices_) = GetParam();
  }
  bool specify_name_;
  bool specify_vertices_;
};

TEST_P(PrintPolygonFilamentTest, CheckPrintPolygonFilament) {
  PolygonFilament polygon_filament;
  if (specify_name_) {
    polygon_filament.name = "filament_3";
  }
  if (specify_vertices_) {
    polygon_filament.vertices.resize(3, 3);
    polygon_filament.vertices.setZero();
  }

  // https://stackoverflow.com/a/33186201
  testing::internal::CaptureStdout();
  PrintPolygonFilament(polygon_filament);
  std::string output = testing::internal::GetCapturedStdout();

  std::string expected_output = "PolygonFilament {\n";
  // PrintPolygonFilament now prints optional name directly via operator<<
  expected_output += "  name: '";
  if (specify_name_) {
    expected_output += "filament_3";
  }
  expected_output += "'\n";
  if (specify_vertices_) {
    expected_output += "  vertices: [3]\n";
  } else {
    expected_output += "  vertices: none\n";
  }
  expected_output += "}\n";

  EXPECT_TRUE(output == expected_output);
}  // CheckPrintPolygonFilament

INSTANTIATE_TEST_SUITE_P(TestMagneticConfigurationLib, PrintPolygonFilamentTest,
                         Combine(Bool(), Bool()));

// -------------------

TEST(TestMagneticConfigurationLib, CheckMoveRadiallyOutwardCircularFilament) {
  const double initial_radius = 3.14;
  const double radial_step = 0.42;

  CircularFilament circular_filament;
  circular_filament.center.set_x(1.23);
  circular_filament.center.set_y(4.56);
  circular_filament.center.set_z(7.89);
  circular_filament.normal.set_x(9.87);
  circular_filament.normal.set_y(6.54);
  circular_filament.normal.set_z(3.21);
  circular_filament.radius = initial_radius;

  absl::Status status = IsCircularFilamentFullyPopulated(circular_filament);
  ASSERT_TRUE(status.ok()) << status.message();

  // check that movement attempt fails because center is not on origin in x and
  // y
  status = MoveRadially(radial_step, /*m_circular_filament=*/circular_filament);
  ASSERT_FALSE(status.ok());

  // fix center to be on origin in x and y
  circular_filament.center.set_x(0.0);
  circular_filament.center.set_y(0.0);

  // check that movement fails because normal is not along z axis
  status = MoveRadially(radial_step, /*m_circular_filament=*/circular_filament);
  ASSERT_FALSE(status.ok());

  // fix normal to be along z axis
  circular_filament.normal.set_x(0.0);
  circular_filament.normal.set_y(0.0);
  circular_filament.normal.set_z(1.0);

  // attempt movement and check that it was successful
  status = MoveRadially(radial_step, /*m_circular_filament=*/circular_filament);
  ASSERT_TRUE(status.ok()) << status.message();

  // check that radius has the expected value
  EXPECT_EQ(circular_filament.radius, initial_radius + radial_step);

  // check that no other members have been changed by successful call to
  // MoveRadially
  EXPECT_EQ(circular_filament.center.x(), 0.0);
  EXPECT_EQ(circular_filament.center.y(), 0.0);
  EXPECT_EQ(circular_filament.center.z(), 7.89);

  EXPECT_EQ(circular_filament.normal.x(), 0.0);
  EXPECT_EQ(circular_filament.normal.y(), 0.0);
  EXPECT_EQ(circular_filament.normal.z(), 1.0);
}  // CheckMoveRadiallyOutwardCircularFilament

TEST(TestMagneticConfigurationLib, CheckMoveRadiallyOutwardPolygonFilament) {
  static constexpr double kTolerance = 1.0e-15;

  const double radial_step = 0.42;

  PolygonFilament polygon_filament;
  polygon_filament.vertices.resize(3, 3);
  polygon_filament.vertices.col(0) = Eigen::Vector3d{1.0, 0.0, 1.3};
  polygon_filament.vertices.col(1) = Eigen::Vector3d{0.0, 1.0, 2.3};
  polygon_filament.vertices.col(2) = Eigen::Vector3d{1.0, 1.0, 3.3};

  absl::Status status = IsPolygonFilamentFullyPopulated(polygon_filament);
  ASSERT_TRUE(status.ok()) << status.message();

  // attempt movement and check that it was successful
  status = MoveRadially(radial_step, /*m_polygon_filament=*/polygon_filament);
  ASSERT_TRUE(status.ok()) << status.message();

  // check that vertices have moved as expected

  // vertex_0 only has x component in x-y plane
  // -> expected to move only along x
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step, polygon_filament.vertices(0, 0),
                            kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, polygon_filament.vertices(1, 0), kTolerance));
  EXPECT_EQ(polygon_filament.vertices(2, 0), 1.3);

  // vertex_1 only has y component in x-y plane
  // -> expected to move only along y
  EXPECT_TRUE(IsCloseRelAbs(0.0, polygon_filament.vertices(0, 1), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step, polygon_filament.vertices(1, 1),
                            kTolerance));
  EXPECT_EQ(polygon_filament.vertices(2, 1), 2.3);

  // vertex_2 has equal components in x and y in x-y plane
  // -> expected to move in equal amounts along both directions
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step / std::sqrt(2),
                            polygon_filament.vertices(0, 2), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step / std::sqrt(2),
                            polygon_filament.vertices(1, 2), kTolerance));
  EXPECT_EQ(polygon_filament.vertices(2, 2), 3.3);
}  // CheckMoveRadiallyOutwardPolygonFilament

TEST(TestMagneticConfigurationLib,
     CheckMoveRadiallyOutwardMagneticConfiguration) {
  MagneticConfiguration magnetic_configuration;
  magnetic_configuration.num_field_periods_ = 1;
  SerialCircuit *serial_circuit = magnetic_configuration.add_serial_circuits();
  serial_circuit->current_ = 1.0;
  Coil *coil = serial_circuit->add_coils();
  coil->num_windings_ = 1.0;

  static constexpr double kTolerance = 1.0e-15;

  const double initial_radius = 3.14;
  const double radial_step = 0.42;

  // Add both a CircularFilament and a PolygonFilament.
  CurrentCarrier *current_carrier_0 = coil->add_current_carriers();
  CircularFilament *circular_filament =
      current_carrier_0->mutable_circular_filament();
  circular_filament->center.set_x(0.0);
  circular_filament->center.set_y(0.0);
  circular_filament->center.set_z(7.89);
  circular_filament->normal.set_x(0.0);
  circular_filament->normal.set_y(0.0);
  circular_filament->normal.set_z(1.0);
  circular_filament->radius = initial_radius;

  CurrentCarrier *current_carrier_1 = coil->add_current_carriers();
  PolygonFilament *polygon_filament =
      current_carrier_1->mutable_polygon_filament();
  polygon_filament->vertices.resize(3, 3);
  polygon_filament->vertices.col(0) = Eigen::Vector3d{1.0, 0.0, 1.3};
  polygon_filament->vertices.col(1) = Eigen::Vector3d{0.0, 1.0, 2.3};
  polygon_filament->vertices.col(2) = Eigen::Vector3d{1.0, 1.0, 3.3};

  // Check that the MagneticConfiguration is fully populated.
  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration);
  ASSERT_TRUE(status.ok()) << status.message();

  // Attempt to radially move the MagneticConfiguration and check that both are
  // moved.
  status = MoveRadially(radial_step,
                        /*m_magnetic_configuration=*/magnetic_configuration);

  // check that radius has the expected value
  EXPECT_EQ(circular_filament->radius, initial_radius + radial_step);

  // check that no other members have been changed by successful call to
  // MoveRadially
  EXPECT_EQ(circular_filament->center.x(), 0.0);
  EXPECT_EQ(circular_filament->center.y(), 0.0);
  EXPECT_EQ(circular_filament->center.z(), 7.89);

  EXPECT_EQ(circular_filament->normal.x(), 0.0);
  EXPECT_EQ(circular_filament->normal.y(), 0.0);
  EXPECT_EQ(circular_filament->normal.z(), 1.0);

  // vertex_0 only has x component in x-y plane
  // -> expected to move only along x
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step, polygon_filament->vertices(0, 0),
                            kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, polygon_filament->vertices(1, 0), kTolerance));
  EXPECT_EQ(polygon_filament->vertices(2, 0), 1.3);

  // vertex_1 only has y component in x-y plane
  // -> expected to move only along y
  EXPECT_TRUE(IsCloseRelAbs(0.0, polygon_filament->vertices(0, 1), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step, polygon_filament->vertices(1, 1),
                            kTolerance));
  EXPECT_EQ(polygon_filament->vertices(2, 1), 2.3);

  // vertex_2 has equal components in x and y in x-y plane
  // -> expected to move in equal amounts along both directions
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step / std::sqrt(2),
                            polygon_filament->vertices(0, 2), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step / std::sqrt(2),
                            polygon_filament->vertices(1, 2), kTolerance));
  EXPECT_EQ(polygon_filament->vertices(2, 2), 3.3);

  // Add an InfiniteStraightFilament (which is not supported to be radially
  // moved).
  CurrentCarrier *current_carrier_2 = coil->add_current_carriers();
  InfiniteStraightFilament *infinite_straight_filament =
      current_carrier_2->mutable_infinite_straight_filament();
  infinite_straight_filament->origin_.emplace();
  infinite_straight_filament->origin_->set_x(1.23);
  infinite_straight_filament->origin_->set_y(4.56);
  infinite_straight_filament->origin_->set_z(7.89);
  infinite_straight_filament->direction_.emplace();
  infinite_straight_filament->direction_->set_x(9.87);
  infinite_straight_filament->direction_->set_y(6.54);
  infinite_straight_filament->direction_->set_z(3.21);

  // Check that the MagneticConfiguration is fully populated.
  status = IsMagneticConfigurationFullyPopulated(magnetic_configuration);
  ASSERT_TRUE(status.ok()) << status.message();

  // Check that the movement of the MagneticConfiguration is rejected (for now).
  status = MoveRadially(radial_step,
                        /*m_magnetic_configuration=*/magnetic_configuration);
  EXPECT_FALSE(status.ok());
}  // CheckMoveRadiallyOutwardMagneticConfiguration

// -------------------
// Regression tests for the std::optional and Eigen refactoring

TEST(TestRegressionOptional, PolygonVerticesAreContiguous) {
  PolygonFilament polygon_filament;
  polygon_filament.vertices.resize(3, 4);
  for (int i = 0; i < 4; ++i) {
    polygon_filament.vertices.col(i) =
        Eigen::Vector3d{1.0 * i, 2.0 * i, 3.0 * i};
  }

  // Verify all values are accessible and correct
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(polygon_filament.vertices(0, i), 1.0 * i);
    EXPECT_EQ(polygon_filament.vertices(1, i), 2.0 * i);
    EXPECT_EQ(polygon_filament.vertices(2, i), 3.0 * i);
  }

  // Verify .cols() returns the correct count
  EXPECT_EQ(polygon_filament.vertices.cols(), 4);
}

TEST(TestRegressionOptional, OptionalFieldsWorkCorrectly) {
  SerialCircuit sc;
  EXPECT_FALSE(sc.name.has_value());
  EXPECT_FALSE(sc.current_.has_value());

  sc.name = "test";
  sc.current_ = 42.0;
  EXPECT_TRUE(sc.name.has_value());
  EXPECT_EQ(sc.name.value(), "test");
  EXPECT_TRUE(sc.current_.has_value());
  EXPECT_EQ(sc.current_.value(), 42.0);

  sc.Clear();
  EXPECT_FALSE(sc.name.has_value());
  EXPECT_FALSE(sc.current_.has_value());
}

TEST(TestRegressionOptional, CircularFilamentFieldsAlwaysAccessible) {
  CircularFilament cf;
  // center, normal, radius should be directly accessible without checks
  cf.center.set_x(1.0);
  cf.center.set_y(2.0);
  cf.center.set_z(3.0);
  cf.normal.set_x(0.0);
  cf.normal.set_y(0.0);
  cf.normal.set_z(1.0);
  cf.radius = 5.0;

  EXPECT_EQ(cf.center.x(), 1.0);
  EXPECT_EQ(cf.center.y(), 2.0);
  EXPECT_EQ(cf.center.z(), 3.0);
  EXPECT_EQ(cf.normal.z(), 1.0);
  EXPECT_EQ(cf.radius, 5.0);

  // name is still optional
  EXPECT_FALSE(cf.name.has_value());
  cf.name = "test_coil";
  EXPECT_EQ(cf.name.value(), "test_coil");
}

}  // namespace magnetics
