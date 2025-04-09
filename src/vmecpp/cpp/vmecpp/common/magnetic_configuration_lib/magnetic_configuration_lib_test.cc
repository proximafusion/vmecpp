// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"

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

  EXPECT_FALSE(magnetic_configuration->has_name());
  ASSERT_TRUE(magnetic_configuration->has_num_field_periods());
  EXPECT_EQ(magnetic_configuration->num_field_periods(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.has_name());
  EXPECT_EQ(serial_circuit.current(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 1);

  const Coil &coil = serial_circuit.coils(0);
  EXPECT_FALSE(coil.has_name());
  ASSERT_TRUE(coil.has_num_windings());
  EXPECT_EQ(coil.num_windings(), 3.0);
  ASSERT_EQ(coil.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier = coil.current_carriers(0);
  ASSERT_TRUE(current_carrier.has_circular_filament());

  const CircularFilament &circular_filament =
      current_carrier.circular_filament();
  ASSERT_TRUE(circular_filament.has_name());
  EXPECT_EQ(circular_filament.name(), "circular_filament");
  ASSERT_TRUE(circular_filament.has_radius());
  EXPECT_EQ(circular_filament.radius(), 1.0);
  ASSERT_TRUE(circular_filament.has_center());
  ASSERT_TRUE(circular_filament.has_normal());

  const Vector3d &center = circular_filament.center();
  EXPECT_EQ(center.x(), 0.0);
  EXPECT_EQ(center.y(), 0.0);
  EXPECT_EQ(center.z(), 2.0);

  const Vector3d &normal = circular_filament.normal();
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

  EXPECT_FALSE(magnetic_configuration->has_name());
  ASSERT_TRUE(magnetic_configuration->has_num_field_periods());
  EXPECT_EQ(magnetic_configuration->num_field_periods(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.has_name());
  ASSERT_TRUE(serial_circuit.has_current());
  EXPECT_EQ(serial_circuit.current(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 1);

  const Coil &coil = serial_circuit.coils(0);
  EXPECT_FALSE(coil.has_name());
  ASSERT_TRUE(coil.has_num_windings());
  EXPECT_EQ(coil.num_windings(), 3.0);
  ASSERT_EQ(coil.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier = coil.current_carriers(0);
  ASSERT_TRUE(current_carrier.has_circular_filament());

  const CircularFilament &circular_filament =
      current_carrier.circular_filament();
  ASSERT_TRUE(circular_filament.has_name());
  EXPECT_EQ(circular_filament.name(), "circular_filament");
  ASSERT_TRUE(circular_filament.has_radius());
  EXPECT_EQ(circular_filament.radius(), 1.0);
  ASSERT_TRUE(circular_filament.has_center());
  ASSERT_TRUE(circular_filament.has_normal());

  const Vector3d &center = circular_filament.center();
  EXPECT_EQ(center.x(), 0.0);
  EXPECT_EQ(center.y(), 0.0);
  EXPECT_EQ(center.z(), 2.0);

  const Vector3d &normal = circular_filament.normal();
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

  EXPECT_FALSE(magnetic_configuration->has_name());
  ASSERT_TRUE(magnetic_configuration->has_num_field_periods());
  EXPECT_EQ(magnetic_configuration->num_field_periods(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.has_name());
  ASSERT_TRUE(serial_circuit.has_current());
  EXPECT_EQ(serial_circuit.current(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 2);

  // first circular filament

  const Coil &coil_1a = serial_circuit.coils(0);
  EXPECT_FALSE(coil_1a.has_name());
  ASSERT_TRUE(coil_1a.has_num_windings());
  EXPECT_EQ(coil_1a.num_windings(), 3.0);
  ASSERT_EQ(coil_1a.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1a = coil_1a.current_carriers(0);
  ASSERT_TRUE(current_carrier_1a.has_circular_filament());

  const CircularFilament &circular_filament_1a =
      current_carrier_1a.circular_filament();
  ASSERT_TRUE(circular_filament_1a.has_name());
  EXPECT_EQ(circular_filament_1a.name(), "circular_filament_1a");
  ASSERT_TRUE(circular_filament_1a.has_radius());
  EXPECT_EQ(circular_filament_1a.radius(), 1.0);
  ASSERT_TRUE(circular_filament_1a.has_center());
  ASSERT_TRUE(circular_filament_1a.has_normal());

  const Vector3d &center_1a = circular_filament_1a.center();
  EXPECT_EQ(center_1a.x(), 0.0);
  EXPECT_EQ(center_1a.y(), 0.0);
  EXPECT_EQ(center_1a.z(), 2.0);

  const Vector3d &normal_1a = circular_filament_1a.normal();
  EXPECT_EQ(normal_1a.x(), 0.0);
  EXPECT_EQ(normal_1a.y(), 0.0);
  EXPECT_EQ(normal_1a.z(), 1.0);

  // second circular filament

  const Coil &coil_1b = serial_circuit.coils(1);
  EXPECT_FALSE(coil_1b.has_name());
  ASSERT_TRUE(coil_1b.has_num_windings());
  EXPECT_EQ(coil_1b.num_windings(), 6.0);
  EXPECT_EQ(coil_1b.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1b = coil_1b.current_carriers(0);
  EXPECT_TRUE(current_carrier_1b.has_circular_filament());

  const CircularFilament &circular_filament_1b =
      current_carrier_1b.circular_filament();
  EXPECT_TRUE(circular_filament_1b.has_name());
  EXPECT_EQ(circular_filament_1b.name(), "circular_filament_1b");
  EXPECT_TRUE(circular_filament_1b.has_radius());
  EXPECT_EQ(circular_filament_1b.radius(), 4.0);
  ASSERT_TRUE(circular_filament_1b.has_center());
  ASSERT_TRUE(circular_filament_1b.has_normal());

  const Vector3d &center_1b = circular_filament_1b.center();
  EXPECT_EQ(center_1b.x(), 0.0);
  EXPECT_EQ(center_1b.y(), 0.0);
  EXPECT_EQ(center_1b.z(), 5.0);

  const Vector3d &normal_1b = circular_filament_1b.normal();
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

  EXPECT_FALSE(magnetic_configuration->has_name());
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 2);

  // first circular filament

  SerialCircuit serial_circuit_1 = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit_1.has_name());
  ASSERT_TRUE(serial_circuit_1.has_current());
  EXPECT_EQ(serial_circuit_1.current(), 1.0);
  ASSERT_EQ(serial_circuit_1.coils_size(), 1);

  const Coil &coil_1 = serial_circuit_1.coils(0);
  EXPECT_FALSE(coil_1.has_name());
  ASSERT_TRUE(coil_1.has_num_windings());
  EXPECT_EQ(coil_1.num_windings(), 3.0);
  ASSERT_EQ(coil_1.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1 = coil_1.current_carriers(0);
  ASSERT_TRUE(current_carrier_1.has_circular_filament());

  const CircularFilament &circular_filament_1 =
      current_carrier_1.circular_filament();
  ASSERT_TRUE(circular_filament_1.has_name());
  EXPECT_EQ(circular_filament_1.name(), "circular_filament_1");
  ASSERT_TRUE(circular_filament_1.has_radius());
  EXPECT_EQ(circular_filament_1.radius(), 1.0);
  ASSERT_TRUE(circular_filament_1.has_center());
  ASSERT_TRUE(circular_filament_1.has_normal());

  const Vector3d &center_1 = circular_filament_1.center();
  EXPECT_EQ(center_1.x(), 0.0);
  EXPECT_EQ(center_1.y(), 0.0);
  EXPECT_EQ(center_1.z(), 2.0);

  const Vector3d &normal_1 = circular_filament_1.normal();
  EXPECT_EQ(normal_1.x(), 0.0);
  EXPECT_EQ(normal_1.y(), 0.0);
  EXPECT_EQ(normal_1.z(), 1.0);

  // second circular filament

  SerialCircuit serial_circuit_2 = magnetic_configuration->serial_circuits(1);
  EXPECT_FALSE(serial_circuit_2.has_name());
  ASSERT_TRUE(serial_circuit_2.has_current());
  EXPECT_EQ(serial_circuit_2.current(), 1.0);
  ASSERT_EQ(serial_circuit_2.coils_size(), 1);

  const Coil &coil_2 = serial_circuit_2.coils(0);
  EXPECT_FALSE(coil_2.has_name());
  ASSERT_TRUE(coil_2.has_num_windings());
  EXPECT_EQ(coil_2.num_windings(), 6.0);
  ASSERT_EQ(coil_2.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_2 = coil_2.current_carriers(0);
  ASSERT_TRUE(current_carrier_2.has_circular_filament());

  const CircularFilament &circular_filament_2 =
      current_carrier_2.circular_filament();
  ASSERT_TRUE(circular_filament_2.has_name());
  EXPECT_EQ(circular_filament_2.name(), "circular_filament_2");
  ASSERT_TRUE(circular_filament_2.has_radius());
  EXPECT_EQ(circular_filament_2.radius(), 4.0);
  ASSERT_TRUE(circular_filament_2.has_center());
  ASSERT_TRUE(circular_filament_2.has_normal());

  const Vector3d &center_2 = circular_filament_2.center();
  EXPECT_EQ(center_2.x(), 0.0);
  EXPECT_EQ(center_2.y(), 0.0);
  EXPECT_EQ(center_2.z(), 5.0);

  const Vector3d &normal_2 = circular_filament_2.normal();
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

  EXPECT_FALSE(magnetic_configuration->has_name());
  ASSERT_TRUE(magnetic_configuration->has_num_field_periods());
  EXPECT_EQ(magnetic_configuration->num_field_periods(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.has_name());
  ASSERT_TRUE(serial_circuit.has_current());
  EXPECT_EQ(serial_circuit.current(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 1);

  const Coil &coil = serial_circuit.coils(0);
  EXPECT_FALSE(coil.has_name());
  ASSERT_TRUE(coil.has_num_windings());
  EXPECT_EQ(coil.num_windings(), 4.0);
  ASSERT_EQ(coil.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier = coil.current_carriers(0);
  ASSERT_TRUE(current_carrier.has_polygon_filament());

  const PolygonFilament &polygon_filament = current_carrier.polygon_filament();
  ASSERT_TRUE(polygon_filament.has_name());
  EXPECT_EQ(polygon_filament.name(), "polygon_filament");
  ASSERT_EQ(polygon_filament.vertices_size(), 2);

  const Vector3d &vertex_0 = polygon_filament.vertices(0);
  EXPECT_EQ(vertex_0.x(), 1.0);
  EXPECT_EQ(vertex_0.y(), 2.0);
  EXPECT_EQ(vertex_0.z(), 3.0);

  const Vector3d &vertex_1 = polygon_filament.vertices(1);
  EXPECT_EQ(vertex_1.x(), 5.0);
  EXPECT_EQ(vertex_1.y(), 6.0);
  EXPECT_EQ(vertex_1.z(), 7.0);
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

  EXPECT_FALSE(magnetic_configuration->has_name());
  ASSERT_TRUE(magnetic_configuration->has_num_field_periods());
  EXPECT_EQ(magnetic_configuration->num_field_periods(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 1);

  SerialCircuit serial_circuit = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit.has_name());
  ASSERT_TRUE(serial_circuit.has_current());
  EXPECT_EQ(serial_circuit.current(), 1.0);
  ASSERT_EQ(serial_circuit.coils_size(), 2);

  // first polygon filament

  const Coil &coil_1a = serial_circuit.coils(0);
  EXPECT_FALSE(coil_1a.has_name());
  ASSERT_TRUE(coil_1a.has_num_windings());
  EXPECT_EQ(coil_1a.num_windings(), 4.0);
  ASSERT_EQ(coil_1a.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1a = coil_1a.current_carriers(0);
  ASSERT_TRUE(current_carrier_1a.has_polygon_filament());

  const PolygonFilament &polygon_filament_1a =
      current_carrier_1a.polygon_filament();
  ASSERT_TRUE(polygon_filament_1a.has_name());
  EXPECT_EQ(polygon_filament_1a.name(), "polygon_filament_1a");
  ASSERT_EQ(polygon_filament_1a.vertices_size(), 2);

  const Vector3d &vertex_1a_0 = polygon_filament_1a.vertices(0);
  EXPECT_EQ(vertex_1a_0.x(), 1.0);
  EXPECT_EQ(vertex_1a_0.y(), 2.0);
  EXPECT_EQ(vertex_1a_0.z(), 3.0);

  const Vector3d &vertex_1a_1 = polygon_filament_1a.vertices(1);
  EXPECT_EQ(vertex_1a_1.x(), 5.0);
  EXPECT_EQ(vertex_1a_1.y(), 6.0);
  EXPECT_EQ(vertex_1a_1.z(), 7.0);

  // second polygon filament

  const Coil &coil_1b = serial_circuit.coils(1);
  EXPECT_FALSE(coil_1b.has_name());
  ASSERT_TRUE(coil_1b.has_num_windings());
  EXPECT_EQ(coil_1b.num_windings(), 4.5);
  ASSERT_EQ(coil_1b.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1b = coil_1b.current_carriers(0);
  ASSERT_TRUE(current_carrier_1b.has_polygon_filament());

  const PolygonFilament &polygon_filament_1b =
      current_carrier_1b.polygon_filament();
  ASSERT_TRUE(polygon_filament_1b.has_name());
  EXPECT_EQ(polygon_filament_1b.name(), "polygon_filament_1b");
  ASSERT_EQ(polygon_filament_1b.vertices_size(), 2);

  const Vector3d &vertex_1b_0 = polygon_filament_1b.vertices(0);
  EXPECT_EQ(vertex_1b_0.x(), 1.5);
  EXPECT_EQ(vertex_1b_0.y(), 2.5);
  EXPECT_EQ(vertex_1b_0.z(), 3.5);

  const Vector3d &vertex_1b_1 = polygon_filament_1b.vertices(1);
  EXPECT_EQ(vertex_1b_1.x(), 5.5);
  EXPECT_EQ(vertex_1b_1.y(), 6.5);
  EXPECT_EQ(vertex_1b_1.z(), 7.5);
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

  EXPECT_FALSE(magnetic_configuration->has_name());
  ASSERT_TRUE(magnetic_configuration->has_num_field_periods());
  EXPECT_EQ(magnetic_configuration->num_field_periods(), 1);
  ASSERT_EQ(magnetic_configuration->serial_circuits_size(), 2);

  // first polygon filament

  SerialCircuit serial_circuit_0 = magnetic_configuration->serial_circuits(0);
  EXPECT_FALSE(serial_circuit_0.has_name());
  ASSERT_TRUE(serial_circuit_0.has_current());
  EXPECT_EQ(serial_circuit_0.current(), 1.0);
  ASSERT_EQ(serial_circuit_0.coils_size(), 1);

  const Coil &coil_0 = serial_circuit_0.coils(0);
  EXPECT_FALSE(coil_0.has_name());
  ASSERT_TRUE(coil_0.has_num_windings());
  EXPECT_EQ(coil_0.num_windings(), 4.0);
  ASSERT_EQ(coil_0.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_0 = coil_0.current_carriers(0);
  ASSERT_TRUE(current_carrier_0.has_polygon_filament());

  const PolygonFilament &polygon_filament_1 =
      current_carrier_0.polygon_filament();
  ASSERT_TRUE(polygon_filament_1.has_name());
  EXPECT_EQ(polygon_filament_1.name(), "polygon_filament_1");
  ASSERT_EQ(polygon_filament_1.vertices_size(), 2);

  const Vector3d &vertex_1_0 = polygon_filament_1.vertices(0);
  EXPECT_EQ(vertex_1_0.x(), 1.0);
  EXPECT_EQ(vertex_1_0.y(), 2.0);
  EXPECT_EQ(vertex_1_0.z(), 3.0);

  const Vector3d &vertex_1_1 = polygon_filament_1.vertices(1);
  EXPECT_EQ(vertex_1_1.x(), 5.0);
  EXPECT_EQ(vertex_1_1.y(), 6.0);
  EXPECT_EQ(vertex_1_1.z(), 7.0);

  // second polygon filament

  SerialCircuit serial_circuit_1 = magnetic_configuration->serial_circuits(1);
  EXPECT_FALSE(serial_circuit_1.has_name());
  ASSERT_TRUE(serial_circuit_1.has_current());
  EXPECT_EQ(serial_circuit_1.current(), 1.0);
  ASSERT_EQ(serial_circuit_1.coils_size(), 1);

  const Coil &coil_1 = serial_circuit_1.coils(0);
  EXPECT_FALSE(coil_1.has_name());
  ASSERT_TRUE(coil_1.has_num_windings());
  EXPECT_EQ(coil_1.num_windings(), 4.5);
  ASSERT_EQ(coil_1.current_carriers_size(), 1);

  const CurrentCarrier &current_carrier_1 = coil_1.current_carriers(0);
  ASSERT_TRUE(current_carrier_1.has_polygon_filament());

  const PolygonFilament &polygon_filament_2 =
      current_carrier_1.polygon_filament();
  ASSERT_TRUE(polygon_filament_2.has_name());
  EXPECT_EQ(polygon_filament_2.name(), "polygon_filament_2");
  ASSERT_EQ(polygon_filament_2.vertices_size(), 2);

  const Vector3d &vertex_2_0 = polygon_filament_2.vertices(0);
  EXPECT_EQ(vertex_2_0.x(), 1.5);
  EXPECT_EQ(vertex_2_0.y(), 2.5);
  EXPECT_EQ(vertex_2_0.z(), 3.5);

  const Vector3d &vertex_2_1 = polygon_filament_2.vertices(1);
  EXPECT_EQ(vertex_2_1.x(), 5.5);
  EXPECT_EQ(vertex_2_1.y(), 6.5);
  EXPECT_EQ(vertex_2_1.z(), 7.5);
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
  std::vector<double> expected_currents = {1.0, 1.0};

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  absl::StatusOr<std::vector<double> > circuit_currents =
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
  std::vector<double> original_currents = {1.0, 1.0};

  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      ImportMagneticConfigurationFromMakegrid(makegrid_coils);
  ASSERT_TRUE(magnetic_configuration.ok());

  // specifying only a single current should be rejected, since two circuits are
  // in the MagneticConfiguration
  std::vector<double> one_current = {2.0};
  absl::Status status_one_current = SetCircuitCurrents(
      one_current, /*m_magnetic_configuration=*/*magnetic_configuration);
  EXPECT_FALSE(status_one_current.ok());

  // check that no change was made to the currents (assume that no other part in
  // the MagneticConfiguration was touched)
  absl::StatusOr<std::vector<double> > currents_after_first_attempt =
      GetCircuitCurrents(*magnetic_configuration);
  ASSERT_TRUE(currents_after_first_attempt.ok());
  EXPECT_THAT(*currents_after_first_attempt,
              ElementsAreArray(original_currents));

  // specifying two currents should be accepted, since two circuits are in the
  // MagneticConfiguration
  std::vector<double> two_currents = {2.0, 3.0};
  absl::Status status_two_current = SetCircuitCurrents(
      two_currents, /*m_magnetic_configuration=*/*magnetic_configuration);
  EXPECT_TRUE(status_two_current.ok());

  // now check that the currents actually appeared in the MagneticConfiguration
  absl::StatusOr<std::vector<double> > currents_after_second_attempt =
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

  EXPECT_EQ(magnetic_configuration.serial_circuits(0).current(), 1.0);
  ASSERT_EQ(magnetic_configuration.serial_circuits(0).coils_size(), 1);
  EXPECT_EQ(magnetic_configuration.serial_circuits(0).coils(0).num_windings(),
            4.0);

  EXPECT_EQ(magnetic_configuration.serial_circuits(1).current(), 1.0);
  ASSERT_EQ(magnetic_configuration.serial_circuits(1).coils_size(), 2);
  EXPECT_EQ(magnetic_configuration.serial_circuits(1).coils(0).num_windings(),
            4.5);
  EXPECT_EQ(magnetic_configuration.serial_circuits(1).coils(1).num_windings(),
            4.5);

  // now make a mutable copy of the MagneticConfiguration
  // and migrate `num_windings` into the circuit currents
  MagneticConfiguration m_magnetic_configuration = magnetic_configuration;

  // call under test
  absl::Status status = NumWindingsToCircuitCurrents(m_magnetic_configuration);
  ASSERT_TRUE(status.ok()) << status.message();

  // now check that currents actually have been migrated into circuit currents
  EXPECT_EQ(m_magnetic_configuration.serial_circuits(0).current(), 4.0);
  ASSERT_EQ(m_magnetic_configuration.serial_circuits(0).coils_size(), 1);
  EXPECT_EQ(m_magnetic_configuration.serial_circuits(0).coils(0).num_windings(),
            1.0);

  EXPECT_EQ(m_magnetic_configuration.serial_circuits(1).current(), 4.5);
  ASSERT_EQ(m_magnetic_configuration.serial_circuits(1).coils_size(), 2);
  EXPECT_EQ(m_magnetic_configuration.serial_circuits(1).coils(0).num_windings(),
            1.0);
  EXPECT_EQ(m_magnetic_configuration.serial_circuits(1).coils(1).num_windings(),
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

  EXPECT_EQ(magnetic_configuration_2.serial_circuits(0).current(), 1.0);
  ASSERT_EQ(magnetic_configuration_2.serial_circuits(0).coils_size(), 2);
  EXPECT_EQ(magnetic_configuration_2.serial_circuits(0).coils(0).num_windings(),
            4.0);
  EXPECT_EQ(magnetic_configuration_2.serial_circuits(0).coils(1).num_windings(),
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
    infinite_straight_filament.set_name("filament_1");
  }
  if (specify_origin_) {
    Vector3d *origin = infinite_straight_filament.mutable_origin();
    if (origin_components_ & (1 << 0)) {
      origin->set_x(1.23);
    }
    if (origin_components_ & (1 << 1)) {
      origin->set_y(4.56);
    }
    if (origin_components_ & (1 << 2)) {
      origin->set_z(7.89);
    }
  }
  if (specify_direction_) {
    Vector3d *direction = infinite_straight_filament.mutable_direction();
    if (direction_components_ & (1 << 0)) {
      direction->set_x(9.87);
    }
    if (direction_components_ & (1 << 1)) {
      direction->set_y(6.54);
    }
    if (direction_components_ & (1 << 2)) {
      direction->set_z(3.21);
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

// The two integer parameters are interpreted as bitfields that control
// which Cartesian components of the center and normal vectors are populated.
// Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
class IsCircularFilamentFullyPopulatedTest
    : public TestWithParam< ::std::tuple<bool, bool, int, bool, int, bool> > {
 protected:
  void SetUp() override {
    std::tie(specify_name_, specify_center_, center_components_,
             specify_normal_, normal_components_, specify_radius_) = GetParam();
  }
  bool specify_name_;
  bool specify_center_;
  int center_components_;
  bool specify_normal_;
  int normal_components_;
  bool specify_radius_;
};

TEST_P(IsCircularFilamentFullyPopulatedTest,
       CheckIsCircularFilamentFullyPopulated) {
  CircularFilament circular_filament;
  if (specify_name_) {
    circular_filament.set_name("filament_1");
  }
  if (specify_center_) {
    Vector3d *center = circular_filament.mutable_center();
    if (center_components_ & (1 << 0)) {
      center->set_x(1.23);
    }
    if (center_components_ & (1 << 1)) {
      center->set_y(4.56);
    }
    if (center_components_ & (1 << 2)) {
      center->set_z(7.89);
    }
  }
  if (specify_normal_) {
    Vector3d *normal = circular_filament.mutable_normal();
    if (normal_components_ & (1 << 0)) {
      normal->set_x(9.87);
    }
    if (normal_components_ & (1 << 1)) {
      normal->set_y(6.54);
    }
    if (normal_components_ & (1 << 2)) {
      normal->set_z(3.21);
    }
  }
  if (specify_radius_) {
    circular_filament.set_radius(3.14);
  }

  absl::Status status = IsCircularFilamentFullyPopulated(circular_filament);
  if (specify_center_ && center_components_ == 7 && specify_normal_ &&
      normal_components_ == 7 && specify_radius_) {
    EXPECT_TRUE(status.ok());
  } else {
    EXPECT_FALSE(status.ok());
  }
}  // CheckIsCircularFilamentFullyPopulated

INSTANTIATE_TEST_SUITE_P(TestMagneticConfigurationLib,
                         IsCircularFilamentFullyPopulatedTest,
                         Combine(Bool(), Bool(), Values(0, 1, 2, 3, 4, 5, 6, 7),
                                 Bool(), Values(0, 1, 2, 3, 4, 5, 6, 7),
                                 Bool()));

// -------------------

// The second integer parameter is interpreted as a bitfield that controls
// which Cartesian components of the vertices are populated.
// Bit 0 controls the x component; x is populated if this bit is 1.
// Bit 1 controls the y component; y is populated if this bit is 1.
// Bit 2 controls the z component; z is populated if this bit is 1.
class IsPolygonFilamentFullyPopulatedTest
    : public TestWithParam< ::std::tuple<bool, int, int> > {
 protected:
  void SetUp() override {
    std::tie(specify_name_, number_of_vertices_, vertex_components_) =
        GetParam();
  }
  bool specify_name_;
  int number_of_vertices_;
  int vertex_components_;
};

TEST_P(IsPolygonFilamentFullyPopulatedTest,
       CheckIsPolygonFilamentFullyPopulated) {
  PolygonFilament polygon_filament;
  if (specify_name_) {
    polygon_filament.set_name("filament_3");
  }
  for (int i = 0; i < number_of_vertices_; ++i) {
    Vector3d *vertex = polygon_filament.add_vertices();
    if (vertex_components_ & (1 << 0)) {
      vertex->set_x(3.14);
    }
    if (vertex_components_ & (1 << 1)) {
      vertex->set_y(2.71);
    }
    if (vertex_components_ & (1 << 2)) {
      vertex->set_z(1.41);
    }
  }

  absl::Status status = IsPolygonFilamentFullyPopulated(polygon_filament);
  if (number_of_vertices_ > 1 && vertex_components_ == 7) {
    EXPECT_TRUE(status.ok());
  } else {
    EXPECT_FALSE(status.ok());
  }
}  // CheckIsPolygonFilamentFullyPopulated

INSTANTIATE_TEST_SUITE_P(TestMagneticConfigurationLib,
                         IsPolygonFilamentFullyPopulatedTest,
                         Combine(Bool(), Values(0, 1, 2, 3),
                                 Values(0, 1, 2, 3, 4, 5, 6, 7)));

// -------------------

class IsMagneticConfigurationFullyPopulatedTest : public Test {
 protected:
  void SetUp() override {
    SerialCircuit *serial_circuit =
        magnetic_configuration_.add_serial_circuits();
    Coil *coil = serial_circuit->add_coils();
    current_carrier_ = coil->add_current_carriers();
  }
  MagneticConfiguration magnetic_configuration_;
  CurrentCarrier *current_carrier_;
};

TEST_F(IsMagneticConfigurationFullyPopulatedTest,
       CheckIsMagneticConfigurationFullyPopulatedWithNoCurrentCarrier) {
  // do not add any current carrier
  // --> noting to test, is also ok and will also not modify the magnetic field
  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_);

  EXPECT_TRUE(status.ok());
}  // CheckIsMagneticConfigurationFullyPopulatedWithNoCurrentCarrier

TEST_F(IsMagneticConfigurationFullyPopulatedTest,
       CheckIsMagneticConfigurationFullyPopulatedWithInfiniteStraightFilament) {
  InfiniteStraightFilament *infinite_straight_filament =
      current_carrier_->mutable_infinite_straight_filament();
  Vector3d *origin = infinite_straight_filament->mutable_origin();
  origin->set_x(1.23);
  origin->set_y(4.56);
  origin->set_z(7.89);
  Vector3d *direction = infinite_straight_filament->mutable_direction();
  direction->set_x(9.87);
  direction->set_y(6.54);
  direction->set_z(3.21);

  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_);

  EXPECT_TRUE(status.ok());
}  // CheckIsMagneticConfigurationFullyPopulatedWithInfiniteStraightFilament

TEST_F(IsMagneticConfigurationFullyPopulatedTest,
       CheckIsMagneticConfigurationFullyPopulatedWithCircularFilament) {
  CircularFilament *circular_filament =
      current_carrier_->mutable_circular_filament();
  Vector3d *center = circular_filament->mutable_center();
  center->set_x(1.23);
  center->set_y(4.56);
  center->set_z(7.89);
  Vector3d *normal = circular_filament->mutable_normal();
  normal->set_x(9.87);
  normal->set_y(6.54);
  normal->set_z(3.21);
  circular_filament->set_radius(3.14);

  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration_);

  EXPECT_TRUE(status.ok());
}  // CheckIsMagneticConfigurationFullyPopulatedWithCircularFilament

TEST_F(IsMagneticConfigurationFullyPopulatedTest,
       CheckIsMagneticConfigurationFullyPopulatedWithPolygonFilament) {
  PolygonFilament *polygon_filament =
      current_carrier_->mutable_polygon_filament();
  Vector3d *vertex_1 = polygon_filament->add_vertices();
  vertex_1->set_x(1.23);
  vertex_1->set_y(4.56);
  vertex_1->set_z(7.89);
  Vector3d *vertex_2 = polygon_filament->add_vertices();
  vertex_2->set_x(9.87);
  vertex_2->set_y(6.54);
  vertex_2->set_z(3.21);

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
    infinite_straight_filament.set_name("filament_1");
  }
  if (specify_origin_) {
    Vector3d *origin = infinite_straight_filament.mutable_origin();
    origin->set_x(1.23);
    origin->set_y(4.56);
    origin->set_z(7.89);
  }
  if (specify_direction_) {
    Vector3d *direction = infinite_straight_filament.mutable_direction();
    direction->set_x(9.87);
    direction->set_y(6.54);
    direction->set_z(3.21);
  }

  testing::internal::CaptureStdout();
  PrintInfiniteStraightFilament(infinite_straight_filament);
  std::string output = testing::internal::GetCapturedStdout();

  std::string expected_output = "InfiniteStraightFilament {\n";
  if (specify_name_) {
    expected_output += "  name: 'filament_1'\n";
  } else {
    expected_output += "  name: none\n";
  }
  if (specify_origin_) {
    expected_output += "  origin: [1.23, 4.56, 7.89]\n";
  } else {
    expected_output += "  origin: none\n";
  }
  if (specify_direction_) {
    expected_output += "  direction: [9.87, 6.54, 3.21]\n";
  } else {
    expected_output += "  direction: none\n";
  }
  expected_output += "}\n";

  EXPECT_TRUE(output == expected_output);
}  // CheckPrintInfiniteStraightFilament

INSTANTIATE_TEST_SUITE_P(TestMagneticConfigurationLib,
                         PrintInfiniteStraightFilamentTest,
                         Combine(Bool(), Bool(), Bool()));

// -------------------

class PrintCircularFilamentTest
    : public TestWithParam< ::std::tuple<bool, bool, bool, bool> > {
 protected:
  void SetUp() override {
    std::tie(specify_name_, specify_center_, specify_normal_, specify_radius_) =
        GetParam();
  }
  bool specify_name_;
  bool specify_center_;
  bool specify_normal_;
  bool specify_radius_;
};

TEST_P(PrintCircularFilamentTest, CheckPrintCircularFilament) {
  CircularFilament circular_filament;
  if (specify_name_) {
    circular_filament.set_name("filament_2");
  }
  if (specify_center_) {
    Vector3d *center = circular_filament.mutable_center();
    center->set_x(1.23);
    center->set_y(4.56);
    center->set_z(7.89);
  }
  if (specify_normal_) {
    Vector3d *normal = circular_filament.mutable_normal();
    normal->set_x(9.87);
    normal->set_y(6.54);
    normal->set_z(3.21);
  }
  if (specify_radius_) {
    circular_filament.set_radius(3.14);
  }

  // https://stackoverflow.com/a/33186201
  testing::internal::CaptureStdout();
  PrintCircularFilament(circular_filament);
  std::string output = testing::internal::GetCapturedStdout();

  std::string expected_output = "CircularFilament {\n";
  if (specify_name_) {
    expected_output += "  name: 'filament_2'\n";
  } else {
    expected_output += "  name: none\n";
  }
  if (specify_center_) {
    expected_output += "  center: [1.23, 4.56, 7.89]\n";
  } else {
    expected_output += "  center: none\n";
  }
  if (specify_normal_) {
    expected_output += "  normal: [9.87, 6.54, 3.21]\n";
  } else {
    expected_output += "  normal: none\n";
  }
  if (specify_radius_) {
    expected_output += "  radius: 3.14\n";
  } else {
    expected_output += "  radius: none\n";
  }
  expected_output += "}\n";

  EXPECT_TRUE(output == expected_output);
}  // CheckPrintCircularFilament

INSTANTIATE_TEST_SUITE_P(TestMagneticConfigurationLib,
                         PrintCircularFilamentTest,
                         Combine(Bool(), Bool(), Bool(), Bool()));

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
    polygon_filament.set_name("filament_3");
  }
  if (specify_vertices_) {
    polygon_filament.add_vertices();
    polygon_filament.add_vertices();
    polygon_filament.add_vertices();
  }

  // https://stackoverflow.com/a/33186201
  testing::internal::CaptureStdout();
  PrintPolygonFilament(polygon_filament);
  std::string output = testing::internal::GetCapturedStdout();

  std::string expected_output = "PolygonFilament {\n";
  if (specify_name_) {
    expected_output += "  name: 'filament_3'\n";
  } else {
    expected_output += "  name: none\n";
  }
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
  Vector3d *center = circular_filament.mutable_center();
  center->set_x(1.23);
  center->set_y(4.56);
  center->set_z(7.89);
  Vector3d *normal = circular_filament.mutable_normal();
  normal->set_x(9.87);
  normal->set_y(6.54);
  normal->set_z(3.21);
  circular_filament.set_radius(initial_radius);

  absl::Status status = IsCircularFilamentFullyPopulated(circular_filament);
  ASSERT_TRUE(status.ok()) << status.message();

  // check that movement attempt fails because center is not on origin in x and
  // y
  status = MoveRadially(radial_step, /*m_circular_filament=*/circular_filament);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.message(),
            "center has to be on origin in x and y to perform radial movement");

  // fix center to be on origin in x and y
  center->set_x(0.0);
  center->set_y(0.0);

  // check that movement fails because normal is not along z axis
  status = MoveRadially(radial_step, /*m_circular_filament=*/circular_filament);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.message(),
            "normal has to be along z axis to perform radial movement");

  // fix normal to be along z axis
  normal->set_x(0.0);
  normal->set_y(0.0);
  normal->set_z(1.0);

  // attempt movement and check that it was successful
  status = MoveRadially(radial_step, /*m_circular_filament=*/circular_filament);
  ASSERT_TRUE(status.ok()) << status.message();

  // check that radius has the expected value
  EXPECT_EQ(circular_filament.radius(), initial_radius + radial_step);

  // check that no other members have been changed by successful call to
  // MoveRadially
  EXPECT_EQ(center->x(), 0.0);
  EXPECT_EQ(center->y(), 0.0);
  EXPECT_EQ(center->z(), 7.89);

  EXPECT_EQ(normal->x(), 0.0);
  EXPECT_EQ(normal->y(), 0.0);
  EXPECT_EQ(normal->z(), 1.0);
}  // CheckMoveRadiallyOutwardCircularFilament

TEST(TestMagneticConfigurationLib, CheckMoveRadiallyOutwardPolygonFilament) {
  static constexpr double kTolerance = 1.0e-15;

  const double radial_step = 0.42;

  PolygonFilament polygon_filament;

  Vector3d *vertex_1 = polygon_filament.add_vertices();
  vertex_1->set_x(1.0);
  vertex_1->set_y(0.0);
  vertex_1->set_z(1.3);

  Vector3d *vertex_2 = polygon_filament.add_vertices();
  vertex_2->set_x(0.0);
  vertex_2->set_y(1.0);
  vertex_2->set_z(2.3);

  Vector3d *vertex_3 = polygon_filament.add_vertices();
  vertex_3->set_x(1.0);
  vertex_3->set_y(1.0);
  vertex_3->set_z(3.3);

  absl::Status status = IsPolygonFilamentFullyPopulated(polygon_filament);
  ASSERT_TRUE(status.ok()) << status.message();

  // attempt movement and check that it was successful
  status = MoveRadially(radial_step, /*m_polygon_filament=*/polygon_filament);
  ASSERT_TRUE(status.ok()) << status.message();

  // check that vertices have moves as expected

  // vertex_1 only has x component in x-y plane
  // -> expected to move only along x
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step, vertex_1->x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, vertex_1->y(), kTolerance));
  EXPECT_EQ(vertex_1->z(), 1.3);

  // vertex_2 only has y component in x-y plane
  // -> expected to move only along y
  EXPECT_TRUE(IsCloseRelAbs(0.0, vertex_2->x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step, vertex_2->y(), kTolerance));
  EXPECT_EQ(vertex_2->z(), 2.3);

  // vertex_3 has equal components in x and y in x-y plane
  // -> expected to move in equal amounts along both directions
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step / std::sqrt(2), vertex_3->x(),
                            kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step / std::sqrt(2), vertex_3->y(),
                            kTolerance));
  EXPECT_EQ(vertex_3->z(), 3.3);
}  // CheckMoveRadiallyOutwardPolygonFilament

TEST(TestMagneticConfigurationLib,
     CheckMoveRadiallyOutwardMagneticConfiguration) {
  MagneticConfiguration magnetic_configuration;
  SerialCircuit *serial_circuit = magnetic_configuration.add_serial_circuits();
  Coil *coil = serial_circuit->add_coils();

  static constexpr double kTolerance = 1.0e-15;

  const double initial_radius = 3.14;
  const double radial_step = 0.42;

  // Add both a CircularFilament and a PolygonFilament.
  CurrentCarrier *current_carrier_0 = coil->add_current_carriers();
  CircularFilament *circular_filament =
      current_carrier_0->mutable_circular_filament();
  Vector3d *center = circular_filament->mutable_center();
  center->set_x(0.0);
  center->set_y(0.0);
  center->set_z(7.89);
  Vector3d *normal = circular_filament->mutable_normal();
  normal->set_x(0.0);
  normal->set_y(0.0);
  normal->set_z(1.0);
  circular_filament->set_radius(initial_radius);

  CurrentCarrier *current_carrier_1 = coil->add_current_carriers();
  PolygonFilament *polygon_filament =
      current_carrier_1->mutable_polygon_filament();
  Vector3d *vertex_1 = polygon_filament->add_vertices();
  vertex_1->set_x(1.0);
  vertex_1->set_y(0.0);
  vertex_1->set_z(1.3);
  Vector3d *vertex_2 = polygon_filament->add_vertices();
  vertex_2->set_x(0.0);
  vertex_2->set_y(1.0);
  vertex_2->set_z(2.3);
  Vector3d *vertex_3 = polygon_filament->add_vertices();
  vertex_3->set_x(1.0);
  vertex_3->set_y(1.0);
  vertex_3->set_z(3.3);

  // Check that the MagneticConfiguration is fully populated.
  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration);
  ASSERT_TRUE(status.ok()) << status.message();

  // Attempt to radially move the MagneticConfigutation and check that both are
  // moved. The correctness of the movement for the individual current carriers
  // is tested stand-alone above.
  status = MoveRadially(radial_step,
                        /*m_magnetic_configuration=*/magnetic_configuration);

  // check that radius has the expected value
  EXPECT_EQ(circular_filament->radius(), initial_radius + radial_step);

  // check that no other members have been changed by successful call to
  // MoveRadially
  EXPECT_EQ(center->x(), 0.0);
  EXPECT_EQ(center->y(), 0.0);
  EXPECT_EQ(center->z(), 7.89);

  EXPECT_EQ(normal->x(), 0.0);
  EXPECT_EQ(normal->y(), 0.0);
  EXPECT_EQ(normal->z(), 1.0);

  // vertex_1 only has x component in x-y plane
  // -> expected to move only along x
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step, vertex_1->x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, vertex_1->y(), kTolerance));
  EXPECT_EQ(vertex_1->z(), 1.3);

  // vertex_2 only has y component in x-y plane
  // -> expected to move only along y
  EXPECT_TRUE(IsCloseRelAbs(0.0, vertex_2->x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step, vertex_2->y(), kTolerance));
  EXPECT_EQ(vertex_2->z(), 2.3);

  // vertex_3 has equal components in x and y in x-y plane
  // -> expected to move in equal amounts along both directions
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step / std::sqrt(2), vertex_3->x(),
                            kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(1.0 + radial_step / std::sqrt(2), vertex_3->y(),
                            kTolerance));
  EXPECT_EQ(vertex_3->z(), 3.3);

  // Add an InfiniteStraightFilament (which is not supported to be radially
  // moved).
  CurrentCarrier *current_carrier_2 = coil->add_current_carriers();
  InfiniteStraightFilament *infinite_straight_filament =
      current_carrier_2->mutable_infinite_straight_filament();
  Vector3d *origin = infinite_straight_filament->mutable_origin();
  origin->set_x(1.23);
  origin->set_y(4.56);
  origin->set_z(7.89);
  Vector3d *direction = infinite_straight_filament->mutable_direction();
  direction->set_x(9.87);
  direction->set_y(6.54);
  direction->set_z(3.21);

  // Check that the MagneticConfiguration is fully populated.
  status = IsMagneticConfigurationFullyPopulated(magnetic_configuration);
  ASSERT_TRUE(status.ok()) << status.message();

  // Check that the movement of the MagneticConfiguration is rejected (for now).
  status = MoveRadially(radial_step,
                        /*m_magnetic_configuration=*/magnetic_configuration);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(),
            "Cannot perform radial movement if an InfiniteStraightFilament is "
            "present "
            "in the MagneticConfiguration");
}  // CheckMoveRadiallyOutwardMagneticConfiguration

}  // namespace magnetics
