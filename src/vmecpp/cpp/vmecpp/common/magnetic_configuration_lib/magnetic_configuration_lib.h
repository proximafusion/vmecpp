// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_MAGNETIC_CONFIGURATION_LIB_MAGNETIC_CONFIGURATION_LIB_H_
#define VMECPP_COMMON_MAGNETIC_CONFIGURATION_LIB_MAGNETIC_CONFIGURATION_LIB_H_

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "vmecpp/common/magnetic_configuration_definition/magnetic_configuration.h"

namespace magnetics {

// Build a MagneticConfiguration from the contents of a MAKEGRID-syle
// "coils-dot" file. Note: A MAKEGRID-style "coils-dot" file can store
// CircularFilament and PolygonFilament current carriers. Note that this
// overload takes the contents of the coils file as a string. Use
// ImportMagneticConfigurationFromCoilsFile to load directly from a file path.
absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromMakegrid(
    const std::string& makegrid_coils);

// Import a MagneticConfiguration from a MAKEGRID-syle "coils-dot" file.
absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromCoilsFile(
    const std::filesystem::path& mgrid_coils_file);

// Get the currents in the SerialCircuits in the given MagneticConfiguration.
// Checks that the given MagneticConfiguration is fully populated.
absl::StatusOr<std::vector<double> > GetCircuitCurrents(
    const MagneticConfiguration& magnetic_configuration);

// Overwrite the currents for the SerialCircuits in the given
// MagneticConfiguration with the currents in the given vector. The length of
// the circuit currents vector has to match the number of SerialCircuits in the
// given MagneticConfiguration.
absl::Status SetCircuitCurrents(
    const std::vector<double>& circuit_currents,
    MagneticConfiguration& m_magnetic_configuration);

// Move num_windings of Coils in each SerialCircuit into common circuit current,
// if num_windings is the same for each Coil in the given MagneticConfiguration.
absl::Status NumWindingsToCircuitCurrents(
    MagneticConfiguration& m_magnetic_configuration);

// ------------------

// Move a CircularFilament in the radial direction.
// Works only if the center of the loop is at the origin
// and its normal is along the z axis.
// This effectively modifies just the radius of the loop.
absl::Status MoveRadially(double radial_step,
                          CircularFilament& m_circular_filament);

// Move all vertices of a polygon filament along the cylindrical R direction by
// the given step.
absl::Status MoveRadially(double radial_step,
                          PolygonFilament& m_polygon_filament);

// Move all current carriers in the given magnetic configuration radially
// outwards, for which this operation is sensible (currently CircularFilament
// and PolygonFilament).
absl::Status MoveRadially(double radial_step,
                          MagneticConfiguration& m_magnetic_configuration);

// ------------------

// Check if the given InfiniteStraightFilament is fully populated.
// Returns true if all required parameters in the current carrier, most of which
// are optional in the specification, are provided. Note that this does NOT
// check if the InfiniteStraightFilament is physically reasonable.
absl::Status IsInfiniteStraightFilamentFullyPopulated(
    const InfiniteStraightFilament& infinite_straight_filament);

// Check if the given CircularFilament is fully populated.
// Returns true if all required parameters in the current carrier, most of which
// are optional in the specification, are provided. Note that this does NOT
// check if the CircularFilament is physically reasonable.
absl::Status IsCircularFilamentFullyPopulated(
    const CircularFilament& circular_filament);

// Check if the given PolygonFilament is fully populated.
// Returns true if all required parameters in the current carrier, most of which
// are optional in the specification, are provided. Note that this does NOT
// check if the PolygonFilament is physically reasonable.
absl::Status IsPolygonFilamentFullyPopulated(
    const PolygonFilament& polygon_filament);

// Check if the given MagneticConfiguration is fully populated.
// Returns true if all required parameters in the current carriers, most of
// which are optional in the specification, are provided. Note that this does
// NOT check if the MagneticConfiguration is physically reasonable.
absl::Status IsMagneticConfigurationFullyPopulated(
    const MagneticConfiguration& magnetic_configuration);

// ------------------

// Print a human-readable summary of the given InfiniteStraightFilament to
// std::cout. Note: Array contents are omitted for brevity. Instead, array
// dimensions are printed.
void PrintInfiniteStraightFilament(
    const InfiniteStraightFilament& infinite_straight_filament,
    int indentation = 0);

// Print a human-readable summary of the given CircularFilament to std::cout.
// Note: Array contents are omitted for brevity. Instead, array dimensions are
// printed.
void PrintCircularFilament(const CircularFilament& circular_filament,
                           int indentation = 0);

// Print a human-readable summary of the given PolygonFilament to std::cout.
// Note: Array contents are omitted for brevity. Instead, array dimensions are
// printed.
void PrintPolygonFilament(const PolygonFilament& polygon_filament,
                          int indentation = 0);

// Print a human-readable summary of the given CurrentCarrier to std::cout.
// Note: Array contents are omitted for brevity. Instead, array dimensions are
// printed.
void PrintCurrentCarrier(const CurrentCarrier& current_carrier,
                         int indentation = 0);

// Print a human-readable summary of the given Coil to std::cout.
// Note: Array contents are omitted for brevity. Instead, array dimensions are
// printed.
void PrintCoil(const Coil& coil, int indentation = 0);

// Print a human-readable summary of the given SerialCircuit to std::cout.
// Note: Array contents are omitted for brevity. Instead, array dimensions are
// printed.
void PrintSerialCircuit(const SerialCircuit& serial_circuit,
                        int indentation = 0);

// Print a human-readable summary of the given MagneticConfiguration to
// std::cout. Note: Array contents are omitted for brevity. Instead, array
// dimensions are printed.
void PrintMagneticConfiguration(
    const MagneticConfiguration& magnetic_configuration, int indentation = 0);

}  // namespace magnetics

#endif  // VMECPP_COMMON_MAGNETIC_CONFIGURATION_LIB_MAGNETIC_CONFIGURATION_LIB_H_
