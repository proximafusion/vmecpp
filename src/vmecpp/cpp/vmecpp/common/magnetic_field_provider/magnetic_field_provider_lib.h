// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_MAGNETIC_FIELD_PROVIDER_MAGNETIC_FIELD_PROVIDER_LIB_H_
#define VMECPP_COMMON_MAGNETIC_FIELD_PROVIDER_MAGNETIC_FIELD_PROVIDER_LIB_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "vmecpp/common/composed_types_definition/composed_types.h"
#include "vmecpp/common/magnetic_configuration_definition/magnetic_configuration.h"

namespace magnetics {

// Compute the magnetic field due to a given InfiniteStraightFilament and a
// given current at given set of evaluation locations. A fatal error occurs if
// the direction vector of the InfiniteStraightFilament has zero length or if
// any of the evaluation positions is located exactly on the filament. The
// magnetic field result is added to the contents of the provided vector for
// easy computation of superpositions. The magnetic field only has been modified
// if an ok status is returned.
absl::Status MagneticField(
    const InfiniteStraightFilament &infinite_straight_filament, double current,
    const std::vector<std::vector<double> > &evaluation_positions,
    std::vector<std::vector<double> > &m_magnetic_field,
    bool check_current_carrier = true);

// Compute the magnetic field due to a given CircularFilament and a given
// current at given set of evaluation locations. The magnetic field result is
// added to the contents of the provided vector for easy computation of
// superpositions. The magnetic field only has been modified if an ok status is
// returned.
absl::Status MagneticField(
    const CircularFilament &circular_filament, double current,
    const std::vector<std::vector<double> > &evaluation_positions,
    std::vector<std::vector<double> > &m_magnetic_field,
    bool check_current_carrier = true);

// Compute the magnetic field due to a given PolygonFilament and a given current
// at given set of evaluation locations. The magnetic field result is added to
// the contents of the provided vector for easy computation of superpositions.
// The magnetic field only has been modified if an ok status is returned.
absl::Status MagneticField(
    const PolygonFilament &polygon_filament, double current,
    const std::vector<std::vector<double> > &evaluation_positions,
    std::vector<std::vector<double> > &m_magnetic_field,
    bool check_current_carrier = true);

// Compute the net magnetic field due to a given MagneticConfiguration at given
// set of evaluation locations. The magnetic field result is added to the
// contents of the provided vector for easy computation of superpositions. The
// magnetic field only has been modified if an ok status is returned.
absl::Status MagneticField(
    const MagneticConfiguration &magnetic_configuration,
    const std::vector<std::vector<double> > &evaluation_positions,
    std::vector<std::vector<double> > &m_magnetic_field,
    bool check_current_carrier = true);

// ----------------

// The magnetic vector potential diverges for an infinite straight filament,
// so there is no method to compute a contribution from it here.

// Compute the magnetic vector potential due to a given CircularFilament and a
// given current at given set of evaluation locations. The magnetic vector
// potential result is added to the contents of the provided vector for easy
// computation of superpositions. The magnetic vector potential only has been
// modified if an ok status is returned.
absl::Status VectorPotential(
    const CircularFilament &circular_filament, double current,
    const std::vector<std::vector<double> > &evaluation_positions,
    std::vector<std::vector<double> > &m_vector_potential,
    bool check_current_carrier = true);

// Compute the magnetic vector potential due to a given PolygonFilament and a
// given current at given set of evaluation locations. The magnetic vector
// potential result is added to the contents of the provided vector for easy
// computation of superpositions. The magnetic vector potential only has been
// modified if an ok status is returned.
absl::Status VectorPotential(
    const PolygonFilament &polygon_filament, double current,
    const std::vector<std::vector<double> > &evaluation_positions,
    std::vector<std::vector<double> > &m_vector_potential,
    bool check_current_carrier = true);

// Compute the net magnetic vector potential due to a given
// MagneticConfiguration at given set of evaluation locations. The magnetic
// vector potential result is added to the contents of the provided vector for
// easy computation of superpositions. The magnetic vector potential only has
// been modified if an ok status is returned.
absl::Status VectorPotential(
    const MagneticConfiguration &magnetic_configuration,
    const std::vector<std::vector<double> > &evaluation_positions,
    std::vector<std::vector<double> > &m_vector_potential,
    bool check_current_carrier = true);

// Compute the linking current between a given magnetic configuration
// and a given closed curve, e.g., the magnetic axis.
// The number of sampling points along the axis is chosen
// automatically to be above 2 times the Nyquist limit.
absl::StatusOr<double> LinkingCurrent(
    const MagneticConfiguration &magnetic_configuration,
    const composed_types::CurveRZFourier &axis_coefficients);

}  // namespace magnetics

#endif  // VMECPP_COMMON_MAGNETIC_FIELD_PROVIDER_MAGNETIC_FIELD_PROVIDER_LIB_H_
