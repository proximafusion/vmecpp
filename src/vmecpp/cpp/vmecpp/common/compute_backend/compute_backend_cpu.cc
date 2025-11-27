// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/compute_backend/compute_backend_cpu.h"

#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

namespace vmecpp {

void ComputeBackendCpu::FourierToReal(const FourierGeometry& physical_x,
                                      const std::vector<double>& xmpq,
                                      const RadialPartitioning& rp,
                                      const Sizes& s,
                                      const RadialProfiles& profiles,
                                      const FourierBasisFastPoloidal& fb,
                                      RealSpaceGeometry& m_geometry) {
  // Delegate to the existing CPU implementation.
  FourierToReal3DSymmFastPoloidal(physical_x, xmpq, rp, s, profiles, fb,
                                  m_geometry);
}

void ComputeBackendCpu::ForcesToFourier(const RealSpaceForces& forces,
                                        const std::vector<double>& xmpq,
                                        const RadialPartitioning& rp,
                                        const FlowControl& fc, const Sizes& s,
                                        const FourierBasisFastPoloidal& fb,
                                        VacuumPressureState vacuum_pressure_state,
                                        FourierForces& m_physical_forces) {
  // Delegate to the existing CPU implementation.
  ForcesToFourier3DSymmFastPoloidal(forces, xmpq, rp, fc, s, fb,
                                    vacuum_pressure_state, m_physical_forces);
}

}  // namespace vmecpp
