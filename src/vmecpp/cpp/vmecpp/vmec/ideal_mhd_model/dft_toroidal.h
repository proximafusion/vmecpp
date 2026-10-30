// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_TOROIDAL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_TOROIDAL_H_

#include <Eigen/Dense>

#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {

void FourierToReal3DSymmFastPoloidal(const FourierGeometry& physical_x,
                                     const Eigen::VectorXd& xmpq,
                                     const RadialPartitioning& r,
                                     const Sizes& s, const RadialProfiles& rp,
                                     const FourierBasisFastPoloidal& fb,
                                     RealSpaceGeometry& m_geometry);

void ForcesToFourier3DSymmFastPoloidal(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

// Non-stellarator-symmetric (lasym) counterparts. The inverse accumulates the
// antisymmetric-parity geometry into the *_asym arrays carried by m_geometry;
// the forward projects the antisymmetric-parity force halves onto the
// frsc / fzcc / flcc coefficients. Both are the cos<->sin mirror of the
// symmetric functions above (educational_VMEC totzspa / tomnspa).
void FourierToReal3DAsymFastPoloidal(const FourierGeometry& physical_x,
                                     const Eigen::VectorXd& xmpq,
                                     const RadialPartitioning& r,
                                     const Sizes& s, const RadialProfiles& rp,
                                     const FourierBasisFastPoloidal& fb,
                                     RealSpaceGeometry& m_geometry);

void ForcesToFourier3DAsymFastPoloidal(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_TOROIDAL_H_
