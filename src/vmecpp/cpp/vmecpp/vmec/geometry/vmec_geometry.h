// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_GEOMETRY_VMEC_GEOMETRY_H_
#define VMECPP_VMEC_GEOMETRY_VMEC_GEOMETRY_H_

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/geometry/geometry.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"

namespace vmecpp {

// State of the R/Z product-basis coefficients supplied to MakeGeometry.
// GatherDataFromThreads returns the solver's m=1-constrained state, whereas
// ComputeOutputQuantities returns the physical coefficients after conversion.
enum class GeometryCoefficientState {
  kSolver,
  kPhysical,
};

Geometry MakeGeometry(
    const VmecINDATA& indata, const VmecInternalResults& internal,
    GeometryCoefficientState state = GeometryCoefficientState::kSolver);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_GEOMETRY_VMEC_GEOMETRY_H_
