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

Geometry MakeGeometry(const VmecINDATA& indata,
                      const VmecInternalResults& internal);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_GEOMETRY_VMEC_GEOMETRY_H_
