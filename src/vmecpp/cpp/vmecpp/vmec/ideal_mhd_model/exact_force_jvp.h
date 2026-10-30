// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_JVP_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_JVP_H_

#include "vmecpp/vmec/ideal_mhd_model/local_force_composition.h"

namespace vmecpp {

// Forward-mode Jacobian-vector product of ComputeLocalForceDensity: given the
// geometry primal and tangent, returns the force-density tangent in dforce in
// one Enzyme forward pass. work/dwork and force/dforce are caller-owned scratch
// sized as ComputeLocalForceDensity requires. Defined in exact_force_jvp.cc,
// which is compiled with the Clang/Enzyme plugin.
void ExactForceDensityJvp(const double* geom, const double* dgeom, double* work,
                          double* dwork, double* force, double* dforce,
                          const LocalForceComposition* c);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_JVP_H_
