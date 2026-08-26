// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_VJP_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_VJP_H_

#include "vmecpp/vmec/ideal_mhd_model/local_force_composition.h"

namespace vmecpp {

// Reverse-mode vector-Jacobian product of ComputeLocalForceDensity: given the
// geometry primal and a force-density cotangent in force_bar, accumulates the
// geometry cotangent J_g^T force_bar into geom_bar in one Enzyme reverse pass.
// geom_bar and work_bar must be zeroed by the caller; work/work_bar/force are
// caller-owned scratch sized as ComputeLocalForceDensity requires. This is the
// transpose of ExactForceDensityJvp and is the nonlinear factor of the
// transposed exact Hessian-vector product. Defined in exact_force_vjp.cc, which
// is compiled with the Clang/Enzyme plugin.
void ExactForceDensityVjp(const double* geom, double* geom_bar, double* work,
                          double* work_bar, double* force, double* force_bar,
                          const LocalForceComposition* c);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_VJP_H_
