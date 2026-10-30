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

// Reverse-mode product of ComputeLocalForceDensity with respect to the
// half-grid profiles c->presH, c->chipH and c->currH at fixed geometry: given
// a force-density cotangent in force_bar, accumulates the profile cotangents
// into presH_bar, chipH_bar and currH_bar (index jH-nsMinH, zeroed by the
// caller). work/work_bar/force are caller-owned scratch as in
// ExactForceDensityVjp.
void ExactForceDensityProfileVjp(const double* geom, double* work,
                                 double* work_bar, double* force,
                                 double* force_bar,
                                 const LocalForceComposition* c,
                                 double* presH_bar, double* chipH_bar,
                                 double* currH_bar);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_EXACT_FORCE_VJP_H_
