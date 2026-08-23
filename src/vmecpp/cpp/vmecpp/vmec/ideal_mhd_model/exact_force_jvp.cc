// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Forward-mode Jacobian-vector product of the local force-density composition.
// This translation unit is compiled with the Clang/Enzyme plugin so that the
// __enzyme_fwddiff call differentiates ComputeLocalForceDensity. The rest of
// VMEC++ is compiled normally; the exact Hessian-vector product calls into here
// for the single nonlinear pass and wraps it with the linear spectral
// transforms.

#include "vmecpp/vmec/ideal_mhd_model/exact_force_jvp.h"

#include "vmecpp/common/enzyme/enzyme.h"
#include "vmecpp/vmec/ideal_mhd_model/local_force_composition.h"

namespace vmecpp {

void ExactForceDensityJvp(const double* geom, const double* dgeom, double* work,
                          double* dwork, double* force, double* dforce,
                          const LocalForceComposition* c) {
  __enzyme_fwddiff<void>(reinterpret_cast<void*>(ComputeLocalForceDensity),
                         enzyme_dup, geom, dgeom, enzyme_dup, work, dwork,
                         enzyme_dup, force, dforce, enzyme_const, c);
}

}  // namespace vmecpp
