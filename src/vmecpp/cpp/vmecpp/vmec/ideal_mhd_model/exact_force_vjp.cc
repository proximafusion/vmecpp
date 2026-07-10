// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Reverse-mode vector-Jacobian product of the local force-density composition.
// Compiled with the Clang/Enzyme plugin so that __enzyme_autodiff
// differentiates ComputeLocalForceDensity in reverse. This is the transpose
// J_g^T of the forward pass in exact_force_jvp.cc; the transposed exact
// Hessian-vector product wraps it with the transposed linear spectral
// transforms.

#include "vmecpp/vmec/ideal_mhd_model/exact_force_vjp.h"

#include "vmecpp/common/enzyme/enzyme.h"
#include "vmecpp/vmec/ideal_mhd_model/local_force_composition.h"

namespace vmecpp {

void ExactForceDensityVjp(const double* geom, double* geom_bar, double* work,
                          double* work_bar, double* force, double* force_bar,
                          const LocalForceComposition* c) {
  // force_bar carries the output cotangent (seed); geom_bar accumulates
  // J_g^T force_bar. geom_bar and work_bar are zeroed by the caller.
  __enzyme_autodiff(reinterpret_cast<void*>(ComputeLocalForceDensity),
                    enzyme_dup, geom, geom_bar, enzyme_dup, work, work_bar,
                    enzyme_dup, force, force_bar, enzyme_const, c);
}

}  // namespace vmecpp
