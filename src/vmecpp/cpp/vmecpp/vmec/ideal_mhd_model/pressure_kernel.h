// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_PRESSURE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_PRESSURE_KERNEL_H_

namespace vmecpp {

// Magnetic pressure |B|^2/2 = 0.5 (B^u B_u + B^v B_v) at each half-grid point.
// This is the field-dependent (nonlinear) part of pressureAndEnergies; the
// kinetic pressure profile and the energy volume integrals stay in the solver.
// Shared, allocation-free over flat buffers (n = number of half-grid points),
// between IdealMhdModel::pressureAndEnergies and the Enzyme autodiff path.
inline void ComputeMagneticPressure(const double* bsupu, const double* bsubu,
                                    const double* bsupv, const double* bsubv,
                                    int n, double* total_pressure) {
  for (int i = 0; i < n; ++i) {
    total_pressure[i] = 0.5 * (bsupu[i] * bsubu[i] + bsupv[i] * bsubv[i]);
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_PRESSURE_KERNEL_H_
