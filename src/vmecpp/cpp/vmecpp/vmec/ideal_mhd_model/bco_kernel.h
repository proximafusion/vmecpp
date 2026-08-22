// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_BCO_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_BCO_KERNEL_H_

namespace vmecpp {

// Lower the index on the contravariant field with the metric tensor:
//   bsubu = guu B^u + guv B^v
//   bsubv = guv B^u + gvv B^v
// In 2D guv is absent and drops out. Shared, allocation-free over flat buffers
// (n = number of half-grid points), between IdealMhdModel::computeBCo and the
// Enzyme autodiff path.
inline void ComputeBCo(const double* guu, const double* guv, const double* gvv,
                       const double* bsupu, const double* bsupv, bool lthreed,
                       int n, double* bsubu, double* bsubv) {
  if (lthreed) {
    for (int i = 0; i < n; ++i) {
      bsubu[i] = guu[i] * bsupu[i] + guv[i] * bsupv[i];
      bsubv[i] = guv[i] * bsupu[i] + gvv[i] * bsupv[i];
    }
  } else {
    for (int i = 0; i < n; ++i) {
      bsubu[i] = guu[i] * bsupu[i];
      bsubv[i] = gvv[i] * bsupv[i];
    }
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_BCO_KERNEL_H_
