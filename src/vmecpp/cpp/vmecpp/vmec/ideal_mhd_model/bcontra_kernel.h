// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_BCONTRA_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_BCONTRA_KERNEL_H_

namespace vmecpp {

// Half-grid contravariant magnetic field from the (already lambda-normalized)
// angular lambda derivatives and the Jacobian gsqrt:
//   bsupv = 0.5 [ (lu_in + lu_out) + sqrtSH (luo_in + luo_out) ] / gsqrt
//   bsupu = 0.5 [ (lv_in + lv_out) + sqrtSH (lvo_in + lvo_out) ] / gsqrt   (3D)
// In 2D bsupu starts at zero here and the chip'/gsqrt term is added by the
// caller. Shared, allocation-free over flat buffers, between
// IdealMhdModel::computeBContra and the Enzyme autodiff path. Same indexing
// conventions as jacobian_kernel.h. lue/luo/lve/lvo are the normalized lambda
// arrays (lu already includes the + phi' term applied by the caller).
inline void ComputeBsupContra(const double* lue, const double* luo,
                              const double* lve, const double* lvo,
                              const double* gsqrt, const double* sqrtSH,
                              bool lthreed, int nZnT, int nsMinF1, int nsMinH,
                              int nsMaxH, double* bsupu, double* bsupv) {
  for (int jH = nsMinH; jH < nsMaxH; ++jH) {
    const double sH = sqrtSH[jH - nsMinH];
    for (int kl = 0; kl < nZnT; ++kl) {
      const int i_in = (jH - nsMinF1) * nZnT + kl;
      const int i_out = (jH + 1 - nsMinF1) * nZnT + kl;
      const int ih = (jH - nsMinH) * nZnT + kl;

      if (lthreed) {
        bsupu[ih] = 0.5 *
                    ((lve[i_in] + lve[i_out]) + sH * (lvo[i_in] + lvo[i_out])) /
                    gsqrt[ih];
      } else {
        bsupu[ih] = 0.0;
      }
      bsupv[ih] = 0.5 *
                  ((lue[i_in] + lue[i_out]) + sH * (luo[i_in] + luo[i_out])) /
                  gsqrt[ih];
    }
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_BCONTRA_KERNEL_H_
