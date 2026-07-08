// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_CONSTRAINT_FORCE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_CONSTRAINT_FORCE_KERNEL_H_

#include <algorithm>

namespace vmecpp {

// The two local (non-transform) pieces of VMEC's spectral-condensation
// constraint force. The Fourier-space bandpass between them is the separate
// shared free function deAliasConstraintForce. All full-grid buffers are
// indexed (jF-nsMinF)*nZnT except sqrtSF, which is indexed (jF-nsMinF1).
// Allocation-free, shared between the solver and the Enzyme autodiff path.

// Effective constraint force gConEff = (rCon - rCon0) ru + (zCon - zCon0) zu.
// No constraint on the axis (it has no poloidal angle), so the axis surface is
// skipped, matching deAliasConstraintForce.
inline void ComputeEffectiveConstraintForce(
    const double* rCon, const double* rCon0, const double* zCon,
    const double* zCon0, const double* ruFull, const double* zuFull, int nZnT,
    int nsMinF, int nsMaxFIncludingLcfs, double* gConEff) {
  int jMin = 0;
  if (nsMinF == 0) {
    jMin = 1;
  }
  for (int jF = std::max(jMin, nsMinF); jF < nsMaxFIncludingLcfs; ++jF) {
    for (int kl = 0; kl < nZnT; ++kl) {
      int idx_kl = (jF - nsMinF) * nZnT + kl;
      gConEff[idx_kl] = (rCon[idx_kl] - rCon0[idx_kl]) * ruFull[idx_kl] +
                        (zCon[idx_kl] - zCon0[idx_kl]) * zuFull[idx_kl];
    }  // kl
  }  // jF
}

// Add the bandpass-filtered constraint force gCon back into the MHD R/Z forces
// (brmn, bzmn) and write the constraint-force outputs (frcon, fzcon).
inline void AddConstraintForces(
    const double* rCon, const double* rCon0, const double* zCon,
    const double* zCon0, const double* ruFull, const double* zuFull,
    const double* gCon, const double* sqrtSF, int nZnT, int nsMinF, int nsMinF1,
    int nsMaxF, double* brmn_e, double* brmn_o, double* bzmn_e, double* bzmn_o,
    double* frcon_e, double* frcon_o, double* fzcon_e, double* fzcon_o) {
  for (int jF = nsMinF; jF < nsMaxF; ++jF) {
    for (int kl = 0; kl < nZnT; ++kl) {
      int idx_kl = (jF - nsMinF) * nZnT + kl;

      double brcon = (rCon[idx_kl] - rCon0[idx_kl]) * gCon[idx_kl];
      double bzcon = (zCon[idx_kl] - zCon0[idx_kl]) * gCon[idx_kl];

      brmn_e[idx_kl] += brcon;
      bzmn_e[idx_kl] += bzcon;
      brmn_o[idx_kl] += brcon * sqrtSF[jF - nsMinF1];
      bzmn_o[idx_kl] += bzcon * sqrtSF[jF - nsMinF1];

      frcon_e[idx_kl] = ruFull[idx_kl] * gCon[idx_kl];
      fzcon_e[idx_kl] = zuFull[idx_kl] * gCon[idx_kl];
      frcon_o[idx_kl] = frcon_e[idx_kl] * sqrtSF[jF - nsMinF1];
      fzcon_o[idx_kl] = fzcon_e[idx_kl] * sqrtSF[jF - nsMinF1];
    }  // kl
  }  // jF
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_CONSTRAINT_FORCE_KERNEL_H_
