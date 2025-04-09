// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"

#include <algorithm>
#include <vector>

namespace vmecpp {

FourierForces::FourierForces(const Sizes* s, const RadialPartitioning* r,
                             int ns)
    : FourierCoeffs(s, r, r->nsMinF, r->nsMaxF, ns),
      frcc(rcc),
      frss(rss),
      frsc(rsc),
      frcs(rcs),
      fzsc(zsc),
      fzcs(zcs),
      fzcc(zcc),
      fzss(zss),
      flsc(lsc),
      flcs(lcs),
      flcc(lcc),
      flss(lss) {}

void FourierForces::zeroZForceForM1() {
  for (int jF = nsMin_; jF < nsMax_; ++jF) {
    for (int n = 0; n < s_.ntor + 1; ++n) {
      int m = 1;
      int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
      if (s_.lthreed) {
        fzcs[idx_fc] = 0.0;
      }
      if (s_.lasym) {
        fzcc[idx_fc] = 0.0;
      }
    }  // n
  }  // j
}

/** Compute the force residuals and write them into the provided [3] array. */
void FourierForces::residuals(std::vector<double>& fRes,
                              bool includeEdgeRZForces) const {
  int jMaxRZ = std::min(nsMax_, ns - 1);
  if (includeEdgeRZForces && r_.nsMaxF1 == ns) {
    jMaxRZ = ns;
  }

  int jMaxIncludeBoundary = nsMax_;
  if (r_.nsMaxF1 == ns) {
    jMaxIncludeBoundary = ns;
  }

  double local_fResR = 0.0;
  double local_fResZ = 0.0;
  double local_fResL = 0.0;
  for (int jF = nsMin_; jF < jMaxIncludeBoundary; ++jF) {
    for (int m = 0; m < s_.mpol; ++m) {
      for (int n = 0; n < s_.ntor + 1; ++n) {
        int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

        if (jF < jMaxRZ) {
          local_fResR += frcc[idx_fc] * frcc[idx_fc];
          local_fResZ += fzsc[idx_fc] * fzsc[idx_fc];
        }
        local_fResL += flsc[idx_fc] * flsc[idx_fc];
        if (s_.lthreed) {
          if (jF < jMaxRZ) {
            local_fResR += frss[idx_fc] * frss[idx_fc];
            local_fResZ += fzcs[idx_fc] * fzcs[idx_fc];
          }
          local_fResL += flcs[idx_fc] * flcs[idx_fc];
        }
        if (s_.lasym) {
          if (jF < jMaxRZ) {
            local_fResR += frsc[idx_fc] * frsc[idx_fc];
            local_fResZ += fzcc[idx_fc] * fzcc[idx_fc];
          }
          local_fResL += flcc[idx_fc] * flcc[idx_fc];
          if (s_.lthreed) {
            if (jF < jMaxRZ) {
              local_fResR += frcs[idx_fc] * frcs[idx_fc];
              local_fResZ += fzss[idx_fc] * fzss[idx_fc];
            }
            local_fResL += flss[idx_fc] * flss[idx_fc];
          }
        }
      }  // n
    }  // m
  }  // j

  fRes[0] = local_fResR;
  fRes[1] = local_fResZ;
  fRes[2] = local_fResL;
}

}  // namespace vmecpp
