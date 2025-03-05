// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
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
  for (int j_f = nsMin_; j_f < nsMax_; ++j_f) {
    for (int n = 0; n < s_.ntor + 1; ++n) {
      int m = 1;
      int idx_fc = ((j_f - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
      if (s_.lthreed) {
        fzcs[idx_fc] = 0.0;
      }
      if (s_.lasym) {
        fzcc[idx_fc] = 0.0;
      }
    }  // n
  }    // j
}

/** Compute the force residuals and write them into the provided [3] array. */
void FourierForces::residuals(std::vector<double>& fRes,
                              bool includeEdgeRZForces) const {
  int j_max_rz = std::min(nsMax_, ns - 1);
  if (includeEdgeRZForces && r_.nsMaxF1 == ns) {
    j_max_rz = ns;
  }

  int j_max_include_boundary = nsMax_;
  if (r_.nsMaxF1 == ns) {
    j_max_include_boundary = ns;
  }

  double local_f_res_r = 0.0;
  double local_f_res_z = 0.0;
  double local_f_res_l = 0.0;
  for (int j_f = nsMin_; j_f < j_max_include_boundary; ++j_f) {
    for (int m = 0; m < s_.mpol; ++m) {
      for (int n = 0; n < s_.ntor + 1; ++n) {
        int idx_fc = ((j_f - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

        if (j_f < j_max_rz) {
          local_f_res_r += frcc[idx_fc] * frcc[idx_fc];
          local_f_res_z += fzsc[idx_fc] * fzsc[idx_fc];
        }
        local_f_res_l += flsc[idx_fc] * flsc[idx_fc];
        if (s_.lthreed) {
          if (j_f < j_max_rz) {
            local_f_res_r += frss[idx_fc] * frss[idx_fc];
            local_f_res_z += fzcs[idx_fc] * fzcs[idx_fc];
          }
          local_f_res_l += flcs[idx_fc] * flcs[idx_fc];
        }
        if (s_.lasym) {
          if (j_f < j_max_rz) {
            local_f_res_r += frsc[idx_fc] * frsc[idx_fc];
            local_f_res_z += fzcc[idx_fc] * fzcc[idx_fc];
          }
          local_f_res_l += flcc[idx_fc] * flcc[idx_fc];
          if (s_.lthreed) {
            if (j_f < j_max_rz) {
              local_f_res_r += frcs[idx_fc] * frcs[idx_fc];
              local_f_res_z += fzss[idx_fc] * fzss[idx_fc];
            }
            local_f_res_l += flss[idx_fc] * flss[idx_fc];
          }
        }
      }  // n
    }    // m
  }      // j

  fRes[0] = local_f_res_r;
  fRes[1] = local_f_res_z;
  fRes[2] = local_f_res_l;
}

}  // namespace vmecpp
