// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_coefficients/fourier_coefficients.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"

namespace vmecpp {

FourierCoeffs::FourierCoeffs(const Sizes* s, const RadialPartitioning* r,
                             int nsMin, int nsMax, int ns)
    : s_(s), r_(r), nsMin_(nsMin), nsMax_(nsMax), ns_(ns) {
  // cannot use r_->nsMaxFIncludingLcfs since need to still obey nsMax
  int j_max_including_boundary = nsMax;
  if (r_->nsMaxF1 == ns_) {
    j_max_including_boundary = ns_;
  }

  int num_fc_rz =
      (j_max_including_boundary - nsMin) * s_->mpol * (s_->ntor + 1);
  int num_fc_l = (j_max_including_boundary - nsMin) * s_->mpol * (s_->ntor + 1);

  rcc.resize(num_fc_rz, 0.0);
  zsc.resize(num_fc_rz, 0.0);
  lsc.resize(num_fc_l, 0.0);
  if (s_->lthreed) {
    rss.resize(num_fc_rz, 0.0);
    zcs.resize(num_fc_rz, 0.0);
    lcs.resize(num_fc_l, 0.0);
  }
  if (s_->lasym) {
    rsc.resize(num_fc_rz, 0.0);
    zcc.resize(num_fc_rz, 0.0);
    lcc.resize(num_fc_l, 0.0);
    if (s_->lthreed) {
      rcs.resize(num_fc_rz, 0.0);
      zss.resize(num_fc_rz, 0.0);
      lss.resize(num_fc_l, 0.0);
    }
  }
}

int FourierCoeffs::nsMin() const { return nsMin_; }

int FourierCoeffs::nsMax() const { return nsMax_; }

void FourierCoeffs::setZero() {
  absl::c_fill(rcc, 0);
  absl::c_fill(zsc, 0);
  absl::c_fill(lsc, 0);
  if (s_->lthreed) {
    absl::c_fill(rss, 0);
    absl::c_fill(zcs, 0);
    absl::c_fill(lcs, 0);
  }
  if (s_->lasym) {
    absl::c_fill(rsc, 0);
    absl::c_fill(zcc, 0);
    absl::c_fill(lcc, 0);
    if (s_->lthreed) {
      absl::c_fill(rcs, 0);
      absl::c_fill(zss, 0);
      absl::c_fill(lss, 0);
    }
  }
}

/** apply even/odd-m decomposition */
void FourierCoeffs::decomposeInto(FourierCoeffs& m_x,
                                  const std::vector<double>& scalxc) const {
  // TODO(jons): understand correct limits in fixed-boundary vs. free-boundary
  int j_max_including_boundary = nsMax_;
  if (r_->nsMaxF1 == ns_) {
    j_max_including_boundary = ns_;
  }

  // int j_max_rz = nsMax;
  int j_max_rz = j_max_including_boundary;

  for (int j_f = nsMin_; j_f < j_max_including_boundary; ++j_f) {
    for (int m = 0; m < s_->mpol; ++m) {
      for (int n = 0; n < s_->ntor + 1; ++n) {
        int idx_fc = ((j_f - nsMin_) * s_->mpol + m) * (s_->ntor + 1) + n;

        int m_parity = m % 2;

        // scalxc is always defined on numFull1
        double scal = scalxc[(j_f - r_->nsMinF1) * 2 + m_parity];

        if (j_f < j_max_rz) {
          m_x.rcc[idx_fc] = rcc[idx_fc] * scal;
          m_x.zsc[idx_fc] = zsc[idx_fc] * scal;
        }
        m_x.lsc[idx_fc] = lsc[idx_fc] * scal;
        if (s_->lthreed) {
          if (j_f < j_max_rz) {
            m_x.rss[idx_fc] = rss[idx_fc] * scal;
            m_x.zcs[idx_fc] = zcs[idx_fc] * scal;
          }
          m_x.lcs[idx_fc] = lcs[idx_fc] * scal;
        }
        if (s_->lasym) {
          if (j_f < j_max_rz) {
            m_x.rsc[idx_fc] = rsc[idx_fc] * scal;
            m_x.zcc[idx_fc] = zcc[idx_fc] * scal;
          }
          m_x.lcc[idx_fc] = lcc[idx_fc] * scal;
          if (s_->lthreed) {
            if (j_f < j_max_rz) {
              m_x.rcs[idx_fc] = rcs[idx_fc] * scal;
              m_x.zss[idx_fc] = zss[idx_fc] * scal;
            }
            m_x.lss[idx_fc] = lss[idx_fc] * scal;
          }
        }
      }  // n
    }  // m
  }  // j_f
}

/** (un)do m=1 constraint to couple R_ss,Z_cs as well as R_sc,Z_cc */
void FourierCoeffs::m1Constraint(double scalingFactor,
                                 std::optional<int> jMax) {
  int ns_max_to_use = nsMax_;
  if (jMax.has_value()) {
    ns_max_to_use = std::min(jMax.value(), ns_max_to_use);
  }

  for (int j_f = nsMin_; j_f < ns_max_to_use; ++j_f) {
    for (int n = 0; n < s_->ntor + 1; ++n) {
      int m = 1;
      int idx_fc = ((j_f - nsMin_) * s_->mpol + m) * (s_->ntor + 1) + n;
      if (s_->lthreed) {
        double old_rss = rss[idx_fc];
        rss[idx_fc] = (old_rss + zcs[idx_fc]) * scalingFactor;
        zcs[idx_fc] = (old_rss - zcs[idx_fc]) * scalingFactor;
      }
      if (s_->lasym) {
        double old_rsc = rsc[idx_fc];
        rsc[idx_fc] = (old_rsc + zcc[idx_fc]) * scalingFactor;
        zcc[idx_fc] = (old_rsc - zcc[idx_fc]) * scalingFactor;
      }
    }  // n
  }  // j_f
}

double FourierCoeffs::rzNorm(bool include_offset, int nsMinHere,
                             int nsMaxHere) const {
  // accumulator for local thread
  double local_norm2 = 0.0;

  for (int j_f = nsMinHere; j_f < nsMaxHere; ++j_f) {
    for (int m = 0; m < s_->mpol; ++m) {
      for (int n = 0; n < s_->ntor + 1; ++n) {
        int idx_fc = ((j_f - nsMin_) * s_->mpol + m) * (s_->ntor + 1) + n;

        if (n > 0 || m > 0 || include_offset) {
          local_norm2 += rcc[idx_fc] * rcc[idx_fc];
        }
        local_norm2 += zsc[idx_fc] * zsc[idx_fc];
        if (s_->lthreed) {
          local_norm2 += rss[idx_fc] * rss[idx_fc];
          local_norm2 += zcs[idx_fc] * zcs[idx_fc];
        }
        if (s_->lasym) {
          local_norm2 += rsc[idx_fc] * rsc[idx_fc];
          if (n > 0 || m > 0 || include_offset) {
            local_norm2 += zcc[idx_fc] * zcc[idx_fc];
          }
          if (s_->lthreed) {
            local_norm2 += rcs[idx_fc] * rcs[idx_fc];
            local_norm2 += zss[idx_fc] * zss[idx_fc];
          }
        }
      }  // n
    }  // m
  }  // j

  return local_norm2;
}

double FourierCoeffs::GetXcElement(int rzl, int idx_basis, int j_f, int n,
                                   int m) const {
  int idx_fc = ((j_f - nsMin_) * s_->mpol + m) * (s_->ntor + 1) + n;

  if (rzl == 0) {
    if (idx_basis == 0) {
      return rcc[idx_fc];
    }
    if (s_->lthreed) {
      if (idx_basis == 1) {
        return rss[idx_fc];
      }
    }
    if (s_->lasym) {
      if ((!s_->lthreed && idx_basis == 1) || (s_->lthreed && idx_basis == 2)) {
        return rsc[idx_fc];
      }
      if (s_->lthreed) {
        if (idx_basis == 3) {
          return rcs[idx_fc];
        }
      }
    }
  } else if (rzl == 1) {
    if (idx_basis == 0) {
      return zsc[idx_fc];
    }
    if (s_->lthreed) {
      if (idx_basis == 1) {
        return zcs[idx_fc];
      }
    }
    if (s_->lasym) {
      if ((!s_->lthreed && idx_basis == 1) || (s_->lthreed && idx_basis == 2)) {
        return zcc[idx_fc];
      }
      if (s_->lthreed) {
        if (idx_basis == 3) {
          return zss[idx_fc];
        }
      }
    }
  } else if (rzl == 2) {
    if (idx_basis == 0) {
      return lsc[idx_fc];
    }
    if (s_->lthreed) {
      if (idx_basis == 1) {
        return lcs[idx_fc];
      }
    }
    if (s_->lasym) {
      if ((!s_->lthreed && idx_basis == 1) || (s_->lthreed && idx_basis == 2)) {
        return lcc[idx_fc];
      }
      if (s_->lthreed) {
        if (idx_basis == 3) {
          return lss[idx_fc];
        }
      }
    }
  }

  std::stringstream error_message;
  error_message << "did not find";
  error_message << " rzl=" << rzl;
  error_message << " idx_basis=" << idx_basis;
  error_message << " j_f=" << j_f;
  error_message << " n=" << n;
  error_message << " m=" << m;
  LOG(FATAL) << error_message.str();

  // should never reach this
  return 0.0;
}

}  // namespace vmecpp
