// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/util/util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"

namespace vmecpp {

int VmecStatusCode(const VmecStatus vmec_status) {
  // from https://stackoverflow.com/a/11421471
  return static_cast<std::underlying_type<VmecStatus>::type>(vmec_status);
}  // VmecErrorCode

std::string VmecStatusAsString(const VmecStatus vmec_status) {
  switch (vmec_status) {
    case VmecStatus::NORMAL_TERMINATION:
      return "NORMAL_TERMINATION";
    case VmecStatus::BAD_JACOBIAN:
      return "BAD_JACOBIAN";
    case VmecStatus::NCURR_NE_1_BLOAT_NE_1:
      return "NCURR_NE_1_BLOAT_NE_1";
    case VmecStatus::JACOBIAN_75_TIMES_BAD:
      return "JACOBIAN_75_TIMES_BAD";
    case VmecStatus::INPUT_PARSING_ERROR:
      return "INPUT_PARSING_ERROR";
    case VmecStatus::PHIEDGE_WRONG_SIGN:
      return "PHIEDGE_WRONG_SIGN";
    case VmecStatus::NS_ERROR:
      return "NS_ERROR";
    case VmecStatus::MISC_ERROR:
      return "MISC_ERROR";
    case VmecStatus::VAC_VMEC_ITOR_MISMATCH:
      return "VAC_VMEC_ITOR_MISMATCH";
    case VmecStatus::SUCCESSFUL_TERMINATION:
      return "SUCCESSFUL_TERMINATION";
  }

  // never reached
  return "UNKNOWN??";
}

// -1 if x<0, 0 if x==0, +1 if x>0
int signum(int x) {
  // https://stackoverflow.com/a/1903975
  return (x > 0) - (x < 0);
}

void TridiagonalSolveSerial(std::vector<double> &m_a, std::vector<double> &m_d,
                            std::vector<double> &m_b,
                            std::vector<std::vector<double>> &m_c, int jMin,
                            int jMax, int nRHS) {
  // If jMin > 0, need to fill in the upper left corner of the matrix
  // with non-intrusive variables that do not change the solution of the system.
  // Thus, put the diagonal elements d to 1 (= implied identity matrix),
  // the off-diagonal matrix elements a and b to 0
  // and the right-hand-side c also to 0.
  for (int j = 0; j < jMin; ++j) {
    m_a[j] = 0.0;
    m_d[j] = 1.0;
    m_b[j] = 0.0;
    for (int k = 0; k < nRHS; ++k) {
      m_c[k][j] = 0.0;
    }  // k
  }  // j

  // Thomas algorithm from
  // https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
  // This works in-place and does not need extra arrays !

  if (m_d[jMin] == 0.0) {
    LOG(FATAL) << "d[jMin] == 0.0 at jMin = " << jMin;
  }
  m_a[jMin] /= m_d[jMin];

  for (int j = jMin + 1; j < jMax - 1; ++j) {
    const double denominator = m_d[j] - m_a[j - 1] * m_b[j];
    if (denominator == 0.0) {
      LOG(FATAL) << "d[j] - a[j - 1] * b[j] == 0.0 at j = " << j;
    }
    m_a[j] /= denominator;
  }  // j

  for (int k = 0; k < nRHS; ++k) {
    // d has not been modified here, so no need to re-check for division-by-zero
    m_c[k][jMin] /= m_d[jMin];

    for (int j = jMin + 1; j < jMax; ++j) {
      const double denominator = m_d[j] - m_a[j - 1] * m_b[j];
      if (denominator == 0.0) {
        LOG(FATAL) << "d[j] - a[j - 1] * b[j] == 0.0 at j = " << j;
      }
      m_c[k][j] = (m_c[k][j] - m_c[k][j - 1] * m_b[j]) / denominator;
    }  // j

    for (int j = jMax - 2; j > jMin - 1; --j) {
      m_c[k][j] -= m_a[j] * m_c[k][j + 1];
    }  // j
  }  // k
}

void TridiagonalSolveOpenMP(
    std::vector<double> &m_ar, std::vector<double> &m_dr,
    std::vector<double> &m_br, std::vector<std::span<double>> &m_cr,
    std::vector<double> &m_az, std::vector<double> &m_dz,
    std::vector<double> &m_bz, std::vector<std::span<double>> &m_cz,
    const std::vector<int> &jMin, int jMax, int mnmax, int nRHS,
    std::vector<std::mutex> &m_mutices, int ncpu, int myid, int nsMinF,
    int nsMaxF, std::vector<double> &m_handover_ar,
    std::vector<std::vector<double>> &m_handover_cr,
    std::vector<double> &m_handover_az,
    std::vector<std::vector<double>> &m_handover_cz) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  if (myid == 0) {
    // this is the rank that has jMin;
    // fill in dummy values into unused entries
    for (int j = 0; j < nsMaxF; ++j) {
      for (int mn = 0; mn < mnmax; ++mn) {
        if (j < jMin[mn]) {  // continue; }
          int idx_mn = (j - nsMinF) * mnmax + mn;
          m_ar[idx_mn] = 0.0;
          m_az[idx_mn] = 0.0;
          m_dr[idx_mn] = 1.0;
          m_dz[idx_mn] = 1.0;
          m_br[idx_mn] = 0.0;
          m_bz[idx_mn] = 0.0;
        }
      }  // mn
    }  // j
    for (int k = 0; k < nRHS; ++k) {
      for (int j = 0; j < nsMaxF; ++j) {
        for (int mn = 0; mn < mnmax; ++mn) {
          if (j < jMin[mn]) {  // continue; }
            int idx_mn = (j - nsMinF) * mnmax + mn;
            m_cr[k][idx_mn] = 0.0;
            m_cz[k][idx_mn] = 0.0;
          }
        }  // mn
      }  // j
    }  // k

    // check that d[jMin] != 0 and if so,
    // compute a[jMin] /= d[jMin] and c[*][jMin] /= d[jMin]
    for (int mn = 0; mn < mnmax; ++mn) {
      int idx_mn = (jMin[mn] - nsMinF) * mnmax + mn;
      if (m_dr[idx_mn] == 0.0) {
        LOG(FATAL) << absl::StrFormat("m_dr[jMin=%d, mn=%d] == 0", jMin[mn],
                                      mn);
      }
      if (m_dz[idx_mn] == 0.0) {
        LOG(FATAL) << absl::StrFormat("m_dz[jMin=%d, mn=%d] == 0", jMin[mn],
                                      mn);
      }
      m_ar[idx_mn] /= m_dr[idx_mn];
      m_az[idx_mn] /= m_dz[idx_mn];
      for (int k = 0; k < nRHS; ++k) {
        m_cr[k][idx_mn] /= m_dr[idx_mn];
        m_cz[k][idx_mn] /= m_dz[idx_mn];
      }  // k
    }  // mn
  }  // myid == 0

  // have each thread take the lock that blocks the next thread
  if (myid + 1 < ncpu) {
    m_mutices[myid + 1].lock();
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // blocks until previous thread is done with this step
  // thread 0 is free to go, since its mutex is unlocked
  m_mutices[myid].lock();

  for (int j = nsMinF; j < std::min(jMax, nsMaxF); ++j) {
    for (int mn = 0; mn < mnmax; ++mn) {
      if (j < jMin[mn] + 1) {
        continue;
      }  // TODO(jons)

      int idx_mn_0 = (j - nsMinF) * mnmax + mn;      //  0
      int idx_mn_m = (j - 1 - nsMinF) * mnmax + mn;  // -1

      double prev_m_ar;
      double prev_az;
      std::vector<std::span<double>> prev_cr(m_handover_cr.size());
      std::vector<std::span<double>> prev_cz(m_handover_cz.size());
      int prev_c_idx;
      if (j == nsMinF) {
        // j-1 is within rank myid-1
        prev_m_ar = m_handover_ar[mn];
        prev_az = m_handover_az[mn];  // all_a[myid-1][(myid-1)->nsMaxF-1][mn]
        for (std::size_t i = 0; i < m_handover_cr.size(); ++i) {
          prev_cr[i] = m_handover_cr[i];
          prev_cz[i] = m_handover_cz[i];
        }

        // // needs to point to (myid-1)->nsMaxF-1 in previous rank
        prev_c_idx = mn;
      } else {
        // j-1 is within current rank
        prev_m_ar = m_ar[idx_mn_m];
        prev_az = m_az[idx_mn_m];
        prev_c_idx = idx_mn_m;
        for (std::size_t i = 0; i < m_cr.size(); ++i) {
          prev_cr[i] = m_cr[i];
          prev_cz[i] = m_cz[i];
        }
      }

      double denom_r = m_dr[idx_mn_0] - prev_m_ar * m_br[idx_mn_0];
      double denom_z = m_dz[idx_mn_0] - prev_az * m_bz[idx_mn_0];

      // make sure to not divide by zero in the following
      if (denom_r == 0.0) {
        LOG(FATAL) << absl::StrFormat("denom_r at j=%d, mn=%d is 0", j, mn);
      }
      if (denom_z == 0.0) {
        LOG(FATAL) << absl::StrFormat("denom_z at j=%d, mn=%d is 0", j, mn);
      }

      if (j < jMax - 1) {
        m_ar[idx_mn_0] /= denom_r;
        m_az[idx_mn_0] /= denom_z;
      }

      for (int k = 0; k < nRHS; ++k) {
        m_cr[k][idx_mn_0] =
            (m_cr[k][idx_mn_0] - prev_cr[k][prev_c_idx] * m_br[idx_mn_0]) /
            denom_r;
        m_cz[k][idx_mn_0] =
            (m_cz[k][idx_mn_0] - prev_cz[k][prev_c_idx] * m_bz[idx_mn_0]) /
            denom_z;
      }  // k

      if (j == nsMaxF - 1) {
        // need to leave handover data for rank myid+1
        m_handover_ar[mn] = m_ar[idx_mn_0];
        m_handover_az[mn] = m_az[idx_mn_0];
        for (int k = 0; k < nRHS; ++k) {
          m_handover_cr[k][mn] = m_cr[k][idx_mn_0];
          m_handover_cz[k][mn] = m_cz[k][idx_mn_0];
        }  // k
      }
    }  // mn
  }  // j

  // first unlock current mutex
  m_mutices[myid].unlock();

  if (myid + 1 < ncpu) {
    // ... now can unlock next one and next thread will immediately continue
    m_mutices[myid + 1].unlock();
  }

  // have each thread lock the mutex that blocks the _previous_ thread
  if (myid > 0) {
    m_mutices[myid - 1].lock();
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  m_mutices[myid].lock();

  for (int j = std::min(jMax - 2, nsMaxF - 1); j >= nsMinF; --j) {
    for (int mn = 0; mn < mnmax; ++mn) {
      if (j < jMin[mn]) {
        continue;
      }  // TODO(jons)

      int idx_mn_p = (j + 1 - nsMinF) * mnmax + mn;  // +1
      int idx_mn_0 = (j - nsMinF) * mnmax + mn;      //  0

      std::vector<std::span<double>> prev_cr(m_handover_cr.size());
      std::vector<std::span<double>> prev_cz(m_handover_cz.size());
      int prev_c_idx;
      if (j == nsMaxF - 1) {
        // j+1 is within rank myid+1
        for (std::size_t i = 0; i < m_handover_cr.size(); ++i) {
          prev_cr[i] = m_handover_cr[i];
          prev_cz[i] = m_handover_cz[i];
        }

        // needs to point to (myid+1)->nsMinF
        prev_c_idx = mn;
      } else {
        // j+1 is within current rank
        for (std::size_t i = 0; i < m_cr.size(); ++i) {
          prev_cr[i] = m_cr[i];
          prev_cz[i] = m_cz[i];
        }
        prev_c_idx = idx_mn_p;
      }

      for (int k = 0; k < nRHS; ++k) {
        m_cr[k][idx_mn_0] -= m_ar[idx_mn_0] * prev_cr[k][prev_c_idx];
        m_cz[k][idx_mn_0] -= m_az[idx_mn_0] * prev_cz[k][prev_c_idx];
      }  // k

      if (j == nsMinF) {
        // need to leave handover data for rank myid-1
        for (int k = 0; k < nRHS; ++k) {
          m_handover_cr[k][mn] = m_cr[k][idx_mn_0];
          m_handover_cz[k][mn] = m_cz[k][idx_mn_0];
        }  // k
      }
    }  // mn
  }  // j

  // first unlock current mutex
  m_mutices[myid].unlock();

  if (myid > 0) {
    // ... now can unlock next one and next thread will immediately continue
    m_mutices[myid - 1].unlock();
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // TriDiagonalSolveOpenMP

int vmec_adjust_num_threads(const int max_threads,
                            const int num_surfaces_to_distribute) {
  // Objective: Distribute num_surfaces_to_distribute among max_threads threads.
  // A minimum of 2 flux surfaces per thread is allowed
  // to have at least a single shared half-grid point in between them.
  // --> maximum number of usable threads for plasma == floor(ns / 2), as done
  // by integer divide
  int num_threads = std::min(max_threads, num_surfaces_to_distribute / 2);

#ifdef _OPENMP
  // This must be done _before_ the '#pragma omp parallel' is entered.
  omp_set_num_threads(num_threads);

  // Explicitly turn off dynamic threads.
  omp_set_dynamic(0);
#endif

  return num_threads;
}

}  // namespace vmecpp
