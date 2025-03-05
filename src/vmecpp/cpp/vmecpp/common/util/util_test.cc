// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/util/util.h"

#include <algorithm>  // for std::transform
#include <random>
#include <vector>

#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"

namespace vmecpp {

namespace {
using testing::IsCloseRelAbs;

using ::testing::Bool;
using ::testing::TestWithParam;
}  // namespace

class TridiagonalSolverSerialTest : public TestWithParam<bool> {
 protected:
  void SetUp() override { random_test_data_ = GetParam(); }
  bool random_test_data_;
};

TEST_P(TridiagonalSolverSerialTest, CheckTridiagonalSolverSerial) {
  // Standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  static constexpr double kTolerance = 1.0e-12;

  // dimensionality of system
  static constexpr int kMatrixDimension = 15;

  // number of right-hand-sides to solve for in one go
  static constexpr int kNumberOfRightHandSides = 1;

  // matrix and right-hand-side
  std::vector<double> mat_diag_low(kMatrixDimension, 0.0);
  std::vector<double> mat_diag(kMatrixDimension, 0.0);
  std::vector<double> mat_diag_up(kMatrixDimension, 0.0);

  std::vector<std::vector<double>> rhs(kNumberOfRightHandSides);
  rhs[0].resize(kMatrixDimension);

  // desired solution vector
  std::vector<double> x(kMatrixDimension, 0.0);

  // fill tri-diagonal matrix and solution vector
  if (random_test_data_) {
    for (int i = 0; i < kMatrixDimension; ++i) {
      if (i > 0) {
        mat_diag_low[i] = dist(rng);
      }
      mat_diag[i] = dist(rng);
      if (i < kMatrixDimension - 1) {
        mat_diag_up[i] = dist(rng);
      }
      x[i] = dist(rng);
    }
  } else {
    for (int i = 0; i < kMatrixDimension; ++i) {
      if (i > 0) {
        mat_diag_low[i] = 2 * i;
      }
      mat_diag[i] = (i + 1) * 10;
      if (i + 1 < kMatrixDimension) {
        mat_diag_up[i] = i + 1;
      }
      x[i] = 2 * abs(i - 4) + 1;
    }
  }

  // compute RHS for known solution
  for (int i = 0; i < kMatrixDimension; ++i) {
    if (i < kMatrixDimension - 1) {
      // sup-diagonal
      rhs[0][i] += mat_diag_up[i] * x[i + 1];
    }

    // diagonal
    rhs[0][i] += mat_diag[i] * x[i];

    if (i > 0) {
      // sub-diagonal
      rhs[0][i] += mat_diag_low[i] * x[i - 1];
    }
  }

  int j_min = 0;
  int j_max = kMatrixDimension;
  TridiagonalSolveSerial(mat_diag_up, mat_diag, mat_diag_low, rhs,
                         j_min, j_max, kNumberOfRightHandSides);

  // check that solution is correct
  for (int i = 0; i < kMatrixDimension; ++i) {
    EXPECT_TRUE(IsCloseRelAbs(x[i], rhs[0][i], kTolerance));
  }
}  // CheckTridiagonalSolverSerial

INSTANTIATE_TEST_SUITE_P(TestUtil, TridiagonalSolverSerialTest, Bool());

TEST(TestUtil, CheckTridiagonalSolveOpenMP) {
  // Standard mersenne_twister_engine
  thread_local std::mt19937 rng(42);
  thread_local std::uniform_real_distribution<> dist(-1., 1.);

  static constexpr double kTolerance = 1.0e-9;

  srand(42);

  int num_basis = 2;
  int ns = 99;

  int mpol = 12;
  int ntor = 12;
  int mnmax = mpol * (ntor + 1);

  // thread-local part of tri-diagonal matrix
  std::vector<std::vector<double>> all_ar;
  std::vector<std::vector<double>> all_az;
  std::vector<std::vector<double>> all_dr;
  std::vector<std::vector<double>> all_dz;
  std::vector<std::vector<double>> all_br;
  std::vector<std::vector<double>> all_bz;

  // on entry: RHS
  // on exit:  solutions
  std::vector<std::vector<std::vector<double>>> all_cr;
  std::vector<std::vector<std::vector<double>>> all_cz;

  // storage to hand over data between ranks
  std::vector<double> handover_ar(mnmax);
  std::vector<double> handover_az(mnmax);

  std::vector<std::vector<double>> handover_cr(num_basis);
  std::vector<std::vector<double>> handover_cz(num_basis);
  for (int k = 0; k < num_basis; ++k) {
    handover_cr[k].resize(mnmax);
    handover_cz[k].resize(mnmax);
  }

  // desired solution vectors
  std::vector<std::vector<double>> xr(num_basis);
  std::vector<std::vector<double>> xz(num_basis);

  for (int k = 0; k < num_basis; ++k) {
    xr[k].resize(ns * mnmax);
    xz[k].resize(ns * mnmax);
  }

  // fill solution vectors
  for (int k = 0; k < num_basis; ++k) {
    for (int j = 0; j < ns; ++j) {
      for (int mn = 0; mn < mnmax; ++mn) {
        int idx_mn = j * mnmax + mn;
        xr[k][idx_mn] = dist(rng) + 1.0e-12;
        xz[k][idx_mn] = dist(rng) + 1.0e-12;
      }  // mn
    }    // j
  }      // k

  std::vector<int> j_min(mnmax);
  int j_max = ns;

#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
#else
  int max_threads = 1;
#endif

  std::vector<std::mutex> mutices(max_threads);

#pragma omp parallel
  {
#ifdef _OPENMP
    int ncpu = omp_get_num_threads();
    int myid = omp_get_thread_num();
#else
    int ncpu = 1;
    int myid = 0;
#endif

#pragma omp single
    {
      all_ar.resize(ncpu);
      all_az.resize(ncpu);
      all_dr.resize(ncpu);
      all_dz.resize(ncpu);
      all_br.resize(ncpu);
      all_bz.resize(ncpu);
      all_cr.resize(ncpu);
      all_cz.resize(ncpu);
    }
    all_cr[myid].resize(num_basis);
    all_cz[myid].resize(num_basis);

    int work_per_cpu = ns / ncpu;
    int work_remainder = ns % ncpu;

    int ns_min_f = myid * work_per_cpu;
    int ns_max_f = (myid + 1) * work_per_cpu;
    if (myid < work_remainder) {
      ns_min_f += myid;
      ns_max_f += myid + 1;
    } else {
      ns_min_f += work_remainder;
      ns_max_f += work_remainder;
    }

    // last entry is never referenced
    all_ar[myid].resize((ns_max_f - ns_min_f) * mnmax);
    all_az[myid].resize((ns_max_f - ns_min_f) * mnmax);

    all_dr[myid].resize((ns_max_f - ns_min_f) * mnmax);
    all_dz[myid].resize((ns_max_f - ns_min_f) * mnmax);

    // first entry is never referenced
    all_br[myid].resize((ns_max_f - ns_min_f) * mnmax);
    all_bz[myid].resize((ns_max_f - ns_min_f) * mnmax);

    for (int k = 0; k < num_basis; ++k) {
      all_cr[myid][k].resize((ns_max_f - ns_min_f) * mnmax);
      all_cz[myid][k].resize((ns_max_f - ns_min_f) * mnmax);
    }  // k

    std::vector<double>& ar = all_ar[myid];
    std::vector<double>& az = all_az[myid];
    std::vector<double>& dr = all_dr[myid];
    std::vector<double>& dz = all_dz[myid];
    std::vector<double>& br = all_br[myid];
    std::vector<double>& bz = all_bz[myid];

    std::vector<std::span<double>> cr(all_cr[myid].begin(), all_cr[myid].end());

    std::vector<std::span<double>> cz(all_cz[myid].begin(), all_cz[myid].end());

    // do not use all_... below here

    // fill tri-diagonal matrix
    for (int j = ns_min_f; j < ns_max_f; ++j) {
      for (int mn = 0; mn < mnmax; ++mn) {
        int idx_mn = (j - ns_min_f) * mnmax + mn;
        if (j < ns - 1) {
          ar[idx_mn] = 0.1 * dist(rng) + 1.0e-12;
          az[idx_mn] = 0.1 * dist(rng) + 1.0e-12;
        }
        dr[idx_mn] = 0.1 * dist(rng) + 1.0;
        dz[idx_mn] = 0.1 * dist(rng) + 1.0;
        if (j > 0) {
          br[idx_mn] = 0.1 * dist(rng) + 1.0e-12;
          bz[idx_mn] = 0.1 * dist(rng) + 1.0e-12;
        }
      }  // mn
    }    // j

    // compute RHS for known solution
    for (int j = ns_min_f; j < ns_max_f; ++j) {
      for (int mn = 0; mn < mnmax; ++mn) {
        int local_idx = (j - ns_min_f) * mnmax + mn;

        int idx_mn_p = (j + 1) * mnmax + mn;  // +1
        int idx_mn_0 = j * mnmax + mn;        //  0
        int idx_mn_m = (j - 1) * mnmax + mn;  // -1

        for (int k = 0; k < num_basis; ++k) {
          // sup-diagonal
          if (j < ns - 1) {
            cr[k][local_idx] += ar[local_idx] * xr[k][idx_mn_p];
            cz[k][local_idx] += az[local_idx] * xz[k][idx_mn_p];
          }

          // diagonal
          cr[k][local_idx] += dr[local_idx] * xr[k][idx_mn_0];
          cz[k][local_idx] += dz[local_idx] * xz[k][idx_mn_0];

          // sub-diagonal
          if (j > 0) {
            cr[k][local_idx] += br[local_idx] * xr[k][idx_mn_m];
            cz[k][local_idx] += bz[local_idx] * xz[k][idx_mn_m];
          }
        }  // k
      }    // mn
    }      // j

#pragma omp barrier

    // solve tri-diagonal system
    TridiagonalSolveOpenMP(ar, dr, br, cr, az, dz, bz, cz, j_min, j_max, mnmax,
                           num_basis, mutices, ncpu, myid, ns_min_f, ns_max_f,
                           handover_ar, handover_cr, handover_az, handover_cz);

#pragma omp barrier

    // check that solution is correct
    for (int j = ns_min_f; j < ns_max_f; ++j) {
      for (int mn = 0; mn < mnmax; ++mn) {
        int local_idx = (j - ns_min_f) * mnmax + mn;
        int idx_mn = j * mnmax + mn;
        for (int k = 0; k < num_basis; ++k) {
          EXPECT_TRUE(
              IsCloseRelAbs(xr[k][idx_mn], cr[k][local_idx], kTolerance))
              << absl::StrFormat("j=%d mn=%d", j, mn);
          EXPECT_TRUE(
              IsCloseRelAbs(xz[k][idx_mn], cz[k][local_idx], kTolerance))
              << absl::StrFormat("j=%d mn=%d", j, mn);
        }  // k
      }    // mn
    }      // j
#pragma omp barrier
  }  // omp parallel
}  // CheckTridiagonalSolveOpenMP

}  // namespace vmecpp
