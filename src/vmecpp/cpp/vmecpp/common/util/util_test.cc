// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
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
  std::vector<double> matDiagLow(kMatrixDimension, 0.0);
  std::vector<double> matDiag(kMatrixDimension, 0.0);
  std::vector<double> matDiagUp(kMatrixDimension, 0.0);

  std::vector<std::vector<double>> rhs(kNumberOfRightHandSides);
  rhs[0].resize(kMatrixDimension);

  // desired solution vector
  std::vector<double> x(kMatrixDimension, 0.0);

  // fill tri-diagonal matrix and solution vector
  if (random_test_data_) {
    for (int i = 0; i < kMatrixDimension; ++i) {
      if (i > 0) {
        matDiagLow[i] = dist(rng);
      }
      matDiag[i] = dist(rng);
      if (i < kMatrixDimension - 1) {
        matDiagUp[i] = dist(rng);
      }
      x[i] = dist(rng);
    }
  } else {
    for (int i = 0; i < kMatrixDimension; ++i) {
      if (i > 0) {
        matDiagLow[i] = 2 * i;
      }
      matDiag[i] = (i + 1) * 10;
      if (i + 1 < kMatrixDimension) {
        matDiagUp[i] = i + 1;
      }
      x[i] = 2 * abs(i - 4) + 1;
    }
  }

  // compute RHS for known solution
  for (int i = 0; i < kMatrixDimension; ++i) {
    if (i < kMatrixDimension - 1) {
      // sup-diagonal
      rhs[0][i] += matDiagUp[i] * x[i + 1];
    }

    // diagonal
    rhs[0][i] += matDiag[i] * x[i];

    if (i > 0) {
      // sub-diagonal
      rhs[0][i] += matDiagLow[i] * x[i - 1];
    }
  }

  int jMin = 0;
  int jMax = kMatrixDimension;
  TridiagonalSolveSerial(matDiagUp, matDiag, matDiagLow, rhs, jMin, jMax,
                         kNumberOfRightHandSides);

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
    }  // j
  }  // k

  std::vector<int> jMin(mnmax);
  int jMax = ns;

#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
#else
  int max_threads = 1;
#endif

  std::vector<std::mutex> mutices(max_threads);

#ifdef _OPENMP
#pragma omp parallel
#endif  // _OPENMP
  {
#ifdef _OPENMP
    int ncpu = omp_get_num_threads();
    int myid = omp_get_thread_num();
#else
    int ncpu = 1;
    int myid = 0;
#endif

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
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

    int work_per_CPU = ns / ncpu;
    int work_remainder = ns % ncpu;

    int nsMinF = myid * work_per_CPU;
    int nsMaxF = (myid + 1) * work_per_CPU;
    if (myid < work_remainder) {
      nsMinF += myid;
      nsMaxF += myid + 1;
    } else {
      nsMinF += work_remainder;
      nsMaxF += work_remainder;
    }

    // last entry is never referenced
    all_ar[myid].resize((nsMaxF - nsMinF) * mnmax);
    all_az[myid].resize((nsMaxF - nsMinF) * mnmax);

    all_dr[myid].resize((nsMaxF - nsMinF) * mnmax);
    all_dz[myid].resize((nsMaxF - nsMinF) * mnmax);

    // first entry is never referenced
    all_br[myid].resize((nsMaxF - nsMinF) * mnmax);
    all_bz[myid].resize((nsMaxF - nsMinF) * mnmax);

    for (int k = 0; k < num_basis; ++k) {
      all_cr[myid][k].resize((nsMaxF - nsMinF) * mnmax);
      all_cz[myid][k].resize((nsMaxF - nsMinF) * mnmax);
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
    for (int j = nsMinF; j < nsMaxF; ++j) {
      for (int mn = 0; mn < mnmax; ++mn) {
        int idx_mn = (j - nsMinF) * mnmax + mn;
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
    }  // j

    // compute RHS for known solution
    for (int j = nsMinF; j < nsMaxF; ++j) {
      for (int mn = 0; mn < mnmax; ++mn) {
        int local_idx = (j - nsMinF) * mnmax + mn;

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
      }  // mn
    }  // j

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

    // solve tri-diagonal system
    TridiagonalSolveOpenMP(ar, dr, br, cr, az, dz, bz, cz, jMin, jMax, mnmax,
                           num_basis, mutices, ncpu, myid, nsMinF, nsMaxF,
                           handover_ar, handover_cr, handover_az, handover_cz);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

    // check that solution is correct
    for (int j = nsMinF; j < nsMaxF; ++j) {
      for (int mn = 0; mn < mnmax; ++mn) {
        int local_idx = (j - nsMinF) * mnmax + mn;
        int idx_mn = j * mnmax + mn;
        for (int k = 0; k < num_basis; ++k) {
          EXPECT_TRUE(
              IsCloseRelAbs(xr[k][idx_mn], cr[k][local_idx], kTolerance))
              << absl::StrFormat("j=%d mn=%d", j, mn);
          EXPECT_TRUE(
              IsCloseRelAbs(xz[k][idx_mn], cz[k][local_idx], kTolerance))
              << absl::StrFormat("j=%d mn=%d", j, mn);
        }  // k
      }  // mn
    }  // j
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
  }  // omp parallel
}  // CheckTridiagonalSolveOpenMP

}  // namespace vmecpp
