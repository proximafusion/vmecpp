// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/block_tridiagonal/block_tridiagonal.h"

#include <random>
#include <vector>

#include <Eigen/Dense>

#include "gtest/gtest.h"

namespace vmecpp {
namespace {

// The block-tridiagonal factorization solves L_j x_{j-1} + D_j x_j + U_j x_{j+1}
// = b_j exactly (up to round-off). Check it against a known solution and against
// the dense (n*k) x (n*k) solve, over a range of block counts and block sizes.
TEST(BlockTridiagonalTest, MatchesKnownSolutionAndDenseSolve) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto rand_mat = [&](int r, int c) {
    return Eigen::MatrixXd::NullaryExpr(r, c, [&] { return dist(rng); });
  };
  const auto rand_vec = [&](int r) {
    return Eigen::VectorXd::NullaryExpr(r, [&] { return dist(rng); });
  };

  for (int trial = 0; trial < 20; ++trial) {
    const int n = 3 + (trial % 8);
    const int k = 2 + (trial % 5);

    std::vector<Eigen::MatrixXd> lower(n), diag(n), upper(n);
    for (int j = 0; j < n; ++j) {
      lower[j] = rand_mat(k, k);
      upper[j] = rand_mat(k, k);
      // Diagonally dominant, so the operator is nonsingular and well
      // conditioned.
      diag[j] = rand_mat(k, k) + (2.0 * k) * Eigen::MatrixXd::Identity(k, k);
    }

    std::vector<Eigen::VectorXd> x_true(n);
    for (int j = 0; j < n; ++j) x_true[j] = rand_vec(k);
    std::vector<Eigen::VectorXd> b(n);
    for (int j = 0; j < n; ++j) {
      b[j] = diag[j] * x_true[j];
      if (j > 0) b[j] += lower[j] * x_true[j - 1];
      if (j < n - 1) b[j] += upper[j] * x_true[j + 1];
    }

    const BlockTridiagonalFactorization fac(lower, diag, upper);
    const std::vector<Eigen::VectorXd> x = fac.Solve(b);

    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n * k, n * k);
    Eigen::VectorXd rhs(n * k);
    for (int j = 0; j < n; ++j) {
      dense.block(j * k, j * k, k, k) = diag[j];
      if (j > 0) dense.block(j * k, (j - 1) * k, k, k) = lower[j];
      if (j < n - 1) dense.block(j * k, (j + 1) * k, k, k) = upper[j];
      rhs.segment(j * k, k) = b[j];
    }
    const Eigen::VectorXd x_dense = dense.fullPivLu().solve(rhs);

    ASSERT_EQ(static_cast<int>(x.size()), n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < k; ++i) {
        EXPECT_NEAR(x[j](i), x_true[j](i), 1e-9) << "trial " << trial;
        EXPECT_NEAR(x[j](i), x_dense(j * k + i), 1e-9) << "trial " << trial;
      }
    }
  }
}

// A linear force F(x) = A x has Jacobian A exactly, so the finite-difference
// assembler must recover the blocks of a block-tridiagonal A to round-off.
TEST(BlockTridiagonalTest, FdAssemblyRecoversLinearJacobian) {
  std::mt19937 rng(777);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto rand_mat = [&](int r, int c) {
    return Eigen::MatrixXd::NullaryExpr(r, c, [&] { return dist(rng); });
  };
  const auto rand_vec = [&](int r) {
    return Eigen::VectorXd::NullaryExpr(r, [&] { return dist(rng); });
  };

  for (int trial = 0; trial < 6; ++trial) {
    const int n = 4 + trial;
    const int k = 3;

    std::vector<Eigen::MatrixXd> lower(n), diag(n), upper(n);
    for (int j = 0; j < n; ++j) {
      lower[j] = Eigen::MatrixXd::Zero(k, k);
      upper[j] = Eigen::MatrixXd::Zero(k, k);
      if (j > 0) lower[j] = rand_mat(k, k);
      if (j < n - 1) upper[j] = rand_mat(k, k);
      diag[j] = rand_mat(k, k) + (2.0 * k) * Eigen::MatrixXd::Identity(k, k);
    }
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n * k, n * k);
    for (int j = 0; j < n; ++j) {
      dense.block(j * k, j * k, k, k) = diag[j];
      if (j > 0) dense.block(j * k, (j - 1) * k, k, k) = lower[j];
      if (j < n - 1) dense.block(j * k, (j + 1) * k, k, k) = upper[j];
    }
    const auto force = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
      return dense * x;
    };

    const BlockTridiagonalBlocks h =
        AssembleFdBlockTridiagonal(force, rand_vec(n * k), n, k, 1e-6);

    for (int j = 0; j < n; ++j) {
      EXPECT_LT((h.diag[j] - diag[j]).cwiseAbs().maxCoeff(), 1e-6)
          << "diag trial " << trial;
      if (j > 0) {
        EXPECT_LT((h.lower[j] - lower[j]).cwiseAbs().maxCoeff(), 1e-6)
            << "lower trial " << trial;
      }
      if (j < n - 1) {
        EXPECT_LT((h.upper[j] - upper[j]).cwiseAbs().maxCoeff(), 1e-6)
            << "upper trial " << trial;
      }
    }
  }
}

}  // namespace
}  // namespace vmecpp
