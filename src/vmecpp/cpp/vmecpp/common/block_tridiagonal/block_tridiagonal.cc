// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/block_tridiagonal/block_tridiagonal.h"

#include "absl/log/check.h"

namespace vmecpp {

BlockTridiagonalFactorization::BlockTridiagonalFactorization(
    const std::vector<Eigen::MatrixXd>& lower,
    const std::vector<Eigen::MatrixXd>& diag,
    const std::vector<Eigen::MatrixXd>& upper)
    : n_(static_cast<int>(diag.size())),
      k_(diag.empty() ? 0 : static_cast<int>(diag[0].rows())) {
  CHECK_GT(n_, 0) << "block-tridiagonal system needs at least one block row";
  CHECK_EQ(static_cast<int>(lower.size()), n_);
  CHECK_EQ(static_cast<int>(upper.size()), n_);

  delta_inverse_.resize(n_);
  multiplier_.resize(n_);
  upper_.resize(n_);

  // Forward elimination of the sub-diagonal (block LU / block-Thomas):
  //   Delta_0 = D_0
  //   M_j     = L_j Delta_{j-1}^{-1}
  //   Delta_j = D_j - M_j U_{j-1}
  Eigen::MatrixXd delta = diag[0];
  CHECK_EQ(delta.rows(), k_);
  CHECK_EQ(delta.cols(), k_);
  delta_inverse_[0] = delta.inverse();
  upper_[0] = upper[0];
  for (int j = 1; j < n_; ++j) {
    CHECK_EQ(diag[j].rows(), k_);
    CHECK_EQ(diag[j].cols(), k_);
    multiplier_[j] = lower[j] * delta_inverse_[j - 1];
    delta = diag[j] - multiplier_[j] * upper[j - 1];
    delta_inverse_[j] = delta.inverse();
    upper_[j] = upper[j];
  }
}

std::vector<Eigen::VectorXd> BlockTridiagonalFactorization::Solve(
    const std::vector<Eigen::VectorXd>& b) const {
  CHECK_EQ(static_cast<int>(b.size()), n_);

  // Forward substitution: y_0 = b_0, y_j = b_j - M_j y_{j-1}.
  std::vector<Eigen::VectorXd> y(n_);
  y[0] = b[0];
  for (int j = 1; j < n_; ++j) {
    y[j] = b[j] - multiplier_[j] * y[j - 1];
  }

  // Back substitution: x_{n-1} = Delta_{n-1}^{-1} y_{n-1},
  //                    x_j     = Delta_j^{-1} (y_j - U_j x_{j+1}).
  std::vector<Eigen::VectorXd> x(n_);
  x[n_ - 1] = delta_inverse_[n_ - 1] * y[n_ - 1];
  for (int j = n_ - 2; j >= 0; --j) {
    x[j] = delta_inverse_[j] * (y[j] - upper_[j] * x[j + 1]);
  }
  return x;
}

BlockTridiagonalBlocks AssembleFdBlockTridiagonal(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& force,
    const Eigen::VectorXd& x0, int n, int k, double eps) {
  CHECK_GT(n, 0);
  CHECK_GT(k, 0);
  CHECK_EQ(x0.size(), static_cast<Eigen::Index>(n) * k);

  BlockTridiagonalBlocks h;
  h.lower.assign(n, Eigen::MatrixXd::Zero(k, k));
  h.diag.assign(n, Eigen::MatrixXd::Zero(k, k));
  h.upper.assign(n, Eigen::MatrixXd::Zero(k, k));

  const auto at = [k](int surface, int dof) { return surface * k + dof; };

  // For each degree of freedom, perturb it on every third surface at once
  // (those surfaces do not share any affected force row), central-difference
  // the force, and read the resulting column into the diagonal / off-diagonal
  // blocks of the affected rows.
  for (int dof = 0; dof < k; ++dof) {
    for (int offset = 0; offset < 3 && offset < n; ++offset) {
      Eigen::VectorXd x_plus = x0;
      Eigen::VectorXd x_minus = x0;
      for (int j = offset; j < n; j += 3) {
        x_plus[at(j, dof)] += eps;
        x_minus[at(j, dof)] -= eps;
      }
      const Eigen::VectorXd df = (force(x_plus) - force(x_minus)) / (2.0 * eps);
      for (int j = offset; j < n; j += 3) {
        h.diag[j].col(dof) = df.segment(at(j, 0), k);
        if (j > 0) {
          h.upper[j - 1].col(dof) = df.segment(at(j - 1, 0), k);
        }
        if (j < n - 1) {
          h.lower[j + 1].col(dof) = df.segment(at(j + 1, 0), k);
        }
      }
    }
  }
  return h;
}

}  // namespace vmecpp
