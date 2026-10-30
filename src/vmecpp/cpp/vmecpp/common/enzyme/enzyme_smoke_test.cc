// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Toolchain smoke test for the Enzyme autodiff build.
//
// It checks the two properties the differentiable VMEC++ kernels rely on:
//   1. The ClangEnzyme plugin is attached and resolves the autodiff intrinsics.
//   2. Enzyme differentiates a scalar objective expressed over Eigen::Map'd
//      caller buffers (the pattern all differentiable kernels here use).
//
// The objective is f(x) = sum_i 0.5 * w_i * x_i^2 + c_i * x_i, with the closed
// form gradient df/dx_i = w_i * x_i + c_i. Reverse- and forward-mode Enzyme
// gradients are checked against that closed form and against central finite
// differences. Exit code 0 on success, 1 on any mismatch.

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>

#include "vmecpp/common/enzyme/enzyme.h"

namespace {

using Eigen::VectorXd;

// Quadratic objective over caller-owned buffers. No allocating Eigen
// expression temporaries cross the differentiated boundary.
__attribute__((noinline)) double Objective(const double* x, const double* w,
                                           const double* c, int n) {
  Eigen::Map<const VectorXd> xv(x, n);
  Eigen::Map<const VectorXd> wv(w, n);
  Eigen::Map<const VectorXd> cv(c, n);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += 0.5 * wv[i] * xv[i] * xv[i] + cv[i] * xv[i];
  }
  return sum;
}

double MaxAbsDiff(const VectorXd& a, const VectorXd& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

}  // namespace

int main() {
  const int n = 8;
  VectorXd x = VectorXd::LinSpaced(n, 1.0, n);
  VectorXd w = VectorXd::LinSpaced(n, 2.0, 2.0 + 0.5 * (n - 1));
  VectorXd c = VectorXd::Constant(n, 0.25);

  VectorXd analytic(n);
  for (int i = 0; i < n; ++i) analytic[i] = w[i] * x[i] + c[i];

  // Reverse mode: gradient accumulates into the shadow buffer.
  VectorXd g_rev = VectorXd::Zero(n);
  __enzyme_autodiff((void*)Objective, enzyme_dup, x.data(), g_rev.data(),
                    enzyme_const, w.data(), enzyme_const, c.data(),
                    enzyme_const, n);

  // Forward mode: one directional derivative per coordinate seed.
  VectorXd g_fwd = VectorXd::Zero(n);
  for (int j = 0; j < n; ++j) {
    VectorXd seed = VectorXd::Zero(n);
    seed[j] = 1.0;
    g_fwd[j] = __enzyme_fwddiff<double>(
        (void*)Objective, enzyme_dup, x.data(), seed.data(), enzyme_const,
        w.data(), enzyme_const, c.data(), enzyme_const, n);
  }

  // Central finite differences.
  VectorXd g_fd = VectorXd::Zero(n);
  const double h = 1e-6;
  for (int j = 0; j < n; ++j) {
    VectorXd xp = x, xm = x;
    xp[j] += h;
    xm[j] -= h;
    g_fd[j] = (Objective(xp.data(), w.data(), c.data(), n) -
               Objective(xm.data(), w.data(), c.data(), n)) /
              (2.0 * h);
  }

  const double err_rev = MaxAbsDiff(g_rev, analytic);
  const double err_fwd = MaxAbsDiff(g_fwd, analytic);
  const double err_fd = MaxAbsDiff(g_rev, g_fd);

  std::printf("enzyme smoke test (n=%d)\n", n);
  std::printf("  max|reverse - analytic| = %.3e\n", err_rev);
  std::printf("  max|forward - analytic| = %.3e\n", err_fwd);
  std::printf("  max|reverse - finite-diff| = %.3e\n", err_fd);

  const bool ok = err_rev < 1e-10 && err_fwd < 1e-10 && err_fd < 1e-5;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
