// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "util/cubic_spline/cubic_spline.h"

#include <Eigen/Dense>
#include <cmath>

#include "absl/status/status.h"
#include "gtest/gtest.h"

using cubic_spline::CubicSpline;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

// Return maximum absolute error between spline and reference f(x)
// sampled at 'm' equally spaced points inside (x0, xn).
template <typename F>
static double MaxError(const CubicSpline& s, const Eigen::VectorXd& x,
                       std::size_t m, F f) {
  double x0 = x(0);
  double xn = x(x.size() - 1);
  double h = (xn - x0) / static_cast<double>(m + 1);
  double max_err = 0.0;

  for (std::size_t i = 1; i <= m; ++i) {
    double xi = x0 + i * h;
    double si = *s.Evaluate(xi);
    double fi = f(xi);
    max_err = std::max(max_err, std::fabs(si - fi));
  }
  return max_err;
}

// ------------------------------------------------------------------
// Analytic reference function and derivative
// ------------------------------------------------------------------
static double f_ref(double x) { return std::sin(x); }
static double fp_ref(double x) { return std::cos(x); }

// ------------------------------------------------------------------
// Test data: non-uniform grid on [0, 2]
// ------------------------------------------------------------------
static Eigen::VectorXd MakeKnots() {
  Eigen::VectorXd x(8);
  x << 0.0, 0.25, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0;
  return x;
}

static Eigen::VectorXd MakeValues(const Eigen::VectorXd& x) {
  return x.array().sin();
}

// ------------------------------------------------------------------
// Branch-coverage macros
// ------------------------------------------------------------------
#define EXPECT_OK(expr)                          \
  do {                                           \
    auto _st_or = (expr);                        \
    EXPECT_TRUE(_st_or.ok()) << _st_or.status(); \
  } while (false)

#define EXPECT_ERR(expr, expected_code)               \
  do {                                                \
    auto _st_or = (expr);                             \
    EXPECT_FALSE(_st_or.ok());                        \
    EXPECT_EQ(_st_or.status().code(), expected_code); \
  } while (false)

// ------------------------------------------------------------------
// 1.  Constructor failure cases
// ------------------------------------------------------------------
TEST(CubicSpline, ConstructorErrors) {
  Eigen::VectorXd x_good = MakeKnots();
  Eigen::VectorXd y_bad = Eigen::VectorXd::Ones(3);  // size mismatch
  CubicSpline::Boundary b;

  auto bad1 = CubicSpline::Create(x_good, y_bad, b, b);
  EXPECT_FALSE(bad1.ok());

  Eigen::VectorXd x_bad = x_good;
  x_bad(3) = x_bad(2);  // not strictly increasing
  auto bad2 = CubicSpline::Create(x_bad, MakeValues(x_bad), b, b);
  EXPECT_FALSE(bad2.ok());
}

// ------------------------------------------------------------------
// 2.  Out-of-range evaluation
// ------------------------------------------------------------------
TEST(CubicSpline, EvaluateOutOfRange) {
  Eigen::VectorXd x = MakeKnots();
  Eigen::VectorXd y = MakeValues(x);

  CubicSpline::Boundary b;
  auto spline_or = CubicSpline::Create(x, y, b, b);
  ASSERT_TRUE(spline_or.ok());
  CubicSpline s = *spline_or;

  EXPECT_ERR(s.Evaluate(-0.1), absl::StatusCode::kOutOfRange);
  EXPECT_ERR(s.Evaluate(5.0), absl::StatusCode::kOutOfRange);
}

// ------------------------------------------------------------------
// Every interpolatory spline must match the data at the knots.
// ------------------------------------------------------------------
TEST(CubicSpline, KnotsExact) {
  Eigen::VectorXd x = MakeKnots();
  Eigen::VectorXd y = MakeValues(x);

  CubicSpline::Boundary l;  // natural / natural
  CubicSpline::Boundary r;

  CubicSpline s = *CubicSpline::Create(x, y, l, r);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(*s.Evaluate(x(i)), y(i), 1e-14);
  }
}

// ------------------------------------------------------------------
// 3.  TriDiagonal solver path  (Natural + Natural)
// ------------------------------------------------------------------
TEST(CubicSpline, TriDiagonalSolverAccuracy) {
  Eigen::VectorXd x = MakeKnots();
  Eigen::VectorXd y = MakeValues(x);

  CubicSpline::Boundary l, r;  // natural/natural -> tridiagonal

  CubicSpline s = *CubicSpline::Create(x, y, l, r);
  CubicSpline ds = s.Derivative();

  // With 8 knots over length 2, a natural spline on a non-uniform grid
  // has global O(h^4) accuracy ~1e-3 and derivative O(h^3).
  EXPECT_LT(MaxError(s, x, 50, f_ref), 2e-3);
  EXPECT_LT(MaxError(ds, x, 50, fp_ref), 5e-2);
}

// ------------------------------------------------------------------
// 4.  AntiSymmetric + Natural (shares natural code path)
// ------------------------------------------------------------------
TEST(CubicSpline, AntiSymmetricLeft) {
  Eigen::VectorXd x = MakeKnots();
  Eigen::VectorXd y = MakeValues(x);

  auto s_or = CubicSpline::Create(
      x, y, CubicSpline::Boundary(CubicSpline::BcType::kAntiSymmetric),
      CubicSpline::Boundary(CubicSpline::BcType::kNatural));

  ASSERT_TRUE(s_or.ok());
  CubicSpline s = *s_or;

  EXPECT_NEAR(*s.Evaluate(1.1), std::sin(1.1), 1e-4);
}

// ------------------------------------------------------------------
// 5.  Clamped + Clamped  (user slopes, still tri-diagonal)
// ------------------------------------------------------------------
TEST(CubicSpline, ClampedBothEnds) {
  Eigen::VectorXd x = MakeKnots();
  Eigen::VectorXd y = MakeValues(x);

  double slope_left = std::cos(0.0);
  double slope_right = std::cos(2.0);

  CubicSpline::Boundary left(CubicSpline::BcType::kClamped, slope_left);
  CubicSpline::Boundary right(CubicSpline::BcType::kClamped, slope_right);

  auto s_or = CubicSpline::Create(x, y, left, right);
  ASSERT_TRUE(s_or.ok());
  CubicSpline s = *s_or;

  EXPECT_NEAR(*s.Evaluate(0.3), std::sin(0.3), 1e-4);
  EXPECT_NEAR(*s.Derivative().Evaluate(0.3), std::cos(0.3), 5e-3);
}

// ------------------------------------------------------------------
// 6.  Not-a-knot left  (band-width 2 -> sparse LU path)
// ------------------------------------------------------------------
TEST(CubicSpline, NotAKnotLeftSparseSolve) {
  Eigen::VectorXd x = MakeKnots();
  Eigen::VectorXd y = MakeValues(x);

  auto s_or = CubicSpline::Create(
      x, y, CubicSpline::Boundary(CubicSpline::BcType::kNotAKnot),
      CubicSpline::Boundary(CubicSpline::BcType::kNatural));

  ASSERT_TRUE(s_or.ok());
  CubicSpline s = *s_or;

  EXPECT_NEAR(*s.Evaluate(1.7), std::sin(1.7), 1e-3);
}

// ------------------------------------------------------------------
// 7.  Not-a-knot right (band-width 2 -> sparse LU path)
// ------------------------------------------------------------------
TEST(CubicSpline, NotAKnotRightSparseSolve) {
  Eigen::VectorXd x = MakeKnots();
  Eigen::VectorXd y = MakeValues(x);

  auto s_or = CubicSpline::Create(
      x, y, CubicSpline::Boundary(CubicSpline::BcType::kSymmetric),
      CubicSpline::Boundary(CubicSpline::BcType::kNotAKnot));

  ASSERT_TRUE(s_or.ok());
  CubicSpline s = *s_or;

  EXPECT_NEAR(*s.Evaluate(1.2), std::sin(1.2), 1e-3);
}

// ------------------------------------------------------------------
// 8.  Periodic + Periodic  (wrap-around entries, sparse LU)
// ------------------------------------------------------------------
TEST(CubicSpline, PeriodicBothEnds) {
  // Use exactly one period of sin on [0, 2*pi]
  const double L = 2.0 * M_PI;
  // use 17 (or 33) knots over one full period; (power-of-two + 1) keeps
  // uniformity
  const int N = 17;
  Eigen::VectorXd x(N);
  for (int i = 0; i < N; ++i) {
    x(i) = i * L / (N - 1);
  }
  Eigen::VectorXd y = x.array().sin();

  auto s_or = CubicSpline::Create(
      x, y, CubicSpline::Boundary(CubicSpline::BcType::kPeriodic),
      CubicSpline::Boundary(CubicSpline::BcType::kPeriodic));

  ASSERT_TRUE(s_or.ok());
  CubicSpline s = *s_or;

  // Evaluate midway to stay strictly inside the domain
  double xi = 0.5 * L;
  EXPECT_NEAR(*s.Evaluate(xi), std::sin(xi), 1e-8);

  // Periodic derivative should equal cos as well
  CubicSpline ds = s.Derivative();
  EXPECT_NEAR(*ds.Evaluate(xi), std::cos(xi), 1e-3);
}
