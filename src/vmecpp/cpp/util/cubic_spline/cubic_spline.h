// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#ifndef UTIL_CUBIC_SPLINE_CUBIC_SPLINE_H_
#define UTIL_CUBIC_SPLINE_CUBIC_SPLINE_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>  // upper_bound
#include <cstddef>    // size_t
#include <limits>     // numeric_limits
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace cubic_spline {

// Cubic spline interpolation with user-selectable boundary conditions
// and automatic choice of tridiagonal versus sparse solve.
class CubicSpline {
 public:
  // Enumeration of the available boundary types.
  enum class BcType {
    kSymmetric,      // S'(x) = 0
    kAntiSymmetric,  // S''(x) = 0   (same as "left natural")
    kNatural,        // S''(x) = 0
    kClamped,        // S'(x) = given slope
    kNotAKnot,       // 3rd derivative continuous
    kPeriodic        // function and two derivatives periodic
  };

  // Convenience structure to pass one boundary.
  struct Boundary {
    BcType type;
    double slope;  // only used when type == kClamped

    Boundary(BcType t = BcType::kNatural, double s = 0.0) : type(t), slope(s) {}
  };

  // Factory: creates and initialises the spline.
  static absl::StatusOr<CubicSpline> Create(const Eigen::VectorXd& x,
                                            const Eigen::VectorXd& y,
                                            const Boundary& left,
                                            const Boundary& right) {
    CubicSpline s;
    s.x_ = x;
    s.y_ = y;
    absl::Status st = s.Build(left, right);
    if (!st.ok()) {
      return st;
    }
    return s;
  }

  // ---------------------------------------------------------------------
  // Evaluate spline at x_query (inclusive domain).
  // ---------------------------------------------------------------------
  absl::StatusOr<double> Evaluate(double x_query) const {
    std::size_t n = static_cast<std::size_t>(x_.size());

    // Fast check for the two end points
    if (x_query == x_(0)) {
      return y_(0);
    }
    if (x_query == x_(static_cast<Eigen::Index>(n - 1))) {
      return y_(static_cast<Eigen::Index>(n - 1));
    }

    // Locate interval i such that  x_i <= x_query < x_{i+1}
    auto it = std::upper_bound(x_.data(), x_.data() + n, x_query);

    if (it == x_.data() || it == x_.data() + n) {
      return absl::OutOfRangeError("x_query outside data range");
    }

    std::size_t i = static_cast<std::size_t>(it - x_.data() - 1);
    double dx = x_query - x_[static_cast<Eigen::Index>(i)];

    // Horner evaluation of the cubic on interval i
    double val = a_[i] + dx * (b_[i] + dx * (c_[i] + dx * d_[i]));
    return val;
  }

  // ---------------------------------------------------------------------
  // Return a spline that represents d/dx of *this*.
  // ---------------------------------------------------------------------
  CubicSpline Derivative() const {
    const std::size_t n = static_cast<std::size_t>(x_.size());
    std::size_t segs = n - 1;

    CubicSpline ds;

    // Copy knot locations.
    ds.x_ = x_;

    // Allocate coefficient arrays.
    ds.a_.resize(segs);
    ds.b_.resize(segs);
    ds.c_.resize(segs);
    ds.d_.resize(segs);

    // ------------------------------------------------------------------
    // For S_i(x) = a_i + b_i dx + c_i dx^2 + d_i dx^3,
    // S'_i(x) = b_i + 2 c_i dx + 3 d_i dx^2.
    // Store that in the same cubic format with d' = 0.
    // ------------------------------------------------------------------
    for (std::size_t i = 0; i < segs; ++i) {
      ds.a_[i] = b_[i];
      ds.b_[i] = 2.0 * c_[i];
      ds.c_[i] = 3.0 * d_[i];
      ds.d_[i] = 0.0;  // cubic term vanishes
    }

    // Build y_ values so Evaluate() works on ds as usual.
    ds.y_.resize(n);

    // Left-side derivatives for the first n-1 knots.
    for (std::size_t i = 0; i < n - 1; ++i) {
      ds.y_(static_cast<Eigen::Index>(i)) = b_[i];
    }

    // Derivative at the last knot: evaluate right end of last segment.
    double h_last = x_(static_cast<Eigen::Index>(n - 1)) -
                    x_(static_cast<Eigen::Index>(n - 2));

    ds.y_(static_cast<Eigen::Index>(n - 1)) = b_[n - 2] +
                                              2.0 * c_[n - 2] * h_last +
                                              3.0 * d_[n - 2] * h_last * h_last;

    return ds;
  }

 private:
  CubicSpline() = default;

  // ======================================================================
  // Build the spline: assemble the linear system and solve for M_i
  // ======================================================================
  absl::Status Build(const Boundary& left_bc, const Boundary& right_bc) {
    const std::size_t n = static_cast<std::size_t>(x_.size());

    // ---------- sanity checks -------------------------------------------
    if (n != static_cast<std::size_t>(y_.size())) {
      return absl::InvalidArgumentError("x and y must have equal length");
    }
    if (n < 3) {
      return absl::InvalidArgumentError("need at least three data points");
    }
    for (std::size_t i = 1; i < n; ++i) {
      if (!(x_[static_cast<Eigen::Index>(i)] >
            x_[static_cast<Eigen::Index>(i - 1)])) {
        return absl::InvalidArgumentError("x must be strictly increasing");
      }
    }

    // ---------- interval widths h[i] = x[i+1] - x[i] --------------------
    Eigen::VectorXd h(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i) {
      h[static_cast<Eigen::Index>(i)] = x_[static_cast<Eigen::Index>(i + 1)] -
                                        x_[static_cast<Eigen::Index>(i)];
    }

    // ---------- detect if the system is strictly tridiagonal -----------
    auto BcIsTri = [](BcType t) {
      return t == BcType::kSymmetric || t == BcType::kAntiSymmetric ||
             t == BcType::kNatural || t == BcType::kClamped;
    };

    bool tri_system = BcIsTri(left_bc.type) && BcIsTri(right_bc.type);

    // ---------- storage for RHS and matrix ------------------------------
    Eigen::VectorXd rhs(n);

    // For tridiagonal: three diagonals as dense vectors.
    Eigen::VectorXd lower, diag, upper;

    // For general case: triplet list for sparse assembly.
    std::vector<Eigen::Triplet<double>> triplets;

    if (tri_system) {
      // allocate three diagonals
      lower = Eigen::VectorXd::Zero(n - 1);
      diag = Eigen::VectorXd::Zero(n);
      upper = Eigen::VectorXd::Zero(n - 1);
    }

    // helper lambda to set a coefficient either in tri vectors
    // or in the triplet list
    auto set_coeff = [&](std::size_t row, std::size_t col, double value) {
      if (tri_system) {
        if (row == col) {
          diag[static_cast<Eigen::Index>(row)] = value;
        } else if (row == col + 1) {
          lower[static_cast<Eigen::Index>(col)] = value;
        } else if (row + 1 == col) {
          upper[static_cast<Eigen::Index>(row)] = value;
        } else {
          // out of tridiagonal band: must convert
          tri_system = false;
          // move existing tri data to triplets
          for (std::size_t i = 0; i < n; ++i) {
            triplets.emplace_back(static_cast<Eigen::Index>(i),
                                  static_cast<Eigen::Index>(i),
                                  diag[static_cast<Eigen::Index>(i)]);
            if (i < n - 1) {
              triplets.emplace_back(static_cast<Eigen::Index>(i + 1),
                                    static_cast<Eigen::Index>(i),
                                    lower[static_cast<Eigen::Index>(i)]);
              triplets.emplace_back(static_cast<Eigen::Index>(i),
                                    static_cast<Eigen::Index>(i + 1),
                                    upper[static_cast<Eigen::Index>(i)]);
            }
          }
          // then fall through to add the current non band entry
          triplets.emplace_back(static_cast<Eigen::Index>(row),
                                static_cast<Eigen::Index>(col), value);
        }
      } else {
        triplets.emplace_back(static_cast<Eigen::Index>(row),
                              static_cast<Eigen::Index>(col), value);
      }
    };

    // ====================================================================
    // 0. left boundary row
    // ====================================================================
    switch (left_bc.type) {
      case BcType::kSymmetric: {
        // 2*M0 + M1 = 6*(y1 - y0)/h0^2
        set_coeff(0, 0, 2.0);
        set_coeff(0, 1, 1.0);
        rhs(0) = 6.0 * (y_(1) - y_(0)) / (h(0) * h(0));
        break;
      }
      case BcType::kAntiSymmetric:
      case BcType::kNatural: {
        // M0 = 0
        set_coeff(0, 0, 1.0);
        rhs(0) = 0.0;
        break;
      }
      case BcType::kClamped: {
        // 2*h0*M0 + h0*M1 = 6*( (y1-y0)/h0 - slope0 )
        double h0 = h(0);
        set_coeff(0, 0, 2.0 * h0);
        set_coeff(0, 1, h0);
        rhs(0) = 6.0 * ((y_(1) - y_(0)) / h0 - left_bc.slope);
        break;
      }
      case BcType::kNotAKnot: {
        // -h1*M0 + (h0+h1)*M1 - h0*M2 = 0
        set_coeff(0, 0, -h(1));
        set_coeff(0, 1, h(0) + h(1));
        set_coeff(0, 2, -h(0));
        rhs(0) = 0.0;
        tri_system = false;  // extra away-band term
        break;
      }
      case BcType::kPeriodic: {
        // M0 - Mn = 0  (first equation)
        set_coeff(0, 0, 1.0);
        set_coeff(0, n - 1, -1.0);
        rhs(0) = 0.0;
        tri_system = false;  // wrap-around
        break;
      }
    }

    // ====================================================================
    // 1..n-2 interior continuity rows
    // ====================================================================
    for (std::size_t i = 1; i < n - 1; ++i) {
      // h_{i-1} * M_{i-1} + 2(h_{i-1}+h_i) * M_i + h_i * M_{i+1} = RHS_i
      set_coeff(i, i - 1, h[static_cast<Eigen::Index>(i - 1)]);
      set_coeff(i, i,
                2.0 * (h[static_cast<Eigen::Index>(i - 1)] +
                       h[static_cast<Eigen::Index>(i)]));
      set_coeff(i, i + 1, h[static_cast<Eigen::Index>(i)]);
      rhs(i) = 6.0 * ((y_[static_cast<Eigen::Index>(i + 1)] -
                       y_[static_cast<Eigen::Index>(i)]) /
                          h[static_cast<Eigen::Index>(i)] -
                      (y_[static_cast<Eigen::Index>(i)] -
                       y_[static_cast<Eigen::Index>(i - 1)]) /
                          h[static_cast<Eigen::Index>(i - 1)]);
    }

    // ====================================================================
    // n. right boundary row
    // ====================================================================
    switch (right_bc.type) {
      case BcType::kSymmetric: {
        // 2*Mn + M_{n-1} = 6*(y_n - y_{n-1}) / h_{n-2}^2
        set_coeff(n - 1, n - 1, 2.0);
        set_coeff(n - 1, n - 2, 1.0);
        double hn1 = h(static_cast<Eigen::Index>(n - 2));
        rhs(n - 1) = -6.0 *
                     (y_(static_cast<Eigen::Index>(n - 1)) -
                      y_(static_cast<Eigen::Index>(n - 2))) /
                     (hn1 * hn1);
        break;
      }
      case BcType::kAntiSymmetric:
      case BcType::kNatural: {
        // Mn = 0
        set_coeff(n - 1, n - 1, 1.0);
        rhs(n - 1) = 0.0;
        break;
      }
      case BcType::kClamped: {
        double hn1 = h(static_cast<Eigen::Index>(n - 2));
        // h_{n-2} * M_{n-2} + 2*h_{n-2} * Mn =
        //   6*( slope_n - (y_n - y_{n-1})/h_{n-2} )
        set_coeff(n - 1, n - 2, hn1);
        set_coeff(n - 1, n - 1, 2.0 * hn1);
        rhs(n - 1) =
            6.0 * (right_bc.slope - (y_(static_cast<Eigen::Index>(n - 1)) -
                                     y_(static_cast<Eigen::Index>(n - 2))) /
                                        hn1);
        break;
      }
      case BcType::kNotAKnot: {
        // -h_{n-2}*M_{n-3} + (h_{n-3}+h_{n-2})*M_{n-2}
        //   - h_{n-3}*M_{n-1} = 0
        set_coeff(n - 1, n - 3, -h(static_cast<Eigen::Index>(n - 2)));
        set_coeff(n - 1, n - 2,
                  h(static_cast<Eigen::Index>(n - 3)) +
                      h(static_cast<Eigen::Index>(n - 2)));
        set_coeff(n - 1, n - 1, -h(static_cast<Eigen::Index>(n - 3)));
        rhs(n - 1) = 0.0;
        tri_system = false;
        break;
      }
      case BcType::kPeriodic: {
        // Periodic spline:  S''(x0) = S''(xn)    ->  M0 - Mn = 0   (row 0)
        //                   S'(x0)  = S'(xn)
        //
        // First-derivative equality produces
        //
        //   h0 * (2 M0 + M1)  -  h_{n-2} * (2 Mn + M_{n-1})
        // = 6 * [ (y1 - y0)/h0  -  (y_n - y_{n-1})/h_{n-2} ]
        //
        double h0 = h(0);
        double hn_1 = h(static_cast<Eigen::Index>(n - 2));

        // Row n-1:
        set_coeff(n - 1, 0, 2.0 * h0);        //  2 h0  * M0
        set_coeff(n - 1, 1, h0);              //    h0  * M1
        set_coeff(n - 1, n - 2, hn_1);        //    hn1 * M_{n-2}
        set_coeff(n - 1, n - 1, 2.0 * hn_1);  //  2 hn1 * M_{n-1}

        rhs(n - 1) = 6.0 * ((y_(1) - y_(0)) / h0 -
                            (y_(static_cast<Eigen::Index>(n - 1)) -
                             y_(static_cast<Eigen::Index>(n - 2))) /
                                hn_1);

        tri_system = false;  // cyclic => sparse solve
        break;
      }
    }

    // ====================================================================
    // Solve for M
    // ====================================================================
    Eigen::VectorXd M(n);

    if (tri_system) {
      std::cout << "tri-diagonal system -> Thomas solve\n";
      // ---------- Thomas forward sweep ---------------------------------
      for (std::size_t i = 1; i < n; ++i) {
        double m = lower(static_cast<Eigen::Index>(i - 1)) /
                   diag(static_cast<Eigen::Index>(i - 1));

        diag(static_cast<Eigen::Index>(i)) -=
            m * upper(static_cast<Eigen::Index>(i - 1));

        rhs(static_cast<Eigen::Index>(i)) -=
            m * rhs(static_cast<Eigen::Index>(i - 1));
      }

      // ---------- Thomas back substitution -----------------------------
      M(n - 1) = rhs(n - 1) / diag(n - 1);
      for (std::size_t i = n - 1; i-- > 0;) {
        M(static_cast<Eigen::Index>(i)) =
            (rhs(static_cast<Eigen::Index>(i)) -
             upper(static_cast<Eigen::Index>(i)) *
                 M(static_cast<Eigen::Index>(i + 1))) /
            diag(static_cast<Eigen::Index>(i));
      }
    } else {
      std::cout << "general (sparse) system -> SparseLU solve\n";
      // ---------- Assemble sparse matrix and solve ---------------------
      Eigen::SparseMatrix<double> A(n, n);
      A.setFromTriplets(triplets.begin(), triplets.end());

      Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
      solver.analyzePattern(A);
      solver.factorize(A);
      if (solver.info() != Eigen::Success) {
        return absl::InternalError("SparseLU factorization failed");
      }
      M = solver.solve(rhs);
      if (solver.info() != Eigen::Success) {
        return absl::InternalError("SparseLU solve failed");
      }
    }

    // ---------- generate cubic coefficients a,b,c,d ---------------------
    std::size_t segs = n - 1;
    a_.resize(segs);
    b_.resize(segs);
    c_.resize(segs);
    d_.resize(segs);

    for (std::size_t i = 0; i < segs; ++i) {
      double hi = h(static_cast<Eigen::Index>(i));

      a_[i] = y_(static_cast<Eigen::Index>(i));
      c_[i] = 0.5 * M(static_cast<Eigen::Index>(i));
      d_[i] = (M(static_cast<Eigen::Index>(i + 1)) -
               M(static_cast<Eigen::Index>(i))) /
              (6.0 * hi);
      b_[i] = (y_(static_cast<Eigen::Index>(i + 1)) -
               y_(static_cast<Eigen::Index>(i))) /
                  hi -
              hi *
                  (2.0 * M(static_cast<Eigen::Index>(i)) +
                   M(static_cast<Eigen::Index>(i + 1))) /
                  6.0;
    }

    return absl::OkStatus();
  }

  // ---------------------------------------------------------------------
  // Data
  // ---------------------------------------------------------------------
  Eigen::VectorXd x_;
  Eigen::VectorXd y_;

  // Interval cubic coefficients
  std::vector<double> a_, b_, c_, d_;
};

}  // namespace cubic_spline

#endif  // UTIL_CUBIC_SPLINE_CUBIC_SPLINE_H_
