// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/singular_integrals/singular_integrals.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <vector>

namespace vmecpp {

namespace {

// A boundary value of the moment recurrence contaminates the moments below it
// by rho^-(distance), rho being the modulus of the growing characteristic
// roots; the boundary-value problem extends far enough above kL to bring that
// below 1e-17 at kL.
constexpr double kMinBoundaryLogDecay = 17.0 * std::numbers::ln10;

// Bounds of that extension. The upper one binds for rho < 1.01, where the
// contamination is rho^-4096 of the error of the upper boundary values,
// which is O(k^-4) + O(rho^-k) at k > 4096.
constexpr int kMinTail = 8;
constexpr int kMaxTail = 4096;

// T_0 = int_{-1}^{1} dt / sqrt(A t^2 + 2 d t + B) for A = a + b2 + c,
// B = a - b2 + c and d = c - a (eq. 6.207 in TNOV):
//   sqrt(A) T_0 = log((2 sqrt(c A) + 2 c + b2) / (2 sqrt(a A) - 2 a - b2)).
// A term of the quotient that is a difference of nearly equal numbers, as the
// denominator is for gvv << guu, is taken in its conjugate form, whose
// numerator 4 a c - b2^2 is the determinant of the metric.
double ComputeT0(double a, double b2, double c) {
  const double A = a + b2 + c;
  const double root = 2.0 * std::sqrt(a * c);
  const double determinant = (root - b2) * (root + b2);
  const double hi = 2.0 * c + b2;
  const double lo = 2.0 * a + b2;
  const double sqrt_ca = 2.0 * std::sqrt(c * A);
  const double sqrt_aa = 2.0 * std::sqrt(a * A);
  const double numerator =
      hi >= 0.0 ? sqrt_ca + hi : determinant / (sqrt_ca - hi);
  const double denominator =
      lo >= 0.0 ? determinant / (sqrt_aa + lo) : sqrt_aa - lo;
  return std::log(numerator / denominator) / std::sqrt(A);
}

// sum_l cmn(l, m, n) t^l, the add-back polynomial of (m, n) before the
// averaging of (6.291) in TNOV. With k = |m - n| and mu = min(m, n),
// Algorithm 1 in TNOV gives
//   cmn(k + 2 q, m, n) = (-1)^{max(0, n - m) + q} (max(m, n) + q)! /
//                        ((mu - q)! (k + q)! q!),
// the coefficients of the radial Zernike polynomial
//   (-1)^{max(0, n - m)} t^k P^{(k, 0)}_mu(1 - 2 t^2).
// The Jacobi polynomial is taken from its three-term recurrence in the degree,
// which is stable on [-1, 1]; the monomial form loses max |cmn| digits.
double EvaluateCmnPolynomial(int m, int n, double t) {
  const int k = std::abs(m - n);
  const int mu = std::min(m, n);
  const double x = 1.0 - 2.0 * t * t;
  double p_previous = 0.0;
  double p = 1.0;
  if (mu >= 1) {
    p_previous = 1.0;
    p = 0.5 * ((k + 2) * x + k);
    for (int j = 1; j < mu; ++j) {
      const double two_j_k = 2.0 * j + k;
      const double c_next = 2.0 * (j + 1) * (j + k + 1) * two_j_k;
      const double c_this =
          (two_j_k + 1.0) * ((two_j_k + 2.0) * two_j_k * x + k * k);
      const double c_previous = 2.0 * (j + k) * j * (two_j_k + 2.0);
      const double p_next = (c_this * p - c_previous * p_previous) / c_next;
      p_previous = p;
      p = p_next;
    }
  }
  const double sign = (std::max(0, n - m) % 2 == 0) ? 1.0 : -1.0;
  return sign * std::pow(t, k) * p;
}

// M_k = int_{-1}^{1} T_k(t) / sqrt(Q(t)) dt, Q = A t^2 + 2 d t + B, for
// A = a + b2_sign b2 + c, B = a - b2_sign b2 + c, d = c - a and k = 0, ..., kL,
// at the first num_points tangential grid points. Integrating
// d/dt (F_k sqrt(Q)) with F_k' = T_k,
// F_k = T_{k+1} / (2 (k + 1)) - T_{k-1} / (2 (k - 1)), gives for k >= 2 the
// five-term recurrence whose factors the constructor tabulates. Its
// characteristic roots are those of Q((z + 1/z) / 2) = 0: a complex pair of
// modulus rho > 1 and their reciprocals, so two homogeneous solutions grow
// and two decay in either direction and neither a forward nor a backward pass
// is stable. The moments decay like k^-2 only, and they are the solution of
// the boundary-value problem whose lower boundary values are M_0 and M_1 and
// whose upper boundary values are the asymptotic
//   M_k = -(1 / sqrt(Q(1)) + (-1)^k / sqrt(Q(-1))) / (k^2 - 1) + O(k^-4),
// which a pentadiagonal elimination solves. The rows of its matrix tend to
// those of the Toeplitz matrix with the symbol Q(cos(theta)) > 0, and the
// elimination runs without pivoting. It is sequential in k and independent
// across grid points, so kChebyshevMomentBatch points are eliminated together,
// over the largest of their extents.
void ComputeChebyshevMoments(const Eigen::VectorXd& a,
                             const Eigen::VectorXd& b2,
                             const Eigen::VectorXd& c, double b2_sign,
                             int num_points, int kL,
                             ChebyshevMomentWorkspace& m_workspace,
                             std::vector<Eigen::VectorXd>& m_moments) {
  using Batch = Eigen::Array<double, kChebyshevMomentBatch, 1>;

  const std::vector<double>& factor_sub2 = m_workspace.factor_sub2;
  const std::vector<double>& factor_sub1 = m_workspace.factor_sub1;
  const std::vector<double>& factor_diag = m_workspace.factor_diag;
  const std::vector<double>& factor_sup1 = m_workspace.factor_sup1;
  const std::vector<double>& factor_sup2 = m_workspace.factor_sup2;
  const std::vector<double>& factor_rhs = m_workspace.factor_rhs;
  auto& upper1 = m_workspace.upper1;
  auto& upper2 = m_workspace.upper2;
  auto& moments = m_workspace.moments;

  // M_0 and M_1 enter as rows of the identity
  upper1.leftCols(2).setZero();
  upper2.leftCols(2).setZero();

  for (int first = 0; first < num_points; first += kChebyshevMomentBatch) {
    Batch A;
    Batch d;
    Batch half_a_plus_b;
    Batch sqrt_q_plus;
    Batch sqrt_q_minus;
    int tail = kMinTail;
    for (int point = 0; point < kChebyshevMomentBatch; ++point) {
      // points past the last one repeat it
      const int kl = std::min(first + point, num_points - 1);
      const double b2_kl = b2_sign * b2[kl];
      A[point] = a[kl] + b2_kl + c[kl];
      d[point] = c[kl] - a[kl];
      half_a_plus_b[point] = 0.5 * A[point] + (a[kl] - b2_kl + c[kl]);
      sqrt_q_plus[point] = 2.0 * std::sqrt(c[kl]);
      sqrt_q_minus[point] = 2.0 * std::sqrt(a[kl]);

      const double m0 = ComputeT0(a[kl], b2_kl, c[kl]);
      moments(point, 0) = m0;
      // from int (A t + d) / sqrt(Q) dt = sqrt(Q(1)) - sqrt(Q(-1))
      moments(point, 1) =
          (sqrt_q_plus[point] - sqrt_q_minus[point] - d[point] * m0) / A[point];

      // The roots of Q lie on the ellipse with foci -1 and 1 whose semi-axes
      // are (rho + 1 / rho) / 2 = (sqrt(Q(1)) + sqrt(Q(-1))) / (2 sqrt(A))
      // and (rho - 1 / rho) / 2 = sqrt((2 sqrt(a c) - b2) / A).
      const double semi_major = 0.5 *
                                (sqrt_q_plus[point] + sqrt_q_minus[point]) /
                                std::sqrt(A[point]);
      const double semi_minor = std::sqrt(
          std::max(2.0 * std::sqrt(a[kl] * c[kl]) - b2_kl, 0.0) / A[point]);
      const double log_rho = std::log(semi_major + semi_minor);
      int tail_kl = kMaxTail;
      if (log_rho > 0.0) {
        tail_kl = static_cast<int>(
            std::min(static_cast<double>(kMaxTail),
                     std::ceil(kMinBoundaryLogDecay / log_rho)));
      }
      tail = std::max(tail, tail_kl);
    }

    // Row k of the eliminated system reads
    //   M_k + upper1_k M_{k+1} + upper2_k M_{k+2} = moments_k.
    const int k_top = kL + tail;
    for (int k = 2; k <= k_top; ++k) {
      const double sign = (k % 2 == 0) ? 1.0 : -1.0;
      const Batch sub2 = A * factor_sub2[k];
      const Batch sub1 = d * factor_sub1[k] - upper1.col(k - 2) * sub2;
      const Batch inverse_pivot =
          (half_a_plus_b - A * factor_diag[k] - upper2.col(k - 2) * sub2 -
           upper1.col(k - 1) * sub1)
              .inverse();
      upper1.col(k) =
          (d * factor_sup1[k] - upper2.col(k - 1) * sub1) * inverse_pivot;
      upper2.col(k) = A * factor_sup2[k] * inverse_pivot;
      moments.col(k) = (-(sqrt_q_plus + sign * sqrt_q_minus) * factor_rhs[k] -
                        moments.col(k - 2) * sub2 - moments.col(k - 1) * sub1) *
                       inverse_pivot;
    }
    for (int k = k_top + 1; k <= k_top + 2; ++k) {
      const double sign = (k % 2 == 0) ? 1.0 : -1.0;
      moments.col(k) =
          -(sqrt_q_plus.inverse() + sign * sqrt_q_minus.inverse()) *
          factor_rhs[k];
    }
    for (int k = k_top; k >= 2; --k) {
      moments.col(k) -= upper1.col(k) * moments.col(k + 1) +
                        upper2.col(k) * moments.col(k + 2);
    }

    const int num_in_batch =
        std::min(kChebyshevMomentBatch, num_points - first);
    for (int k = 0; k <= kL; ++k) {
      m_moments[k].segment(first, num_in_batch) =
          moments.col(k).head(num_in_batch);
    }
  }  // first
}  // ComputeChebyshevMoments

}  // namespace

SingularIntegrals::SingularIntegrals(const Sizes* s,
                                     const FourierBasisFastToroidal* fb,
                                     const TangentialPartitioning* tp,
                                     const SurfaceGeometry* sg, int nf, int mf)
    : s_(*s), fb_(*fb), tp_(*tp), sg_(*sg), nf(nf), mf(mf) {
  numSC = mf * (nf + 1);
  numCS = (mf + 1) * nf;
  nzLen = numSC + numCS;

  chebyshev_coefficients.resize((1 + nf + mf) * (nf + 1) * (mf + 1));

  // -------------

  // thread-local tangential grid point range
  int numLocal = tp_.ztMax - tp_.ztMin;

  ap.resize(numLocal);
  am.resize(numLocal);
  d.resize(numLocal);
  sqrtc2.resize(numLocal);
  sqrta2.resize(numLocal);
  delta4.resize(numLocal);

  Ap.resize(numLocal);
  Am.resize(numLocal);
  D.resize(numLocal);
  R1p.resize(numLocal);
  R1m.resize(numLocal);
  R0p.resize(numLocal);
  R0m.resize(numLocal);
  Ra1p.resize(numLocal);
  Ra1m.resize(numLocal);

  chebyshev_moments_p.resize(mf + nf + 1);
  chebyshev_moments_m.resize(mf + nf + 1);
  chebyshev_s_moments_p.resize(mf + nf + 1);
  chebyshev_s_moments_m.resize(mf + nf + 1);
  for (int k = 0; k < mf + nf + 1; ++k) {
    chebyshev_moments_p[k].resize(numLocal);
    chebyshev_moments_m[k].resize(numLocal);
    chebyshev_s_moments_p[k].resize(numLocal);
    chebyshev_s_moments_m[k].resize(numLocal);
  }

  // The moment recurrence at index k, for k = 2, ..., kL + kMaxTail, reads
  //   A f_sup2 M_{k+2} + d f_sup1 M_{k+1} + (A / 2 + B - A f_diag) M_k
  //     + d f_sub1 M_{k-1} + A f_sub2 M_{k-2}
  //     = -f_rhs (sqrtc2 + (-1)^k sqrta2);
  // the boundary values above the last index take f_rhs two entries further.
  const int k_max = mf + nf + kMaxTail;
  moment_workspace_.factor_sub2.resize(k_max + 3);
  moment_workspace_.factor_sub1.resize(k_max + 3);
  moment_workspace_.factor_diag.resize(k_max + 3);
  moment_workspace_.factor_sup1.resize(k_max + 3);
  moment_workspace_.factor_sup2.resize(k_max + 3);
  moment_workspace_.factor_rhs.resize(k_max + 3);
  for (int k = 2; k < k_max + 3; ++k) {
    const double kd = k;
    moment_workspace_.factor_sub2[k] = (kd - 2.0) / (4.0 * (kd - 1.0));
    moment_workspace_.factor_sub1[k] = (2.0 * kd - 3.0) / (2.0 * (kd - 1.0));
    moment_workspace_.factor_diag[k] = 1.0 / (2.0 * (kd * kd - 1.0));
    moment_workspace_.factor_sup1[k] = (2.0 * kd + 3.0) / (2.0 * (kd + 1.0));
    moment_workspace_.factor_sup2[k] = (kd + 2.0) / (4.0 * (kd + 1.0));
    moment_workspace_.factor_rhs[k] = 1.0 / (kd * kd - 1.0);
  }
  moment_workspace_.upper1.resize(kChebyshevMomentBatch, k_max + 3);
  moment_workspace_.upper2.resize(kChebyshevMomentBatch, k_max + 3);
  moment_workspace_.moments.resize(kChebyshevMomentBatch, k_max + 3);

  const int mnfull = (2 * nf + 1) * (mf + 1);
  bvec_sin.setZero(mnfull);
  grpmn_sin.setZero(mnfull * numLocal);
  if (s->lasym) {
    bvec_cos.setZero(mnfull);
    grpmn_cos.setZero(mnfull * numLocal);
  }

  // -------------

  computeCoefficients();
}

void SingularIntegrals::computeCoefficients() {
  const int kL = mf + nf;
  const int num_nodes = kL + 1;

  // p_mn has degree m + n <= kL, so its Chebyshev coefficients follow exactly
  // from its values at the kL + 1 Chebyshev-Gauss nodes.
  std::vector<double> theta(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    theta[i] = (i + 0.5) * std::numbers::pi / num_nodes;
  }

  // sum_l cmn(l, m, n) t_i^l at the nodes
  std::vector<double> cmn_values(static_cast<std::size_t>(nf + 1) * (mf + 1) *
                                 num_nodes);
  for (int n = 0; n < nf + 1; ++n) {
    for (int m = 0; m < mf + 1; ++m) {
      for (int i = 0; i < num_nodes; ++i) {
        cmn_values[(n * (mf + 1) + m) * num_nodes + i] =
            EvaluateCmnPolynomial(m, n, std::cos(theta[i]));
      }
    }
  }

  chebyshev_coefficients.setZero();
  std::vector<double> p(num_nodes);
  for (int n = 0; n < nf + 1; ++n) {
    for (int m = 0; m < mf + 1; ++m) {
      // cmns from cmn: (6.291) in TNOV
      const int n_m_ = (n * (mf + 1) + m) * num_nodes;
      const int n1m_ = ((n - 1) * (mf + 1) + m) * num_nodes;
      const int n_m1 = (n * (mf + 1) + (m - 1)) * num_nodes;
      const int n1m1 = ((n - 1) * (mf + 1) + (m - 1)) * num_nodes;
      for (int i = 0; i < num_nodes; ++i) {
        if (m == 0 && n == 0) {
          p[i] = cmn_values[n_m_ + i];
        } else if (m == 0 && n > 0) {
          p[i] = (cmn_values[n_m_ + i] + cmn_values[n1m_ + i]) / 2;
        } else if (m > 0 && n == 0) {
          p[i] = (cmn_values[n_m_ + i] + cmn_values[n_m1 + i]) / 2;
        } else {
          p[i] = (cmn_values[n_m_ + i] + cmn_values[n1m_ + i] +
                  cmn_values[n_m1 + i] + cmn_values[n1m1 + i]) /
                 2;
        }
      }  // i

      // discrete cosine transform; the coefficients above m + n vanish
      for (int k = 0; k <= m + n; ++k) {
        double sum = 0.0;
        for (int i = 0; i < num_nodes; ++i) {
          sum += p[i] * std::cos(k * theta[i]);
        }
        const int knm = (k * (nf + 1) + n) * (mf + 1) + m;
        chebyshev_coefficients[knm] = (k == 0 ? 1.0 : 2.0) * sum / num_nodes;
      }  // k
    }  // m
  }  // n
}  // computeCoefficients

void SingularIntegrals::update(const Eigen::VectorXd& bDotN, bool fullUpdate) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  prepareUpdate(sg_.guu, sg_.guv, sg_.gvv, sg_.auu, sg_.auv, sg_.avv,
                fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  performUpdate(bDotN, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // update

void SingularIntegrals::prepareUpdate(
    const Eigen::VectorXd& a, const Eigen::VectorXd& b2,
    const Eigen::VectorXd& c, const Eigen::VectorXd& A,
    const Eigen::VectorXd& B2, const Eigen::VectorXd& C, bool fullUpdate) {
  const int kL = mf + nf;
  int numLocal = tp_.ztMax - tp_.ztMin;

  // T^- exchanges ap and am, which is b2 -> -b2
  ComputeChebyshevMoments(a, b2, c, /*b2_sign=*/1.0, numLocal, kL,
                          moment_workspace_, chebyshev_moments_p);
  ComputeChebyshevMoments(a, b2, c, /*b2_sign=*/-1.0, numLocal, kL,
                          moment_workspace_, chebyshev_moments_m);

  for (int kl = 0; kl < numLocal; ++kl) {
    // initialize constants (along expansion in l)
    ap[kl] = a[kl] + b2[kl] + c[kl];
    am[kl] = a[kl] - b2[kl] + c[kl];
    d[kl] = c[kl] - a[kl];
    sqrtc2[kl] = 2.0 * sqrt(c[kl]);
    sqrta2[kl] = 2.0 * sqrt(a[kl]);

    if (fullUpdate) {
      delta4[kl] = ap[kl] * am[kl] - d[kl] * d[kl];

      Ap[kl] = A[kl] + B2[kl] + C[kl];
      Am[kl] = A[kl] - B2[kl] + C[kl];
      D[kl] = C[kl] - A[kl];

      R1p[kl] = (Ap[kl] * (delta4[kl] - d[kl] * d[kl]) / ap[kl] -
                 Am[kl] * ap[kl] + 2 * D[kl] * d[kl]) /
                delta4[kl];
      R1m[kl] = (Am[kl] * (delta4[kl] - d[kl] * d[kl]) / am[kl] -
                 Ap[kl] * am[kl] + 2 * D[kl] * d[kl]) /
                delta4[kl];
      R0p[kl] = (-Ap[kl] * am[kl] * d[kl] / ap[kl] - Am[kl] * d[kl] +
                 2 * D[kl] * am[kl]) /
                delta4[kl];
      R0m[kl] = (-Am[kl] * ap[kl] * d[kl] / am[kl] - Ap[kl] * d[kl] +
                 2 * D[kl] * ap[kl]) /
                delta4[kl];
      Ra1p[kl] = Ap[kl] / ap[kl];
      Ra1m[kl] = Am[kl] / am[kl];

      // Eq. (A17) maps the polynomial p to
      //   R1 int t p' / sqrt(Q) + Ra1 int p / sqrt(Q) + R0 int p' / sqrt(Q)
      //     - (R0 + R1) p(1) / sqrtc2 + (R0 - R1) p(-1) / sqrta2,
      // which is S_l for p = t^l. For p = T_k, T_k' = k U_{k-1} and
      // t U_{k-1} = (U_k + U_{k-2}) / 2, with the moments of U_k = 2 T_k +
      // U_{k-2} accumulated in N_k.
      const auto s_moments = [&](const std::vector<Eigen::VectorXd>& moments,
                                 double R0, double R1, double Ra1,
                                 std::vector<Eigen::VectorXd>& m_s_moments) {
        const double at_plus = -(R0 + R1) / sqrtc2[kl];
        const double at_minus = (R0 - R1) / sqrta2[kl];
        double n_km2 = 0.0;  // N_{k-2}
        double n_km1 = 0.0;  // N_{k-1}
        for (int k = 0; k <= kL; ++k) {
          const double n_k = (k == 0 ? 1.0 : 2.0) * moments[k][kl] + n_km2;
          m_s_moments[k][kl] = R1 * k * 0.5 * (n_k + n_km2) +
                               Ra1 * moments[k][kl] + R0 * k * n_km1 + at_plus +
                               (k % 2 == 0 ? at_minus : -at_minus);
          n_km2 = n_km1;
          n_km1 = n_k;
        }
      };
      s_moments(chebyshev_moments_p, R0p[kl], R1p[kl], Ra1p[kl],
                chebyshev_s_moments_p);
      s_moments(chebyshev_moments_m, R0m[kl], R1m[kl], Ra1m[kl],
                chebyshev_s_moments_m);
    }  // fullUpdate
  }  // kl
}  // prepareUpdate

void SingularIntegrals::performUpdate(const Eigen::VectorXd& bDotN,
                                      bool fullUpdate) {
  const int numLocal = tp_.ztMax - tp_.ztMin;

  bvec_sin.setZero();
  if (s_.lasym) {
    bvec_cos.setZero();
  }

  if (fullUpdate) {
    grpmn_sin.setZero();
    if (s_.lasym) {
      grpmn_cos.setZero();
    }
  }

  // order: index of the Chebyshev polynomial T_order
  for (int order = 0; order < 1 + nf + mf; ++order) {
    for (int n = 0; n < nf + 1; ++n) {
      for (int m = 0; m < mf + 1; ++m) {
        const int idx_m_posn = (nf + n) * (mf + 1) + m;
        const int idx_m_negn = (nf - n) * (mf + 1) + m;

        // p_mn has degree m + n
        if (order > m + n) {
          continue;
        }

        const int idx_knm = (order * (nf + 1) + n) * (mf + 1) + m;
        const double gamma_factor =
            chebyshev_coefficients[idx_knm] / (fb_.mscale[m] * fb_.nscale[n]);

        if (n == 0 || m == 0) {
          // analysum

          for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
            const int l = kl / s_.nZeta;
            const int k = kl % s_.nZeta;
            const int klRel = kl - tp_.ztMin;

            // The poloidal basis cosmu/sinmu is stored only on the reduced
            // theta range [0, nThetaReduced). The lasym free-boundary path
            // integrates over the full theta range, so for l >= nThetaReduced
            // reflect via stellarator symmetry (theta -> -theta): cos(m*theta)
            // is even, sin(m*theta) is odd.
            const int lr = (l < s_.nThetaReduced) ? l : (s_.nThetaEven - l);
            const double sgnmu = (l < s_.nThetaReduced) ? 1.0 : -1.0;
            const int idx_lm = lr * (s_.mnyq2 + 1) + m;
            const int idx_nk = n * s_.nZeta + k;

            // sin(mu - |n|v) * gamma_k(m,n)
            const double sinp = (sgnmu * fb_.sinmu[idx_lm] * fb_.cosnv[idx_nk] -
                                 fb_.cosmu[idx_lm] * fb_.sinnv[idx_nk]) *
                                gamma_factor;

            bvec_sin[idx_m_posn] += (chebyshev_moments_p[order][klRel] +
                                     chebyshev_moments_m[order][klRel]) *
                                    bDotN[klRel] * s_.wInt[l] * sinp;
            if (fullUpdate) {
              grpmn_sin[idx_m_posn * numLocal + klRel] +=
                  (chebyshev_s_moments_p[order][klRel] +
                   chebyshev_s_moments_m[order][klRel]) *
                  sinp;
            }

            if (s_.lasym) {
              // cos(mu - |n|v) * gamma_k(m,n)
              const double cosp =
                  (fb_.cosmu[idx_lm] * fb_.cosnv[idx_nk] +
                   sgnmu * fb_.sinmu[idx_lm] * fb_.sinnv[idx_nk]) *
                  gamma_factor;

              bvec_cos[idx_m_posn] += (chebyshev_moments_p[order][klRel] +
                                       chebyshev_moments_m[order][klRel]) *
                                      bDotN[klRel] * s_.wInt[l] * cosp;
              if (fullUpdate) {
                grpmn_cos[idx_m_posn * numLocal + klRel] +=
                    (chebyshev_s_moments_p[order][klRel] +
                     chebyshev_s_moments_m[order][klRel]) *
                    cosp;
              }
            }
          }  // kl

        } else {
          // analysum2
          //
          // M^- and S^- carry sin(mu - |n|v) (mode (m, +|n|)) and M^+ and S^+
          // carry sin(mu + |n|v) (mode (m, -|n|)): this is the assignment
          // under which the analytic Fourier coefficients equal those of the
          // tangent-plane kernels subtracted in RegularizedIntegrals, with the
          // metric and curvature cross terms as stored. The opposite
          // assignment (the one PARVMEC's analyt makes by passing slm, tlm,
          // slp, tlp to analysum2's dummy arguments slp, tlp, slm, tlm) gives
          // mode (m, n) the coefficient of (m, -n): an error that is first
          // order in the non-axisymmetric shaping and independent of the
          // resolution.

          for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
            const int l = kl / s_.nZeta;
            int k = kl % s_.nZeta;
            // Reflect onto the reduced theta range for l >= nThetaReduced
            // (sin(m*theta) is odd under theta -> -theta).
            const int lr = (l < s_.nThetaReduced) ? l : (s_.nThetaEven - l);
            const double sgnmu = (l < s_.nThetaReduced) ? 1.0 : -1.0;
            const int idx_lm = lr * (s_.mnyq2 + 1) + m;
            const int remaining = std::min(s_.nZeta - k, tp_.ztMax - kl);

            const double coeff1 = sgnmu * fb_.sinmu[idx_lm] * gamma_factor;
            const double coeff2 = fb_.cosmu[idx_lm] * gamma_factor;

            std::array<double, 4> buf_m_posn{};
            std::array<double, 4> buf_m_negn{};

            int i = 0;
            for (; i + 3 < remaining; i += 4, k += 4, kl += 4) {
              // in here l is constant and k always increases
              const int klRel = kl - tp_.ztMin;
              const int idx_nk = n * s_.nZeta + k;

              const double c0 = bDotN[klRel + 0] * s_.wInt[l];
              const double c1 = bDotN[klRel + 1] * s_.wInt[l];
              const double c2 = bDotN[klRel + 2] * s_.wInt[l];
              const double c3 = bDotN[klRel + 3] * s_.wInt[l];

              // sin(mu - |n|v) * gamma_k(m,n)
              const double sinp0 = coeff1 * fb_.cosnv[idx_nk + 0] -
                                   coeff2 * fb_.sinnv[idx_nk + 0];
              const double sinp1 = coeff1 * fb_.cosnv[idx_nk + 1] -
                                   coeff2 * fb_.sinnv[idx_nk + 1];
              const double sinp2 = coeff1 * fb_.cosnv[idx_nk + 2] -
                                   coeff2 * fb_.sinnv[idx_nk + 2];
              const double sinp3 = coeff1 * fb_.cosnv[idx_nk + 3] -
                                   coeff2 * fb_.sinnv[idx_nk + 3];

              buf_m_posn[0] +=
                  chebyshev_moments_m[order][klRel + 0] * c0 * sinp0;
              buf_m_posn[1] +=
                  chebyshev_moments_m[order][klRel + 1] * c1 * sinp1;
              buf_m_posn[2] +=
                  chebyshev_moments_m[order][klRel + 2] * c2 * sinp2;
              buf_m_posn[3] +=
                  chebyshev_moments_m[order][klRel + 3] * c3 * sinp3;

              // sin(mu + |n|v) * gamma_k(m,n)
              const double sinm0 = coeff1 * fb_.cosnv[idx_nk + 0] +
                                   coeff2 * fb_.sinnv[idx_nk + 0];
              const double sinm1 = coeff1 * fb_.cosnv[idx_nk + 1] +
                                   coeff2 * fb_.sinnv[idx_nk + 1];
              const double sinm2 = coeff1 * fb_.cosnv[idx_nk + 2] +
                                   coeff2 * fb_.sinnv[idx_nk + 2];
              const double sinm3 = coeff1 * fb_.cosnv[idx_nk + 3] +
                                   coeff2 * fb_.sinnv[idx_nk + 3];

              buf_m_negn[0] +=
                  chebyshev_moments_p[order][klRel + 0] * c0 * sinm0;
              buf_m_negn[1] +=
                  chebyshev_moments_p[order][klRel + 1] * c1 * sinm1;
              buf_m_negn[2] +=
                  chebyshev_moments_p[order][klRel + 2] * c2 * sinm2;
              buf_m_negn[3] +=
                  chebyshev_moments_p[order][klRel + 3] * c3 * sinm3;

              if (fullUpdate) {
                grpmn_sin[idx_m_posn * numLocal + klRel + 0] +=
                    chebyshev_s_moments_m[order][klRel + 0] * sinp0;
                grpmn_sin[idx_m_posn * numLocal + klRel + 1] +=
                    chebyshev_s_moments_m[order][klRel + 1] * sinp1;
                grpmn_sin[idx_m_posn * numLocal + klRel + 2] +=
                    chebyshev_s_moments_m[order][klRel + 2] * sinp2;
                grpmn_sin[idx_m_posn * numLocal + klRel + 3] +=
                    chebyshev_s_moments_m[order][klRel + 3] * sinp3;

                grpmn_sin[idx_m_negn * numLocal + klRel + 0] +=
                    chebyshev_s_moments_p[order][klRel + 0] * sinm0;
                grpmn_sin[idx_m_negn * numLocal + klRel + 1] +=
                    chebyshev_s_moments_p[order][klRel + 1] * sinm1;
                grpmn_sin[idx_m_negn * numLocal + klRel + 2] +=
                    chebyshev_s_moments_p[order][klRel + 2] * sinm2;
                grpmn_sin[idx_m_negn * numLocal + klRel + 3] +=
                    chebyshev_s_moments_p[order][klRel + 3] * sinm3;
              }
            }

            bvec_sin[idx_m_posn] +=
                buf_m_posn[0] + buf_m_posn[1] + buf_m_posn[2] + buf_m_posn[3];
            bvec_sin[idx_m_negn] +=
                buf_m_negn[0] + buf_m_negn[1] + buf_m_negn[2] + buf_m_negn[3];

            if (i != remaining) {
              for (; i < remaining; ++i, ++k, ++kl) {
                // in here l is constant and k always increases
                const int klRel = kl - tp_.ztMin;
                const int idx_nk = n * s_.nZeta + k;

                const double coeff1 = sgnmu * fb_.sinmu[idx_lm] *
                                      fb_.cosnv[idx_nk] * gamma_factor;
                const double coeff2 =
                    fb_.cosmu[idx_lm] * fb_.sinnv[idx_nk] * gamma_factor;

                // sin(mu + |n|v) * gamma_k(m,n)
                const double sinm = coeff1 + coeff2;

                // sin(mu - |n|v) * gamma_k(m,n)
                const double sinp = coeff1 - coeff2;

                const double c = bDotN[klRel] * s_.wInt[l];
                bvec_sin[idx_m_posn] +=
                    chebyshev_moments_m[order][klRel] * c * sinp;
                bvec_sin[idx_m_negn] +=
                    chebyshev_moments_p[order][klRel] * c * sinm;

                if (fullUpdate) {
                  grpmn_sin[idx_m_posn * numLocal + klRel] +=
                      chebyshev_s_moments_m[order][klRel] * sinp;
                  grpmn_sin[idx_m_negn * numLocal + klRel] +=
                      chebyshev_s_moments_p[order][klRel] * sinm;
                }
              }
            }

            // adjust for the ++kl that's coming from the outer loop
            --kl;
          }  // kl

          if (s_.lasym) {
            for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
              const int l = kl / s_.nZeta;
              const int k = kl % s_.nZeta;
              const int klRel = kl - tp_.ztMin;

              // Reflect onto the reduced theta range for l >= nThetaReduced
              // (sin(m*theta) is odd under theta -> -theta).
              const int lr = (l < s_.nThetaReduced) ? l : (s_.nThetaEven - l);
              const double sgnmu = (l < s_.nThetaReduced) ? 1.0 : -1.0;
              const int idx_lm = lr * (s_.mnyq2 + 1) + m;
              const int idx_nk = n * s_.nZeta + k;

              const double coeff1 =
                  fb_.cosmu[idx_lm] * fb_.cosnv[idx_nk] * gamma_factor;
              const double coeff2 =
                  sgnmu * fb_.sinmu[idx_lm] * fb_.sinnv[idx_nk] * gamma_factor;

              // cos(mu + |n|v) * gamma_k(m,n)
              const double cosm = coeff1 - coeff2;

              // cos(mu - |n|v) * gamma_k(m,n)
              const double cosp = coeff1 + coeff2;

              bvec_cos[idx_m_posn] += chebyshev_moments_m[order][klRel] *
                                      bDotN[klRel] * s_.wInt[l] * cosp;
              bvec_cos[idx_m_negn] += chebyshev_moments_p[order][klRel] *
                                      bDotN[klRel] * s_.wInt[l] * cosm;
              if (fullUpdate) {
                grpmn_cos[idx_m_posn * numLocal + klRel] +=
                    chebyshev_s_moments_m[order][klRel] * cosp;
                grpmn_cos[idx_m_negn * numLocal + klRel] +=
                    chebyshev_s_moments_p[order][klRel] * cosm;
              }
            }
          }  // kl
        }  // m == 0 or n == 0
      }  // m
    }  // n

  }  // order
}  // performUpdate

}  // namespace vmecpp
