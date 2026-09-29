// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_
#define VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_

#include <Eigen/Dense>
#include <vector>

#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

// Number of tangential grid points whose Chebyshev-moment problems
// SingularIntegrals eliminates together.
inline constexpr int kChebyshevMomentBatch = 8;

// Factors and scratch of the boundary-value problems that SingularIntegrals
// solves for the Chebyshev moments of the tangent-plane kernel.
struct ChebyshevMomentWorkspace {
  // k-dependent factors of the five-term moment recurrence, indexed by k
  std::vector<double> factor_sub2;
  std::vector<double> factor_sub1;
  std::vector<double> factor_diag;
  std::vector<double> factor_sup1;
  std::vector<double> factor_sup2;
  std::vector<double> factor_rhs;

  // Rows of the eliminated systems of one batch, [point, k]: the two bands
  // above the unit diagonal, and the right-hand side, which the back
  // substitution turns into the moments.
  Eigen::Array<double, kChebyshevMomentBatch, Eigen::Dynamic> upper1;
  Eigen::Array<double, kChebyshevMomentBatch, Eigen::Dynamic> upper2;
  Eigen::Array<double, kChebyshevMomentBatch, Eigen::Dynamic> moments;
};

class SingularIntegrals {
 public:
  SingularIntegrals(const Sizes* s, const FourierBasisFastToroidal* fb,
                    const TangentialPartitioning* tp, const SurfaceGeometry* sg,
                    int nf, int mf);

  void update(const Eigen::VectorXd& bDotN, bool fullUpdate);

  int numSC;
  int numCS;
  int nzLen;  // non-zero length

  // Chebyshev coefficients gamma_k(m, n) of the add-back polynomials
  //   p_mn(t) = sum_l cmns(l, m, n) t^l = sum_k gamma_k(m, n) T_k(t),
  // indexed (k * (nf + 1) + n) * (mf + 1) + m. cmns are the coefficients of
  // (6.291) in TNOV; |p_mn| <= 1 on [-1, 1] and |gamma_k| <= 1, while cmns
  // reaches 1e11 at m + n = 33.
  Eigen::VectorXd chebyshev_coefficients;

  Eigen::VectorXd ap;
  Eigen::VectorXd am;
  Eigen::VectorXd d;
  Eigen::VectorXd sqrtc2;
  Eigen::VectorXd sqrta2;
  Eigen::VectorXd delta4;

  Eigen::VectorXd Ap;
  Eigen::VectorXd Am;
  Eigen::VectorXd D;

  Eigen::VectorXd R1p;
  Eigen::VectorXd R1m;
  Eigen::VectorXd R0p;
  Eigen::VectorXd R0m;
  Eigen::VectorXd Ra1p;
  Eigen::VectorXd Ra1m;

  // Chebyshev moments of the tangent-plane kernel along the expansion
  // variable, [k][kl]:
  //   M^+_k = int_{-1}^{1} T_k(t) / sqrt(ap t^2 + 2 d t + am) dt,
  // and M^-_k with ap and am exchanged.
  std::vector<Eigen::VectorXd> chebyshev_moments_p;
  std::vector<Eigen::VectorXd> chebyshev_moments_m;

  // The functional of Eq. (A17), which maps t^l to S^{+/-}_l, applied to
  // T_k(t), [k][kl].
  std::vector<Eigen::VectorXd> chebyshev_s_moments_p;
  std::vector<Eigen::VectorXd> chebyshev_s_moments_m;

  // sum_kl { M^- * sin(mu - nv), M^+ * sin(mu + nv) } for modes (m, n), (m, -n)
  Eigen::VectorXd bvec_sin;

  // sum_kl { M^- * cos(mu - nv), M^+ * cos(mu + nv) }
  Eigen::VectorXd bvec_cos;

  // S^- * sin(mu - nv), S^+ * sin(mu + nv)
  Eigen::VectorXd grpmn_sin;

  // S^- * cos(mu - nv), S^+ * cos(mu + nv)
  Eigen::VectorXd grpmn_cos;

  void prepareUpdate(const Eigen::VectorXd& a, const Eigen::VectorXd& b2,
                     const Eigen::VectorXd& c, const Eigen::VectorXd& A,
                     const Eigen::VectorXd& B2, const Eigen::VectorXd& C,
                     bool fullUpdate);

 private:
  const Sizes& s_;
  const FourierBasisFastToroidal& fb_;
  const TangentialPartitioning& tp_;
  const SurfaceGeometry& sg_;

  void computeCoefficients();

  void performUpdate(const Eigen::VectorXd& bDotN, bool fullUpdate);

  int nf;
  int mf;

  ChebyshevMomentWorkspace moment_workspace_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_
