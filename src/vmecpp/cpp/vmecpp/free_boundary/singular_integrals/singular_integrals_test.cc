// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/singular_integrals/singular_integrals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"

namespace vmecpp {

using testing::IsCloseRelAbs;

// 64-point Gauss-Legendre quadrature on [-1, 1], stored as {weight, abscissa}
// pairs; exact for polynomials up to degree 127.
constexpr int kGLNodes = 64;
constexpr std::array<std::array<double, 2>, kGLNodes> kGaussLegendre64 = {{
    {0.0486909570091397, -0.0243502926634244},
    {0.0486909570091397, 0.0243502926634244},
    {0.0485754674415034, -0.0729931217877990},
    {0.0485754674415034, 0.0729931217877990},
    {0.0483447622348030, -0.1214628192961206},
    {0.0483447622348030, 0.1214628192961206},
    {0.0479993885964583, -0.1696444204239928},
    {0.0479993885964583, 0.1696444204239928},
    {0.0475401657148303, -0.2174236437400071},
    {0.0475401657148303, 0.2174236437400071},
    {0.0469681828162100, -0.2646871622087674},
    {0.0469681828162100, 0.2646871622087674},
    {0.0462847965813144, -0.3113228719902110},
    {0.0462847965813144, 0.3113228719902110},
    {0.0454916279274181, -0.3572201583376681},
    {0.0454916279274181, 0.3572201583376681},
    {0.0445905581637566, -0.4022701579639916},
    {0.0445905581637566, 0.4022701579639916},
    {0.0435837245293235, -0.4463660172534641},
    {0.0435837245293235, 0.4463660172534641},
    {0.0424735151236536, -0.4894031457070530},
    {0.0424735151236536, 0.4894031457070530},
    {0.0412625632426235, -0.5312794640198946},
    {0.0412625632426235, 0.5312794640198946},
    {0.0399537411327203, -0.5718956462026340},
    {0.0399537411327203, 0.5718956462026340},
    {0.0385501531786156, -0.6111553551723933},
    {0.0385501531786156, 0.6111553551723933},
    {0.0370551285402400, -0.6489654712546573},
    {0.0370551285402400, 0.6489654712546573},
    {0.0354722132568824, -0.6852363130542333},
    {0.0354722132568824, 0.6852363130542333},
    {0.0338051618371416, -0.7198818501716109},
    {0.0338051618371416, 0.7198818501716109},
    {0.0320579283548516, -0.7528199072605319},
    {0.0320579283548516, 0.7528199072605319},
    {0.0302346570724025, -0.7839723589433414},
    {0.0302346570724025, 0.7839723589433414},
    {0.0283396726142595, -0.8132653151227975},
    {0.0283396726142595, 0.8132653151227975},
    {0.0263774697150547, -0.8406292962525803},
    {0.0263774697150547, 0.8406292962525803},
    {0.0243527025687109, -0.8659993981540928},
    {0.0243527025687109, 0.8659993981540928},
    {0.0222701738083833, -0.8893154459951141},
    {0.0222701738083833, 0.8893154459951141},
    {0.0201348231535302, -0.9105221370785028},
    {0.0201348231535302, 0.9105221370785028},
    {0.0179517157756973, -0.9295691721319396},
    {0.0179517157756973, 0.9295691721319396},
    {0.0157260304760247, -0.9464113748584028},
    {0.0157260304760247, 0.9464113748584028},
    {0.0134630478967186, -0.9610087996520538},
    {0.0134630478967186, 0.9610087996520538},
    {0.0111681394601311, -0.9733268277899110},
    {0.0111681394601311, 0.9733268277899110},
    {0.0088467598263639, -0.9833362538846260},
    {0.0088467598263639, 0.9833362538846260},
    {0.0065044579689784, -0.9910133714767443},
    {0.0065044579689784, 0.9910133714767443},
    {0.0041470332605625, -0.9963401167719553},
    {0.0041470332605625, 0.9963401167719553},
    {0.0017832807216964, -0.9993050417357722},
    {0.0017832807216964, 0.9993050417357722},
}};

// Composite Gauss-Legendre rule on [x0, x1] in num_panels panels.
template <typename F>
static double Integrate(F f, double x0, double x1, int num_panels) {
  double sum = 0.0;
  const double width = (x1 - x0) / num_panels;
  for (int panel = 0; panel < num_panels; ++panel) {
    const double mid = x0 + (panel + 0.5) * width;
    for (const auto& [w, x] : kGaussLegendre64) {
      sum += 0.5 * width * w * f(mid + 0.5 * width * x);
    }
  }
  return sum;
}

static double ChebyshevT(int k, double t) {
  return std::cos(k * std::acos(std::clamp(t, -1.0, 1.0)));
}

// value at the monomial t^l of a linear functional given on T_0, ..., T_l:
//   t^l = 2^{1-l} sum_{j=0}^{floor(l/2)} binom(l, j) T_{l-2j},
// with the T_0 term halved. The coefficients are positive and sum to one.
static double AtMonomial(int l, const std::vector<double>& at_chebyshev) {
  double sum = 0.0;
  double binomial = 1.0;
  for (int j = 0; 2 * j <= l; ++j) {
    const double weight = (l - 2 * j == 0) ? 0.5 : 1.0;
    sum += weight * binomial * at_chebyshev[l - 2 * j];
    binomial *= static_cast<double>(l - j) / (j + 1);
  }
  return sum * std::ldexp(1.0, 1 - l);
}

// sum_k gamma_k(m, n) T_k(t)
static double AddBackPolynomial(const SingularIntegrals& si, int nf, int mf,
                                int m, int n, double t) {
  double sum = 0.0;
  for (int k = 0; k <= mf + nf; ++k) {
    const int knm = (k * (nf + 1) + n) * (mf + 1) + m;
    sum += si.chebyshev_coefficients[knm] * ChebyshevT(k, t);
  }
  return sum;
}

// The Chebyshev coefficients describe the polynomials whose monomial
// coefficients are cmns: cmn from its closed form (6.182 in TNOV), cmns from
// cmn by (6.291). At mf + nf = 13 the monomial form still evaluates to 1e-12.
TEST(TestSingularIntegrals, ChebyshevCoefficientsReproduceCmns) {
  static constexpr double kTolerance = 1.0e-11;

  const bool lasym = false;
  const int nfp = 5;
  const int mpol = 6;
  const int ntor = 6;
  // will be auto-adjusted by Sizes
  const int ntheta = 0;
  const int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);
  SurfaceGeometry sg(&s, &fb, &tp);

  const int nf = ntor;
  const int mf = mpol + 1;
  SingularIntegrals si(&s, &fb, &tp, &sg, nf, mf);

  // cmn(l, m, n), zero outside |m - n| <= l <= m + n and for l - m - n odd
  const auto cmn = [](int l, int m, int n) {
    if (m < 0 || n < 0 || l < std::abs(m - n) || l > m + n ||
        (l - m - n) % 2 != 0) {
      return 0.0;
    }
    const int sign = ((l - m + n) / 2) % 2 == 0 ? 1 : -1;
    // exp(lgamma(n + 1) - lgamma(m + 1)) == n! / m!
    const double numFac = std::lgamma((m + n + l) / 2 + 1);
    const double denFac1 = std::lgamma((m + n - l) / 2 + 1);
    const double denFac2 = std::lgamma((l + std::abs(m - n)) / 2 + 1);
    const double denFac3 = std::lgamma((l - std::abs(m - n)) / 2 + 1);
    return sign * std::exp(numFac - denFac1 - denFac2 - denFac3);
  };

  for (int n = 0; n < nf + 1; ++n) {
    for (int m = 0; m < mf + 1; ++m) {
      for (const double t : {-1.0, -0.9, -0.35, 0.2, 0.7, 1.0}) {
        double expected = 0.0;
        for (int l = 0; l <= m + n; ++l) {
          const double cmns = (cmn(l, m, n) + cmn(l, m, n - 1) +
                               cmn(l, m - 1, n) + cmn(l, m - 1, n - 1)) /
                              ((m == 0 && n == 0) ? 1.0 : 2.0);
          expected += cmns * std::pow(t, l);
        }
        EXPECT_TRUE(IsCloseRelAbs(
            expected, AddBackPolynomial(si, nf, mf, m, n, t), kTolerance))
            << "(m, n) = (" << m << ", " << n << ") at t = " << t;
      }
    }  // m
  }  // n
}  // ChebyshevCoefficientsReproduceCmns

// cmns reaches 1e11 at mf + nf = 33 while the polynomial it describes stays
// below one, which is what the Chebyshev coefficients have to reflect. The
// radial Zernike polynomial of (m, n) is (-1)^n at t = 1 and (-1)^m at
// t = -1, so (6.291) leaves p_mn(1) = [n == 0] and p_mn(-1) = [m == 0].
TEST(TestSingularIntegrals, ChebyshevCoefficientsStayBounded) {
  static constexpr double kTolerance = 1.0e-13;

  const bool lasym = false;
  const int nfp = 5;
  const int mpol = 16;
  const int ntor = 16;
  const int ntheta = 0;
  const int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);
  SurfaceGeometry sg(&s, &fb, &tp);

  const int nf = ntor;
  const int mf = mpol + 1;
  SingularIntegrals si(&s, &fb, &tp, &sg, nf, mf);

  EXPECT_LE(si.chebyshev_coefficients.cwiseAbs().maxCoeff(), 1.0 + kTolerance);
  for (int n = 0; n < nf + 1; ++n) {
    for (int m = 0; m < mf + 1; ++m) {
      double at_plus = 0.0;
      double at_minus = 0.0;
      for (int k = 0; k <= mf + nf; ++k) {
        const int knm = (k * (nf + 1) + n) * (mf + 1) + m;
        at_plus += si.chebyshev_coefficients[knm];
        at_minus += (k % 2 == 0 ? 1.0 : -1.0) * si.chebyshev_coefficients[knm];
        if (k > m + n) {
          EXPECT_EQ(si.chebyshev_coefficients[knm], 0.0);
        }
      }
      EXPECT_TRUE(IsCloseRelAbs(n == 0 ? 1.0 : 0.0, at_plus, kTolerance))
          << "(m, n) = (" << m << ", " << n << ")";
      EXPECT_TRUE(IsCloseRelAbs(m == 0 ? 1.0 : 0.0, at_minus, kTolerance))
          << "(m, n) = (" << m << ", " << n << ")";
    }  // m
  }  // n
}  // ChebyshevCoefficientsStayBounded

// prepareUpdate has to deliver the Chebyshev moments
//   M^+_k = int_{-1}^{1} T_k(t) / sqrt(ap t^2 + 2 d t + am) dt
// (M^-_k with ap and am exchanged) for all k in [0, kL], at a range of
// resolutions and metric coefficients, and the functional of Eq. (A17) on T_k,
//   S^+_k = int_{-1}^{1} T_k(t) (Ap t^2 + 2 D t + Am) /
//                        (ap t^2 + 2 d t + am)^{3/2} dt
// (S^-_k with ap, am and Ap, Am exchanged). The reference is a composite
// Gauss-Legendre quadrature of the defining integrals, which is independent of
// the recurrence; the monomial moments T^{+/-}_l that follow from the
// Chebyshev ones are held to the quadrature of their own integral as well.
struct MomentCase {
  int mpol;
  int ntor;
  // metric coefficients (a, b2, c) = (guu, 2 guv, gvv) of the tangent plane
  double a;
  double b2;
  double c;
  // panels of the reference quadrature
  int panels;
  double tolerance;
  // R0 and R1 of Eq. (A17) divide by 4 a c - b2^2
  double s_tolerance;
};

class ChebyshevMomentsAccuracyTest
    : public ::testing::TestWithParam<MomentCase> {};

TEST_P(ChebyshevMomentsAccuracyTest, MatchesQuadrature) {
  const auto [mpol, ntor, a_val, b2_val, c_val, panels, tolerance,
              s_tolerance] = GetParam();

  const bool lasym = false;
  const int nfp = 5;
  const int ntheta = 0;
  // Sizes requires nzeta >= 2 * ntor + 1
  const int nzeta = 4 * (ntor + 1);

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);
  SurfaceGeometry sg(&s, &fb, &tp);

  const int nf = ntor;
  const int mf = mpol + 1;
  const int kL = mf + nf;
  SingularIntegrals si(&s, &fb, &tp, &sg, nf, mf);

  const double ap = a_val + b2_val + c_val;
  const double am = a_val - b2_val + c_val;
  const double d = c_val - a_val;
  ASSERT_LT(d * d, ap * am) << "test setup: need a positive-definite metric";

  const int numLocal = tp.ztMax - tp.ztMin;
  Eigen::VectorXd a = Eigen::VectorXd::Constant(numLocal, a_val);
  Eigen::VectorXd b2 = Eigen::VectorXd::Constant(numLocal, b2_val);
  Eigen::VectorXd c = Eigen::VectorXd::Constant(numLocal, c_val);
  // second-fundamental-form coefficients (A, B2, C), not proportional to the
  // metric
  const double A_val = 0.062236;
  const double B2_val = -0.004955;
  const double C_val = 0.050789;
  const double Ap = A_val + B2_val + C_val;
  const double Am = A_val - B2_val + C_val;
  const double D = C_val - A_val;
  Eigen::VectorXd A = Eigen::VectorXd::Constant(numLocal, A_val);
  Eigen::VectorXd B2 = Eigen::VectorXd::Constant(numLocal, B2_val);
  Eigen::VectorXd C = Eigen::VectorXd::Constant(numLocal, C_val);

  si.prepareUpdate(a, b2, c, A, B2, C, /*fullUpdate=*/true);

  std::vector<double> moments_p(kL + 1);
  std::vector<double> moments_m(kL + 1);
  for (int k = 0; k <= kL; ++k) {
    const double reference_p = Integrate(
        [&](double t) {
          return ChebyshevT(k, t) / std::sqrt(ap * t * t + 2.0 * d * t + am);
        },
        -1.0, 1.0, panels);
    const double reference_m = Integrate(
        [&](double t) {
          return ChebyshevT(k, t) / std::sqrt(am * t * t + 2.0 * d * t + ap);
        },
        -1.0, 1.0, panels);
    // Coefficients are uniform, so every kl must give the same value.
    for (int kl = 0; kl < numLocal; ++kl) {
      ASSERT_TRUE(
          IsCloseRelAbs(reference_p, si.chebyshev_moments_p[k][kl], tolerance))
          << "M^+ at k = " << k << ", kl = " << kl;
      ASSERT_TRUE(
          IsCloseRelAbs(reference_m, si.chebyshev_moments_m[k][kl], tolerance))
          << "M^- at k = " << k << ", kl = " << kl;
    }
    moments_p[k] = si.chebyshev_moments_p[k][0];
    moments_m[k] = si.chebyshev_moments_m[k][0];
  }

  for (int l = 0; l <= kL; ++l) {
    const double reference_p = Integrate(
        [&](double t) {
          return std::pow(t, l) / std::sqrt(ap * t * t + 2.0 * d * t + am);
        },
        -1.0, 1.0, panels);
    const double reference_m = Integrate(
        [&](double t) {
          return std::pow(t, l) / std::sqrt(am * t * t + 2.0 * d * t + ap);
        },
        -1.0, 1.0, panels);
    EXPECT_TRUE(IsCloseRelAbs(reference_p, AtMonomial(l, moments_p), tolerance))
        << "T^+ at l = " << l;
    EXPECT_TRUE(IsCloseRelAbs(reference_m, AtMonomial(l, moments_m), tolerance))
        << "T^- at l = " << l;
  }

  for (int k = 0; k <= kL; ++k) {
    const double reference_p = Integrate(
        [&](double t) {
          const double q = ap * t * t + 2.0 * d * t + am;
          return ChebyshevT(k, t) * (Ap * t * t + 2.0 * D * t + Am) /
                 (q * std::sqrt(q));
        },
        -1.0, 1.0, panels);
    const double reference_m = Integrate(
        [&](double t) {
          const double q = am * t * t + 2.0 * d * t + ap;
          return ChebyshevT(k, t) * (Am * t * t + 2.0 * D * t + Ap) /
                 (q * std::sqrt(q));
        },
        -1.0, 1.0, panels);
    EXPECT_TRUE(
        IsCloseRelAbs(reference_p, si.chebyshev_s_moments_p[k][0], s_tolerance))
        << "S^+ at k = " << k;
    EXPECT_TRUE(
        IsCloseRelAbs(reference_m, si.chebyshev_s_moments_m[k][0], s_tolerance))
        << "S^- at k = " << k;
  }
}

// The first three cases share ap = 0.7, am = 3.3, d = 0.4 at kL = 15, 27 and
// 45. The fourth takes the metric of the cth_like_free_bdy boundary at the
// point of its largest cross term, at kL = 33. The next two have
// gvv = guu / 100 and guu = gvv / 100 with a cross term at 90 percent of its
// limit, where the closed form of T_0 cancels. The last has its cross term at
// 99.995 percent of the limit: the characteristic roots lie within 0.005 of
// the unit circle, the extent of the boundary-value problem is at its cap, and
// the integrand is peaked enough to need 1024 panels. The condition number of
// the problem, max Q / min Q = 4e4, sets the tolerances there.
INSTANTIATE_TEST_SUITE_P(
    ResolutionSweep, ChebyshevMomentsAccuracyTest,
    ::testing::Values(
        MomentCase{6, 8, 0.8, -1.3, 1.2, 16, 1.0e-13, 1.0e-13},
        MomentCase{12, 14, 0.8, -1.3, 1.2, 16, 1.0e-13, 1.0e-13},
        MomentCase{20, 24, 0.8, -1.3, 1.2, 16, 1.0e-13, 1.0e-13},
        MomentCase{16, 16, 1.360e-02, 2.241e-02, 4.626e-02, 16, 1.0e-13,
                   1.0e-11},
        MomentCase{16, 16, 1.0, 0.18, 1.0e-02, 16, 1.0e-13, 1.0e-11},
        MomentCase{16, 16, 1.0e-02, -0.18, 1.0, 16, 1.0e-13, 1.0e-11},
        MomentCase{16, 16, 1.0, 1.9999, 1.0, 1024, 1.0e-11, 1.0e-8}),
    [](const ::testing::TestParamInfo<MomentCase>& info) {
      return "case" + std::to_string(info.index) + "_mpol" +
             std::to_string(info.param.mpol) + "_ntor" +
             std::to_string(info.param.ntor);
    });

// The pentadiagonal elimination runs without pivoting and the extent of the
// boundary-value problem follows the characteristic roots, so the moments are
// checked over the metrics a boundary can present: gvv / guu from 1e-2 to 1e2
// and a cross term up to 95 percent of its limit, a different metric at every
// grid point of one update.
TEST(TestSingularIntegrals, ChebyshevMomentsOverTheMetricRange) {
  static constexpr double kTolerance = 1.0e-13;
  static constexpr int kPanels = 16;

  const bool lasym = false;
  const int nfp = 5;
  const int mpol = 16;
  const int ntor = 16;
  const int ntheta = 0;
  const int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);
  SurfaceGeometry sg(&s, &fb, &tp);

  const int nf = ntor;
  const int mf = mpol + 1;
  const int kL = mf + nf;
  SingularIntegrals si(&s, &fb, &tp, &sg, nf, mf);

  const int numLocal = tp.ztMax - tp.ztMin;
  Eigen::VectorXd a(numLocal);
  Eigen::VectorXd b2(numLocal);
  Eigen::VectorXd c(numLocal);
  for (int kl = 0; kl < numLocal; ++kl) {
    // both ratios sweep their range, at incommensurate rates
    const double log_ratio = 2.0 * std::sin(0.37 * kl);
    const double cross = 0.95 * std::sin(1.13 * kl + 0.5);
    a[kl] = 1.0;
    c[kl] = std::pow(10.0, log_ratio);
    b2[kl] = 2.0 * cross * std::sqrt(a[kl] * c[kl]);
  }
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(numLocal);

  si.prepareUpdate(a, b2, c, zero, zero, zero, /*fullUpdate=*/false);

  // every 7th point keeps the reference quadrature short
  int checked = 0;
  for (int kl = 0; kl < numLocal; kl += 7) {
    const double ap = a[kl] + b2[kl] + c[kl];
    const double am = a[kl] - b2[kl] + c[kl];
    const double d = c[kl] - a[kl];
    for (int k = 0; k <= kL; ++k) {
      const double reference_p = Integrate(
          [&](double t) {
            return ChebyshevT(k, t) / std::sqrt(ap * t * t + 2.0 * d * t + am);
          },
          -1.0, 1.0, kPanels);
      const double reference_m = Integrate(
          [&](double t) {
            return ChebyshevT(k, t) / std::sqrt(am * t * t + 2.0 * d * t + ap);
          },
          -1.0, 1.0, kPanels);
      EXPECT_TRUE(
          IsCloseRelAbs(reference_p, si.chebyshev_moments_p[k][kl], kTolerance))
          << "M^+ at k = " << k << " for gvv / guu = " << c[kl]
          << ", b2 = " << b2[kl];
      EXPECT_TRUE(
          IsCloseRelAbs(reference_m, si.chebyshev_moments_m[k][kl], kTolerance))
          << "M^- at k = " << k << " for gvv / guu = " << c[kl]
          << ", b2 = " << b2[kl];
    }
    ++checked;
  }
  EXPECT_GT(checked, 100);
}  // ChebyshevMomentsOverTheMetricRange

// Reference 2D Fourier coefficients of the tangent-plane kernels that
// RegularizedIntegrals subtracts, for one set of metric (a, b2, c) and
// second-fundamental-form (A, B2, C) coefficients:
//   F1(m, n) = int cos(m du - n dv) / sqrt(a tu^2 + b2 tu tv + c tv^2)
//   F2(m, n) = int cos(m du - n dv) (A tu^2 + B2 tu tv + C tv^2)
//                  / (a tu^2 + b2 tu tv + c tv^2)^{3/2}
// over (du, dv) in (-pi, pi)^2 with tu = 2 tan(du/2), tv = 2 tan(dv/2).
// The integrands are 1/r singular at the origin; polar coordinates about it
// make r * kernel smooth, and a tensor Gauss-Legendre rule on angular panels
// (the square's corners are panel boundaries) converges geometrically. The
// panels are subdivided with m + |n| so that each carries a few wavelengths
// of cos(m du - n dv).
static std::pair<double, double> TangentPlaneKernelReference(
    int m, int n, double a, double b2, double c, double A, double B2,
    double C) {
  const int subdivisions = 1 + (m + std::abs(n)) / 8;
  double f1 = 0.0;
  double f2 = 0.0;
  for (int panel = 0; panel < 8 * subdivisions; ++panel) {
    const double p0 = panel * M_PI / (4.0 * subdivisions);
    const double p1 = (panel + 1) * M_PI / (4.0 * subdivisions);
    for (const auto& [wp, xp] : kGaussLegendre64) {
      const double psi = 0.5 * (p1 - p0) * xp + 0.5 * (p1 + p0);
      const double wpsi = 0.5 * (p1 - p0) * wp;
      const double cp = std::cos(psi);
      const double sp = std::sin(psi);
      const double rmax = M_PI / std::max(std::abs(cp), std::abs(sp));
      const auto radial = [&](double r) {
        const double du = r * cp;
        const double dv = r * sp;
        const double tu = 2.0 * std::tan(du / 2.0);
        const double tv = 2.0 * std::tan(dv / 2.0);
        const double q1 = a * tu * tu + b2 * tu * tv + c * tv * tv;
        const double q2 = A * tu * tu + B2 * tu * tv + C * tv * tv;
        const double cs = std::cos(m * du - n * dv);
        return std::pair<double, double>(r * cs / std::sqrt(q1),
                                         r * cs * q2 / (q1 * std::sqrt(q1)));
      };
      f1 += wpsi * Integrate([&](double r) { return radial(r).first; }, 0.0,
                             rmax, subdivisions);
      f2 += wpsi * Integrate([&](double r) { return radial(r).second; }, 0.0,
                             rmax, subdivisions);
    }
  }
  return {f1, f2};
}

// The analytic add-back must reproduce the Fourier coefficients of exactly
// the kernels subtracted numerically, including the metric and curvature
// cross terms (guv, auv) that are non-zero on any non-axisymmetric surface.
// With a delta source at grid point (l0, k0), bvec_sin[(m, n)] and
// grpmn_sin[(m, n), kl0] equal F1(m, n) / (2 pi) * sin(m u0 - n v0) and
// F2(m, n) / (2 pi) * sin(m u0 - n v0).
//
// The second resolution has mf + nf = 41. There the monomial coefficients of
// the add-back polynomials reach 1e14, and a sum over them would leave the
// highest modes with an error of 1e-3 to 1e-2.
struct AddBackCase {
  bool lasym;
  int mpol;
  int ntor;
  int nzeta;
  // modes with m + |n| below this are skipped
  int min_band;
  int min_checked;
};

class AnalyticAddBackTest : public ::testing::TestWithParam<AddBackCase> {};

TEST_P(AnalyticAddBackTest, MatchesSubtractedKernels) {
  static constexpr double kTolerance = 1.0e-9;

  const auto [lasym, mpol, ntor, nzeta, min_band, min_checked] = GetParam();
  const int nfp = 2;
  const int ntheta = 0;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);
  SurfaceGeometry sg(&s, &fb, &tp);

  const int nf = ntor;
  const int mf = mpol + 1;
  SingularIntegrals si(&s, &fb, &tp, &sg, nf, mf);

  // coefficients of a helically deformed circular torus (R0 = 1, a = 0.3,
  // 0.05 cos(theta - 2 phi) deformation) at one surface point
  const double a = 0.100265;
  const double b2 = -0.012765;
  const double c = 0.387789;
  const double A = 0.062236;
  const double B2 = -0.004955;
  const double C = 0.050789;

  const int numLocal = tp.ztMax - tp.ztMin;
  sg.guu = Eigen::VectorXd::Constant(numLocal, a);
  sg.guv = Eigen::VectorXd::Constant(numLocal, b2);
  sg.gvv = Eigen::VectorXd::Constant(numLocal, c);
  sg.auu = Eigen::VectorXd::Constant(numLocal, A);
  sg.auv = Eigen::VectorXd::Constant(numLocal, B2);
  sg.avv = Eigen::VectorXd::Constant(numLocal, C);

  const int l0 = 3;
  const int k0 = 5;
  const int kl0 = l0 * s.nZeta + k0;
  Eigen::VectorXd bDotN = Eigen::VectorXd::Zero(numLocal);
  bDotN[kl0] = 1.0 / s.wInt[l0];
  si.update(bDotN, /*fullUpdate=*/true);

  const double u0 = 2.0 * M_PI * l0 / s.nThetaEven;
  const double v0 = 2.0 * M_PI * k0 / s.nZeta;
  int checked = 0;
  for (int n = -nf; n <= nf; ++n) {
    for (int m = 0; m <= mf; ++m) {
      // m = 0 keeps only n >= 0 in NESTOR's basis
      if (m == 0 && n < 0) continue;
      if (m + std::abs(n) < min_band) continue;
      const double sn = std::sin(m * u0 - n * v0);
      if (std::abs(sn) < 0.2) continue;
      const auto [f1, f2] =
          TangentPlaneKernelReference(m, n, a, b2, c, A, B2, C);
      const int idx = (nf + n) * (mf + 1) + m;
      const double expected_bvec = f1 / (2.0 * M_PI) * sn;
      const double expected_grpmn = f2 / (2.0 * M_PI) * sn;
      if (lasym) {
        const double cs = std::cos(m * u0 - n * v0);
        EXPECT_TRUE(
            IsCloseRelAbs(f1 / (2.0 * M_PI) * cs, si.bvec_cos[idx], kTolerance))
            << "bvec_cos at (m, n) = (" << m << ", " << n << ")";
        EXPECT_TRUE(IsCloseRelAbs(
            f2 / (2.0 * M_PI) * cs,
            si.grpmn_cos[static_cast<std::size_t>(idx) * numLocal + kl0],
            kTolerance))
            << "grpmn_cos at (m, n) = (" << m << ", " << n << ")";
      }
      EXPECT_TRUE(IsCloseRelAbs(expected_bvec, si.bvec_sin[idx], kTolerance))
          << "bvec_sin at (m, n) = (" << m << ", " << n << "): expected "
          << expected_bvec << ", got " << si.bvec_sin[idx];
      EXPECT_TRUE(IsCloseRelAbs(
          expected_grpmn,
          si.grpmn_sin[static_cast<std::size_t>(idx) * numLocal + kl0],
          kTolerance))
          << "grpmn_sin at (m, n) = (" << m << ", " << n << "): expected "
          << expected_grpmn << ", got "
          << si.grpmn_sin[static_cast<std::size_t>(idx) * numLocal + kl0];
      ++checked;
    }
  }
  EXPECT_GT(checked, min_checked);
}

INSTANTIATE_TEST_SUITE_P(
    Resolutions, AnalyticAddBackTest,
    ::testing::Values(AddBackCase{false, 8, 4, 24, 0, 40},
                      AddBackCase{true, 8, 4, 24, 0, 40},
                      AddBackCase{false, 20, 20, 48, 37, 12},
                      AddBackCase{true, 20, 20, 48, 37, 12}),
    [](const ::testing::TestParamInfo<AddBackCase>& info) {
      return std::string(info.param.lasym ? "lasym" : "symmetric") + "_mpol" +
             std::to_string(info.param.mpol) + "_ntor" +
             std::to_string(info.param.ntor);
    });

}  // namespace vmecpp
