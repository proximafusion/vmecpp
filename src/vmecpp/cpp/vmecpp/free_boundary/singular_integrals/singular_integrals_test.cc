// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/singular_integrals/singular_integrals.h"

#include <array>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"

namespace vmecpp {

using testing::IsCloseRelAbs;

// 64-point Gauss-Legendre quadrature on [-1, 1]. Stored as {weight, abscissa}
// pairs. Exact for polynomials up to degree 127, and for smooth non-polynomial
// integrands (like t^l / sqrt(am + 2*d*t + ap*t^2) with realistic coefficients)
// it reaches machine precision long before L = mf + nf = 13.
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

// Reference T^{+/-}_l via numerical quadrature of the defining integral
//   T^+_l = integral_{-1}^{+1} t^l / sqrt(am + 2*d*t + ap*t^2) dt
//   T^-_l = integral_{-1}^{+1} t^l / sqrt(ap + 2*d*t + am*t^2) dt
// (the integrands are smooth on [-1,+1] as long as the discriminant
// d^2 - ap*am < 0, i.e. b2 != 0).
// 64-point Gauss-Legendre reaches machine precision for these integrands well
// beyond l = L = mf + nf of interest (the integrand is analytic in a strip
// around [-1, 1] whose width scales with the distance from the roots of the
// square-root radicand; see Trefethen, "Is Gauss Quadrature Better Than
// Clenshaw-Curtis?").
static double TlpReference(int l, double ap, double am, double d) {
  double sum = 0.0;
  for (const auto& [w, t] : kGaussLegendre64) {
    const double tl = std::pow(t, l);
    const double radicand = am + 2.0 * d * t + ap * t * t;
    sum += w * tl / std::sqrt(radicand);
  }
  return sum;
}

static double TlmReference(int l, double ap, double am, double d) {
  // T^-_l swaps ap <-> am in the radicand (see eq. 6.206 in the numerics doc).
  double sum = 0.0;
  for (const auto& [w, t] : kGaussLegendre64) {
    const double tl = std::pow(t, l);
    const double radicand = ap + 2.0 * d * t + am * t * t;
    sum += w * tl / std::sqrt(radicand);
  }
  return sum;
}

TEST(TestSingularIntegrals, CheckConstants) {
  static constexpr double kTolerance = 1.0e-12;

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

  for (int n = 0; n < nf + 1; ++n) {
    for (int m = 0; m < mf + 1; ++m) {
      for (int l = std::abs(m - n); l <= m + n; l += 2) {
        const int lnm = (l * (nf + 1) + n) * (mf + 1) + m;

        const int sign = ((l - m + n) / 2) % 2 == 0 ? 1 : -1;

        // need to compute n! / m!
        // Note: n! = gamma(n + 1)
        // Note: lgamma(n + 1) == log(n!)
        // exp(lgamma(n +1) - lgamma(m+1)) == n! / m!
        const double numFac = std::lgamma((m + n + l) / 2 + 1);
        const double denFac1 = std::lgamma((m + n - l) / 2 + 1);
        const double denFac2 = std::lgamma((l + std::abs(m - n)) / 2 + 1);
        const double denFac3 = std::lgamma((l - std::abs(m - n)) / 2 + 1);

        const double cmnRef =
            sign * std::exp(numFac - denFac1 - denFac2 - denFac3);

        EXPECT_TRUE(IsCloseRelAbs(cmnRef, si.cmn[lnm], kTolerance));
      }  // l
    }  // m
  }  // n
}  // CheckConstants

// Verify that prepareUpdate computes T^{+/-}_l accurately for ALL l in [0, L]
// at a range of Fourier resolutions.
//
// Resolutions span the low-kL regime (forward recurrence is accurate, so the
// forward branch of prepareUpdate is exercised) all the way up to high-kL
// (Miller backward recurrence fires; forward would lose all digits).
//
// Reference: 64-point Gauss-Legendre quadrature of the defining integral.
// This is independent of both recurrence directions and reaches ~1e-13
// relative accuracy at the chosen coefficients up to l = 45.
class TlpTlmAccuracyTest
    : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(TlpTlmAccuracyTest, MatchesQuadrature) {
  // GL-64 reference noise at l near kL is ~1e-13; 1e-11 leaves a safe margin
  // while staying far below the forward-recurrence error that Miller corrects.
  static constexpr double kTolerance = 1.0e-11;

  const auto [mpol, ntor] = GetParam();

  const bool lasym = false;
  const int nfp = 5;
  const int ntheta = 0;
  // Sizes requires nzeta >= 2*ntor+1 (to resolve all toroidal modes).
  // Pick the smallest power-of-two-friendly value that works for each case.
  const int nzeta = 4 * (ntor + 1);

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);
  SurfaceGeometry sg(&s, &fb, &tp);

  const int nf = ntor;
  const int mf = mpol + 1;
  const int kL = mf + nf;
  SingularIntegrals si(&s, &fb, &tp, &sg, nf, mf);

  // Geometry coefficients chosen so am/ap ~ 4.7 and the discriminant
  // d^2 - ap*am = 0.16 - 2.31 < 0 (so the integrand is smooth on [-1, 1]).
  //   ap = a + b2 + c = 0.7
  //   am = a - b2 + c = 3.3
  //   d  = c - a     = 0.4
  // At these coefficients the forward recurrence loses ~kL * log10(am/ap)
  // ~ 0.67 * kL significant digits; for kL >= 15 it loses > 10 digits and the
  // Miller activation threshold (growth > 1e10) triggers.
  const double a_val = 0.8;
  const double b2_val = -1.3;
  const double c_val = 1.2;
  const double ap = a_val + b2_val + c_val;
  const double am = a_val - b2_val + c_val;
  const double d = c_val - a_val;
  ASSERT_GT(am, ap) << "test setup: need am > ap to exercise Miller path";
  ASSERT_LT(d * d, ap * am) << "test setup: need smooth integrand on [-1,1]";

  const int numLocal = tp.ztMax - tp.ztMin;
  std::vector<double> a(numLocal, a_val);
  std::vector<double> b2(numLocal, b2_val);
  std::vector<double> c(numLocal, c_val);
  std::vector<double> A(numLocal, 0.0);
  std::vector<double> B2(numLocal, 0.0);
  std::vector<double> C(numLocal, 0.0);

  si.prepareUpdate(a, b2, c, A, B2, C, /*fullUpdate=*/false);

  // Reference values via Gauss-Legendre, independent of either recurrence.
  std::vector<double> Tp_ref(kL + 1);
  std::vector<double> Tm_ref(kL + 1);
  for (int l = 0; l <= kL; ++l) {
    Tp_ref[l] = TlpReference(l, ap, am, d);
    Tm_ref[l] = TlmReference(l, ap, am, d);
  }

  // Check every l in [0, kL], across every local grid point.
  // Coefficients are uniform, so every kl must give the same value.
  for (int l = 0; l <= kL; ++l) {
    for (int kl = 0; kl < numLocal; ++kl) {
      EXPECT_TRUE(IsCloseRelAbs(Tp_ref[l], si.Tlp[l][kl], kTolerance))
          << "Tlp mismatch at mpol=" << mpol << ", ntor=" << ntor << ", l=" << l
          << ", kl=" << kl << ": quadrature ref = " << Tp_ref[l]
          << ", computed = " << si.Tlp[l][kl]
          << ", abs diff = " << std::abs(Tp_ref[l] - si.Tlp[l][kl]);
      EXPECT_TRUE(IsCloseRelAbs(Tm_ref[l], si.Tlm[l][kl], kTolerance))
          << "Tlm mismatch at mpol=" << mpol << ", ntor=" << ntor << ", l=" << l
          << ", kl=" << kl << ": quadrature ref = " << Tm_ref[l]
          << ", computed = " << si.Tlm[l][kl]
          << ", abs diff = " << std::abs(Tm_ref[l] - si.Tlm[l][kl]);
    }
  }
}

// (mpol, ntor) pairs spanning the forward-stable regime (kL=15, Miller just
// barely fires) up to high-kL where forward would lose >30 digits.
INSTANTIATE_TEST_SUITE_P(
    ResolutionSweep, TlpTlmAccuracyTest,
    ::testing::Values(std::pair<int, int>{6, 8}, std::pair<int, int>{12, 14},
                      std::pair<int, int>{20, 24}),
    [](const ::testing::TestParamInfo<std::pair<int, int>>& info) {
      return "mpol" + std::to_string(info.param.first) + "_ntor" +
             std::to_string(info.param.second);
    });

}  // namespace vmecpp
