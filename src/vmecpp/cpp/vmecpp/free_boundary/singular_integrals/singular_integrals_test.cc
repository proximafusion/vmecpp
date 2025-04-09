// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/singular_integrals/singular_integrals.h"

#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"

namespace vmecpp {

using testing::IsCloseRelAbs;

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

}  // namespace vmecpp
