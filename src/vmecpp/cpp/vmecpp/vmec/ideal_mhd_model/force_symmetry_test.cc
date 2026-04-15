// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/force_symmetry.h"

#include <vector>

#include "gtest/gtest.h"

namespace vmecpp {
namespace {

TEST(ForceSymmetryTest, StandardParityDecompositionMatchesReflectionFormula) {
  const Sizes sizes(/*lasym=*/true, /*nfp=*/1, /*mpol=*/2, /*ntor=*/0,
                    /*ntheta=*/8, /*nzeta=*/4);

  std::vector<double> full(sizes.nZnT);
  for (int k = 0; k < sizes.nZeta; ++k) {
    for (int l = 0; l < sizes.nThetaEff; ++l) {
      full[k * sizes.nThetaEff + l] = 100.0 * k + l;
    }
  }

  std::vector<double> sym(sizes.nZeta * sizes.nThetaReduced);
  std::vector<double> asym(sizes.nZeta * sizes.nThetaReduced);
  DecomposeForceComponent(sizes, full, ReflectionParity::kStandard, sym, asym);

  for (int k = 0; k < sizes.nZeta; ++k) {
    const int k_ref = (sizes.nZeta - k) % sizes.nZeta;
    for (int l = 0; l < sizes.nThetaReduced; ++l) {
      const int l_ref = l == 0 ? 0 : sizes.nThetaEven - l;
      const double value = full[k * sizes.nThetaEff + l];
      const double reflected = full[k_ref * sizes.nThetaEff + l_ref];
      const int idx = k * sizes.nThetaReduced + l;

      EXPECT_DOUBLE_EQ(sym[idx], 0.5 * (value + reflected));
      EXPECT_DOUBLE_EQ(asym[idx], 0.5 * (value - reflected));
    }
  }
}

TEST(ForceSymmetryTest, ReversedParityDecompositionMatchesReflectionFormula) {
  const Sizes sizes(/*lasym=*/true, /*nfp=*/1, /*mpol=*/2, /*ntor=*/0,
                    /*ntheta=*/8, /*nzeta=*/4);

  std::vector<double> full(sizes.nZnT * 2);
  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      for (int l = 0; l < sizes.nThetaEff; ++l) {
        const int idx = ((j * sizes.nZeta + k) * sizes.nThetaEff) + l;
        full[idx] = 1000.0 * j + 100.0 * k + l;
      }
    }
  }

  std::vector<double> sym(2 * sizes.nZeta * sizes.nThetaReduced);
  std::vector<double> asym(2 * sizes.nZeta * sizes.nThetaReduced);
  DecomposeForceComponent(sizes, full, ReflectionParity::kReversed, sym, asym);

  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      const int k_ref = (sizes.nZeta - k) % sizes.nZeta;
      for (int l = 0; l < sizes.nThetaReduced; ++l) {
        const int l_ref = l == 0 ? 0 : sizes.nThetaEven - l;
        const int idx_full = ((j * sizes.nZeta + k) * sizes.nThetaEff) + l;
        const int idx_ref =
            ((j * sizes.nZeta + k_ref) * sizes.nThetaEff) + l_ref;
        const int idx = ((j * sizes.nZeta + k) * sizes.nThetaReduced) + l;

        EXPECT_DOUBLE_EQ(sym[idx], 0.5 * (full[idx_full] - full[idx_ref]));
        EXPECT_DOUBLE_EQ(asym[idx], 0.5 * (full[idx_full] + full[idx_ref]));
      }
    }
  }
}

}  // namespace
}  // namespace vmecpp
