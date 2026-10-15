// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/jacobian_kernel.h"

#include <array>
#include <cmath>

#include "gtest/gtest.h"

namespace vmecpp {
namespace {

TEST(JacobianKernelTest,
     UsesFourthOrderRadialDerivativeWhenStencilIsAvailable) {
  constexpr int kNumFull = 8;
  constexpr int kNumHalf = kNumFull - 1;
  const double delta_s = 1.0 / (kNumFull - 1);
  std::array<double, kNumFull> r1e{};
  std::array<double, kNumFull> r1o{};
  std::array<double, kNumFull> z1e{};
  std::array<double, kNumFull> z1o{};
  std::array<double, kNumFull> ru_e{};
  std::array<double, kNumFull> ru_o{};
  std::array<double, kNumFull> zu_e{};
  std::array<double, kNumFull> zu_o{};
  std::array<double, kNumHalf> sqrt_sh{};
  std::array<double, kNumHalf> r12{};
  std::array<double, kNumHalf> ru12{};
  std::array<double, kNumHalf> zu12{};
  std::array<double, kNumHalf> rs{};
  std::array<double, kNumHalf> zs{};
  std::array<double, kNumHalf> tau{};

  for (int j = 0; j < kNumFull; ++j) {
    const double s = j * delta_s;
    r1e[j] = s * s * s;
  }
  for (int j = 0; j < kNumHalf; ++j) {
    sqrt_sh[j] = std::sqrt((j + 0.5) * delta_s);
  }

  ComputeHalfGridJacobian(
      r1e.data(), r1o.data(), z1e.data(), z1o.data(), ru_e.data(), ru_o.data(),
      zu_e.data(), zu_o.data(), sqrt_sh.data(), delta_s, 0.25,
      /*nZnT=*/1, /*nsMinF1=*/0, /*nsMinH=*/0, /*nsMaxH=*/kNumHalf, r12.data(),
      ru12.data(), zu12.data(), rs.data(), zs.data(), tau.data());

  for (int j = 0; j < kNumHalf; ++j) {
    const double s_half = (j + 0.5) * delta_s;
    const bool has_four_point_stencil = j > 0 && j + 2 < kNumFull;
    const double expected =
        has_four_point_stencil
            ? 3.0 * s_half * s_half
            : (std::pow((j + 1) * delta_s, 3) - std::pow(j * delta_s, 3)) /
                  delta_s;
    EXPECT_NEAR(rs[j], expected, 1e-13);
  }
}

}  // namespace
}  // namespace vmecpp
