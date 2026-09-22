// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/step_limit_kernel.h"

#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/vmec/ideal_mhd_model/jacobian_kernel.h"

namespace vmecpp {
namespace {

TEST(StepLimitKernel, FirstRootInUnitInterval) {
  // no root: 2 x^2 - x + 1 > 0
  EXPECT_EQ(FirstRootInUnitInterval(2.0, -1.0, 1.0), 1.0);
  // linear: -4 x + 1 = 0 at x = 0.25
  EXPECT_DOUBLE_EQ(FirstRootInUnitInterval(0.0, -4.0, 1.0), 0.25);
  // linear and increasing: no root
  EXPECT_EQ(FirstRootInUnitInterval(0.0, 3.0, 1.0), 1.0);
  // (x - 0.2)(x - 0.5) = x^2 - 0.7 x + 0.1: the first root
  EXPECT_NEAR(FirstRootInUnitInterval(1.0, -0.7, 0.1), 0.2, 1e-15);
  // (x - 2)(x - 3): both roots beyond the step
  EXPECT_EQ(FirstRootInUnitInterval(1.0, -5.0, 6.0), 1.0);
  // -(x + 1)(x - 0.5) = -x^2 - 0.5 x + 0.5: the positive root
  EXPECT_NEAR(FirstRootInUnitInterval(-1.0, -0.5, 0.5), 0.5, 1e-15);
}

// The step fraction lands where tau keeps exactly the retained fraction at the
// binding point and at least that everywhere along the shortened step.
TEST(StepLimitKernel, LargestStepFractionStopsAtTheRetainedFraction) {
  // tau(alpha) = tau0 + b alpha + c alpha^2 on three points of both signs
  const std::vector<double> tau0 = {-2.0, 3.0, 1.0};
  const std::vector<double> b = {3.0, -1.0, -3.5};
  const std::vector<double> c = {0.5, 0.2, 0.9};
  std::vector<double> tau1(3);
  std::vector<double> tau_d(3);
  for (int i = 0; i < 3; ++i) {
    tau1[i] = tau0[i] + b[i] + c[i];
    tau_d[i] = c[i];
  }
  const double retained = 0.01;
  const double alpha =
      LargestStepFraction(tau0.data(), tau1.data(), tau_d.data(), 3, retained);
  ASSERT_GT(alpha, 0.0);
  ASSERT_LT(alpha, 1.0);
  double smallest_ratio = 1.0;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k <= 100; ++k) {
      const double a = alpha * k / 100.0;
      const double ratio = (tau0[i] + b[i] * a + c[i] * a * a) / tau0[i];
      EXPECT_GE(ratio, retained - 1e-14) << i << " " << a;
    }
    smallest_ratio =
        std::min(smallest_ratio,
                 (tau0[i] + b[i] * alpha + c[i] * alpha * alpha) / tau0[i]);
  }
  EXPECT_NEAR(smallest_ratio, retained, 1e-14);

  // a step along which tau keeps enough of its value is not shortened
  const std::vector<double> growing = {-2.5, 3.5, 1.2};
  const std::vector<double> none = {0.0, 0.0, 0.0};
  EXPECT_EQ(LargestStepFraction(tau0.data(), growing.data(), none.data(), 3,
                                retained),
            1.0);
}

// tau is a quadratic form in the geometry: along g0 + alpha (g1 - g0) it
// equals tau0 + alpha (tau1 - tau0 - tau_d) + alpha^2 tau_d, with tau_d the
// kernel applied to g1 - g0.
TEST(StepLimitKernel, JacobianIsQuadraticAlongAStep) {
  const int nZnT = 12;
  const int num_full = 6;
  const int num_half = num_full - 1;
  const double delta_s = 1.0 / num_half;
  const double d_shalf_ds_interp = 0.25;
  std::vector<double> sqrt_s_half(num_half);
  for (int j = 0; j < num_half; ++j) {
    sqrt_s_half[j] = std::sqrt((j + 0.5) * delta_s);
  }
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  const int n = num_full * nZnT;
  auto random_geometry = [&]() {
    std::vector<std::vector<double>> g(8, std::vector<double>(n));
    for (auto& array : g) {
      for (double& value : array) {
        value = uniform(rng);
      }
    }
    return g;
  };
  auto jacobian = [&](const std::vector<std::vector<double>>& g) {
    std::vector<double> r12(num_half * nZnT), ru12(num_half * nZnT),
        zu12(num_half * nZnT), rs(num_half * nZnT), zs(num_half * nZnT),
        tau(num_half * nZnT);
    ComputeHalfGridJacobian(g[0].data(), g[1].data(), g[2].data(), g[3].data(),
                            g[4].data(), g[5].data(), g[6].data(), g[7].data(),
                            sqrt_s_half.data(), delta_s, d_shalf_ds_interp,
                            nZnT, /*nsMinF1=*/0, /*nsMinH=*/0,
                            /*nsMaxH=*/num_half, r12.data(), ru12.data(),
                            zu12.data(), rs.data(), zs.data(), tau.data());
    return tau;
  };
  const auto g0 = random_geometry();
  const auto g1 = random_geometry();
  auto combination = [&](double alpha) {
    auto g = g0;
    for (int k = 0; k < 8; ++k) {
      for (int i = 0; i < n; ++i) {
        g[k][i] = g0[k][i] + alpha * (g1[k][i] - g0[k][i]);
      }
    }
    return g;
  };
  const auto tau0 = jacobian(g0);
  const auto tau1 = jacobian(g1);
  auto difference = g1;
  for (int k = 0; k < 8; ++k) {
    for (int i = 0; i < n; ++i) {
      difference[k][i] = g1[k][i] - g0[k][i];
    }
  }
  const auto tau_d = jacobian(difference);
  for (const double alpha : {0.1, 0.37, 0.8, 1.6}) {
    const auto tau = jacobian(combination(alpha));
    for (int i = 0; i < num_half * nZnT; ++i) {
      const double quadratic = tau0[i] +
                               alpha * (tau1[i] - tau0[i] - tau_d[i]) +
                               alpha * alpha * tau_d[i];
      EXPECT_NEAR(tau[i], quadratic, 1e-12 * (1.0 + std::abs(tau[i])))
          << alpha << " " << i;
    }
  }
}

}  // namespace
}  // namespace vmecpp
