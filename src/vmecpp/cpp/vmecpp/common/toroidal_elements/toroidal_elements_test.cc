// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/toroidal_elements/toroidal_elements.h"

#include <cmath>
#include <numbers>

#include "gtest/gtest.h"

namespace vmecpp {
namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

TEST(TestToroidalElements, GaussLobattoIsExactToDegreeTwoPMinusOne) {
  for (int degree = 1; degree <= 8; ++degree) {
    Eigen::VectorXd points;
    Eigen::VectorXd weights;
    Eigen::MatrixXd derivative;
    GaussLobatto(degree, points, weights, derivative);
    for (int k = 0; k <= 2 * degree - 1; ++k) {
      const double exact = k % 2 == 0 ? 2.0 / (k + 1) : 0.0;
      EXPECT_NEAR(weights.dot(points.array().pow(k).matrix()), exact, 1.0e-14)
          << "degree " << degree << ", x^" << k;
    }
    // D x^k = k x^(k - 1) for k <= degree
    for (int k = 1; k <= degree; ++k) {
      const Eigen::VectorXd values = points.array().pow(k);
      const Eigen::VectorXd slope = k * points.array().pow(k - 1);
      EXPECT_LT((derivative * values - slope).cwiseAbs().maxCoeff(), 1.0e-12)
          << "degree " << degree << ", x^" << k;
    }
  }
}

TEST(TestToroidalElements, BasisIsOrthonormalAndMirrorSymmetric) {
  const Eigen::VectorXd breaks{{0.0, 0.4, 1.9, std::numbers::pi}};
  const int degree = 4;
  // with the quadrature points at the nodes, where the mass is lumped
  absl::StatusOr<ToroidalElements> elements =
      MakeToroidalElements(breaks, degree, /*nfp=*/3, degree + 1);
  ASSERT_TRUE(elements.ok()) << elements.status();
  const int num_slots = elements->ntor + 1;
  EXPECT_EQ(elements->ntor, 3 * degree);
  EXPECT_EQ(elements->num_points, 2 * 3 * (degree + 1));

  double weight_sum = 0.0;
  for (int k = 0; k < elements->num_points; ++k) {
    weight_sum += elements->weight[k];
    const int mirror = elements->mirror[k];
    EXPECT_EQ(elements->mirror[mirror], k);
    EXPECT_NEAR(elements->zeta[mirror], kTwoPi - elements->zeta[k], 1.0e-14);
  }
  EXPECT_NEAR(weight_sum, kTwoPi, 1.0e-13);

  // scaled functions orthonormal under the nodal quadrature
  for (int n = 0; n < num_slots; ++n) {
    for (int n2 = 0; n2 < num_slots; ++n2) {
      double even = 0.0;
      double odd = 0.0;
      for (int k = 0; k < elements->num_points; ++k) {
        const double w = elements->weight[k] / kTwoPi * elements->scale[n] *
                         elements->scale[n2];
        even += w * elements->even[k * num_slots + n] *
                elements->even[k * num_slots + n2];
        odd += w * elements->odd[k * num_slots + n] *
               elements->odd[k * num_slots + n2];
      }
      const bool has_odd = 0 < n && n < elements->ntor;
      EXPECT_NEAR(even, n == n2 ? 1.0 : 0.0, 1.0e-13) << n << " " << n2;
      EXPECT_NEAR(odd, n == n2 && has_odd ? 1.0 : 0.0, 1.0e-13)
          << n << " " << n2;
    }
  }

  // the constant function in the scaled even coordinates
  for (int k = 0; k < elements->num_points; ++k) {
    double value = 0.0;
    for (int n = 0; n < num_slots; ++n) {
      value += elements->constant[n] * elements->scale[n] *
               elements->even[k * num_slots + n];
    }
    EXPECT_NEAR(value, 1.0, 1.0e-14);
  }
}

TEST(TestToroidalElements, StiffnessApproachesFourierWavenumbers) {
  const int nfp = 2;
  absl::StatusOr<ToroidalElements> elements =
      MakeToroidalElements(Eigen::VectorXd::LinSpaced(5, 0.0, std::numbers::pi),
                           /*degree=*/8, nfp, /*num_quadrature=*/0);
  ASSERT_TRUE(elements.ok()) << elements.status();
  // lowest eigenvalues (n nfp)^2, even from n = 0 and odd from n = 1
  for (int n = 0; n < 5; ++n) {
    EXPECT_NEAR(elements->even_eigenvalues[n], n * nfp * n * nfp,
                1.0e-6 * (1.0 + n * n * nfp * nfp))
        << n;
  }
  for (int n = 1; n < 5; ++n) {
    EXPECT_NEAR(elements->odd_eigenvalues[n], n * nfp * n * nfp,
                1.0e-6 * n * n * nfp * nfp)
        << n;
  }
}

TEST(TestToroidalElements, OverIntegrationGivesTheExactMass) {
  const Eigen::VectorXd breaks{{0.0, 0.9, 2.0, std::numbers::pi}};
  const int degree = 5;
  absl::StatusOr<ToroidalElements> exact =
      MakeToroidalElements(breaks, degree, /*nfp=*/2, degree + 2);
  absl::StatusOr<ToroidalElements> finer =
      MakeToroidalElements(breaks, degree, /*nfp=*/2, degree + 9);
  ASSERT_TRUE(exact.ok()) << exact.status();
  ASSERT_TRUE(finer.ok()) << finer.status();
  EXPECT_LT((exact->even_mass - finer->even_mass).cwiseAbs().maxCoeff(),
            1.0e-13);
  EXPECT_LT((exact->odd_mass - finer->odd_mass).cwiseAbs().maxCoeff(), 1.0e-13);
  // V^T M V = I with the stiffness diagonal in the eigenbasis
  const int num_slots = exact->ntor + 1;
  const Eigen::MatrixXd& v = exact->even_eigenvectors;
  EXPECT_LT((v.transpose() * exact->even_mass * v -
             Eigen::MatrixXd::Identity(num_slots, num_slots))
                .cwiseAbs()
                .maxCoeff(),
            1.0e-12);
  // the default is degree + 2, and the nodes are the same for every quadrature
  absl::StatusOr<ToroidalElements> by_default =
      MakeToroidalElements(breaks, degree, /*nfp=*/2, /*num_quadrature=*/0);
  absl::StatusOr<ToroidalElements> at_nodes =
      MakeToroidalElements(breaks, degree, /*nfp=*/2, degree + 1);
  ASSERT_TRUE(by_default.ok()) << by_default.status();
  ASSERT_TRUE(at_nodes.ok()) << at_nodes.status();
  EXPECT_EQ(by_default->num_quadrature, degree + 2);
  EXPECT_EQ(at_nodes->node_zeta, exact->node_zeta);
  EXPECT_EQ(exact->num_points, 2 * 3 * (degree + 2));
}

TEST(TestToroidalElements, RejectsInvalidBreaksAndDegree) {
  const Eigen::VectorXd valid = Eigen::VectorXd::LinSpaced(3, 0.0, 3.0);
  EXPECT_FALSE(MakeToroidalElements(valid, 2, 1, 0).ok());  // ends before pi
  const Eigen::VectorXd decreasing{{0.0, 2.0, 1.0, std::numbers::pi}};
  EXPECT_FALSE(MakeToroidalElements(decreasing, 2, 1, 0).ok());
  const Eigen::VectorXd breaks{{0.0, std::numbers::pi}};
  EXPECT_FALSE(MakeToroidalElements(breaks, 0, 1, 0).ok());
  // fewer quadrature points than nodes per element
  EXPECT_FALSE(MakeToroidalElements(breaks, 3, 1, 3).ok());
}

TEST(TestToroidalElements, FourierIntegralsOfTheInterpolant) {
  const Eigen::VectorXd breaks{{0.0, 0.7, 1.5, 2.3, std::numbers::pi}};
  absl::StatusOr<ToroidalElements> elements = MakeToroidalElements(
      breaks, /*degree=*/12, /*nfp=*/1, /*num_quadrature=*/0);
  ASSERT_TRUE(elements.ok()) << elements.status();
  const int max_n = 6;
  Eigen::MatrixXd m_cos;
  Eigen::MatrixXd m_sin;
  FourierIntegrals(*elements, max_n, m_cos, m_sin);
  // f = 1 + cos(2 v) + sin(3 v), resolved by the interpolant to roundoff
  Eigen::VectorXd f(elements->num_points);
  for (int k = 0; k < elements->num_points; ++k) {
    const double v = elements->zeta[k];
    f[k] = 1.0 + std::cos(2.0 * v) + std::sin(3.0 * v);
  }
  const Eigen::VectorXd cos_coefficients = m_cos.transpose() * f;
  const Eigen::VectorXd sin_coefficients = m_sin.transpose() * f;
  for (int n = 0; n <= max_n; ++n) {
    EXPECT_NEAR(cos_coefficients[n], n == 0 ? 1.0 : (n == 2 ? 0.5 : 0.0),
                1.0e-10)
        << n;
    EXPECT_NEAR(sin_coefficients[n], n == 3 ? 0.5 : 0.0, 1.0e-10) << n;
  }
}

}  // namespace
}  // namespace vmecpp
