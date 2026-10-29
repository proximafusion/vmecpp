// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/toroidal_elements/toroidal_elements.h"

#include <cmath>
#include <numbers>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace vmecpp {

namespace {

// P_0 ... P_degree at the points, one column per order
Eigen::MatrixXd LegendreColumns(const Eigen::VectorXd& x, int degree) {
  Eigen::MatrixXd p(x.size(), degree + 1);
  p.col(0).setOnes();
  if (degree >= 1) {
    p.col(1) = x;
  }
  for (int k = 2; k <= degree; ++k) {
    p.col(k) = ((2.0 * k - 1.0) * x.cwiseProduct(p.col(k - 1)) -
                (k - 1.0) * p.col(k - 2)) /
               k;
  }
  return p;
}

// nodal Lagrange functions and their derivatives at the points x
void LagrangeAtPoints(const Eigen::VectorXd& nodes, const Eigen::VectorXd& x,
                      Eigen::MatrixXd& m_values, Eigen::MatrixXd& m_slopes) {
  const int num_nodes = static_cast<int>(nodes.size());
  m_values.resize(x.size(), num_nodes);
  m_slopes.resize(x.size(), num_nodes);
  for (int i = 0; i < x.size(); ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      double value = 1.0;
      double slope = 0.0;
      for (int l = 0; l < num_nodes; ++l) {
        if (l == j) {
          continue;
        }
        double product = 1.0 / (nodes[j] - nodes[l]);
        for (int m = 0; m < num_nodes; ++m) {
          if (m != j && m != l) {
            product *= (x[i] - nodes[m]) / (nodes[j] - nodes[m]);
          }
        }
        slope += product;
        value *= (x[i] - nodes[l]) / (nodes[j] - nodes[l]);
      }
      m_values(i, j) = value;
      m_slopes(i, j) = slope;
    }
  }
}

}  // namespace

void GaussLobatto(int degree, Eigen::VectorXd& m_points,
                  Eigen::VectorXd& m_weights, Eigen::MatrixXd& m_derivative) {
  const int n = degree;
  Eigen::VectorXd x(n + 1);
  for (int i = 0; i <= n; ++i) {
    x[i] = -std::cos(std::numbers::pi * i / n);
  }
  // Newton iteration for the roots of (1 - x^2) P_n'(x), which leaves +-1 fixed
  for (int iteration = 0; iteration < 100; ++iteration) {
    const Eigen::MatrixXd p = LegendreColumns(x, n);
    const Eigen::VectorXd step = (x.cwiseProduct(p.col(n)) - p.col(n - 1))
                                     .cwiseQuotient((n + 1) * p.col(n));
    x -= step;
    if (step.cwiseAbs().maxCoeff() < 1.0e-16) {
      break;
    }
  }
  // exact symmetry about 0
  for (int i = 0; i <= n / 2; ++i) {
    const double symmetric = 0.5 * (x[n - i] - x[i]);
    x[i] = -symmetric;
    x[n - i] = symmetric;
  }
  if (n % 2 == 0) {
    x[n / 2] = 0.0;
  }
  const Eigen::VectorXd p_n = LegendreColumns(x, n).col(n);

  m_points = x;
  m_weights = 2.0 / (n * (n + 1.0) * p_n.array().square());
  m_derivative = Eigen::MatrixXd::Zero(n + 1, n + 1);
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i != j) {
        m_derivative(i, j) = p_n[i] / (p_n[j] * (x[i] - x[j]));
      }
    }
  }
  m_derivative(0, 0) = -0.25 * n * (n + 1.0);
  m_derivative(n, n) = 0.25 * n * (n + 1.0);
}

absl::StatusOr<ToroidalElements> MakeToroidalElements(
    const Eigen::VectorXd& breaks, int degree, int nfp, int num_quadrature) {
  static constexpr double kPi = std::numbers::pi;
  if (degree < 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "toroidal_element_degree must be at least 1, but is %d", degree));
  }
  if (num_quadrature == 0) {
    num_quadrature = degree + 2;
  }
  if (num_quadrature < degree + 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("toroidal_element_quadrature must be 0 or at least "
                        "toroidal_element_degree + 1 = %d, but is %d",
                        degree + 1, num_quadrature));
  }
  if (breaks.size() < 2 || breaks[0] != 0.0 ||
      std::abs(breaks[breaks.size() - 1] - kPi) > 1.0e-12) {
    return absl::InvalidArgumentError(
        "toroidal_element_breaks must run from 0 to pi");
  }
  for (int e = 0; e + 1 < breaks.size(); ++e) {
    if (!(breaks[e + 1] > breaks[e])) {
      return absl::InvalidArgumentError(
          "toroidal_element_breaks must increase strictly");
    }
  }

  ToroidalElements t;
  const int num_half = static_cast<int>(breaks.size()) - 1;
  t.num_half_elements = num_half;
  t.degree = degree;
  t.num_quadrature = num_quadrature;
  t.nfp = nfp;
  t.ntor = num_half * degree;
  t.num_points = 2 * num_half * num_quadrature;
  const int num_elements = 2 * num_half;
  const int num_nodes = 2 * t.ntor;
  const int num_slots = t.ntor + 1;
  const bool at_nodes = num_quadrature == degree + 1;

  // reference nodes, quadrature points and nodal functions at the points
  Eigen::VectorXd xi;
  Eigen::VectorXd omega;
  Eigen::MatrixXd derivative;
  GaussLobatto(degree, xi, omega, derivative);
  Eigen::VectorXd xq = xi;
  Eigen::VectorXd wq = omega;
  Eigen::MatrixXd values = Eigen::MatrixXd::Identity(degree + 1, degree + 1);
  Eigen::MatrixXd slopes = derivative;
  if (!at_nodes) {
    Eigen::MatrixXd unused;
    GaussLobatto(num_quadrature - 1, xq, wq, unused);
    LagrangeAtPoints(xi, xq, values, slopes);
  }

  // node g of point j of element e
  auto node = [&](int e, int j) { return (e * degree + j) % num_nodes; };

  t.zeta.resize(t.num_points);
  t.weight.resize(t.num_points);
  t.mirror.resize(t.num_points);
  t.element_start.resize(num_elements);
  t.element_end.resize(num_elements);
  std::vector<double> half_width(num_elements);
  std::vector<double> node_weight(num_nodes, 0.0);
  for (int e = 0; e < num_elements; ++e) {
    double start = 0.0;
    double end = 0.0;
    if (e < num_half) {
      start = breaks[e];
      end = breaks[e + 1];
    } else {
      const int mirrored = num_elements - 1 - e;
      start = 2.0 * kPi - breaks[mirrored + 1];
      end = 2.0 * kPi - breaks[mirrored];
    }
    half_width[e] = 0.5 * (end - start);
    t.element_start[e] = start;
    t.element_end[e] = end;
    for (int j = 0; j <= degree; ++j) {
      node_weight[node(e, j)] += half_width[e] * omega[j];
    }
    for (int i = 0; i < num_quadrature; ++i) {
      const int k = e * num_quadrature + i;
      t.zeta[k] = 0.5 * (start + end) + half_width[e] * xq[i];
      t.weight[k] = half_width[e] * wq[i];
      t.mirror[k] =
          (num_elements - 1 - e) * num_quadrature + (num_quadrature - 1 - i);
    }
  }

  // the node at v = 0 is also the last node of the period, at 2 pi
  t.node_zeta.resize(num_nodes);
  for (int e = num_elements - 1; e >= 0; --e) {
    for (int j = degree; j >= 0; --j) {
      t.node_zeta[node(e, j)] =
          0.5 * (t.element_start[e] + t.element_end[e]) + half_width[e] * xi[j];
    }
  }

  // slot of a node and the sign of its odd function
  auto slot = [&](int g) { return g <= t.ntor ? g : num_nodes - g; };
  auto odd_sign = [&](int g) { return g <= t.ntor ? 1.0 : -1.0; };
  auto has_odd = [&](int n) { return 0 < n && n < t.ntor; };

  t.scale.resize(num_slots);
  for (int n = 0; n < num_slots; ++n) {
    double mass = node_weight[n];
    if (n != 0 && n != t.ntor) {
      mass += node_weight[num_nodes - n];
    }
    t.scale[n] = std::sqrt(2.0 * kPi / mass);
  }

  t.even.assign(t.num_points * num_slots, 0.0);
  t.odd.assign(t.num_points * num_slots, 0.0);
  t.even_dv.assign(t.num_points * num_slots, 0.0);
  t.odd_dv.assign(t.num_points * num_slots, 0.0);
  for (int e = 0; e < num_elements; ++e) {
    for (int i = 0; i < num_quadrature; ++i) {
      const int k = e * num_quadrature + i;
      for (int j = 0; j <= degree; ++j) {
        const int g = node(e, j);
        const int n = slot(g);
        const double d = slopes(i, j) / half_width[e];
        t.even[k * num_slots + n] += values(i, j);
        t.even_dv[k * num_slots + n] += d;
        if (has_odd(n)) {
          t.odd[k * num_slots + n] += values(i, j) * odd_sign(g);
          t.odd_dv[k * num_slots + n] += d * odd_sign(g);
        }
      }
    }
  }

  // mass and stiffness in the scaled coordinates, with d / d\phi = nfp d / dv
  t.even_mass = Eigen::MatrixXd::Identity(num_slots, num_slots);
  t.odd_mass = Eigen::MatrixXd::Identity(num_slots, num_slots);
  if (!at_nodes) {
    t.even_mass.setZero();
    t.odd_mass.setZero();
  }
  Eigen::MatrixXd even_stiffness = Eigen::MatrixXd::Zero(num_slots, num_slots);
  Eigen::MatrixXd odd_stiffness = Eigen::MatrixXd::Zero(num_slots, num_slots);
  for (int k = 0; k < t.num_points; ++k) {
    const double w_mass = t.weight[k] / (2.0 * kPi);
    const double w = t.weight[k] / (2.0 * kPi) * nfp * nfp;
    for (int n = 0; n < num_slots; ++n) {
      const double even_n = t.even_dv[k * num_slots + n] * t.scale[n];
      const double odd_n = t.odd_dv[k * num_slots + n] * t.scale[n];
      for (int n2 = 0; n2 < num_slots; ++n2) {
        even_stiffness(n, n2) +=
            w * even_n * t.even_dv[k * num_slots + n2] * t.scale[n2];
        odd_stiffness(n, n2) +=
            w * odd_n * t.odd_dv[k * num_slots + n2] * t.scale[n2];
        if (!at_nodes) {
          t.even_mass(n, n2) += w_mass * t.even[k * num_slots + n] *
                                t.scale[n] * t.even[k * num_slots + n2] *
                                t.scale[n2];
          t.odd_mass(n, n2) += w_mass * t.odd[k * num_slots + n] * t.scale[n] *
                               t.odd[k * num_slots + n2] * t.scale[n2];
        }
      }
    }
  }
  t.odd_mass(0, 0) = 1.0;
  t.odd_mass(t.ntor, t.ntor) = 1.0;

  // clamp roundoff below the zero eigenvalue of the constant function
  auto eigenbasis = [&](const Eigen::MatrixXd& stiffness,
                        const Eigen::MatrixXd& mass,
                        Eigen::Ref<Eigen::VectorXd> m_eigenvalues,
                        Eigen::Ref<Eigen::MatrixXd> m_eigenvectors) {
    if (at_nodes) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(stiffness);
      m_eigenvalues = solver.eigenvalues().cwiseMax(0.0);
      m_eigenvectors = solver.eigenvectors();
    } else {
      Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
          stiffness, mass);
      m_eigenvalues = solver.eigenvalues().cwiseMax(0.0);
      m_eigenvectors = solver.eigenvectors();
    }
  };
  t.even_eigenvalues.resize(num_slots);
  t.even_eigenvectors.resize(num_slots, num_slots);
  eigenbasis(even_stiffness, t.even_mass, t.even_eigenvalues,
             t.even_eigenvectors);

  t.odd_eigenvalues = Eigen::VectorXd::Zero(num_slots);
  t.odd_eigenvectors = Eigen::MatrixXd::Identity(num_slots, num_slots);
  const int num_odd = t.ntor - 1;
  if (num_odd > 0) {
    eigenbasis(odd_stiffness.block(1, 1, num_odd, num_odd),
               t.odd_mass.block(1, 1, num_odd, num_odd),
               t.odd_eigenvalues.segment(1, num_odd),
               t.odd_eigenvectors.block(1, 1, num_odd, num_odd));
  }

  t.constant.resize(num_slots);
  for (int n = 0; n < num_slots; ++n) {
    t.constant[n] = 1.0 / t.scale[n];
  }
  return t;
}

void FourierIntegrals(const ToroidalElements& elements, int max_n,
                      Eigen::MatrixXd& m_cos, Eigen::MatrixXd& m_sin) {
  // the interpolant through the quadrature points of an element
  const int degree = elements.num_quadrature - 1;
  Eigen::VectorXd xi;
  Eigen::VectorXd omega;
  Eigen::MatrixXd derivative;
  GaussLobatto(degree, xi, omega, derivative);

  m_cos = Eigen::MatrixXd::Zero(elements.num_points, max_n + 1);
  m_sin = Eigen::MatrixXd::Zero(elements.num_points, max_n + 1);
  const int num_elements = static_cast<int>(elements.element_start.size());
  for (int e = 0; e < num_elements; ++e) {
    const double half_width =
        0.5 * (elements.element_end[e] - elements.element_start[e]);
    const double middle =
        0.5 * (elements.element_end[e] + elements.element_start[e]);
    // Gauss-Legendre points resolving the interpolant times cos(max_n v)
    const int num_gauss =
        degree + 10 + static_cast<int>(std::ceil(2.0 * max_n * half_width));
    Eigen::VectorXd x(num_gauss);
    Eigen::VectorXd w(num_gauss);
    for (int g = 0; g < num_gauss; ++g) {
      double root = std::cos(std::numbers::pi * (g + 0.75) / (num_gauss + 0.5));
      for (int iteration = 0; iteration < 100; ++iteration) {
        double p0 = 1.0;
        double p1 = root;
        for (int k = 2; k <= num_gauss; ++k) {
          const double p2 = ((2.0 * k - 1.0) * root * p1 - (k - 1.0) * p0) / k;
          p0 = p1;
          p1 = p2;
        }
        const double dp = num_gauss * (root * p1 - p0) / (root * root - 1.0);
        const double step = p1 / dp;
        root -= step;
        if (std::abs(step) < 1.0e-16) {
          double q0 = 1.0;
          double q1 = root;
          for (int k = 2; k <= num_gauss; ++k) {
            const double q2 =
                ((2.0 * k - 1.0) * root * q1 - (k - 1.0) * q0) / k;
            q0 = q1;
            q1 = q2;
          }
          const double dq = num_gauss * (root * q1 - q0) / (root * root - 1.0);
          w[g] = 2.0 / ((1.0 - root * root) * dq * dq);
          break;
        }
      }
      x[g] = root;
    }
    for (int g = 0; g < num_gauss; ++g) {
      const double v = middle + half_width * x[g];
      const double weight = half_width * w[g] / (2.0 * std::numbers::pi);
      for (int i = 0; i <= degree; ++i) {
        double lagrange = 1.0;
        for (int j = 0; j <= degree; ++j) {
          if (j != i) {
            lagrange *= (x[g] - xi[j]) / (xi[i] - xi[j]);
          }
        }
        const int k = e * (degree + 1) + i;
        for (int n = 0; n <= max_n; ++n) {
          m_cos(k, n) += weight * lagrange * std::cos(n * v);
          m_sin(k, n) += weight * lagrange * std::sin(n * v);
        }
      }  // i
    }  // g
  }  // e
}

}  // namespace vmecpp
