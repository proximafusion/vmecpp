// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_TOROIDAL_ELEMENTS_TOROIDAL_ELEMENTS_H_
#define VMECPP_COMMON_TOROIDAL_ELEMENTS_TOROIDAL_ELEMENTS_H_

#include <vector>

#include "Eigen/Dense"
#include "absl/status/statusor.h"

namespace vmecpp {

// Spectral elements in v = nfp \phi in place of the toroidal Fourier basis
struct ToroidalElements {
  int num_half_elements = 0;
  int degree = 0;
  int num_quadrature = 0;
  int nfp = 1;

  // number of slots minus one: num_half_elements * degree
  int ntor = 0;

  // 2 * num_half_elements * num_quadrature
  int num_points = 0;

  // [num_points] v of each quadrature point, in [0, 2 \pi]
  std::vector<double> zeta;

  // [num_points] quadrature weights in v; they sum to 2 \pi
  std::vector<double> weight;

  // [num_points] index of the point at 2 \pi - v
  std::vector<int> mirror;

  // [2 * ntor] v of the nodes; slot n holds nodes n and 2 * ntor - n
  std::vector<double> node_zeta;

  // [2 * num_half_elements] start and end of each element in v
  std::vector<double> element_start;
  std::vector<double> element_end;

  // [ntor + 1] scales normalizing the slots under the nodal quadrature
  std::vector<double> scale;

  // [num_points * (ntor + 1)] unscaled even and odd functions and d / dv
  std::vector<double> even;
  std::vector<double> odd;
  std::vector<double> even_dv;
  std::vector<double> odd_dv;

  // [(ntor + 1), (ntor + 1)] mass matrices of the scaled functions
  Eigen::MatrixXd even_mass;
  Eigen::MatrixXd odd_mass;

  // eigenpairs of the toroidal stiffness relative to the mass, V^T M V = I
  Eigen::VectorXd even_eigenvalues;
  Eigen::MatrixXd even_eigenvectors;
  Eigen::VectorXd odd_eigenvalues;
  Eigen::MatrixXd odd_eigenvectors;

  // [ntor + 1] the constant function in the scaled even coordinates
  Eigen::VectorXd constant;
};

// Gauss-Lobatto points and weights on [-1, 1], D(i, j) = \ell_j'(\xi_i)
void GaussLobatto(int degree, Eigen::VectorXd& m_points,
                  Eigen::VectorXd& m_weights, Eigen::MatrixXd& m_derivative);

// breaks increase from 0 to \pi; num_quadrature = 0 selects degree + 2
absl::StatusOr<ToroidalElements> MakeToroidalElements(
    const Eigen::VectorXd& breaks, int degree, int nfp, int num_quadrature);

// (1 / 2 \pi) \int L_k(v) cos(n v) dv and sin(n v) likewise, n <= max_n
void FourierIntegrals(const ToroidalElements& elements, int max_n,
                      Eigen::MatrixXd& m_cos, Eigen::MatrixXd& m_sin);

}  // namespace vmecpp

#endif  // VMECPP_COMMON_TOROIDAL_ELEMENTS_TOROIDAL_ELEMENTS_H_
