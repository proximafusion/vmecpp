// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_
#define VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_

#include <Eigen/Dense>
#include <vector>

#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class LaplaceSolver {
 public:
  LaplaceSolver(const Sizes* s, const FourierBasisFastToroidal* fb,
                const TangentialPartitioning* tp, int nf, int mf,
                std::span<double> matrixShare, std::span<int> iPiv,
                std::span<double> bvecShare);

  void TransformGreensFunctionDerivative(const std::vector<double>& greenp);
  void SymmetriseSourceTerm(const std::vector<double>& gstore);
  void AccumulateFullGrpmn(const std::vector<double>& grpmn_sin_singular);
  void PerformToroidalFourierTransforms();
  void PerformPoloidalFourierTransforms();

  void BuildMatrix();
  void DecomposeMatrix();
  void SolveForPotential(const std::vector<double>& bvec_sin_singular);

  // Green's function derivative Fourier transform, non-singular part,
  // stellarator-symmetric.
  // Logically [mnpd x numLocal] matrix stored as flat vector in column-major
  // order.
  Eigen::VectorXd grpmn_sin;

  // Green's function derivative Fourier transform, non-singular part,
  // non-stellarator-symmetric.
  // Logically [mnpd x numLocal] matrix stored as flat vector in column-major
  // order.
  Eigen::VectorXd grpmn_cos;

  // Symmetrized source term, stellarator-symmetric.
  // Logically [nThetaReduced x nZeta] matrix stored as flat vector in row-major
  // order.
  Eigen::VectorXd gstore_symm;

  // Fourier transform intermediate results.
  // Logically [(2*nf+1) x nThetaReduced] matrices stored as flat vectors.
  Eigen::VectorXd bcos;
  Eigen::VectorXd bsin;

  // Intermediate matrix transform results.
  // Logically [mnpd * (2*nf+1) x nThetaEff] stored as flat vectors.
  Eigen::VectorXd actemp;
  Eigen::VectorXd astemp;

  // Linear system to be solved.
  // bvec_sin: vector of size [mnpd]
  // amat_sin_sin: logically [mnpd x mnpd] matrix stored as flat vector
  Eigen::VectorXd bvec_sin;
  Eigen::VectorXd amat_sin_sin;

 private:
  const Sizes& s_;
  const FourierBasisFastToroidal& fb_;
  const TangentialPartitioning& tp_;

  int nf;
  int mf;

  // needed for LAPACK's dgetrf
  // non-owning pointers
  std::span<double> matrixShare;
  std::span<int> iPiv;
  std::span<double> bvecShare;

  // ----------------

  int numLocal;

  // Pre-computed scaled Fourier basis matrices for efficient transforms
  // cosnv_scaled: [(nf+1) x nZeta] - scaled toroidal cosine basis
  // sinnv_scaled: [(nf+1) x nZeta] - scaled toroidal sine basis
  Eigen::MatrixXd cosnv_scaled;
  Eigen::MatrixXd sinnv_scaled;

  // cosmui_scaled: [nThetaReduced x (mf+1)] - scaled poloidal cosine basis
  // sinmui_scaled: [nThetaReduced x (mf+1)] - scaled poloidal sine basis
  Eigen::MatrixXd cosmui_scaled;
  Eigen::MatrixXd sinmui_scaled;

  Eigen::VectorXd grpOdd;
  Eigen::VectorXd grpEvn;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_
