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
  void AccumulateFullGrpmn(const std::vector<double>& grpmn_sin_singular,
                           const std::vector<double>& grpmn_cos_singular);
  void PerformToroidalFourierTransforms();
  void PerformPoloidalFourierTransforms();

  void BuildMatrix();
  void DecomposeMatrix();
  // For lasym = false, only bvec_sin_singular is consumed and bvec_cos_singular
  // may be an empty span. For lasym = true both halves are required and
  // bvecShare must be sized for 2 * mnpd entries.
  void SolveForPotential(const std::vector<double>& bvec_sin_singular,
                         const std::vector<double>& bvec_cos_singular = {});

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

  // Symmetrized source term, sin (anti-symmetric under (u,v) -> (-u,-v))
  // half. Logically [nThetaReduced x nZeta] in row-major order.
  Eigen::VectorXd gstore_symm;

  // Symmetrized source term, cos (symmetric under (u,v) -> (-u,-v)) half.
  // Same layout as gstore_symm; only allocated for lasym.
  Eigen::VectorXd gstore_asym;

  // Fourier transform intermediate results for the sin-basis source.
  // Logically [(2*nf+1) x nThetaReduced] matrices stored as flat vectors.
  Eigen::VectorXd bcos;
  Eigen::VectorXd bsin;

  // Fourier transform intermediate results for the cos-basis source
  // (lasym only). Same shape as bcos / bsin.
  Eigen::VectorXd bcos_asym;
  Eigen::VectorXd bsin_asym;

  // Intermediate matrix transform results for the sin-basis kernel.
  // Logically [mnpd * (2*nf+1) x nThetaEff] stored as flat vectors.
  Eigen::VectorXd actemp;
  Eigen::VectorXd astemp;

  // Intermediate matrix transform results for the cos-basis kernel
  // (lasym only). Same shape as actemp / astemp.
  Eigen::VectorXd actemp_cos;
  Eigen::VectorXd astemp_cos;

  // Linear system blocks. The full lasym amatsq has four (mnpd x mnpd)
  // quadrants: sin-sin (top-left), sin-cos (top-right), cos-sin
  // (bottom-left), cos-cos (bottom-right). For lasym = false only
  // amat_sin_sin and bvec_sin are populated.
  Eigen::VectorXd bvec_sin;
  Eigen::VectorXd bvec_cos;
  Eigen::VectorXd amat_sin_sin;
  Eigen::VectorXd amat_sin_cos;
  Eigen::VectorXd amat_cos_sin;
  Eigen::VectorXd amat_cos_cos;

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
