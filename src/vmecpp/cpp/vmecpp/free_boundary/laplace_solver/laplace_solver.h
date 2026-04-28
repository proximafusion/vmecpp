// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_
#define VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_

#include <Eigen/Dense>

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

  void TransformGreensFunctionDerivative(const Eigen::VectorXd& greenp);
  void SymmetriseSourceTerm(const Eigen::VectorXd& gstore);
  void AccumulateFullGrpmn(const Eigen::VectorXd& grpmn_sin_singular);
  void PerformToroidalFourierTransforms();
  void PerformPoloidalFourierTransforms();

  void BuildMatrix();
  void DecomposeMatrix();
  void SolveForPotential(const Eigen::VectorXd& bvec_sin_singular);

  // Green's function derivative Fourier transform, non-singular part,
  // stellarator-symmetric
  Eigen::VectorXd grpmn_sin;

  // Green's function derivative Fourier transform, non-singular part,
  // non-stellarator-symmetric
  Eigen::VectorXd grpmn_cos;

  // symmetrized source term, stellarator-symmetric
  Eigen::VectorXd gstore_symm;

  Eigen::VectorXd bcos;
  Eigen::VectorXd bsin;

  Eigen::VectorXd actemp;
  Eigen::VectorXd astemp;

  // linear system to be solved
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

  Eigen::VectorXd grpOdd;
  Eigen::VectorXd grpEvn;

  // Precomputed Fourier basis matrices for GEMM acceleration.
  // cosnv_scaled_(n, k) = fb_.cosnv[n*nZeta+k] / fb_.nscale[n]
  // Shape: (nf+1, nZeta)
  Eigen::MatrixXd cosnv_scaled_;
  Eigen::MatrixXd sinnv_scaled_;

  // sinmui_mat_(l, m) = fb_.sinmui[l*(mnyq2+1)+m] / fb_.mscale[m]
  // cosmui_mat_(l, m) = fb_.cosmui[l*(mnyq2+1)+m] / fb_.mscale[m]
  // Shape: (nThetaReduced, mf+1)
  Eigen::MatrixXd sinmui_mat_;
  Eigen::MatrixXd cosmui_mat_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_
