// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_
#define VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_

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
  // stellarator-symmetric
  std::vector<double> grpmn_sin;

  // Green's function derivative Fourier transform, non-singular part,
  // non-stellarator-symmetric
  std::vector<double> grpmn_cos;

  // symmetrized source term, stellarator-symmetric
  std::vector<double> gstore_symm;

  std::vector<double> bcos;
  std::vector<double> bsin;

  std::vector<double> actemp;
  std::vector<double> astemp;

  // linear system to be solved
  std::vector<double> bvec_sin;
  std::vector<double> amat_sin_sin;

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

  std::vector<double> grpOdd;
  std::vector<double> grpEvn;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_
