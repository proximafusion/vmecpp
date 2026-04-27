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
                std::span<real_t> matrixShare, std::span<int> iPiv,
                std::span<real_t> bvecShare);

  void TransformGreensFunctionDerivative(const std::vector<real_t>& greenp);
  void SymmetriseSourceTerm(const std::vector<real_t>& gstore);
  void AccumulateFullGrpmn(const std::vector<real_t>& grpmn_sin_singular);
  void PerformToroidalFourierTransforms();
  void PerformPoloidalFourierTransforms();

  void BuildMatrix();
  void DecomposeMatrix();
  void SolveForPotential(const std::vector<real_t>& bvec_sin_singular);

  // Green's function derivative Fourier transform, non-singular part,
  // stellarator-symmetric
  std::vector<real_t> grpmn_sin;

  // Green's function derivative Fourier transform, non-singular part,
  // non-stellarator-symmetric
  std::vector<real_t> grpmn_cos;

  // symmetrized source term, stellarator-symmetric
  std::vector<real_t> gstore_symm;

  std::vector<real_t> bcos;
  std::vector<real_t> bsin;

  std::vector<real_t> actemp;
  std::vector<real_t> astemp;

  // linear system to be solved
  std::vector<real_t> bvec_sin;
  std::vector<real_t> amat_sin_sin;

 private:
  const Sizes& s_;
  const FourierBasisFastToroidal& fb_;
  const TangentialPartitioning& tp_;

  int nf;
  int mf;

  // needed for LAPACK's dgetrf
  // non-owning pointers
  std::span<real_t> matrixShare;
  std::span<int> iPiv;
  std::span<real_t> bvecShare;

  // ----------------

  int numLocal;

  std::vector<real_t> grpOdd;
  std::vector<real_t> grpEvn;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_LAPLACE_SOLVER_LAPLACE_SOLVER_H_
