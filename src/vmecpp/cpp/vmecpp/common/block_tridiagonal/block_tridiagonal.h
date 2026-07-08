// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_BLOCK_TRIDIAGONAL_BLOCK_TRIDIAGONAL_H_
#define VMECPP_COMMON_BLOCK_TRIDIAGONAL_BLOCK_TRIDIAGONAL_H_

#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace vmecpp {

// The three block diagonals of a block-tridiagonal operator with n block rows
// of size k x k: lower[j] = L_j (couples row j to j-1), diag[j] = D_j, upper[j]
// = U_j (couples row j to j+1). lower[0] and upper[n-1] are zero/unused.
struct BlockTridiagonalBlocks {
  std::vector<Eigen::MatrixXd> lower;
  std::vector<Eigen::MatrixXd> diag;
  std::vector<Eigen::MatrixXd> upper;
};

// Assemble the block-tridiagonal Jacobian dF/dx of a force field F that has
// nearest-neighbour radial coupling (F_j depends only on x_{j-1}, x_j, x_{j+1}),
// by central finite differences about x0. There are n radial surfaces with k
// degrees of freedom each; x0 and the force are flattened surface-major, i.e.
// entry (surface, dof) lives at surface * k + dof.
//
// The coupling is exploited: perturbing surface j only changes F at j-1, j, j+1,
// so surfaces 3 apart are independent and are perturbed together. The assembly
// therefore costs 6 * k force evaluations (2 central-difference directions times
// k dofs times 3 stride offsets), independent of n.
//
// Convention matches BlockTridiagonalFactorization: dF_j/dx_j -> diag[j],
// dF_j/dx_{j-1} -> lower[j], dF_j/dx_{j+1} -> upper[j].
BlockTridiagonalBlocks AssembleFdBlockTridiagonal(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& force,
    const Eigen::VectorXd& x0, int n, int k, double eps);

// Direct solver for a block-tridiagonal linear system with n block rows of size
// k x k:
//
//   L_j x_{j-1} + D_j x_j + U_j x_{j+1} = b_j,   j = 0 .. n-1
//
// (L_0 and U_{n-1} are unused). This is the operator the VMEC 2D preconditioner
// factors: the diagonal block D_j couples the poloidal modes and field
// components at radial surface j, and L_j / U_j are the radial coupling to the
// neighbouring surfaces.
//
// The factorization is the serial block-Thomas sweep (a block LU): forward
// eliminate the sub-diagonal, back-substitute. It is O(n k^3) and exact up to
// round-off. PARVMEC's BCYCLIC is the parallel cyclic-reduction algorithm that
// produces the same solution; this serial form is the reference and the
// single-node implementation. The factorization is built once (from one
// preconditioner update) and reused across many Solve() calls (one per force
// iteration).
class BlockTridiagonalFactorization {
 public:
  // Factor the operator. lower/diag/upper each hold n blocks; lower[0] and
  // upper[n-1] are ignored. Every block must be k x k with the same k.
  BlockTridiagonalFactorization(const std::vector<Eigen::MatrixXd>& lower,
                                const std::vector<Eigen::MatrixXd>& diag,
                                const std::vector<Eigen::MatrixXd>& upper);

  // Solve for the right-hand side b (n blocks of length k), returning x.
  std::vector<Eigen::VectorXd> Solve(
      const std::vector<Eigen::VectorXd>& b) const;

  int num_blocks() const { return n_; }
  int block_size() const { return k_; }

 private:
  int n_;
  int k_;
  // Inverses of the reduced diagonal blocks Delta_j and the elimination
  // multipliers M_j = L_j Delta_{j-1}^{-1}; upper blocks are kept for the
  // back-substitution. Explicit inverses keep the reference implementation
  // simple (k is modest); an LU-based, inverse-free variant is a later
  // refinement.
  std::vector<Eigen::MatrixXd> delta_inverse_;
  std::vector<Eigen::MatrixXd> multiplier_;
  std::vector<Eigen::MatrixXd> upper_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_BLOCK_TRIDIAGONAL_BLOCK_TRIDIAGONAL_H_
