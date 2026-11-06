// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_VMEC_ANDERSON_ACCELERATION_H_
#define VMECPP_VMEC_VMEC_ANDERSON_ACCELERATION_H_

#include <Eigen/Dense>
#include <deque>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"

namespace vmecpp {

// Per-thread state for Anderson acceleration of the descent iteration.
//
// One Garabedian time step maps the Fourier state x_k to g_k. Anderson
// acceleration treats that map as a fixed-point iteration: it keeps a short
// history of map outputs g and residuals r = g - x, finds the least-squares
// combination of the recent residual differences that best cancels the
// current residual, and applies the same combination to the map outputs.
//
// Each thread holds the history of its own slice of the state, taken over the
// full stored radial range including the satellite points, so a linear
// combination of exchange-consistent iterates is again exchange-consistent at
// the partition seams. The normal equations of the least-squares problem are
// the only cross-thread quantities; Vmec reduces them over the team and hands
// the resulting coefficients back to every thread.
class AndersonAcceleration {
 public:
  AndersonAcceleration(const Sizes* s, int window);

  // Drop the history, e.g. after a restart or a multigrid step.
  void Reset();

  // Record the state before the time step.
  void CapturePreStep(const FourierGeometry& x);

  // Record the state after the time step and its residual against the
  // captured pre-step state, dropping the oldest pair beyond the window.
  void PushPostStep(const FourierGeometry& x);

  // Number of residual differences available for the least-squares problem.
  int NumDifferences() const;

  // This thread's contribution to the normal equations over the residual
  // differences: `m_gram` receives the k*k Gram matrix in row-major order and
  // `m_rhs` the k inner products with the current residual, k =
  // NumDifferences().
  void LocalNormalEquations(double* m_gram, double* m_rhs) const;

  // Overwrite this thread's slice of the state with the accelerated
  // combination g_last - sum_i gamma[i] * (g_{i+1} - g_i).
  void ApplyCombination(const double* gamma, FourierGeometry& m_x) const;

 private:
  void Pack(const FourierGeometry& x, Eigen::VectorXd& m_out) const;

  const Sizes* s_;
  int window_;

  Eigen::VectorXd pre_step_;
  std::deque<Eigen::VectorXd> map_outputs_;
  std::deque<Eigen::VectorXd> residuals_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_VMEC_ANDERSON_ACCELERATION_H_
