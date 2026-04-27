// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_ANDERSON_ACCELERATOR_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_ANDERSON_ACCELERATOR_H_

#include <Eigen/Dense>

namespace vmecpp {

// Anderson(1) acceleration for the free-boundary vacuum-pressure coupling.
//
// The free-boundary VMEC solver uses a Picard iteration to couple the plasma
// equilibrium to the vacuum magnetic pressure computed by Nestor. This class
// accelerates that fixed-point iteration using the Walker-Ni two-point mixing
// (Anderson(1)).
//
// Given consecutive Nestor outputs f_{k-1} and f_k with residuals r_{k-1}
// and r_k, the accelerated update is:
//   x_{k+1} = f_k - theta * (f_k - f_{k-1})
// where theta minimises ||r_k + theta * (r_{k-1} - r_k)||^2:
//   theta = r_k . (r_k - r_{k-1}) / ||r_k - r_{k-1}||^2
// and is clamped to [0, 2] for robustness.
//
// For a linear fixed-point iteration with contraction factor c, Anderson(1)
// converges in ONE Nestor call once two samples are available.
//
// Usage (must be called from an omp single context):
//   SaveOldPressure(p)  before each full Nestor update (ivacskip == 0)
//   Apply(p)            after each full Nestor update (when kActive)
//   Reset()             on iteration restart
class AndersonAccelerator {
 public:
  // history_size is retained for API compatibility but only the most recent
  // pair of iterates (Anderson(1)) is used internally.
  explicit AndersonAccelerator(int history_size);

  // Save a copy of the current vacuum pressure before the Nestor call.
  // This is the "input" x_k for the current Picard step.
  // Must be called from an omp single context.
  void SaveOldPressure(const Eigen::VectorXd& vacuum_pressure);

  // Apply Anderson(1) acceleration to vacuum_pressure after a full Nestor
  // update (ivacskip == 0). Modifies vacuum_pressure in place with the
  // accelerated estimate. Must be called from an omp single context.
  void Apply(Eigen::VectorXd& vacuum_pressure);

  // Clear the history. Call when the iteration is restarted.
  void Reset();

 private:
  int history_size_;

  // Vacuum pressure used as boundary condition since the last Nestor call.
  // Corresponds to x_k in the fixed-point notation.
  Eigen::VectorXd old_pressure_;

  // Previous Nestor output f_{k-1} and its residual r_{k-1}.
  // Used for the two-point Walker-Ni mixing formula.
  Eigen::VectorXd prev_nestor_output_;
  Eigen::VectorXd prev_residual_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_ANDERSON_ACCELERATOR_H_
