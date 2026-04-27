// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/anderson_accelerator.h"

#include <algorithm>

namespace vmecpp {

AndersonAccelerator::AndersonAccelerator(int history_size)
    : history_size_(history_size) {}

void AndersonAccelerator::Reset() {
  old_pressure_.resize(0);
  prev_nestor_output_.resize(0);
  prev_residual_.resize(0);
}

void AndersonAccelerator::SaveOldPressure(
    const Eigen::VectorXd& vacuum_pressure) {
  old_pressure_ = vacuum_pressure;
}

void AndersonAccelerator::Apply(Eigen::VectorXd& vacuum_pressure) {
  // Only apply when we have a saved previous pressure (x_k).
  if (old_pressure_.empty()) {
    return;
  }

  // Residual r_k = f_k - x_k (new Nestor output minus previous input).
  const Eigen::VectorXd r_k = vacuum_pressure - old_pressure_;

  // Guard: if the current residual is negligibly small relative to the vacuum
  // pressure magnitude, the iteration is already at (or very near) the fixed
  // point. Pure Picard is optimal there and Anderson mixing can only perturb
  // a settled iterate. Clear history to avoid stale data on the next call.
  const double pressure_scale = std::max(1.0, vacuum_pressure.norm());
  if (r_k.norm() < 1.0e-9 * pressure_scale) {
    prev_nestor_output_.clear();
    prev_residual_.clear();
    return;
  }

  // Anderson(1) / Walker-Ni two-point mixing.
  //
  // Given the current Nestor output f_k (= vacuum_pressure on entry) and the
  // previous output f_{k-1}, the mixed update is:
  //
  //   x_{k+1} = f_k - theta * (f_k - f_{k-1})
  //
  // where theta minimises ||r_k + theta*(r_{k-1} - r_k)||^2:
  //
  //   theta = r_k . (r_k - r_{k-1}) / ||r_k - r_{k-1}||^2
  //
  // This reduces to a single well-conditioned scalar problem (no matrix
  // inversion) and converges in ONE Nestor call for linear fixed-point
  // iterations with any contraction factor.
  if (!prev_nestor_output_.empty()) {
    const Eigen::VectorXd delta_r = r_k - prev_residual_;
    const double denom = delta_r.squaredNorm();

    // Skip mixing when delta_r is negligibly small: consecutive residuals are
    // nearly identical, meaning no useful extrapolation information exists.
    if (denom > 1.0e-20 * r_k.squaredNorm()) {
      const double theta_raw = r_k.dot(delta_r) / denom;
      // Clamp theta to [0, 2]: values below 0 indicate a sign flip (rare in
      // practice) and values above 2 indicate aggressive overshoot. Both are
      // discarded in favour of the safe range.
      const double theta = std::max(0.0, std::min(2.0, theta_raw));
      vacuum_pressure -= theta * (vacuum_pressure - prev_nestor_output_);
    }
  }

  // Store the ORIGINAL (pre-mixing) f_k and r_k for the next Apply call.
  // Since vacuum_pressure may have been modified above, we reconstruct f_k
  // from the identity f_k = x_k + r_k = old_pressure_ + r_k.
  prev_nestor_output_ = old_pressure_ + r_k;
  prev_residual_ = r_k;
}

}  // namespace vmecpp
