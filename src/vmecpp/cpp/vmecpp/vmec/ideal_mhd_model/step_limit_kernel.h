// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_STEP_LIMIT_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_STEP_LIMIT_KERNEL_H_

#include <algorithm>
#include <cmath>

namespace vmecpp {

// Smallest root in (0, 1] of c x^2 + a x + f with f > 0, or 1 without one.
inline double FirstRootInUnitInterval(double c, double a, double f) {
  double root = 1.0;
  if (c == 0.0) {
    if (a < 0.0) {
      root = std::min(root, -f / a);
    }
    return root;
  }
  const double discriminant = a * a - 4.0 * c * f;
  if (discriminant < 0.0) {
    return root;
  }
  // roots q / c and f / q, formed without cancellation
  const double q = -0.5 * (a + std::copysign(std::sqrt(discriminant), a));
  if (q != 0.0) {
    const double first = q / c;
    const double second = f / q;
    if (first > 0.0) {
      root = std::min(root, first);
    }
    if (second > 0.0) {
      root = std::min(root, second);
    }
  }
  return root;
}

// Largest fraction alpha in (0, 1] of a step along which the Jacobian tau keeps
// at least retained_fraction of its value at every one of the n points. tau is
// a quadratic form in the geometry, so with tau0 and tau1 its values at the
// start and the end of the step and tau_d its value for the change in the
// geometry, tau(alpha) = tau0 + alpha (tau1 - tau0 - tau_d) + alpha^2 tau_d.
inline double LargestStepFraction(const double* tau0, const double* tau1,
                                  const double* tau_d, int n,
                                  double retained_fraction) {
  const double allowed_decrease = 1.0 - retained_fraction;
  double alpha = 1.0;
  for (int i = 0; i < n; ++i) {
    if (tau0[i] == 0.0) {
      continue;
    }
    // tau(alpha) / tau0 - retained_fraction = c alpha^2 + a alpha +
    // allowed_decrease
    const double a = (tau1[i] - tau0[i] - tau_d[i]) / tau0[i];
    const double c = tau_d[i] / tau0[i];
    alpha = std::min(alpha, FirstRootInUnitInterval(c, a, allowed_decrease));
  }
  return alpha;
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_STEP_LIMIT_KERNEL_H_
