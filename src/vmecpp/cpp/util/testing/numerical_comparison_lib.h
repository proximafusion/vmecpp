// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef UTIL_TESTING_NUMERICAL_COMPARISON_LIB_H_
#define UTIL_TESTING_NUMERICAL_COMPARISON_LIB_H_

#include <Eigen/Dense>
#include <vector>

namespace testing {

/// Check if two values are approximately equal within a prescribed tolerance.
///
/// For values much smaller than 1, this is similar to a comparison of the
/// absolute values. For values much greater than 1, this is similar to a
/// comparison of the relative values. This method is described in Gill, Murray
/// & Wright, "Practical Optimization" (1984).
///
/// If the expected value is NaN, the actual value is checked to also be NaN.
/// If the expected value is not NaN, the actual value is checked to also not be
/// NaN.
///
/// @param expected  expected result
/// @param actual    actual result
/// @param tolerance relative or absolute tolerance on the mismatch between the
/// expected and the actual values
/// @return true if the values match within the prescribed tolerance; false
/// otherwise
bool IsCloseRelAbs(double expected, double actual, double tolerance);

/// Check that two STL vectors have same size and their elements pass the
/// IsCloseRelAbs check pairwise.
bool IsVectorCloseRelAbs(const std::vector<double>& expected,
                         const std::vector<double>& actual, double tolerance);

/// Check that two Eigen vectors have same size and their elements pass the
/// IsCloseRelAbs check pairwise.
bool IsVectorCloseRelAbs(const Eigen::VectorXd& expected,
                         const Eigen::VectorXd& actual, double tolerance);
}  // namespace testing

#endif  // UTIL_TESTING_NUMERICAL_COMPARISON_LIB_H_
