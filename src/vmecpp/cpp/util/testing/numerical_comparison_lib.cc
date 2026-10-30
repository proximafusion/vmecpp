// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/testing/numerical_comparison_lib.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

#include "absl/strings/str_format.h"

bool testing::IsCloseRelAbs(double expected, double actual, double tolerance) {
  const double rel_abs_error = (actual - expected) / (1.0 + std::abs(expected));
  if (std::abs(rel_abs_error) > tolerance ||
      (std::isnan(expected) != std::isnan(actual))) {
    std::cerr << absl::StrFormat(
        "out-of-tolerance: |% .3e| > % .3e\n  expected = % .20e\n    "
        "actual = % .20e\n",
        rel_abs_error, tolerance, expected, actual);
    return false;
  }
  return true;
}

bool testing::IsVectorCloseRelAbs(const std::vector<double>& expected,
                                  const std::vector<double>& actual,
                                  double tolerance) {
  const auto& expected_eigen = Eigen::Map<const Eigen::VectorXd>(
      expected.data(), static_cast<Eigen::Index>(expected.size()));
  const auto& actual_eigen = Eigen::Map<const Eigen::VectorXd>(
      actual.data(), static_cast<Eigen::Index>(actual.size()));

  return IsVectorCloseRelAbs(expected_eigen, actual_eigen, tolerance);
}

bool testing::IsVectorCloseRelAbs(const Eigen::VectorXd& expected,
                                  const Eigen::VectorXd& actual,
                                  double tolerance) {
  if (expected.size() != actual.size()) {
    return false;
  }

  for (int i = 0; i < expected.size(); ++i) {
    if (!IsCloseRelAbs(expected[i], actual[i], tolerance)) {
      return false;
    }
  }

  return true;
}
