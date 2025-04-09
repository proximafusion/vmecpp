// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <limits>

#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"

namespace testing {

TEST(TestNumericalComparison, CheckIsCloseRelAbs) {
  EXPECT_TRUE(IsCloseRelAbs(1.0, 1.0, 0.0));
  EXPECT_TRUE(IsCloseRelAbs(1.0, 1.0, 1.0e-15));

  // like absolute error for small numbers
  EXPECT_TRUE(IsCloseRelAbs(1.0e-15, 2.0e-15, 1.0e-14));

  // like a relative error for relatively-small numbers
  EXPECT_TRUE(IsCloseRelAbs(1.0, 1.0 + 1.0e-15, 1.0e-14));

  // like a relative error for large numbers
  EXPECT_TRUE(IsCloseRelAbs(1.0e15, 1.0e15 + 1, 1.0e-14));

  EXPECT_FALSE(IsCloseRelAbs(1.0, 2.0, 1.0e-3));

  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(IsCloseRelAbs(1.0, nan, 1.0e-3));
  EXPECT_FALSE(IsCloseRelAbs(nan, 1.0, 1.0e-3));

  EXPECT_TRUE(IsCloseRelAbs(nan, nan, 1.0e-3));
}  // CheckIsCloseRelAbs

}  // namespace testing
