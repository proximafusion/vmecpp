// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_UTIL_REAL_TYPE_H_
#define VMECPP_COMMON_UTIL_REAL_TYPE_H_

#include <Eigen/Dense>

namespace vmecpp {

// Primary floating-point type used throughout all physics computations.
// On x86-64 Linux with GCC/Clang this is 80-bit extended precision
// (16 bytes, ~18-19 significant decimal digits), providing ~3 extra digits
// of precision beyond double.
using real_t = long double;

// Row-major dynamic Eigen matrix of real_t, used pervasively for 2-D Fourier
// coefficient arrays and geometry data. Row-major matches numpy's default
// memory layout for efficient pybind11 array transfers after conversion.
using RowMatrixXr =
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace vmecpp

#endif  // VMECPP_COMMON_UTIL_REAL_TYPE_H_
