// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_COMPOSED_TYPES_LIB_COMPOSED_TYPES_LIB_H_
#define VMECPP_COMMON_COMPOSED_TYPES_LIB_COMPOSED_TYPES_LIB_H_

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "vmecpp/common/composed_types_definition/composed_types.h"

namespace composed_types {

absl::Status IsVector3dFullyPopulated(const Vector3d& vector,
                                      absl::string_view vector_name = "");

// Compute the length of a three-dimensional Cartesian vector,
// that is, sqrt(x^2 + y^2 + z^2).
double Length(const Vector3d& vector);

// Re-scale a vector to have a given length without changing its direction.
Vector3d ScaleTo(const Vector3d& vector, double desired_length);

// Normalize a vector to unit length without changing its direction.
Vector3d Normalize(const Vector3d& vector);

// Compute the component-wise sum of two vectors.
// Let (x1, y1, z1) be the components of vector_1
// and (x2, y2, z2) be the components of vector_2,
// then the result is (x1 + x2, y1 + y2, z1 + z2).
Vector3d Add(const Vector3d& vector_1, const Vector3d& vector_2);

// Compute the component-wise difference of two vectors.
// Let (x1, y1, z1) be the components of vector_1
// and (x2, y2, z2) be the components of vector_2,
// then the result is (x1 - x2, y1 - y2, z1 - z2).
Vector3d Subtract(const Vector3d& vector_1, const Vector3d& vector_2);

// Compute the dot product of two vectors.
// Let (x1, y1, z1) be the components of vector_1
// and (x2, y2, z2) be the components of vector_2,
// then the result is x1 *x2 + y1 * y2 + z1 * z2.
double DotProduct(const Vector3d& vector_1, const Vector3d& vector_2);

// Compute the cross product of two vectors (sometimes also called vector
// product). Let (x1, y1, z1) be the components of vector_1 and (x2, y2, z2) be
// the components of vector_2, then the result is (y1 * z2 - z1 * y2, z1 * x2 -
// x1 * z2, x1 * y2 - y1 * x2).
Vector3d CrossProduct(const Vector3d& vector_1, const Vector3d& vector_2);

// Return that Cartesian unit vector (e_x, e_y or e_z) which is "most
// perpendicular" to the given vector `axis`. Here, perpendicularity is measured
// by the projection of the `axis` on the respective coordinate axes using the
// dot product. In this metric, the "most perpendiular" coordinate axis is the
// one which has the smallest dot product with the given `axis`.
// Effectively, this is accomplished by looking for the minimum among the
// Cartesian components of `axis`.
Vector3d MostPerpendicularCoordinateAxis(const Vector3d& axis);

// Compute an ortho-normal coordinate frame around a given direction `axis`
// and the Cartesian coordinate axis most-perpendicular to it (see
// `MostPerpendicularCoordinateAxis`) using the Gram-Schmidt process
// (https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process). It is not
// required that the given `axis` has unit length. The returned set of unit
// vectors each have unit length.
std::array<Vector3d, 3> OrthonormalFrameAroundAxis(const Vector3d& axis);

absl::Status IsFourierCoefficient1DFullyPopulated(
    const FourierCoefficient1D& fourier_coefficient,
    absl::string_view fourier_coefficient_name = "");

absl::Status IsFourierCoefficient2DFullyPopulated(
    const FourierCoefficient2D& fourier_coefficient,
    absl::string_view fourier_coefficient_name = "");

absl::Status IsCurveRZFourierFullyPopulated(const CurveRZFourier& curve);
absl::StatusOr<CurveRZFourier> CurveRZFourierFromCsv(
    const std::string& axis_coefficients_csv);
absl::StatusOr<std::string> CurveRZFourierToCsv(const CurveRZFourier& axis);
absl::StatusOr<std::vector<int>> ModeNumbers(const CurveRZFourier& curve);
absl::StatusOr<std::vector<double>> CoefficientsRCos(
    const CurveRZFourier& curve);
absl::StatusOr<std::vector<double>> CoefficientsZSin(
    const CurveRZFourier& curve);
absl::StatusOr<std::vector<double>> CoefficientsRSin(
    const CurveRZFourier& curve);
absl::StatusOr<std::vector<double>> CoefficientsZCos(
    const CurveRZFourier& curve);

absl::Status IsSurfaceRZFourierFullyPopulated(const SurfaceRZFourier& surface);
absl::StatusOr<SurfaceRZFourier> SurfaceRZFourierFromCsv(
    const std::string& boundary_coefficients_csv);
absl::StatusOr<std::string> SurfaceRZFourierToCsv(
    const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<int>> PoloidalModeNumbers(
    const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<int>> ToroidalModeNumbers(
    const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<double>> CoefficientsRCos(
    const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<double>> CoefficientsZSin(
    const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<double>> CoefficientsRSin(
    const SurfaceRZFourier& surface);
absl::StatusOr<std::vector<double>> CoefficientsZCos(
    const SurfaceRZFourier& surface);

}  // namespace composed_types

#endif  // VMECPP_COMMON_COMPOSED_TYPES_LIB_COMPOSED_TYPES_LIB_H_
