// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"

#include <cmath>
#include <numbers>
#include <string>

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"

namespace composed_types {

using testing::IsCloseRelAbs;

using std::numbers::inv_sqrt3;
using std::numbers::sqrt2;
using std::numbers::sqrt3;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(ComposedTypesLibTest, CheckLength) {
  static constexpr double kTolerance = 1.0e-15;

  Vector3d vector_x;
  vector_x.set_x(1.0);
  vector_x.set_y(0.0);
  vector_x.set_z(0.0);
  EXPECT_TRUE(IsCloseRelAbs(1.0, Length(vector_x), kTolerance));

  Vector3d vector_y;
  vector_y.set_x(0.0);
  vector_y.set_y(1.0);
  vector_y.set_z(0.0);
  EXPECT_TRUE(IsCloseRelAbs(1.0, Length(vector_y), kTolerance));

  Vector3d vector_z;
  vector_z.set_x(0.0);
  vector_z.set_y(0.0);
  vector_z.set_z(1.0);
  EXPECT_TRUE(IsCloseRelAbs(1.0, Length(vector_z), kTolerance));

  Vector3d vector_xy;
  vector_xy.set_x(1.0);
  vector_xy.set_y(1.0);
  vector_xy.set_z(0.0);
  EXPECT_TRUE(IsCloseRelAbs(sqrt2, Length(vector_xy), kTolerance));

  Vector3d vector_xz;
  vector_xz.set_x(1.0);
  vector_xz.set_y(0.0);
  vector_xz.set_z(1.0);
  EXPECT_TRUE(IsCloseRelAbs(sqrt2, Length(vector_xz), kTolerance));

  Vector3d vector_yz;
  vector_yz.set_x(0.0);
  vector_yz.set_y(1.0);
  vector_yz.set_z(1.0);
  EXPECT_TRUE(IsCloseRelAbs(sqrt2, Length(vector_yz), kTolerance));

  Vector3d vector_xyz;
  vector_xyz.set_x(1.0);
  vector_xyz.set_y(1.0);
  vector_xyz.set_z(1.0);
  EXPECT_TRUE(IsCloseRelAbs(sqrt3, Length(vector_xyz), kTolerance));
}  // CheckLength

TEST(ComposedTypesLibTest, CheckScaleToInPlane) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kDesiredLength = 1.23;

  const double kExpectedComponent = kDesiredLength / sqrt2;

  Vector3d vector_xy;
  vector_xy.set_x(1.0);
  vector_xy.set_y(1.0);
  vector_xy.set_z(0.0);
  Vector3d normalized_xy = ScaleTo(vector_xy, kDesiredLength);
  EXPECT_TRUE(IsCloseRelAbs(kExpectedComponent, normalized_xy.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(kExpectedComponent, normalized_xy.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, normalized_xy.z(), kTolerance));

  Vector3d vector_xz;
  vector_xz.set_x(1.0);
  vector_xz.set_y(0.0);
  vector_xz.set_z(1.0);
  Vector3d normalized_xz = ScaleTo(vector_xz, kDesiredLength);
  EXPECT_TRUE(IsCloseRelAbs(kExpectedComponent, normalized_xz.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, normalized_xz.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(kExpectedComponent, normalized_xz.z(), kTolerance));

  Vector3d vector_yz;
  vector_yz.set_x(0.0);
  vector_yz.set_y(1.0);
  vector_yz.set_z(1.0);
  Vector3d normalized_yz = ScaleTo(vector_yz, kDesiredLength);
  EXPECT_TRUE(IsCloseRelAbs(0.0, normalized_yz.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(kExpectedComponent, normalized_yz.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(kExpectedComponent, normalized_yz.z(), kTolerance));
}  // CheckScaleToInPlane

TEST(ComposedTypesLibTest, CheckScaleToInVolume) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kDesiredLength = 1.23;

  const double kExpectedComponent = kDesiredLength / sqrt3;

  Vector3d vector_xyz;
  vector_xyz.set_x(1.0);
  vector_xyz.set_y(1.0);
  vector_xyz.set_z(1.0);
  Vector3d normalized_xyz = ScaleTo(vector_xyz, kDesiredLength);
  EXPECT_TRUE(
      IsCloseRelAbs(kExpectedComponent, normalized_xyz.x(), kTolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(kExpectedComponent, normalized_xyz.y(), kTolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(kExpectedComponent, normalized_xyz.z(), kTolerance));
}  // CheckScaleToInVolume

TEST(ComposedTypesLibTest, CheckNormalize) {
  static constexpr double kTolerance = 1.0e-15;

  static constexpr double kInverseSqrt2 = 1.0 / sqrt2;

  Vector3d vector_xy;
  vector_xy.set_x(1.0);
  vector_xy.set_y(1.0);
  vector_xy.set_z(0.0);
  Vector3d normalized_xy = Normalize(vector_xy);
  EXPECT_TRUE(IsCloseRelAbs(kInverseSqrt2, normalized_xy.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(kInverseSqrt2, normalized_xy.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, normalized_xy.z(), kTolerance));

  Vector3d vector_xz;
  vector_xz.set_x(1.0);
  vector_xz.set_y(0.0);
  vector_xz.set_z(1.0);
  Vector3d normalized_xz = Normalize(vector_xz);
  EXPECT_TRUE(IsCloseRelAbs(kInverseSqrt2, normalized_xz.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, normalized_xz.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(kInverseSqrt2, normalized_xz.z(), kTolerance));

  Vector3d vector_yz;
  vector_yz.set_x(0.0);
  vector_yz.set_y(1.0);
  vector_yz.set_z(1.0);
  Vector3d normalized_yz = Normalize(vector_yz);
  EXPECT_TRUE(IsCloseRelAbs(0.0, normalized_yz.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(kInverseSqrt2, normalized_yz.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(kInverseSqrt2, normalized_yz.z(), kTolerance));

  Vector3d vector_xyz;
  vector_xyz.set_x(1.0);
  vector_xyz.set_y(1.0);
  vector_xyz.set_z(1.0);
  Vector3d normalized_xyz = Normalize(vector_xyz);
  EXPECT_TRUE(IsCloseRelAbs(inv_sqrt3, normalized_xyz.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(inv_sqrt3, normalized_xyz.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(inv_sqrt3, normalized_xyz.z(), kTolerance));
}  // CheckNormalize

TEST(ComposedTypesLibTest, CheckAdd) {
  static constexpr double kTolerance = 1.0e-15;

  Vector3d vector_1;
  vector_1.set_x(1.23);
  vector_1.set_y(4.56);
  vector_1.set_z(7.89);

  Vector3d vector_2;
  vector_2.set_x(3.14);
  vector_2.set_y(2.71);
  vector_2.set_z(1.41);

  Vector3d sum = Add(vector_1, vector_2);

  EXPECT_TRUE(IsCloseRelAbs(1.23 + 3.14, sum.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(4.56 + 2.71, sum.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(7.89 + 1.41, sum.z(), kTolerance));
}  // CheckAdd

TEST(ComposedTypesLibTest, CheckSubtract) {
  static constexpr double kTolerance = 1.0e-15;

  Vector3d vector_1;
  vector_1.set_x(1.23);
  vector_1.set_y(4.56);
  vector_1.set_z(7.89);

  Vector3d vector_2;
  vector_2.set_x(3.14);
  vector_2.set_y(2.71);
  vector_2.set_z(1.41);

  Vector3d difference = Subtract(vector_1, vector_2);

  EXPECT_TRUE(IsCloseRelAbs(1.23 - 3.14, difference.x(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(4.56 - 2.71, difference.y(), kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(7.89 - 1.41, difference.z(), kTolerance));
}  // CheckSubtract

TEST(ComposedTypesLibTest, CheckDotProduct) {
  static constexpr double kTolerance = 1.0e-15;

  Vector3d vector_1;
  vector_1.set_x(1.23);
  vector_1.set_y(4.56);
  vector_1.set_z(7.89);

  Vector3d vector_2;
  vector_2.set_x(3.14);
  vector_2.set_y(2.71);
  vector_2.set_z(1.41);

  const double dot_product = DotProduct(vector_1, vector_2);

  double dot_product_reference = 0.0;
  dot_product_reference += vector_1.x() * vector_2.x();
  dot_product_reference += vector_1.y() * vector_2.y();
  dot_product_reference += vector_1.z() * vector_2.z();

  EXPECT_TRUE(IsCloseRelAbs(dot_product_reference, dot_product, kTolerance));
}  // CheckDotProduct

TEST(ComposedTypesLibTest, CheckCrossProduct) {
  static constexpr double kTolerance = 1.0e-15;

  Vector3d vector_1;
  vector_1.set_x(1.23);
  vector_1.set_y(4.56);
  vector_1.set_z(7.89);

  Vector3d vector_2;
  vector_2.set_x(3.14);
  vector_2.set_y(2.71);
  vector_2.set_z(1.41);

  Vector3d cross_product = CrossProduct(vector_1, vector_2);

  EXPECT_TRUE(
      IsCloseRelAbs(vector_1.y() * vector_2.z() - vector_1.z() * vector_2.y(),
                    cross_product.x(), kTolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(vector_1.z() * vector_2.x() - vector_1.x() * vector_2.z(),
                    cross_product.y(), kTolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(vector_1.x() * vector_2.y() - vector_1.y() * vector_2.x(),
                    cross_product.z(), kTolerance));
}  // CheckCrossProduct

TEST(ComposedTypesLibTest, CheckMostPerpendicularCoordinateAxis) {
  Vector3d vector_1;
  vector_1.set_x(1.23);
  vector_1.set_y(4.56);
  vector_1.set_z(7.89);
  Vector3d most_perp_to_1 = MostPerpendicularCoordinateAxis(vector_1);

  // smallest component of `vector_1` is in x
  // -> expect unit vector along x
  EXPECT_EQ(most_perp_to_1.x(), 1.0);
  EXPECT_EQ(most_perp_to_1.y(), 0.0);
  EXPECT_EQ(most_perp_to_1.z(), 0.0);

  Vector3d vector_2;
  vector_2.set_x(3.14);
  vector_2.set_y(2.71);
  vector_2.set_z(1.41);
  Vector3d most_perp_to_2 = MostPerpendicularCoordinateAxis(vector_2);

  // smallest component of `vector_2` is in z
  // -> expect unit vector along z
  EXPECT_EQ(most_perp_to_2.x(), 0.0);
  EXPECT_EQ(most_perp_to_2.y(), 0.0);
  EXPECT_EQ(most_perp_to_2.z(), 1.0);

  Vector3d vector_3;
  vector_3.set_x(-3.14);
  vector_3.set_y(-2.71);
  vector_3.set_z(-1.41);
  Vector3d most_perp_to_3 = MostPerpendicularCoordinateAxis(vector_3);

  // smallest component (in magnitude) of `vector_3` is in z
  // -> expect unit vector along z
  EXPECT_EQ(most_perp_to_3.x(), 0.0);
  EXPECT_EQ(most_perp_to_3.y(), 0.0);
  EXPECT_EQ(most_perp_to_3.z(), 1.0);
}  // CheckMostPerpendicularCoordinateAxis

TEST(ComposedTypesLibTest, CheckOrthonormalFrameAroundAxis) {
  static constexpr double kTolerance = 1.0e-15;

  // test around x axis -> expect y and z axis
  // -> order is (x, y, z), which is an even permutation of (x, y, z),
  //    hence no axis flipped its direction
  Vector3d vector_1;
  vector_1.set_x(3.14);
  vector_1.set_y(0.0);
  vector_1.set_z(0.0);
  std::array<Vector3d, 3> axes_around_1 = OrthonormalFrameAroundAxis(vector_1);

  EXPECT_EQ(axes_around_1[0].x(), 1.0);
  EXPECT_EQ(axes_around_1[0].y(), 0.0);
  EXPECT_EQ(axes_around_1[0].z(), 0.0);

  EXPECT_EQ(axes_around_1[1].x(), 0.0);
  EXPECT_EQ(axes_around_1[1].y(), 1.0);
  EXPECT_EQ(axes_around_1[1].z(), 0.0);

  EXPECT_EQ(axes_around_1[2].x(), 0.0);
  EXPECT_EQ(axes_around_1[2].y(), 0.0);
  EXPECT_EQ(axes_around_1[2].z(), 1.0);

  // test around y axis -> expect x and -z axis
  // -> order is (y, x, z), which is an odd permutation of (x, y, z),
  //    hence one of the axes had to flip its direction
  Vector3d vector_2;
  vector_2.set_x(0.0);
  vector_2.set_y(3.14);
  vector_2.set_z(0.0);
  std::array<Vector3d, 3> axes_around_2 = OrthonormalFrameAroundAxis(vector_2);

  EXPECT_EQ(axes_around_2[0].x(), 0.0);
  EXPECT_EQ(axes_around_2[0].y(), 1.0);
  EXPECT_EQ(axes_around_2[0].z(), 0.0);

  EXPECT_EQ(axes_around_2[1].x(), 1.0);
  EXPECT_EQ(axes_around_2[1].y(), 0.0);
  EXPECT_EQ(axes_around_2[1].z(), 0.0);

  EXPECT_EQ(axes_around_2[2].x(), 0.0);
  EXPECT_EQ(axes_around_2[2].y(), 0.0);
  EXPECT_EQ(axes_around_2[2].z(), -1.0);

  // test around z axis -> expect x and y axis
  // -> order is (z, x, y), which is an even permutation of (x, y, z),
  //    hence no axis flipped its direction
  Vector3d vector_3;
  vector_3.set_x(0.0);
  vector_3.set_y(0.0);
  vector_3.set_z(3.14);
  std::array<Vector3d, 3> axes_around_3 = OrthonormalFrameAroundAxis(vector_3);

  EXPECT_EQ(axes_around_3[0].x(), 0.0);
  EXPECT_EQ(axes_around_3[0].y(), 0.0);
  EXPECT_EQ(axes_around_3[0].z(), 1.0);

  EXPECT_EQ(axes_around_3[1].x(), 1.0);
  EXPECT_EQ(axes_around_3[1].y(), 0.0);
  EXPECT_EQ(axes_around_3[1].z(), 0.0);

  EXPECT_EQ(axes_around_3[2].x(), 0.0);
  EXPECT_EQ(axes_around_3[2].y(), 1.0);
  EXPECT_EQ(axes_around_3[2].z(), 0.0);

  // test around an arbitrary vector
  // and test that projections among each other vanish
  Vector3d vector_4;
  vector_4.set_x(3.14);
  vector_4.set_y(2.71);
  vector_4.set_z(1.41);
  std::array<Vector3d, 3> axes_around_4 = OrthonormalFrameAroundAxis(vector_4);

  EXPECT_TRUE(IsCloseRelAbs(0.0, DotProduct(axes_around_4[0], axes_around_4[1]),
                            kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, DotProduct(axes_around_4[0], axes_around_4[2]),
                            kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(0.0, DotProduct(axes_around_4[1], axes_around_4[2]),
                            kTolerance));
}  // CheckOrthonormalFrameAroundAxis

TEST(TestReadCoefficientsFromCsv, CheckReadAxisCoefficientsFromCsv) {
  std::string axis_coefficients_csv = R"(n,raxis_c,zaxis_s,raxis_s,zaxis_c
0,3.999,0,0,0
1,1.026,1.58,0,0
2,-0.068,0.01,0,0
3,0,0,0,0
4,0,0,0,0
5,0,0,0,0)";

  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok()) << axis_coefficients.status();

  absl::StatusOr<std::vector<int>> mode_numbers =
      ModeNumbers(*axis_coefficients);
  ASSERT_TRUE(mode_numbers.ok()) << mode_numbers.status();
  EXPECT_THAT(*mode_numbers, ElementsAre(0, 1, 2, 3, 4, 5));

  absl::StatusOr<std::vector<double>> r_cos =
      CoefficientsRCos(*axis_coefficients);
  ASSERT_TRUE(r_cos.ok()) << r_cos.status();
  EXPECT_THAT(*r_cos, ElementsAre(3.999, 1.026, -0.068, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> z_sin =
      CoefficientsZSin(*axis_coefficients);
  ASSERT_TRUE(z_sin.ok()) << z_sin.status();
  EXPECT_THAT(*z_sin, ElementsAre(0.0, 1.58, 0.01, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> r_sin =
      CoefficientsRSin(*axis_coefficients);
  ASSERT_TRUE(r_sin.ok()) << r_sin.status();
  EXPECT_THAT(*r_sin, ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> z_cos =
      CoefficientsZCos(*axis_coefficients);
  ASSERT_TRUE(z_cos.ok()) << z_cos.status();
  EXPECT_THAT(*z_cos, ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}  // CheckReadAxisCoefficientsFromCsv

// check round-trip serialization/deserialization
// rely on CheckReadAxisCoefficientsFromCsv for de-serialization
TEST(TestWriteCoefficientsToCsv, CheckWriteAxisCoefficientsToCsv) {
  std::string original_axis_coefficients_csv =
      R"(n,raxis_c,zaxis_s,raxis_s,zaxis_c
0,3.999,0,0,0
1,1.026,1.58,0,0
2,-0.068,0.01,0,0
3,0,0,0,0
4,0,0,0,0
5,0,0,0,0)";

  absl::StatusOr<CurveRZFourier> original_axis_coefficients =
      CurveRZFourierFromCsv(original_axis_coefficients_csv);
  ASSERT_TRUE(original_axis_coefficients.ok())
      << original_axis_coefficients.status();

  // call under test here
  absl::StatusOr<std::string> axis_coefficients_csv =
      CurveRZFourierToCsv(*original_axis_coefficients);
  ASSERT_TRUE(axis_coefficients_csv.ok()) << axis_coefficients_csv.status();

  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok()) << axis_coefficients.status();

  absl::StatusOr<std::vector<int>> mode_numbers =
      ModeNumbers(*axis_coefficients);
  ASSERT_TRUE(mode_numbers.ok()) << mode_numbers.status();
  EXPECT_THAT(*mode_numbers, ElementsAre(0, 1, 2, 3, 4, 5));

  absl::StatusOr<std::vector<double>> r_cos =
      CoefficientsRCos(*axis_coefficients);
  ASSERT_TRUE(r_cos.ok()) << r_cos.status();
  EXPECT_THAT(*r_cos, ElementsAre(3.999, 1.026, -0.068, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> z_sin =
      CoefficientsZSin(*axis_coefficients);
  ASSERT_TRUE(z_sin.ok()) << z_sin.status();
  EXPECT_THAT(*z_sin, ElementsAre(0.0, 1.58, 0.01, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> r_sin =
      CoefficientsRSin(*axis_coefficients);
  ASSERT_TRUE(r_sin.ok()) << r_sin.status();
  EXPECT_THAT(*r_sin, ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> z_cos =
      CoefficientsZCos(*axis_coefficients);
  ASSERT_TRUE(z_cos.ok()) << z_cos.status();
  EXPECT_THAT(*z_cos, ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}  // CheckWriteAxisCoefficientsToCsv

TEST(TestReadCoefficientsFromCsv, CheckReadBoundaryCoefficientsFromCsv) {
  std::string boundary_coefficients_csv = R"(n,m,rbc,zbs,rbs,zbc
0,0,3.999,0,0,0
0,1,1.026,1.58,0,0
0,2,-0.068,0.01,0,0
0,3,0,0,0,0
0,4,0,0,0,0
0,5,0,0,0,0)";

  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok()) << boundary_coefficients.status();

  absl::StatusOr<std::vector<int>> poloidal_mode_numbers =
      PoloidalModeNumbers(*boundary_coefficients);
  ASSERT_TRUE(poloidal_mode_numbers.ok()) << poloidal_mode_numbers.status();
  EXPECT_THAT(*poloidal_mode_numbers, ElementsAre(0, 1, 2, 3, 4, 5));

  absl::StatusOr<std::vector<int>> toroidal_mode_numbers =
      ToroidalModeNumbers(*boundary_coefficients);
  ASSERT_TRUE(toroidal_mode_numbers.ok()) << toroidal_mode_numbers.status();
  EXPECT_THAT(*toroidal_mode_numbers, ElementsAre(0, 0, 0, 0, 0, 0));

  absl::StatusOr<std::vector<double>> r_cos =
      CoefficientsRCos(*boundary_coefficients);
  ASSERT_TRUE(r_cos.ok()) << r_cos.status();
  EXPECT_THAT(*r_cos, ElementsAre(3.999, 1.026, -0.068, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> z_sin =
      CoefficientsZSin(*boundary_coefficients);
  ASSERT_TRUE(z_sin.ok()) << z_sin.status();
  EXPECT_THAT(*z_sin, ElementsAre(0.0, 1.58, 0.01, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> r_sin =
      CoefficientsRSin(*boundary_coefficients);
  ASSERT_TRUE(r_sin.ok()) << r_sin.status();
  EXPECT_THAT(*r_sin, ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> z_cos =
      CoefficientsZCos(*boundary_coefficients);
  ASSERT_TRUE(z_cos.ok()) << z_cos.status();
  EXPECT_THAT(*z_cos, ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}  // CheckReadBoundaryCoefficientsFromCsv

// check round-trip serialization/deserialization
// rely on CheckReadBoundaryCoefficientsFromCsv for de-serialization
TEST(TestWriteCoefficientsToCsv, CheckWriteCoefficientsToCsv) {
  std::string original_boundary_coefficients_csv = R"(n,m,rbc,zbs,rbs,zbc
0,0,3.999,0,0,0
0,1,1.026,1.58,0,0
0,2,-0.068,0.01,0,0
0,3,0,0,0,0
0,4,0,0,0,0
0,5,0,0,0,0)";

  absl::StatusOr<SurfaceRZFourier> original_boundary_coefficients =
      SurfaceRZFourierFromCsv(original_boundary_coefficients_csv);
  ASSERT_TRUE(original_boundary_coefficients.ok())
      << original_boundary_coefficients.status();

  // call under test here
  absl::StatusOr<std::string> boundary_coefficients_csv =
      SurfaceRZFourierToCsv(*original_boundary_coefficients);
  ASSERT_TRUE(boundary_coefficients_csv.ok())
      << boundary_coefficients_csv.status();

  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok()) << boundary_coefficients.status();

  absl::StatusOr<std::vector<int>> poloidal_mode_numbers =
      PoloidalModeNumbers(*boundary_coefficients);
  ASSERT_TRUE(poloidal_mode_numbers.ok()) << poloidal_mode_numbers.status();
  EXPECT_THAT(*poloidal_mode_numbers, ElementsAre(0, 1, 2, 3, 4, 5));

  absl::StatusOr<std::vector<int>> toroidal_mode_numbers =
      ToroidalModeNumbers(*boundary_coefficients);
  ASSERT_TRUE(toroidal_mode_numbers.ok()) << toroidal_mode_numbers.status();
  EXPECT_THAT(*toroidal_mode_numbers, ElementsAre(0, 0, 0, 0, 0, 0));

  absl::StatusOr<std::vector<double>> r_cos =
      CoefficientsRCos(*boundary_coefficients);
  ASSERT_TRUE(r_cos.ok()) << r_cos.status();
  EXPECT_THAT(*r_cos, ElementsAre(3.999, 1.026, -0.068, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> z_sin =
      CoefficientsZSin(*boundary_coefficients);
  ASSERT_TRUE(z_sin.ok()) << z_sin.status();
  EXPECT_THAT(*z_sin, ElementsAre(0.0, 1.58, 0.01, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> r_sin =
      CoefficientsRSin(*boundary_coefficients);
  ASSERT_TRUE(r_sin.ok()) << r_sin.status();
  EXPECT_THAT(*r_sin, ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

  absl::StatusOr<std::vector<double>> z_cos =
      CoefficientsZCos(*boundary_coefficients);
  ASSERT_TRUE(z_cos.ok()) << z_cos.status();
  EXPECT_THAT(*z_cos, ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}  // CheckWriteCoefficientsToCsv

}  // namespace composed_types
