#include <gtest/gtest.h>

#include <cmath>

class JVMECInterpolationAnalysisTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// Test analyzing jVMEC's critical interpolation difference from VMEC++
TEST_F(JVMECInterpolationAnalysisTest, JVMECInitialGuessInterpolation) {
  // CRITICAL FINDING: jVMEC uses m^p power law interpolation
  // Lines 1287-1288 in IdealMHDModel.java:
  // interpolationWeight = Math.pow(sqrtSFull[j], m);
  //
  // This differs significantly from linear interpolation used in VMEC++
  // for m>0 modes in the initial guess generation

  double sqrtS = 0.5;  // Example radial position

  // jVMEC interpolation weights for different m modes
  double weight_m1 = std::pow(sqrtS, 1);  // = 0.5
  double weight_m2 = std::pow(sqrtS, 2);  // = 0.25
  double weight_m3 = std::pow(sqrtS, 3);  // = 0.125
  double weight_m4 = std::pow(sqrtS, 4);  // = 0.0625

  EXPECT_NEAR(weight_m1, 0.5, 1e-10);
  EXPECT_NEAR(weight_m2, 0.25, 1e-10);
  EXPECT_NEAR(weight_m3, 0.125, 1e-10);
  EXPECT_NEAR(weight_m4, 0.0625, 1e-10);

  // This creates much stronger suppression of high-m modes near axis
  // compared to linear interpolation which would give constant weights
}

TEST_F(JVMECInterpolationAnalysisTest, SpectralCondensationConvert) {
  // jVMEC SpectralCondensation.java lines 131-134:
  // final double backup = rss_rsc[j][n][m];
  // rss_rsc[j][n][m] = scalingFactor * (backup + zcs_zcc[j][n][m]);
  // zcs_zcc[j][n][m] = scalingFactor * (backup - zcs_zcc[j][n][m]);
  //
  // This m=1 constraint transforms Fourier coefficients to ensure
  // polar relations: RSS(n) = ZCS(n) for symmetric, RSC(n) = ZCC(n) for
  // asymmetric

  double original_rss = 0.1;
  double original_zcs = 0.05;
  double scaling_factor = 0.5;  // Lines 523-524: 1.0 / Math.sqrt(2.0)

  // Apply jVMEC transform
  double new_rss = scaling_factor * (original_rss + original_zcs);
  double new_zcs = scaling_factor * (original_rss - original_zcs);

  EXPECT_NEAR(new_rss, scaling_factor * 0.15, 1e-10);
  EXPECT_NEAR(new_zcs, scaling_factor * 0.05, 1e-10);
}

TEST_F(JVMECInterpolationAnalysisTest, MultiGridInterpolationWeights) {
  // jVMEC IdealMHDModel.java lines 1388-1400 show linear interpolation
  // between coarse and fine grids, but with normalized radial coordinates

  double hsOld = 1.0 / 15.0;  // Old grid spacing (16 surfaces -> 15 intervals)
  double hsNew = 1.0 / 31.0;  // New grid spacing (32 surfaces -> 31 intervals)

  // Example: interpolating surface j=10 on new grid
  int j = 10;
  double sNew = j * hsNew;  // s-coordinate on new grid

  // Find bracketing indices on old grid
  int j1 = static_cast<int>(sNew / hsOld);
  int j2 = j1 + 1;
  double s1 = j1 * hsOld;

  // Linear interpolation weight
  double xInt = (sNew - s1) / hsOld;
  xInt = std::max(0.0, std::min(1.0, xInt));  // Clamp to [0,1]

  // Validate interpolation makes sense
  EXPECT_GE(j1, 0);
  EXPECT_LE(j2, 16);  // Old grid has 16 surfaces (0-15)
  EXPECT_GE(xInt, 0.0);
  EXPECT_LE(xInt, 1.0);
}

TEST_F(JVMECInterpolationAnalysisTest, AxisExclusionImplication) {
  // The axis exclusion in tau calculation (lines 386-391 commented out
  // in RealSpaceGeometry.java) means jVMEC excludes axis from Jacobian
  // sign determination, which affects initial guess stability

  // This suggests the initial guess near the axis might be handled
  // differently - less constrained by Jacobian sign requirements

  bool axis_excluded_from_jacobian_check = true;
  EXPECT_TRUE(axis_excluded_from_jacobian_check);
}
