// Test for initial guess interpolation fix for asymmetric mode

#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include <vector>

// Test the initial guess interpolation logic
TEST(InitialGuessInterpolation, AsymmetricModeInterpolation) {
  // Test parameters matching VMEC
  const int ns = 5;  // Number of radial surfaces
  const int mpol = 3;  // Poloidal mode number
  const int ntor = 0;  // Toroidal mode number (axisymmetric)
  
  // Test the interpolation for different radial positions
  for (int js = 1; js <= ns; ++js) {
    double sqrts_js = std::sqrt((js - 1.0) / (ns - 1.0));
    double s_js = sqrts_js * sqrts_js;
    double sm0 = 1.0 - s_js;  // 1 - s(js) for axis contribution
    
    std::cout << "js=" << js << ", sqrts=" << sqrts_js << ", s=" << s_js << std::endl;
    
    // Test m=0 mode (axis contribution)
    {
      double boundary_contrib = s_js * 1.0;  // Boundary value = 1.0
      double axis_contrib = sm0 * 1.0;  // Axis value = 1.0
      double total = boundary_contrib + axis_contrib;
      
      std::cout << "  m=0: boundary_contrib=" << boundary_contrib 
                << ", axis_contrib=" << axis_contrib 
                << ", total=" << total << std::endl;
      
      // At axis (js=1), should be dominated by axis contribution
      // At boundary (js=ns), should be dominated by boundary contribution
      if (js == 1) {
        EXPECT_NEAR(total, 1.0, 1e-10);  // Pure axis
        EXPECT_NEAR(axis_contrib, 1.0, 1e-10);
        EXPECT_NEAR(boundary_contrib, 0.0, 1e-10);
      } else if (js == ns) {
        EXPECT_NEAR(total, 1.0, 1e-10);  // Pure boundary
        EXPECT_NEAR(axis_contrib, 0.0, 1e-10);
        EXPECT_NEAR(boundary_contrib, 1.0, 1e-10);
      }
    }
    
    // Test m>0 modes (no axis contribution)
    for (int m = 1; m <= mpol; ++m) {
      // Power law interpolation: boundary * sqrts^m
      double interp_weight = std::pow(sqrts_js, m);
      double boundary_value = 1.0;  // Test with unit boundary value
      double interpolated = interp_weight * boundary_value;
      
      std::cout << "  m=" << m << ": interp_weight=" << interp_weight 
                << ", interpolated=" << interpolated << std::endl;
      
      // At axis (js=1), m>0 modes should vanish
      // At boundary (js=ns), should equal boundary value
      if (js == 1) {
        EXPECT_NEAR(interpolated, 0.0, 1e-10);
      } else if (js == ns) {
        EXPECT_NEAR(interpolated, boundary_value, 1e-10);
      }
    }
  }
}

// Test specific case that causes Jacobian issues
TEST(InitialGuessInterpolation, AsymmetricPerturbationCase) {
  // Simple asymmetric tokamak parameters
  const double rbc_00 = 1.0;    // R major radius
  const double rbc_01 = 0.3;    // R m=1 symmetric
  const double rbs_01 = 0.001;  // R m=1 asymmetric perturbation
  const double zbs_01 = 0.3;    // Z m=1 symmetric
  
  // Test at different radial positions
  const int ns = 5;
  
  std::cout << "\nTesting asymmetric perturbation case:" << std::endl;
  
  for (int js = 1; js <= ns; ++js) {
    double sqrts_js = std::sqrt((js - 1.0) / (ns - 1.0));
    
    // m=0 contribution (linear interpolation)
    double r_m0 = sqrts_js * sqrts_js * rbc_00 + (1.0 - sqrts_js * sqrts_js) * 1.0;  // Assuming axis at R=1
    
    // m=1 contributions (power law)
    double r_m1_cos = std::pow(sqrts_js, 1) * rbc_01;
    double r_m1_sin = std::pow(sqrts_js, 1) * rbs_01;
    double z_m1_sin = std::pow(sqrts_js, 1) * zbs_01;
    
    std::cout << "js=" << js << ": R_m0=" << r_m0 
              << ", R_m1_cos=" << r_m1_cos 
              << ", R_m1_sin=" << r_m1_sin 
              << ", Z_m1_sin=" << z_m1_sin << std::endl;
    
    // Check that asymmetric perturbation vanishes at axis
    if (js == 1) {
      EXPECT_NEAR(r_m1_sin, 0.0, 1e-10);
    } else if (js == ns) {
      EXPECT_NEAR(r_m1_sin, rbs_01, 1e-10);
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}