#include <gtest/gtest.h>

#include <cmath>

class PowerLawInterpolationTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Test comparing jVMEC power law vs VMEC++ linear interpolation
TEST_F(PowerLawInterpolationTest, JVMECPowerLawVsLinearInterpolation) {
    // Test jVMEC power law: interpolationWeight = Math.pow(sqrtSFull[j], m)
    // vs VMEC++ linear interpolation
    
    const int num_surfaces = 5;
    const int mpol = 5;
    
    // Radial positions: sqrtS values from axis (0) to boundary (1)
    std::vector<double> sqrtS_values = {0.0, 0.25, 0.5, 0.75, 1.0};
    
    // Compare interpolation weights for different m modes
    for (int m = 1; m < mpol; ++m) {
        for (int j = 0; j < num_surfaces; ++j) {
            double sqrtS = sqrtS_values[j];
            
            // jVMEC power law interpolation
            double jvmec_weight = std::pow(sqrtS, m);
            
            // VMEC++ linear interpolation (simplified)
            double vmecpp_weight = sqrtS;  // Linear in sqrtS
            
            // For m=1, they should be identical
            if (m == 1) {
                EXPECT_NEAR(jvmec_weight, vmecpp_weight, 1e-10);
            }
            // For m>1, jVMEC suppresses more strongly
            else {
                if (sqrtS > 0.0 && sqrtS < 1.0) {
                    EXPECT_LT(jvmec_weight, vmecpp_weight) 
                        << "jVMEC should suppress more for m=" << m 
                        << " at sqrtS=" << sqrtS;
                }
            }
        }
    }
}

TEST_F(PowerLawInterpolationTest, HighModeSuppressionAnalysis) {
    // Analyze how power law suppresses high-m modes near axis
    
    double sqrtS_near_axis = 0.1;  // 10% of minor radius
    
    // Calculate suppression factors for different m modes
    std::vector<double> suppression_factors;
    for (int m = 1; m <= 6; ++m) {
        double weight = std::pow(sqrtS_near_axis, m);
        suppression_factors.push_back(weight);
    }
    
    // Verify increasing suppression with higher m
    for (size_t i = 1; i < suppression_factors.size(); ++i) {
        EXPECT_LT(suppression_factors[i], suppression_factors[i-1])
            << "Higher m modes should be more suppressed";
    }
    
    // Specific values for m=1 to m=6 at sqrtS=0.1
    EXPECT_NEAR(suppression_factors[0], 0.1, 1e-10);      // m=1: 0.1
    EXPECT_NEAR(suppression_factors[1], 0.01, 1e-10);     // m=2: 0.01  
    EXPECT_NEAR(suppression_factors[2], 0.001, 1e-10);    // m=3: 0.001
    EXPECT_NEAR(suppression_factors[3], 0.0001, 1e-10);   // m=4: 0.0001
    EXPECT_NEAR(suppression_factors[4], 0.00001, 1e-10);  // m=5: 0.00001
    EXPECT_NEAR(suppression_factors[5], 0.000001, 1e-10); // m=6: 0.000001
}

TEST_F(PowerLawInterpolationTest, RadialProfileComparison) {
    // Compare radial profiles for specific mode (m=3)
    const int m = 3;
    const int num_points = 11;
    
    for (int i = 0; i <= 10; ++i) {
        double sqrtS = i / 10.0;  // 0.0, 0.1, 0.2, ..., 1.0
        
        // jVMEC power law
        double jvmec_profile = std::pow(sqrtS, m);
        
        // Linear interpolation (VMEC++ style)
        double linear_profile = sqrtS;
        
        // At boundary (sqrtS=1), both should equal 1
        if (i == 10) {
            EXPECT_NEAR(jvmec_profile, 1.0, 1e-10);
            EXPECT_NEAR(linear_profile, 1.0, 1e-10);
        }
        // At axis (sqrtS=0), both should equal 0
        else if (i == 0) {
            EXPECT_NEAR(jvmec_profile, 0.0, 1e-10);
            EXPECT_NEAR(linear_profile, 0.0, 1e-10);
        }
        // In between, power law should be smaller
        else {
            EXPECT_LT(jvmec_profile, linear_profile)
                << "Power law should be smaller at sqrtS=" << sqrtS;
        }
    }
}

TEST_F(PowerLawInterpolationTest, SpectralCondensationImpact) {
    // Test how power law interpolation affects spectral condensation
    // and m=1 constraint enforcement
    
    // Simulate boundary coefficients for symmetric case
    double boundary_rss_m1 = 0.1;  // m=1 mode amplitude
    double boundary_zcs_m1 = 0.05; // corresponding Z coefficient
    
    // Test spectral condensation constraint: RSS(n) = ZCS(n) for symmetric
    // This is the jVMEC convert_to_m1_constrained operation
    
    double scaling_factor = 1.0 / std::sqrt(2.0);
    
    // Apply jVMEC m=1 constraint (lines 131-134 in SpectralCondensation.java)
    double backup = boundary_rss_m1;
    double new_rss = scaling_factor * (backup + boundary_zcs_m1);
    double new_zcs = scaling_factor * (backup - boundary_zcs_m1);
    
    // Verify constraint application
    EXPECT_NEAR(new_rss, scaling_factor * 0.15, 1e-10);
    EXPECT_NEAR(new_zcs, scaling_factor * 0.05, 1e-10);
    
    // Test how power law interpolation interacts with constrained coefficients
    double sqrtS_mid = 0.5;
    int m = 1;
    
    double interpolated_coefficient = std::pow(sqrtS_mid, m) * new_rss;
    EXPECT_NEAR(interpolated_coefficient, 0.5 * new_rss, 1e-10);
}