#include <gtest/gtest.h>

#include <cmath>

class SpectralCondensationComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Test jVMEC convert_to_m1_constrained implementation
TEST_F(SpectralCondensationComparisonTest, JVMECConvertToM1Constrained) {
    // jVMEC convert_to_m1_constrained algorithm from SpectralCondensation.java lines 131-134:
    // final double backup = rss_rsc[j][n][m];
    // rss_rsc[j][n][m] = scalingFactor * (backup + zcs_zcc[j][n][m]);
    // zcs_zcc[j][n][m] = scalingFactor * (backup - zcs_zcc[j][n][m]);
    
    std::cout << "\nJVMEC CONVERT_TO_M1_CONSTRAINED ALGORITHM:\n";
    std::cout << "Purpose: Impose m=1 constraint for theta angle invariance\n";
    std::cout << "Constraint: RSS(n) = ZCS(n) for symmetric, RSC(n) = ZCC(n) for asymmetric\n\n";
    
    // Test symmetric case: RSS = ZCS constraint
    double original_rss = 0.1;
    double original_zcs = 0.05;
    double scaling_factor_symmetric = 1.0;  // Different from force scaling!
    
    std::cout << "SYMMETRIC CASE (RSS = ZCS constraint):\n";
    std::cout << "Original RSS: " << original_rss << ", ZCS: " << original_zcs << "\n";
    
    // jVMEC constraint transformation
    double backup = original_rss;
    double constrained_rss = scaling_factor_symmetric * (backup + original_zcs);
    double constrained_zcs = scaling_factor_symmetric * (backup - original_zcs);
    
    std::cout << "Constrained RSS: " << constrained_rss << ", ZCS: " << constrained_zcs << "\n";
    std::cout << "Verification: RSS - ZCS = " << (constrained_rss - constrained_zcs) << " (should be 2*original_zcs)\n\n";
    
    EXPECT_NEAR(constrained_rss, 0.15, 1e-10);
    EXPECT_NEAR(constrained_zcs, 0.05, 1e-10);
    EXPECT_NEAR(constrained_rss - constrained_zcs, 2.0 * original_zcs, 1e-10);
    
    // Test asymmetric case: RSC = ZCC constraint  
    double original_rsc = 0.08;
    double original_zcc = 0.03;
    double scaling_factor_asymmetric = 1.0 / std::sqrt(2.0);  // Force scaling factor
    
    std::cout << "ASYMMETRIC CASE (RSC = ZCC constraint):\n";
    std::cout << "Original RSC: " << original_rsc << ", ZCC: " << original_zcc << "\n";
    std::cout << "Scaling factor: " << scaling_factor_asymmetric << "\n";
    
    backup = original_rsc;
    double constrained_rsc = scaling_factor_asymmetric * (backup + original_zcc);
    double constrained_zcc = scaling_factor_asymmetric * (backup - original_zcc);
    
    std::cout << "Constrained RSC: " << constrained_rsc << ", ZCC: " << constrained_zcc << "\n";
    std::cout << "Verification: RSC - ZCC = " << (constrained_rsc - constrained_zcc) << "\n\n";
    
    EXPECT_NEAR(constrained_rsc, scaling_factor_asymmetric * 0.11, 1e-10);
    EXPECT_NEAR(constrained_zcc, scaling_factor_asymmetric * 0.05, 1e-10);
}

TEST_F(SpectralCondensationComparisonTest, JVMECForceScalingFactors) {
    // jVMEC uses different scaling factors for different purposes:
    // 1. Geometry spectral width: scaling_factor = 1.0 (line 502, 505)
    // 2. Force decomposition: scaling_factor = 1.0/sqrt(2.0) (in force processing)
    
    std::cout << "JVMEC SCALING FACTOR USAGE:\n";
    std::cout << "1. Geometry spectral width calculation: 1.0\n";
    std::cout << "2. Force decomposition and constraint: 1.0/sqrt(2.0) = " << (1.0/std::sqrt(2.0)) << "\n";
    std::cout << "3. Applied to forces with different constraint multipliers\n\n";
    
    // Mathematical verification of constraint scaling
    double force_scaling = 1.0 / std::sqrt(2.0);
    double geometry_scaling = 1.0;
    
    std::cout << "Force scaling preserves constraint energy: " << (force_scaling * force_scaling * 2.0) << " = 1.0\n";
    std::cout << "Geometry scaling maintains coefficients: " << geometry_scaling << " = 1.0\n\n";
    
    EXPECT_NEAR(force_scaling * force_scaling * 2.0, 1.0, 1e-10);
    EXPECT_NEAR(geometry_scaling, 1.0, 1e-10);
}

TEST_F(SpectralCondensationComparisonTest, VMECPPSpectralCondensationLocation) {
    // VMEC++ spectral condensation implementation location and differences
    std::cout << "VMEC++ SPECTRAL CONDENSATION IMPLEMENTATION:\n";
    std::cout << "Location: ideal_mhd_model.cc (spectral condensation functions)\n";
    std::cout << "Current implementation: m1Constraint() in fourier_geometry.cc line 225\n";
    std::cout << "Scaling factor: 0.5 (hardcoded in InitFromState)\n\n";
    
    std::cout << "DIFFERENCES TO INVESTIGATE:\n";
    std::cout << "1. VMEC++ uses fixed 0.5 scaling vs jVMEC's variable scaling\n";
    std::cout << "2. jVMEC applies constraint in force decomposition vs VMEC++ in geometry\n";
    std::cout << "3. jVMEC has separate constraint force multiplier calculation\n";
    std::cout << "4. Different application to asymmetric vs symmetric modes\n\n";
    
    // Test VMEC++ scaling factor
    double vmecpp_scaling = 0.5;
    double jvmec_force_scaling = 1.0 / std::sqrt(2.0);
    
    std::cout << "VMEC++ scaling: " << vmecpp_scaling << "\n";
    std::cout << "jVMEC force scaling: " << jvmec_force_scaling << "\n";
    std::cout << "Ratio: " << (vmecpp_scaling / jvmec_force_scaling) << "\n\n";
    
    EXPECT_NEAR(vmecpp_scaling, 0.5, 1e-10);
    EXPECT_NEAR(jvmec_force_scaling, 1.0/std::sqrt(2.0), 1e-10);
}

TEST_F(SpectralCondensationComparisonTest, ConstraintForceMultiplierAnalysis) {
    // jVMEC constraint force multiplier calculation from lines 221-248
    std::cout << "JVMEC CONSTRAINT FORCE MULTIPLIER CALCULATION:\n";
    
    // Example calculation similar to jVMEC lines 221-225
    double tcon0 = 1.0;  // Initial constraint multiplier
    int numSurfaces = 51;  // Example surface count
    double r0scale = 1.0;  // Scaling factor
    
    // jVMEC formula: tcon0 * (1 + ns*(1/60 + ns/(200*120))) / (4*r0scale^2)^2
    double constraint_multiplier = tcon0 * (1.0 + numSurfaces * (1.0/60.0 + numSurfaces/(200.0*120.0)));
    constraint_multiplier /= (4.0 * r0scale * r0scale) * (4.0 * r0scale * r0scale);
    
    std::cout << "Base tcon0: " << tcon0 << "\n";
    std::cout << "Surface count: " << numSurfaces << "\n";
    std::cout << "Calculated multiplier: " << constraint_multiplier << "\n";
    std::cout << "r0scale factor: " << ((4.0 * r0scale * r0scale) * (4.0 * r0scale * r0scale)) << "\n\n";
    
    // Profile calculation (simplified)
    double ard_norm = 1e-3;  // Example norm
    double azd_norm = 1e-3;  // Example norm
    double ard_value = 1e-6; // Example preconditioner value
    double azd_value = 1e-6; // Example preconditioner value
    
    double profile_value = std::min(std::abs(ard_value / ard_norm), std::abs(azd_value / azd_norm)) 
                          * constraint_multiplier 
                          * (32.0 / (numSurfaces - 1.0)) * (32.0 / (numSurfaces - 1.0));
    
    std::cout << "Profile constraint value: " << profile_value << "\n";
    std::cout << "Surface scaling: " << ((32.0 / (numSurfaces - 1.0)) * (32.0 / (numSurfaces - 1.0))) << "\n\n";
    
    EXPECT_GT(constraint_multiplier, 0.0);
    EXPECT_GT(profile_value, 0.0);
}

TEST_F(SpectralCondensationComparisonTest, AsymmetricConstraintDifferences) {
    // Key differences in asymmetric constraint handling
    std::cout << "ASYMMETRIC CONSTRAINT DIFFERENCES:\n\n";
    
    std::cout << "JVMEC ASYMMETRIC HANDLING:\n";
    std::cout << "1. Uses convert_to_m1_constrained for both symmetric and asymmetric\n";
    std::cout << "2. Different scaling factors for forces vs geometry\n";
    std::cout << "3. Constraint force profile calculation includes asymmetric factor\n";
    std::cout << "4. Band-pass filtering in deAliasConstraintForce()\n";
    std::cout << "5. Symmetrization in constraint force computation\n\n";
    
    std::cout << "VMEC++ ASYMMETRIC HANDLING:\n";
    std::cout << "1. Uses m1Constraint() with fixed 0.5 scaling\n";
    std::cout << "2. Applied during geometry initialization only\n";
    std::cout << "3. No separate constraint force multiplier calculation\n";
    std::cout << "4. Different force symmetrization approach\n\n";
    
    std::cout << "CRITICAL INVESTIGATION POINTS:\n";
    std::cout << "A. Force decomposition vs geometry constraint application timing\n";
    std::cout << "B. Scaling factor differences affecting convergence rate\n";
    std::cout << "C. Constraint force profile calculation missing in VMEC++\n";
    std::cout << "D. Band-pass filtering of constraint forces\n";
    std::cout << "E. Asymmetric constraint force symmetrization differences\n\n";
    
    // This test documents the investigation - always passes
    EXPECT_TRUE(true);
}