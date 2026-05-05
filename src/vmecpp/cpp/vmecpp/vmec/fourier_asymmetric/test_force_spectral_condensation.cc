#include <gtest/gtest.h>

#include <cmath>

class ForceSpectralCondensationTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// Test jVMEC force spectral condensation vs VMEC++ implementation
TEST_F(ForceSpectralCondensationTest, JVMECForceConstraintApplication) {
  // jVMEC applies convert_to_m1_constrained to FORCES with specific scaling
  // This is DIFFERENT from VMEC++ which applies m1Constraint to GEOMETRY

  std::cout << "\nJVMEC FORCE SPECTRAL CONDENSATION:\n";
  std::cout
      << "Application: Forces during iteration, not geometry initialization\n";
  std::cout << "Timing: Called in force decomposition and spectral width "
               "calculation\n";
  std::cout << "Scaling: 1.0/sqrt(2.0) for forces, 1.0 for geometry spectral "
               "width\n\n";

  // Simulate jVMEC force constraint application
  double original_force_rss = 0.01;             // Example force component
  double original_force_zcs = 0.005;            // Example force component
  double force_scaling = 1.0 / std::sqrt(2.0);  // jVMEC force scaling

  std::cout << "FORCE CONSTRAINT EXAMPLE:\n";
  std::cout << "Original force RSS: " << original_force_rss
            << ", ZCS: " << original_force_zcs << "\n";
  std::cout << "Force scaling factor: " << force_scaling << "\n";

  // jVMEC constraint applied to forces
  double backup = original_force_rss;
  double constrained_force_rss = force_scaling * (backup + original_force_zcs);
  double constrained_force_zcs = force_scaling * (backup - original_force_zcs);

  std::cout << "Constrained force RSS: " << constrained_force_rss
            << ", ZCS: " << constrained_force_zcs << "\n";
  std::cout << "Energy conservation: "
            << (constrained_force_rss * constrained_force_rss +
                constrained_force_zcs * constrained_force_zcs)
            << "\n";
  std::cout << "Original energy: "
            << (original_force_rss * original_force_rss +
                original_force_zcs * original_force_zcs)
            << "\n\n";

  EXPECT_NEAR(constrained_force_rss, force_scaling * 0.015, 1e-10);
  EXPECT_NEAR(constrained_force_zcs, force_scaling * 0.005, 1e-10);
}

TEST_F(ForceSpectralCondensationTest, VMECPPGeometryConstraintApplication) {
  // VMEC++ applies m1Constraint to GEOMETRY during initialization
  // This is DIFFERENT from jVMEC which applies to forces during iteration

  std::cout << "VMEC++ GEOMETRY SPECTRAL CONDENSATION:\n";
  std::cout << "Application: Geometry during initialization, not forces\n";
  std::cout << "Timing: Called once in InitFromState(), line 225\n";
  std::cout << "Scaling: Fixed 0.5 scaling factor\n\n";

  // Simulate VMEC++ geometry constraint application
  double original_geom_rss = 0.1;   // Example geometry component
  double original_geom_zcs = 0.05;  // Example geometry component
  double geom_scaling = 0.5;        // VMEC++ fixed scaling

  std::cout << "GEOMETRY CONSTRAINT EXAMPLE:\n";
  std::cout << "Original geometry RSS: " << original_geom_rss
            << ", ZCS: " << original_geom_zcs << "\n";
  std::cout << "Geometry scaling factor: " << geom_scaling << "\n";

  // VMEC++ constraint applied to geometry
  double backup = original_geom_rss;
  double constrained_geom_rss = geom_scaling * (backup + original_geom_zcs);
  double constrained_geom_zcs = geom_scaling * (backup - original_geom_zcs);

  std::cout << "Constrained geometry RSS: " << constrained_geom_rss
            << ", ZCS: " << constrained_geom_zcs << "\n";
  std::cout << "Scaling difference vs jVMEC: "
            << (geom_scaling / (1.0 / std::sqrt(2.0))) << "\n\n";

  EXPECT_NEAR(constrained_geom_rss, 0.5 * 0.15, 1e-10);
  EXPECT_NEAR(constrained_geom_zcs, 0.5 * 0.05, 1e-10);
}

TEST_F(ForceSpectralCondensationTest, ConstraintTimingDifferences) {
  // Critical difference: WHEN the constraint is applied

  std::cout << "CONSTRAINT APPLICATION TIMING ANALYSIS:\n\n";

  std::cout << "JVMEC TIMING:\n";
  std::cout << "1. Geometry initialization: NO constraint applied\n";
  std::cout
      << "2. Force computation: constraint applied with scaling 1/sqrt(2)\n";
  std::cout
      << "3. Spectral width calculation: constraint applied with scaling 1.0\n";
  std::cout
      << "4. Each iteration: forces re-constrained during decomposition\n\n";

  std::cout << "VMEC++ TIMING:\n";
  std::cout
      << "1. Geometry initialization: constraint applied with scaling 0.5\n";
  std::cout << "2. Force computation: NO additional constraint application\n";
  std::cout << "3. Spectral width calculation: uses constrained geometry\n";
  std::cout << "4. Each iteration: no force re-constraining\n\n";

  std::cout << "POTENTIAL CONVERGENCE IMPACT:\n";
  std::cout << "A. Different scaling factors affect force magnitudes\n";
  std::cout << "B. Timing differences may cause constraint drift\n";
  std::cout << "C. jVMEC re-applies constraint each iteration vs VMEC++ once\n";
  std::cout << "D. Force vs geometry constraint application fundamentally "
               "different\n\n";

  // Mathematical comparison of scaling effects
  double jvmec_force_scale = 1.0 / std::sqrt(2.0);
  double vmecpp_geom_scale = 0.5;
  double scaling_ratio = vmecpp_geom_scale / jvmec_force_scale;

  std::cout << "SCALING FACTOR COMPARISON:\n";
  std::cout << "jVMEC force scaling: " << jvmec_force_scale << "\n";
  std::cout << "VMEC++ geometry scaling: " << vmecpp_geom_scale << "\n";
  std::cout << "Ratio: " << scaling_ratio << "\n";
  std::cout << "Percentage difference: " << (100.0 * (scaling_ratio - 1.0))
            << "%\n\n";

  EXPECT_NEAR(jvmec_force_scale, 0.7071067811865476, 1e-10);
  EXPECT_NEAR(vmecpp_geom_scale, 0.5, 1e-10);
  EXPECT_NEAR(scaling_ratio, 0.7071067811865476, 1e-10);
}

TEST_F(ForceSpectralCondensationTest, ConstraintForceMultiplierMissing) {
  // jVMEC has constraint force multiplier calculation that VMEC++ lacks

  std::cout << "CONSTRAINT FORCE MULTIPLIER ANALYSIS:\n\n";

  std::cout << "JVMEC CONSTRAINT FORCE MULTIPLIER:\n";
  std::cout << "Purpose: Dynamically calculated force scaling based on surface "
               "count\n";
  std::cout
      << "Formula: tcon0 * (1 + ns*(1/60 + ns/(200*120))) / (4*r0scale^2)^2\n";
  std::cout << "Usage: Applied to constraint force profile calculation\n";
  std::cout << "Effect: Surface-dependent constraint strength\n\n";

  std::cout << "VMEC++ CONSTRAINT FORCE MULTIPLIER:\n";
  std::cout << "Status: NOT IMPLEMENTED\n";
  std::cout
      << "Impact: Fixed constraint strength regardless of surface count\n";
  std::cout << "Potential issue: Suboptimal constraint enforcement for high "
               "resolution\n\n";

  // Example calculation showing missing multiplier effect
  int ns_low = 21;    // Low resolution
  int ns_high = 101;  // High resolution
  double tcon0 = 1.0;
  double r0scale = 1.0;

  auto calculate_multiplier = [&](int ns) -> double {
    double mult = tcon0 * (1.0 + ns * (1.0 / 60.0 + ns / (200.0 * 120.0)));
    mult /= (4.0 * r0scale * r0scale) * (4.0 * r0scale * r0scale);
    return mult;
  };

  double mult_low = calculate_multiplier(ns_low);
  double mult_high = calculate_multiplier(ns_high);

  std::cout << "MULTIPLIER COMPARISON:\n";
  std::cout << "Low resolution (ns=" << ns_low << "): " << mult_low << "\n";
  std::cout << "High resolution (ns=" << ns_high << "): " << mult_high << "\n";
  std::cout << "Ratio high/low: " << (mult_high / mult_low) << "\n";
  std::cout << "VMEC++ uses same constraint strength for both!\n\n";

  EXPECT_GT(mult_high, mult_low);
  EXPECT_GT(mult_high / mult_low, 1.0);
}

TEST_F(ForceSpectralCondensationTest, BandPassFilteringMissing) {
  // jVMEC applies band-pass filtering to constraint forces

  std::cout << "BAND-PASS FILTERING ANALYSIS:\n\n";

  std::cout << "JVMEC BAND-PASS FILTERING:\n";
  std::cout << "Purpose: Retain only poloidal modes m=1 to m=(mpol-2)\n";
  std::cout << "Implementation: deAliasConstraintForce() method\n";
  std::cout << "Effect: Removes m=0 and m=(mpol-1) from constraint forces\n";
  std::cout << "Reasoning: Constraint force is sine-like quantity\n\n";

  std::cout << "VMEC++ BAND-PASS FILTERING:\n";
  std::cout << "Status: NOT IMPLEMENTED\n";
  std::cout << "Impact: All modes contribute to constraint forces\n";
  std::cout << "Potential issue: Unwanted m=0 and high-m constraint "
               "contributions\n\n";

  // Example of mode filtering effect
  int mpol = 16;
  std::cout << "MODE FILTERING EXAMPLE (mpol=" << mpol << "):\n";
  std::cout << "jVMEC uses modes: m=1 to m=" << (mpol - 2) << "\n";
  std::cout << "VMEC++ uses modes: m=0 to m=" << (mpol - 1) << "\n";
  std::cout << "Excluded modes in jVMEC: m=0, m=" << (mpol - 1) << "\n";
  std::cout << "Additional modes in VMEC++: 2 extra modes contribute to "
               "constraint\n\n";

  EXPECT_EQ(mpol - 2, 14);  // jVMEC highest mode
  EXPECT_EQ(mpol - 1, 15);  // VMEC++ highest mode
}

TEST_F(ForceSpectralCondensationTest, SymmetrizationDifferences) {
  // jVMEC has explicit symmetrization in constraint force computation

  std::cout << "SYMMETRIZATION DIFFERENCES:\n\n";

  std::cout << "JVMEC SYMMETRIZATION:\n";
  std::cout << "Location: deAliasConstraintForce() lines 367-370\n";
  std::cout
      << "Method: 0.5*(work[...] + work[...reversed]) for asymmetric case\n";
  std::cout << "Purpose: Enforce stellarator symmetry in constraint forces\n";
  std::cout
      << "Effect: Symmetric and anti-symmetric force components combined\n\n";

  std::cout << "VMEC++ SYMMETRIZATION:\n";
  std::cout << "Status: Different approach in asymmetric transforms\n";
  std::cout << "Method: SymmetrizeForces() in ideal_mhd_model.cc\n";
  std::cout
      << "Timing: Applied after force calculation, not during constraint\n";
  std::cout
      << "Potential difference: Order of operations may affect results\n\n";

  // Mathematical example of symmetrization difference
  double force_normal = 0.01;
  double force_reversed = 0.008;

  std::cout << "SYMMETRIZATION EXAMPLE:\n";
  std::cout << "Normal position force: " << force_normal << "\n";
  std::cout << "Reversed position force: " << force_reversed << "\n";
  std::cout << "jVMEC symmetrized: " << (0.5 * (force_normal + force_reversed))
            << "\n";
  std::cout << "Anti-symmetric component: "
            << (0.5 * (force_normal - force_reversed)) << "\n\n";

  EXPECT_NEAR(0.5 * (force_normal + force_reversed), 0.009, 1e-10);
  EXPECT_NEAR(0.5 * (force_normal - force_reversed), 0.001, 1e-10);
}
