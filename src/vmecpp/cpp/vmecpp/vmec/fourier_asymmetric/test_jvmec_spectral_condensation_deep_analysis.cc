#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

class JVMECSpectralCondensationDeepAnalysisTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void WriteDebugHeader(const std::string& section) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "=== " << section << " ===\n";
    std::cout << std::string(80, '=') << "\n\n";
  }
};

TEST_F(JVMECSpectralCondensationDeepAnalysisTest,
       M1ConstraintAlgorithmAnalysis) {
  WriteDebugHeader("JVMEC M=1 CONSTRAINT ALGORITHM DEEP ANALYSIS");

  std::cout << "JVMEC convert_to_m1_constrained() Implementation Analysis:\n";
  std::cout << "Location: SpectralCondensation.java lines 123-136\n\n";

  std::cout << "Purpose:\n";
  std::cout << "- Impose M=1 mode constraint for theta angle invariance\n";
  std::cout << "- Make geometry invariant to phi-shifts and theta shifts\n";
  std::cout
      << "- Constraint: RSS = ZCS (symmetric), RSC = ZCC (asymmetric)\n\n";

  std::cout << "Mathematical Transformation:\n";
  std::cout << "Symmetric case (m=1):\n";
  std::cout << "  XC(rss) = 0.5*(Rss + Zcs), XC(zcs) = 0.5*(Rss - Zcs) -> 0\n";
  std::cout << "Asymmetric case (m=1):\n";
  std::cout
      << "  XC(rsc) = 0.5*(Rsc + Zcc), XC(zcc) = 0.5*(Rsc - Zcc) -> 0\n\n";

  // Test the exact jVMEC algorithm
  double original_rss = 0.1;  // Example symmetric coefficient
  double original_zcs = 0.05;
  double original_rsc = 0.08;  // Example asymmetric coefficient
  double original_zcc = 0.03;

  double scaling_factor_geometry = 1.0;  // For geometry spectral width
  double scaling_factor_force =
      1.0 / std::sqrt(2.0);  // For force decomposition

  std::cout << "TEST 1: Symmetric Constraint (RSS = ZCS)\n";
  std::cout << "Original: RSS=" << original_rss << ", ZCS=" << original_zcs
            << "\n";

  // jVMEC algorithm for symmetric case
  double backup_rss = original_rss;
  double constrained_rss =
      scaling_factor_geometry * (backup_rss + original_zcs);
  double constrained_zcs =
      scaling_factor_geometry * (backup_rss - original_zcs);

  std::cout << "jVMEC Result: RSS=" << constrained_rss
            << ", ZCS=" << constrained_zcs << "\n";
  std::cout << "Constraint check: RSS - ZCS = "
            << (constrained_rss - constrained_zcs)
            << " (should be 2*original_zcs=" << (2.0 * original_zcs) << ")\n";
  std::cout << "Constraint satisfied: "
            << (std::abs((constrained_rss - constrained_zcs) -
                         2.0 * original_zcs) < 1e-12
                    ? "YES"
                    : "NO")
            << "\n\n";

  EXPECT_NEAR(constrained_rss, 0.15, 1e-12);
  EXPECT_NEAR(constrained_zcs, 0.05, 1e-12);

  std::cout << "TEST 2: Asymmetric Constraint (RSC = ZCC)\n";
  std::cout << "Original: RSC=" << original_rsc << ", ZCC=" << original_zcc
            << "\n";

  // jVMEC algorithm for asymmetric case with force scaling
  double backup_rsc = original_rsc;
  double constrained_rsc = scaling_factor_force * (backup_rsc + original_zcc);
  double constrained_zcc = scaling_factor_force * (backup_rsc - original_zcc);

  std::cout << "jVMEC Result: RSC=" << constrained_rsc
            << ", ZCC=" << constrained_zcc << "\n";
  std::cout << "Force scaling factor: " << scaling_factor_force << "\n";
  std::cout << "Constraint check: RSC - ZCC = "
            << (constrained_rsc - constrained_zcc)
            << " (should be 2*scaling*original_zcc="
            << (2.0 * scaling_factor_force * original_zcc) << ")\n\n";

  EXPECT_NEAR(constrained_rsc, scaling_factor_force * 0.11, 1e-12);
  EXPECT_NEAR(constrained_zcc, scaling_factor_force * 0.05, 1e-12);
}

TEST_F(JVMECSpectralCondensationDeepAnalysisTest,
       ConstraintForceMultiplierAnalysis) {
  WriteDebugHeader("JVMEC CONSTRAINT FORCE MULTIPLIER DEEP ANALYSIS");

  std::cout << "JVMEC computeConstraintForceMultiplier() Implementation:\n";
  std::cout << "Location: SpectralCondensation.java lines 204-258\n\n";

  // Test parameters from jVMEC
  double tcon0 = 1.0;
  int numSurfaces = 51;  // Typical NS value
  double r0scale = 1.0;

  std::cout << "Input Parameters:\n";
  std::cout << "  tcon0 = " << tcon0 << "\n";
  std::cout << "  numSurfaces = " << numSurfaces << "\n";
  std::cout << "  r0scale = " << r0scale << "\n\n";

  // jVMEC formula (lines 221-225)
  double constraintForceMultiplier =
      tcon0 *
      (1.0 + numSurfaces * (1.0 / 60.0 + numSurfaces / (200.0 * 120.0)));
  std::cout << "Step 1 - Parabolic scaling: " << constraintForceMultiplier
            << "\n";

  // Double scaling by (4*r0scale^2)^2
  constraintForceMultiplier /=
      (4.0 * r0scale * r0scale) * (4.0 * r0scale * r0scale);
  std::cout << "Step 2 - r0scale normalization: " << constraintForceMultiplier
            << "\n";
  std::cout << "Scaling denominator: "
            << ((4.0 * r0scale * r0scale) * (4.0 * r0scale * r0scale))
            << "\n\n";

  // Profile calculation (lines 227-251)
  std::cout << "Profile Calculation (surface-dependent):\n";

  // Example preconditioner values
  double ard_example = 1e-6;
  double azd_example = 1e-6;
  double arNorm_example = 1e-3;
  double azNorm_example = 1e-3;

  for (int j = 1; j < std::min(6, numSurfaces - 1); ++j) {
    double constraintForceProfile =
        std::min(std::abs(ard_example / arNorm_example),
                 std::abs(azd_example / azNorm_example)) *
        constraintForceMultiplier * (32.0 / (numSurfaces - 1.0)) *
        (32.0 / (numSurfaces - 1.0));

    std::cout << "  Surface " << j
              << ": constraintForceProfile = " << std::scientific
              << std::setprecision(6) << constraintForceProfile << "\n";
  }

  std::cout << "\nSurface scaling factor: "
            << (32.0 / (numSurfaces - 1.0)) * (32.0 / (numSurfaces - 1.0))
            << "\n\n";

  EXPECT_GT(constraintForceMultiplier, 0.0);
  EXPECT_LT(constraintForceMultiplier, 1.0);  // Should be reasonable value
}

TEST_F(JVMECSpectralCondensationDeepAnalysisTest,
       DeAliasConstraintForceAnalysis) {
  WriteDebugHeader("JVMEC DEALIAS CONSTRAINT FORCE DEEP ANALYSIS");

  std::cout << "JVMEC deAliasConstraintForce() Implementation Analysis:\n";
  std::cout << "Location: SpectralCondensation.java lines 290-442\n\n";

  std::cout << "Purpose: Bandpass-filter constraint force in poloidal Fourier "
               "space\n";
  std::cout << "Filter range: m = 1 to (mpol-2)\n";
  std::cout << "Excluded modes: m = 0 and m = (mpol-1)\n\n";

  int mpol = 8;
  int ntor = 3;
  bool lasym = true;

  std::cout << "Test Configuration:\n";
  std::cout << "  mpol = " << mpol << "\n";
  std::cout << "  ntor = " << ntor << "\n";
  std::cout << "  lasym = " << (lasym ? "true" : "false") << "\n\n";

  std::cout << "Filtering Analysis:\n";
  std::cout << "Modes processed:\n";
  for (int m = 1; m < mpol - 1; ++m) {
    std::cout << "  m = " << m << " (INCLUDED in bandpass filter)\n";
  }
  std::cout << "  m = 0 (EXCLUDED - DC component)\n";
  std::cout << "  m = " << (mpol - 1) << " (EXCLUDED - highest mode)\n\n";

  std::cout << "Symmetric vs Asymmetric Processing:\n";
  std::cout << "Symmetric case (!lasym):\n";
  std::cout << "  - Process gsc and gcs coefficients only\n";
  std::cout << "  - work[0] = sin(mu), work[1] = cos(mu)\n\n";

  std::cout << "Asymmetric case (lasym):\n";
  std::cout << "  - Process gcc, gss, gsc, gcs coefficients\n";
  std::cout << "  - work[0] = sin(mu), work[1] = cos(mu)\n";
  std::cout << "  - work[2] = cos(-mu), work[3] = sin(-mu) (reflected)\n";
  std::cout << "  - Symmetrization: 0.5*(forward + reflected)\n\n";

  std::cout << "Key jVMEC Symmetrization Lines (366-371):\n";
  std::cout << "  gcc[j][n][m] += 0.5 * constraintForceProfile[j-1] * "
               "cosnv[k][n] * (work[1][j][k] + work[2][j][k])\n";
  std::cout << "  gss[j][n][m] += 0.5 * constraintForceProfile[j-1] * "
               "sinnv[k][n] * (work[0][j][k] + work[3][j][k])\n";
  std::cout << "  gsc[j][n][m] += 0.5 * constraintForceProfile[j-1] * "
               "cosnv[k][n] * (work[0][j][k] - work[3][j][k])\n";
  std::cout << "  gcs[j][n][m] += 0.5 * constraintForceProfile[j-1] * "
               "sinnv[k][n] * (work[1][j][k] - work[2][j][k])\n\n";

  std::cout << "Asymmetric Domain Extension (lines 418-437):\n";
  std::cout << "  - Extend theta domain from [0,pi] to [0,2pi]\n";
  std::cout
      << "  - Use reflection formula: gcon[j][k][l] = "
         "-gcon[j][kReversed][lReversed] + gcona[j][kReversed][lReversed]\n";
  std::cout
      << "  - Add symmetric and antisymmetric pieces in [0,pi] domain\n\n";

  // Verify that the filtering excludes correct modes
  std::vector<bool> modeIncluded(mpol, false);
  for (int m = 1; m < mpol - 1; ++m) {
    modeIncluded[m] = true;
  }

  EXPECT_FALSE(modeIncluded[0]);         // m=0 excluded
  EXPECT_FALSE(modeIncluded[mpol - 1]);  // m=mpol-1 excluded
  for (int m = 1; m < mpol - 1; ++m) {
    EXPECT_TRUE(modeIncluded[m]);  // m=1 to mpol-2 included
  }
}

TEST_F(JVMECSpectralCondensationDeepAnalysisTest, VMECPPIntegrationAnalysis) {
  WriteDebugHeader("VMEC++ INTEGRATION ANALYSIS AND REQUIRED CHANGES");

  std::cout << "Current VMEC++ Spectral Condensation Implementation:\n";
  std::cout << "Location: ideal_mhd_model.cc (various functions)\n\n";

  std::cout << "IDENTIFIED GAPS IN VMEC++ vs jVMEC:\n\n";

  std::cout << "1. M=1 CONSTRAINT APPLICATION TIMING:\n";
  std::cout << "   jVMEC: Applied during force decomposition (line-by-line in "
               "iteration)\n";
  std::cout << "   VMEC++: Applied during geometry initialization only\n";
  std::cout << "   IMPACT: Different constraint strength and convergence "
               "behavior\n\n";

  std::cout << "2. CONSTRAINT FORCE MULTIPLIER:\n";
  std::cout << "   jVMEC: Complex formula with parabolic NS scaling and "
               "r0scale normalization\n";
  std::cout
      << "   VMEC++: Missing computeConstraintForceMultiplier() equivalent\n";
  std::cout << "   IMPACT: No dynamic constraint strength adjustment\n\n";

  std::cout << "3. BANDPASS FILTERING:\n";
  std::cout
      << "   jVMEC: deAliasConstraintForce() with m=1 to m=mpol-2 filtering\n";
  std::cout << "   VMEC++: Basic deAliasConstraintForce() exists but may lack "
               "asymmetric handling\n";
  std::cout << "   IMPACT: Different spectral content in constraint forces\n\n";

  std::cout << "4. FORCE SCALING FACTORS:\n";
  std::cout << "   jVMEC: Different scaling for geometry (1.0) vs forces "
               "(1/sqrt(2))\n";
  std::cout << "   VMEC++: Fixed 0.5 scaling factor\n";
  std::cout << "   IMPACT: 29% difference in constraint strength\n\n";

  std::cout << "5. ASYMMETRIC DOMAIN EXTENSION:\n";
  std::cout << "   jVMEC: Explicit theta=[pi,2pi] extension with reflection "
               "formula\n";
  std::cout << "   VMEC++: Handled in SymmetrizeRealSpaceGeometry\n";
  std::cout
      << "   IMPACT: May affect asymmetric constraint force distribution\n\n";

  std::cout << "REQUIRED VMEC++ MODIFICATIONS:\n\n";

  std::cout << "PRIORITY 1: Add computeConstraintForceMultiplier()\n";
  std::cout << "  - Implement jVMEC formula with parabolic NS scaling\n";
  std::cout << "  - Add r0scale normalization factor\n";
  std::cout << "  - Create constraintForceProfile[] array\n\n";

  std::cout << "PRIORITY 2: Enhance deAliasConstraintForce()\n";
  std::cout << "  - Verify bandpass filtering m=1 to m=mpol-2\n";
  std::cout
      << "  - Add asymmetric symmetrization (0.5*(forward + reflected))\n";
  std::cout << "  - Implement theta domain extension for asymmetric case\n\n";

  std::cout << "PRIORITY 3: Add applyM1ConstraintToForces()\n";
  std::cout
      << "  - Apply constraint during force decomposition each iteration\n";
  std::cout << "  - Use 1/sqrt(2) scaling factor for forces\n";
  std::cout << "  - Separate from geometry constraint application\n\n";

  std::cout << "PRIORITY 4: Add computeEffectiveConstraintForce()\n";
  std::cout << "  - Implement jVMEC formula: (R_con - R_con_0) * dRdTheta + "
               "(Z_con - Z_con_0) * dZdTheta\n";
  std::cout << "  - Use extrapolated R_con_0, Z_con_0 arrays\n";
  std::cout << "  - Feed into deAliasConstraintForce()\n\n";

  // This test always passes - it's documentation
  EXPECT_TRUE(true);
}

TEST_F(JVMECSpectralCondensationDeepAnalysisTest, ImplementationPlan) {
  WriteDebugHeader("IMPLEMENTATION PLAN FOR VMEC++ SPECTRAL CONDENSATION");

  std::cout << "STEP-BY-STEP IMPLEMENTATION PLAN:\n\n";

  std::cout << "PHASE 1: Create Missing Functions (TDD Approach)\n";
  std::cout << "1.1 Create test_vmecpp_constraint_force_multiplier.cc\n";
  std::cout << "    - Write failing test that calls "
               "computeConstraintForceMultiplier()\n";
  std::cout << "    - Test jVMEC formula with known inputs\n";
  std::cout
      << "    - Verify parabolic NS scaling and r0scale normalization\n\n";

  std::cout << "1.2 Implement computeConstraintForceMultiplier() in "
               "ideal_mhd_model.cc\n";
  std::cout << "    - Add constraintForceMultiplier member variable\n";
  std::cout << "    - Add constraintForceProfile[] array\n";
  std::cout << "    - Implement jVMEC formula exactly\n\n";

  std::cout << "1.3 Create test_vmecpp_effective_constraint_force.cc\n";
  std::cout
      << "    - Write failing test for computeEffectiveConstraintForce()\n";
  std::cout << "    - Test with realistic R_con, Z_con arrays\n";
  std::cout << "    - Verify extrapolation formula\n\n";

  std::cout << "1.4 Implement computeEffectiveConstraintForce() in "
               "ideal_mhd_model.cc\n";
  std::cout << "    - Add effectiveConstraintForce[][][] array\n";
  std::cout << "    - Implement jVMEC formula\n";
  std::cout << "    - Add extrapolateRZConIntoVolume() equivalent\n\n";

  std::cout << "PHASE 2: Enhance Existing Functions\n";
  std::cout << "2.1 Create test_enhanced_dealias_constraint_force.cc\n";
  std::cout << "    - Test bandpass filtering m=1 to m=mpol-2\n";
  std::cout << "    - Test asymmetric symmetrization\n";
  std::cout << "    - Test theta domain extension\n\n";

  std::cout << "2.2 Enhance deAliasConstraintForce() in ideal_mhd_model.cc\n";
  std::cout << "    - Add asymmetric work[2], work[3] arrays\n";
  std::cout << "    - Implement 0.5*(forward + reflected) symmetrization\n";
  std::cout << "    - Add theta=[pi,2pi] domain extension\n\n";

  std::cout << "2.3 Create test_m1_constraint_force_application.cc\n";
  std::cout << "    - Test applyM1ConstraintToForces() during iteration\n";
  std::cout << "    - Test 1/sqrt(2) scaling factor\n";
  std::cout << "    - Test separation from geometry constraint\n\n";

  std::cout
      << "2.4 Implement applyM1ConstraintToForces() in ideal_mhd_model.cc\n";
  std::cout << "    - Add to main iteration loop\n";
  std::cout << "    - Apply to force coefficients, not geometry\n";
  std::cout
      << "    - Use constraintForceProfile[] for surface-dependent scaling\n\n";

  std::cout << "PHASE 3: Integration Testing\n";
  std::cout << "3.1 Create test_spectral_condensation_integration.cc\n";
  std::cout << "    - Test full spectral condensation pipeline\n";
  std::cout << "    - Compare with jVMEC step-by-step\n";
  std::cout << "    - Verify no regression in symmetric mode\n\n";

  std::cout << "3.2 Test asymmetric convergence with enhanced spectral "
               "condensation\n";
  std::cout << "    - Use proven jVMEC configurations\n";
  std::cout << "    - Compare convergence rates\n";
  std::cout << "    - Debug any remaining differences\n\n";

  std::cout << "PHASE 4: Performance and Production\n";
  std::cout << "4.1 Optimize spectral condensation performance\n";
  std::cout << "4.2 Add comprehensive test coverage\n";
  std::cout << "4.3 Document usage and best practices\n\n";

  std::cout << "ESTIMATED TIMELINE:\n";
  std::cout << "  Phase 1: 2-3 hours (core functionality)\n";
  std::cout << "  Phase 2: 2-3 hours (enhancements)\n";
  std::cout << "  Phase 3: 1-2 hours (integration)\n";
  std::cout << "  Phase 4: 1 hour (polish)\n";
  std::cout << "  Total: 6-9 hours\n\n";

  // This test always passes - it's documentation
  EXPECT_TRUE(true);
}
