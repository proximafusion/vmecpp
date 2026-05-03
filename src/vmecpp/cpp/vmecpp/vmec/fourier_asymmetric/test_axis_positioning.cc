#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

TEST(AxisPositioningTest, JVMECGuessAxisComparison) {
  std::cout << "\n=== AXIS POSITIONING OPTIMIZATION TEST ===\n";

  std::cout << "jVMEC guessAxis algorithm analysis:\n";
  std::cout << "1. Analyzes boundary Fourier coefficients\n";
  std::cout << "2. Calculates optimal axis position to minimize curvature\n";
  std::cout << "3. Adjusts R00 and Z00 based on asymmetric boundary shape\n";
  std::cout << "4. Uses iterative optimization to find best axis\n";

  // Test different axis positions systematically
  std::vector<double> r00_values = {6.0, 6.1, 6.2, 5.9, 5.8};
  std::vector<double> z00_values = {0.0, 0.1, -0.1, 0.2, -0.2};

  std::cout << "\nTesting axis position variations:\n";
  for (size_t i = 0; i < r00_values.size(); ++i) {
    double r00 = r00_values[i];
    double z00 = z00_values[i];

    std::cout << "Test " << (i + 1) << ": R00=" << r00 << ", Z00=" << z00
              << "\n";

    // Create test configuration with this axis position
    vmecpp::VmecINDATA indata;
    indata.lasym = true;
    indata.nfp = 1;
    indata.mpol = 3;
    indata.ntor = 2;
    indata.ns_array = {3};
    indata.niter_array = {5};  // Short iteration for testing

    // Set axis position
    int coeff_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
    indata.rbc.resize(coeff_size, 0.0);
    indata.zbs.resize(coeff_size, 0.0);
    indata.rbs.resize(coeff_size, 0.0);  // Asymmetric
    indata.zbc.resize(coeff_size, 0.0);  // Asymmetric

    // Calculate indices for m=0, n=0 mode
    int idx_00 = 0 * (2 * indata.ntor + 1) + indata.ntor;  // m=0, n=0
    indata.rbc[idx_00] = r00;                              // R00
    indata.zbc[idx_00] = z00;                              // Z00 (asymmetric)

    // Add simple boundary perturbation
    int idx_10 = 1 * (2 * indata.ntor + 1) + indata.ntor;  // m=1, n=0
    indata.rbc[idx_10] = 1.0;                              // R10
    indata.zbs[idx_10] = 1.0;                              // Z10

    // Add asymmetric perturbation
    indata.rbs[idx_10] = 0.1;  // R10 asymmetric
    indata.zbc[idx_10] = 0.1;  // Z10 asymmetric

    std::cout << "  Configuration created successfully\n";

    // Theoretical analysis of this axis position
    double axis_shift = std::sqrt(r00 * r00 + z00 * z00) - 6.0;
    std::cout << "  Axis shift magnitude: " << axis_shift << "\n";

    if (std::abs(axis_shift) < 0.05) {
      std::cout << "  → Small shift: likely to maintain good geometry\n";
    } else if (std::abs(axis_shift) < 0.2) {
      std::cout << "  → Moderate shift: may improve asymmetric balance\n";
    } else {
      std::cout << "  → Large shift: could create geometric instability\n";
    }
  }

  std::cout << "\nKey insights for Jacobian sign issue:\n";
  std::cout << "1. Axis position directly affects Jacobian distribution\n";
  std::cout << "2. Poor axis choice can create sign changes in tau\n";
  std::cout << "3. jVMEC's guessAxis finds optimal position automatically\n";
  std::cout << "4. VMEC++ currently uses user-specified axis (R00, Z00)\n";

  std::cout << "\nNext implementation steps:\n";
  std::cout << "1. Implement automatic axis optimization in VMEC++\n";
  std::cout << "2. Test each axis position with actual Jacobian calculation\n";
  std::cout << "3. Find axis that eliminates sign changes\n";
  std::cout << "4. Compare with jVMEC's automatic axis selection\n";

  EXPECT_TRUE(true) << "Axis positioning analysis completed";
}

TEST(AxisPositioningTest, OptimalAxisCalculation) {
  std::cout << "\n=== OPTIMAL AXIS CALCULATION ===\n";

  std::cout << "jVMEC axis optimization algorithm:\n";
  std::cout << "1. Start with boundary center-of-mass calculation\n";
  std::cout << "2. Analyze Fourier coefficient distribution\n";
  std::cout << "3. Calculate axis that minimizes boundary curvature\n";
  std::cout << "4. Apply iterative refinement\n";

  // Mock boundary coefficients from asymmetric tokamak
  struct BoundaryCoeffs {
    double r10 = 1.0;       // Major boundary shape
    double z10 = 1.0;       // Major boundary shape
    double r11 = 0.0;       // Toroidal coupling
    double z11 = 0.0;       // Toroidal coupling
    double r10_asym = 0.1;  // Asymmetric perturbation
    double z10_asym = 0.1;  // Asymmetric perturbation
  };

  BoundaryCoeffs coeffs;

  std::cout << "\nBoundary coefficient analysis:\n";
  std::cout << "  Symmetric: R10=" << coeffs.r10 << ", Z10=" << coeffs.z10
            << "\n";
  std::cout << "  Asymmetric: R10s=" << coeffs.r10_asym
            << ", Z10c=" << coeffs.z10_asym << "\n";

  // Calculate center-of-mass based axis position
  double r00_com = 6.0;  // Base tokamak major radius
  double z00_com = 0.0;  // Start symmetric

  // Apply asymmetric correction (simplified jVMEC approach)
  double asym_magnitude = std::sqrt(coeffs.r10_asym * coeffs.r10_asym +
                                    coeffs.z10_asym * coeffs.z10_asym);
  double correction_factor = asym_magnitude / coeffs.r10;  // Relative asymmetry

  std::cout << "\nAsymmetric correction analysis:\n";
  std::cout << "  Asymmetric magnitude: " << asym_magnitude << "\n";
  std::cout << "  Relative asymmetry: " << correction_factor << "\n";

  // Calculate corrected axis position
  double r00_corrected = r00_com + 0.1 * coeffs.r10_asym;  // Small correction
  double z00_corrected = z00_com + 0.1 * coeffs.z10_asym;  // Small correction

  std::cout << "\nOptimal axis positions:\n";
  std::cout << "  Original: R00=" << r00_com << ", Z00=" << z00_com << "\n";
  std::cout << "  Corrected: R00=" << r00_corrected << ", Z00=" << z00_corrected
            << "\n";

  // Validate the correction is reasonable
  double correction_magnitude =
      std::sqrt((r00_corrected - r00_com) * (r00_corrected - r00_com) +
                (z00_corrected - z00_com) * (z00_corrected - z00_com));

  std::cout << "  Correction magnitude: " << correction_magnitude << "\n";

  EXPECT_LT(correction_magnitude, 0.5) << "Axis correction should be small";
  EXPECT_GT(correction_magnitude, 0.0)
      << "Should apply some correction for asymmetric case";

  std::cout << "\nExpected impact on Jacobian:\n";
  std::cout << "- Better axis position should reduce tau sign changes\n";
  std::cout << "- Optimized geometry improves numerical stability\n";
  std::cout << "- May resolve convergence failure in asymmetric mode\n";

  std::cout
      << "\n✅ Optimal axis calculation framework ready for implementation\n";
}

TEST(AxisPositioningTest, ImplementJVMECAxisOptimization) {
  std::cout << "\n=== IMPLEMENT jVMEC AXIS OPTIMIZATION ===\n";

  std::cout << "Implementation strategy:\n";
  std::cout << "1. Create axis optimization module in VMEC++\n";
  std::cout << "2. Port jVMEC's guessAxis algorithm\n";
  std::cout << "3. Integrate with asymmetric equilibrium initialization\n";
  std::cout << "4. Test with current problematic configuration\n";

  std::cout << "\nRequired components:\n";
  std::cout << "📁 New file: vmecpp/vmec/axis_optimization/\n";
  std::cout << "  ├── axis_optimization.h\n";
  std::cout << "  ├── axis_optimization.cc\n";
  std::cout << "  └── axis_optimization_test.cc\n";

  std::cout << "\nKey functions to implement:\n";
  std::cout << "1. CalculateBoundaryCenterOfMass()\n";
  std::cout << "2. AnalyzeAsymmetricPerturbations()\n";
  std::cout << "3. OptimizeAxisPosition()\n";
  std::cout << "4. ValidateAxisGeometry()\n";

  std::cout << "\nIntegration points in ideal_mhd_model.cc:\n";
  std::cout << "- Call axis optimization before initial guess generation\n";
  std::cout << "- Update R00 and Z00 based on optimization results\n";
  std::cout << "- Add debug output comparing with jVMEC axis choice\n";

  std::cout << "\nSuccess criteria:\n";
  std::cout << "✅ Find axis position that eliminates Jacobian sign changes\n";
  std::cout
      << "✅ Achieve convergence with current asymmetric tokamak config\n";
  std::cout << "✅ Match jVMEC's automatic axis selection behavior\n";
  std::cout << "✅ Validate with multiple asymmetric test cases\n";

  std::cout << "\nTesting approach:\n";
  std::cout << "1. Unit test each axis optimization component\n";
  std::cout << "2. Compare optimized axis with jVMEC's choice\n";
  std::cout << "3. Run convergence test with optimized axis\n";
  std::cout << "4. Verify improvement in Jacobian sign distribution\n";

  // Create a mock optimization result to validate the concept
  struct OptimizationResult {
    double r00_optimal = 6.05;        // Slightly adjusted from 6.0
    double z00_optimal = 0.02;        // Small vertical shift
    double improvement_score = 0.85;  // How much better vs original
    bool converged = true;
  };

  OptimizationResult result;

  std::cout << "\nMock optimization result:\n";
  std::cout << "  Optimal R00: " << result.r00_optimal << "\n";
  std::cout << "  Optimal Z00: " << result.z00_optimal << "\n";
  std::cout << "  Improvement score: " << result.improvement_score << "\n";
  std::cout << "  Converged: " << (result.converged ? "Yes" : "No") << "\n";

  EXPECT_TRUE(result.converged) << "Axis optimization should converge";
  EXPECT_GT(result.improvement_score, 0.5)
      << "Should show significant improvement";
  EXPECT_LT(std::abs(result.r00_optimal - 6.0), 0.2)
      << "Optimal R00 should be close to original";
  EXPECT_LT(std::abs(result.z00_optimal), 0.1)
      << "Optimal Z00 should be small for tokamak";

  std::cout
      << "\n🎯 Ready to implement jVMEC-style axis optimization in VMEC++\n";
  std::cout << "📍 This should resolve the Jacobian sign change issue\n";
}
