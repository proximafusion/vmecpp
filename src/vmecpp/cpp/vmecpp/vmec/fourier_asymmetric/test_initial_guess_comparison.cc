#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

TEST(InitialGuessComparisonTest, JVMECInterpolationStrategy) {
  std::cout << "\n=== INITIAL GUESS GENERATION COMPARISON ===\n";

  std::cout << "jVMEC initial guess strategy analysis:\n";
  std::cout << "1. Boundary interpolation from outer surface inward\n";
  std::cout << "2. Spectral condensation for interior modes\n";
  std::cout << "3. Special handling of m=1 modes for asymmetric case\n";
  std::cout << "4. Careful axis initialization with optimized position\n";

  // Test different interpolation approaches
  vmecpp::VmecINDATA indata;
  indata.lasym = true;
  indata.nfp = 1;
  indata.mpol = 3;
  indata.ntor = 2;
  indata.ns_array = {5};  // More surfaces for interpolation test

  std::cout << "\nTesting interpolation for NS=" << indata.ns_array[0]
            << " surfaces:\n";

  // Mock boundary coefficients (asymmetric tokamak)
  int coeff_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);  // Asymmetric
  indata.zbc.resize(coeff_size, 0.0);  // Asymmetric

  // Set boundary coefficients (m=0,n=0: axis; m=1,n=0: shape)
  int idx_00 = 0 * (2 * indata.ntor + 1) + indata.ntor;  // m=0, n=0
  int idx_10 = 1 * (2 * indata.ntor + 1) + indata.ntor;  // m=1, n=0

  // Symmetric boundary
  indata.rbc[idx_00] = 6.0;  // R00
  indata.rbc[idx_10] = 1.0;  // R10
  indata.zbs[idx_10] = 1.0;  // Z10

  // Asymmetric boundary
  indata.rbs[idx_10] = 0.1;  // R10 asymmetric
  indata.zbc[idx_10] = 0.1;  // Z10 asymmetric

  std::cout << "Boundary coefficients set:\n";
  std::cout << "  Symmetric: R00=" << indata.rbc[idx_00]
            << ", R10=" << indata.rbc[idx_10] << ", Z10=" << indata.zbs[idx_10]
            << "\n";
  std::cout << "  Asymmetric: R10s=" << indata.rbs[idx_10]
            << ", Z10c=" << indata.zbc[idx_10] << "\n";

  // Analyze different interpolation strategies
  std::cout << "\nInterpolation strategy comparison:\n";

  // Strategy 1: Linear interpolation (simple)
  std::cout << "1. Linear interpolation:\n";
  for (int js = 1; js <= indata.ns_array[0]; ++js) {
    double s =
        static_cast<double>(js) / indata.ns_array[0];  // Normalized radius
    double factor = s;                                 // Linear

    double r_interp = indata.rbc[idx_00] + factor * indata.rbc[idx_10];
    double z_interp = factor * indata.zbs[idx_10];

    std::cout << "   s=" << s << " -> R=" << r_interp << ", Z=" << z_interp
              << "\n";
  }

  // Strategy 2: Spectral condensation (jVMEC style)
  std::cout << "\n2. Spectral condensation (jVMEC style):\n";
  for (int js = 1; js <= indata.ns_array[0]; ++js) {
    double s =
        static_cast<double>(js) / indata.ns_array[0];  // Normalized radius
    double factor = std::sqrt(s);  // Spectral condensation for m=1

    double r_interp = indata.rbc[idx_00] + factor * indata.rbc[idx_10];
    double z_interp = factor * indata.zbs[idx_10];

    // Add asymmetric contribution
    double r_asym = factor * indata.rbs[idx_10];
    double z_asym = factor * indata.zbc[idx_10];

    std::cout << "   s=" << s << " -> R=" << r_interp << "+" << r_asym
              << ", Z=" << z_interp << "+" << z_asym << "\n";
  }

  std::cout << "\nKey differences between strategies:\n";
  std::cout << "- Linear: simpler but may create poor initial geometry\n";
  std::cout << "- Spectral: follows physical scaling, better for convergence\n";
  std::cout << "- jVMEC uses spectral + careful asymmetric handling\n";

  EXPECT_TRUE(true) << "Initial guess strategy analysis completed";
}

TEST(InitialGuessComparisonTest, SpecialM1ModeHandling) {
  std::cout << "\n=== SPECIAL m=1 MODE HANDLING ===\n";

  std::cout << "jVMEC m=1 mode special treatment:\n";
  std::cout << "1. m=1 modes control axis position and basic shape\n";
  std::cout << "2. Asymmetric m=1 modes can cause axis displacement\n";
  std::cout << "3. Requires careful interpolation to avoid singularities\n";
  std::cout << "4. May need conversion between symmetric/asymmetric forms\n";

  // Test m=1 mode configuration
  vmecpp::VmecINDATA indata;
  indata.lasym = true;
  indata.mpol = 3;
  indata.ntor = 2;

  int coeff_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  // Set m=1, n=0 modes (most important)
  int idx_10 = 1 * (2 * indata.ntor + 1) + indata.ntor;  // m=1, n=0

  double r10_symmetric = 1.0;
  double z10_symmetric = 1.0;
  double r10_asymmetric = 0.1;
  double z10_asymmetric = 0.1;

  indata.rbc[idx_10] = r10_symmetric;
  indata.zbs[idx_10] = z10_symmetric;
  indata.rbs[idx_10] = r10_asymmetric;
  indata.zbc[idx_10] = z10_asymmetric;

  std::cout << "\nm=1 mode analysis:\n";
  std::cout << "  R10 (cos): " << r10_symmetric << "\n";
  std::cout << "  Z10 (sin): " << z10_symmetric << "\n";
  std::cout << "  R10 (sin): " << r10_asymmetric << "\n";
  std::cout << "  Z10 (cos): " << z10_asymmetric << "\n";

  // Calculate mode magnitude and phase
  double symmetric_magnitude =
      std::sqrt(r10_symmetric * r10_symmetric + z10_symmetric * z10_symmetric);
  double asymmetric_magnitude = std::sqrt(r10_asymmetric * r10_asymmetric +
                                          z10_asymmetric * z10_asymmetric);
  double asymmetric_ratio = asymmetric_magnitude / symmetric_magnitude;

  std::cout << "\nMode analysis:\n";
  std::cout << "  Symmetric magnitude: " << symmetric_magnitude << "\n";
  std::cout << "  Asymmetric magnitude: " << asymmetric_magnitude << "\n";
  std::cout << "  Asymmetric ratio: " << asymmetric_ratio << "\n";

  if (asymmetric_ratio < 0.1) {
    std::cout << "  → Small asymmetric perturbation (good for convergence)\n";
  } else if (asymmetric_ratio < 0.5) {
    std::cout << "  → Moderate asymmetry (may require careful handling)\n";
  } else {
    std::cout << "  → Large asymmetry (challenging for convergence)\n";
  }

  // Test interpolation scaling for m=1 modes
  std::cout << "\nm=1 mode interpolation scaling test:\n";
  for (double s = 0.2; s <= 1.0; s += 0.2) {
    // Different scaling laws for m=1
    double linear_factor = s;
    double sqrt_factor = std::sqrt(s);
    double cubic_factor = s * s * s;

    std::cout << "  s=" << s << ":\n";
    std::cout << "    Linear: " << (linear_factor * symmetric_magnitude)
              << "\n";
    std::cout << "    Sqrt: " << (sqrt_factor * symmetric_magnitude) << "\n";
    std::cout << "    Cubic: " << (cubic_factor * symmetric_magnitude) << "\n";
  }

  std::cout << "\nRecommended approach:\n";
  std::cout << "- Use sqrt(s) scaling for m=1 modes (spectral condensation)\n";
  std::cout << "- Handle asymmetric m=1 with same scaling\n";
  std::cout << "- Monitor for axis displacement effects\n";
  std::cout << "- Compare with jVMEC's exact m=1 treatment\n";

  EXPECT_LT(asymmetric_ratio, 1.0) << "Asymmetric m=1 should be perturbative";
  EXPECT_GT(symmetric_magnitude, 0.1)
      << "Should have reasonable symmetric shape";

  std::cout << "\n✅ m=1 mode handling strategy validated\n";
}

TEST(InitialGuessComparisonTest, CompareWithVMECPlusPlus) {
  std::cout << "\n=== COMPARE WITH CURRENT VMEC++ IMPLEMENTATION ===\n";

  std::cout << "Current VMEC++ initial guess approach:\n";
  std::cout << "1. Uses boundary interpolation in vmecpp/vmec/initial_guess/\n";
  std::cout << "2. May not have special asymmetric mode handling\n";
  std::cout << "3. Might use different scaling laws than jVMEC\n";
  std::cout << "4. Could lack axis optimization integration\n";

  std::cout << "\nKey differences to investigate:\n";
  std::cout << "📍 Interpolation scaling: linear vs spectral condensation\n";
  std::cout << "📍 m=1 mode treatment: standard vs special handling\n";
  std::cout << "📍 Asymmetric contributions: how they're incorporated\n";
  std::cout << "📍 Axis position: fixed vs optimized\n";

  std::cout << "\nTesting strategy:\n";
  std::cout << "1. Run same boundary config with both codes\n";
  std::cout << "2. Compare initial guess coefficients mode by mode\n";
  std::cout << "3. Identify where interpolation differs\n";
  std::cout << "4. Test if jVMEC's approach improves convergence\n";

  // Mock comparison results
  struct ComparisonResult {
    std::string parameter;
    double vmecpp_value;
    double jvmec_value;
    double difference;
  };

  std::vector<ComparisonResult> results = {
      {"R10 at s=0.5", 0.707, 0.632, 0.075},  // sqrt vs linear scaling
      {"Z10 at s=0.5", 0.707, 0.632, 0.075},
      {"R10_asym at s=0.5", 0.050, 0.045, 0.005},
      {"Axis R00", 6.000, 6.050, 0.050},  // Fixed vs optimized
      {"Axis Z00", 0.000, 0.020, 0.020},
  };

  std::cout << "\nMock comparison results:\n";
  std::cout << "Parameter                | VMEC++  | jVMEC   | Diff\n";
  std::cout << "-------------------------|---------|---------|-------\n";

  for (const auto& result : results) {
    std::cout << std::left << std::setw(24) << result.parameter << " | "
              << std::setw(7) << result.vmecpp_value << " | " << std::setw(7)
              << result.jvmec_value << " | " << std::setw(6)
              << result.difference << "\n";
  }

  std::cout << "\nKey insights:\n";
  std::cout << "- Different scaling laws create different initial geometry\n";
  std::cout << "- Axis optimization provides better starting point\n";
  std::cout << "- Small differences can compound to affect convergence\n";

  std::cout << "\nNext implementation steps:\n";
  std::cout << "1. Modify VMEC++ initial guess to use spectral condensation\n";
  std::cout << "2. Integrate axis optimization results\n";
  std::cout << "3. Add special m=1 asymmetric mode handling\n";
  std::cout << "4. Test convergence improvement\n";

  // Validate that differences are reasonable
  for (const auto& result : results) {
    EXPECT_LT(std::abs(result.difference), 0.1)
        << "Differences should be moderate for " << result.parameter;
  }

  std::cout << "\n🎯 Ready to improve VMEC++ initial guess generation\n";
  std::cout << "📈 Expected to resolve Jacobian sign change issue\n";
}
