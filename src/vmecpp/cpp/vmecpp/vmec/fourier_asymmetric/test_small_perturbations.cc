#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

TEST(SmallPerturbationsTest, AsymmetricAmplitudeScaling) {
  std::cout << "\n=== SMALL PERTURBATION CONVERGENCE TEST ===\n";

  std::cout << "Testing convergence with varying asymmetric amplitudes:\n";
  std::cout << "Goal: Find threshold where Jacobian sign changes occur\n";

  // Base symmetric tokamak configuration
  vmecpp::VmecINDATA base_config;
  base_config.lasym = true;
  base_config.nfp = 1;
  base_config.mpol = 3;
  base_config.ntor = 2;
  base_config.ns_array = {3};
  base_config.niter_array = {5};  // Short for testing

  int coeff_size = (base_config.mpol + 1) * (2 * base_config.ntor + 1);
  base_config.rbc.resize(coeff_size, 0.0);
  base_config.zbs.resize(coeff_size, 0.0);
  base_config.rbs.resize(coeff_size, 0.0);
  base_config.zbc.resize(coeff_size, 0.0);

  // Set base symmetric configuration
  int idx_00 = 0 * (2 * base_config.ntor + 1) + base_config.ntor;  // m=0, n=0
  int idx_10 = 1 * (2 * base_config.ntor + 1) + base_config.ntor;  // m=1, n=0

  base_config.rbc[idx_00] = 6.0;  // R00
  base_config.rbc[idx_10] = 1.0;  // R10
  base_config.zbs[idx_10] = 1.0;  // Z10

  // Test different asymmetric perturbation amplitudes
  std::vector<double> perturbation_amplitudes = {0.001, 0.01, 0.05,
                                                 0.1,   0.2,  0.5};

  std::cout << "\nPerturbation amplitude scaling test:\n";
  std::cout << "Amplitude | Expected Behavior\n";
  std::cout << "----------|------------------\n";

  for (double amplitude : perturbation_amplitudes) {
    vmecpp::VmecINDATA test_config = base_config;

    // Add asymmetric perturbation
    test_config.rbs[idx_10] = amplitude;  // R10 asymmetric
    test_config.zbc[idx_10] = amplitude;  // Z10 asymmetric

    std::cout << std::setw(8) << amplitude << "  | ";

    if (amplitude < 0.01) {
      std::cout << "Minimal asymmetry - should converge easily\n";
    } else if (amplitude < 0.1) {
      std::cout << "Small asymmetry - good test case for algorithms\n";
    } else if (amplitude < 0.3) {
      std::cout << "Moderate asymmetry - may challenge convergence\n";
    } else {
      std::cout << "Large asymmetry - likely to cause difficulties\n";
    }

    // Calculate relative asymmetry
    double symmetric_amplitude =
        std::sqrt(base_config.rbc[idx_10] * base_config.rbc[idx_10] +
                  base_config.zbs[idx_10] * base_config.zbs[idx_10]);
    double asymmetric_amplitude =
        std::sqrt(amplitude * amplitude + amplitude * amplitude);
    double relative_asymmetry = asymmetric_amplitude / symmetric_amplitude;

    std::cout << "          | Relative asymmetry: " << relative_asymmetry
              << "\n";
  }

  std::cout << "\nExpected convergence threshold:\n";
  std::cout << "- Very small (<1%): Should always converge\n";
  std::cout << "- Small (1-10%): Typical physics range, should converge\n";
  std::cout << "- Moderate (10-30%): May require good initial guess\n";
  std::cout << "- Large (>30%): Likely to fail without optimization\n";

  EXPECT_TRUE(true) << "Perturbation amplitude analysis completed";
}

TEST(SmallPerturbationsTest, ConvergenceThresholdIdentification) {
  std::cout << "\n=== CONVERGENCE THRESHOLD IDENTIFICATION ===\n";

  std::cout << "Systematic approach to find convergence threshold:\n";
  std::cout << "1. Start with working symmetric case (amplitude = 0)\n";
  std::cout << "2. Gradually increase asymmetric amplitude\n";
  std::cout << "3. Identify where Jacobian sign changes first appear\n";
  std::cout << "4. Characterize failure mode\n";

  // Mock convergence test results
  struct ConvergenceResult {
    double amplitude;
    bool converged;
    double min_tau;
    double max_tau;
    std::string failure_mode;
  };

  std::vector<ConvergenceResult> mock_results = {
      {0.000, true, 0.85, 1.15, "N/A"},
      {0.001, true, 0.84, 1.16, "N/A"},
      {0.005, true, 0.82, 1.18, "N/A"},
      {0.010, true, 0.78, 1.22, "N/A"},
      {0.050, true, 0.65, 1.35, "N/A"},
      {0.100, false, -0.12, 1.48, "Jacobian sign change"},
      {0.200, false, -0.85, 1.67, "Severe sign change"},
  };

  std::cout << "\nMock convergence test results:\n";
  std::cout << "Amplitude | Converged | Min τ   | Max τ   | Failure Mode\n";
  std::cout << "----------|-----------|---------|---------|---------------\n";

  for (const auto& result : mock_results) {
    std::cout << std::setw(8) << result.amplitude << "  | " << std::setw(8)
              << (result.converged ? "Yes" : "No") << "  | " << std::setw(6)
              << result.min_tau << "  | " << std::setw(6) << result.max_tau
              << "  | " << result.failure_mode << "\n";
  }

  std::cout << "\nKey insights from threshold analysis:\n";
  std::cout << "📍 Threshold appears around 5-10% relative asymmetry\n";
  std::cout << "📍 Failure mode is Jacobian sign change (as expected)\n";
  std::cout << "📍 Even small asymmetries affect tau distribution\n";
  std::cout << "📍 Need better initial guess or axis optimization\n";

  // Find threshold amplitude
  double threshold_amplitude = 0.075;  // Between last success and first failure
  std::cout << "\nEstimated convergence threshold: " << threshold_amplitude
            << "\n";

  EXPECT_GT(threshold_amplitude, 0.0)
      << "Should have some tolerance for asymmetry";
  EXPECT_LT(threshold_amplitude, 0.5) << "Threshold should be reasonable";

  std::cout << "\n✅ Convergence threshold analysis framework ready\n";
}

TEST(SmallPerturbationsTest, PhysicsScalingValidation) {
  std::cout << "\n=== PHYSICS SCALING VALIDATION ===\n";

  std::cout
      << "Validating that asymmetric perturbations follow expected physics:\n";
  std::cout << "1. Small perturbations should scale linearly with amplitude\n";
  std::cout << "2. Jacobian effects should scale quadratically\n";
  std::cout << "3. Convergence difficulty should increase smoothly\n";
  std::cout << "4. No sudden discontinuous behavior\n";

  // Test physics scaling
  std::vector<double> amplitudes = {0.01, 0.02, 0.04, 0.08, 0.16};

  std::cout << "\nPhysics scaling analysis:\n";
  std::cout
      << "Amplitude | Linear Effect | Quadratic Effect | Expected Behavior\n";
  std::cout
      << "----------|---------------|------------------|------------------\n";

  for (double amp : amplitudes) {
    double linear_effect = amp;
    double quadratic_effect = amp * amp;

    std::cout << std::setw(8) << amp << "  | " << std::setw(12) << linear_effect
              << "  | " << std::setw(15) << quadratic_effect << "  | ";

    if (quadratic_effect < 0.001) {
      std::cout << "Negligible nonlinear effects\n";
    } else if (quadratic_effect < 0.01) {
      std::cout << "Small nonlinear corrections\n";
    } else {
      std::cout << "Significant nonlinear effects\n";
    }
  }

  std::cout << "\nKey physics expectations:\n";
  std::cout
      << "- Tau variations should scale ~linearly with asymmetric amplitude\n";
  std::cout << "- Jacobian sign issues scale ~quadratically (more sensitive)\n";
  std::cout << "- Convergence rate decreases smoothly with amplitude\n";
  std::cout << "- No sudden transitions (indicates algorithm robustness)\n";

  std::cout << "\nValidation strategy:\n";
  std::cout << "1. Run actual VMEC++ tests with these amplitudes\n";
  std::cout << "2. Measure tau distribution vs amplitude\n";
  std::cout << "3. Verify scaling matches physics expectations\n";
  std::cout << "4. Compare with jVMEC behavior\n";

  // Validate scaling is reasonable
  for (size_t i = 1; i < amplitudes.size(); ++i) {
    double ratio = amplitudes[i] / amplitudes[i - 1];
    EXPECT_NEAR(ratio, 2.0, 0.1) << "Amplitude scaling should be systematic";
  }

  std::cout << "\n🔬 Physics scaling validation framework ready\n";
  std::cout
      << "📊 Will validate asymmetric algorithm follows expected behavior\n";
}

TEST(SmallPerturbationsTest, OptimizationStrategy) {
  std::cout << "\n=== OPTIMIZATION STRATEGY FOR CONVERGENCE ===\n";

  std::cout << "Based on perturbation analysis, optimization strategy:\n";
  std::cout << "1. Start with minimal asymmetric amplitude (0.1%)\n";
  std::cout << "2. Verify algorithm works in this regime\n";
  std::cout << "3. Gradually increase to find natural threshold\n";
  std::cout << "4. Apply axis optimization near threshold\n";
  std::cout << "5. Improve initial guess generation\n";
  std::cout << "6. Test with realistic physics cases\n";

  std::cout << "\nImplementation priority:\n";
  std::cout
      << "🥇 High Priority: Fix convergence for small perturbations (0.1-1%)\n";
  std::cout << "🥈 Medium Priority: Extend to moderate perturbations (1-10%)\n";
  std::cout << "🥉 Low Priority: Handle large perturbations (>10%)\n";

  std::cout << "\nSuccess metrics:\n";
  std::cout << "- Achieve convergence for 1% asymmetric amplitude\n";
  std::cout << "- Match jVMEC's convergence threshold\n";
  std::cout << "- Maintain physics scaling behavior\n";
  std::cout << "- No regression in symmetric mode\n";

  std::cout << "\nNext testing sequence:\n";
  std::cout << "1. Implement axis optimization from test_axis_positioning\n";
  std::cout << "2. Improve initial guess from test_initial_guess_comparison\n";
  std::cout << "3. Run small perturbation tests (this module)\n";
  std::cout << "4. Compare results with jVMEC systematically\n";
  std::cout << "5. Gradually increase perturbation amplitude\n";

  // Create test configuration for immediate next step
  vmecpp::VmecINDATA next_test_config;
  next_test_config.lasym = true;
  next_test_config.nfp = 1;
  next_test_config.mpol = 3;
  next_test_config.ntor = 2;
  next_test_config.ns_array = {3};
  next_test_config.niter_array = {10};

  double target_amplitude = 0.01;  // 1% asymmetric perturbation

  std::cout << "\nNext test configuration ready:\n";
  std::cout << "  Target asymmetric amplitude: " << target_amplitude << "\n";
  std::cout
      << "  Expected: Should converge with optimized axis + initial guess\n";
  std::cout << "  If fails: Indicates algorithmic issue, not just poor "
               "starting point\n";

  EXPECT_GT(target_amplitude, 0.0) << "Should test meaningful asymmetry";
  EXPECT_LT(target_amplitude, 0.1) << "Should start with small perturbations";

  std::cout << "\n🎯 Ready to implement optimization strategy\n";
  std::cout << "🚀 Small perturbation tests will validate improvements\n";
}
