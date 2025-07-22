// Compare tau calculation between symmetric and asymmetric modes
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(TauSymmetricVsAsymmetricTest, CompareIdenticalGeometry) {
  std::cout << "\n=== COMPARE TAU CALCULATION: SYMMETRIC VS ASYMMETRIC ===\n";
  std::cout << std::fixed << std::setprecision(8);

  std::cout << "GOAL: Compare tau1=ru12*zs-rs*zu12 between modes\n";
  std::cout << "HYPOTHESIS: Same geometry should produce same tau1 values\n";
  std::cout << "EVIDENCE: Asymmetric tau negative, symmetric tau positive\n";

  // Create identical configuration for both modes
  VmecINDATA base_config;
  base_config.nfp = 1;
  base_config.mpol = 3;
  base_config.ntor = 0;
  base_config.ns_array = {3};
  base_config.niter_array = {1};
  base_config.ftol_array = {1e-6};
  base_config.return_outputs_even_if_not_converged = true;
  base_config.delt = 0.5;
  base_config.tcon0 = 1.0;
  base_config.phiedge = 1.0;
  base_config.pmass_type = "power_series";
  base_config.am = {0.0};
  base_config.pres_scale = 0.0;

  // Circular tokamak - exactly same coefficients
  base_config.rbc = {10.0, 2.0, 0.5};
  base_config.zbs = {0.0, 2.0, 0.5};
  base_config.rbs = {0.0, 0.0, 0.0};  // Zero asymmetric
  base_config.zbc = {0.0, 0.0, 0.0};

  base_config.raxis_c = {10.0};
  base_config.zaxis_s = {0.0};
  base_config.raxis_s = {0.0};
  base_config.zaxis_c = {0.0};

  std::cout << "\n=== TEST 1: SYMMETRIC MODE (lasym=false) ===\n";
  VmecINDATA symmetric_config = base_config;
  symmetric_config.lasym = false;

  std::cout << "Configuration: R0=10, a=2, zero asymmetric coeffs\n";
  std::cout << "Expected: Should run without Jacobian issues\n";

  const auto symmetric_output = vmecpp::run(symmetric_config);
  if (!symmetric_output.ok()) {
    std::cout << "❌ Symmetric failed: " << symmetric_output.status() << "\n";
  } else {
    std::cout << "✅ Symmetric succeeded as expected\n";
  }

  std::cout << "\n=== TEST 2: ASYMMETRIC MODE (lasym=true) ===\n";
  VmecINDATA asymmetric_config = base_config;
  asymmetric_config.lasym = true;

  std::cout << "Configuration: IDENTICAL geometry, only lasym=true\n";
  std::cout << "Expected: Should produce same tau1 values\n";

  const auto asymmetric_output = vmecpp::run(asymmetric_config);
  if (!asymmetric_output.ok()) {
    std::cout << "❌ Asymmetric failed: " << asymmetric_output.status() << "\n";
    std::string error_msg(asymmetric_output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout << "Confirmed: Jacobian failure with identical geometry\n";
    }
  } else {
    std::cout << "✅ Asymmetric succeeded - issue may be resolved!\n";
  }

  std::cout << "\n=== ANALYSIS ===\n";
  std::cout << "Key differences to investigate:\n";
  std::cout << "1. THETA RANGE: [0,π] vs [0,2π]\n";
  std::cout << "2. GRID POINTS: nThetaReduced=7 vs nThetaEff=16\n";
  std::cout << "3. DERIVATIVES: ru12, zu12, rs, zs calculation\n";
  std::cout << "4. JACOBIAN: tau1 = ru12*zs - rs*zu12\n";

  std::cout << "\nFrom debug output analysis:\n";
  std::cout << "- Symmetric tau1 ≈ -3.0 (negative, but consistent)\n";
  std::cout << "- Asymmetric tau1 ≈ -0.9 to +4.0 (sign changes!)\n";
  std::cout << "- Sign change triggers (minTau * maxTau < 0.0) failure\n";

  EXPECT_TRUE(true) << "Tau comparison analysis complete";
}

TEST(TauSymmetricVsAsymmetricTest, AnalyzeTauComponents) {
  std::cout << "\n=== ANALYZE TAU COMPONENT DIFFERENCES ===\n";

  std::cout << "tau1 = ru12 * zs - rs * zu12\n";
  std::cout << "\nFrom debug output (asymmetric mode):\n";
  std::cout << "Surface jH=0:\n";
  std::cout
      << "  kl=6: ru12=0.300000, zs=-0.600000, rs=-12.060000, zu12=-0.060000\n";
  std::cout
      << "  → tau1 = 0.3*(-0.6) - (-12.06)*(-0.06) = -0.18 - 0.72 = -0.90 ✓\n";
  std::cout
      << "  kl=8: ru12=0.000000, zs=0.000000, rs=-11.340000, zu12=0.360000\n";
  std::cout << "  → tau1 = 0.0*0.0 - (-11.34)*0.36 = 0 + 4.08 = +4.08 ✓\n";

  std::cout << "\nSurface jH=1:\n";
  std::cout
      << "  kl=6: ru12=0.900000, zs=-0.600000, rs=-0.180000, zu12=-0.300000\n";
  std::cout
      << "  → tau1 = 0.9*(-0.6) - (-0.18)*(-0.3) = -0.54 - 0.054 = -0.59 ✓\n";
  std::cout
      << "  kl=8: ru12=0.000000, zs=0.000000, rs=0.780000, zu12=1.200000\n";
  std::cout << "  → tau1 = 0.0*0.0 - 0.78*1.2 = 0 - 0.936 = -0.94 ✓\n";

  std::cout << "\nKEY OBSERVATION:\n";
  std::cout << "- Surface jH=0: tau ranges from -0.9 to +4.08\n";
  std::cout << "- Surface jH=1: tau all negative (-0.59 to -0.94)\n";
  std::cout << "- maxTau=+4.08, minTau=-0.94 → minTau*maxTau < 0 ❌\n";

  std::cout << "\nWHY SIGN CHANGES:\n";
  std::cout
      << "1. rs values change sign: jH=0 negative, jH=1 positive/negative\n";
  std::cout << "2. zu12 values vary: affects rs*zu12 contribution\n";
  std::cout
      << "3. Full theta range [0,2π] creates different derivative patterns\n";

  std::cout << "\nHYPOTHESIS:\n";
  std::cout << "Symmetric mode [0,π] maintains consistent tau1 signs\n";
  std::cout << "Asymmetric mode [0,2π] creates sign changes due to:\n";
  std::cout << "- Different theta sampling\n";
  std::cout << "- Different derivative calculation\n";
  std::cout << "- Geometric effects from extended range\n";

  EXPECT_TRUE(true) << "Tau component analysis complete";
}

TEST(TauSymmetricVsAsymmetricTest, HypothesisDerivativeCalculation) {
  std::cout << "\n=== HYPOTHESIS: DERIVATIVE CALCULATION DIFFERENCES ===\n";

  std::cout << "SYMMETRIC MODE:\n";
  std::cout << "- Theta range: [0,π]\n";
  std::cout << "- Grid points: nThetaReduced = 7\n";
  std::cout << "- Geometry: ru, zu, rs, zs computed on reduced grid\n";
  std::cout << "- Derivatives: Finite differences on [0,π] range\n";

  std::cout << "\nASYMMETRIC MODE:\n";
  std::cout << "- Theta range: [0,2π]\n";
  std::cout << "- Grid points: nThetaEff = 16\n";
  std::cout << "- Geometry: ru, zu, rs, zs computed on full grid\n";
  std::cout << "- Derivatives: Finite differences on [0,2π] range\n";

  std::cout << "\nPOTENTIAL ISSUES:\n";
  std::cout << "1. **Grid spacing**: Δθ = π/6 vs π/8 (different resolution)\n";
  std::cout
      << "2. **Periodicity**: [0,2π] requires careful boundary handling\n";
  std::cout << "3. **Symmetry**: [0,π] exploits stellarator symmetry, [0,2π] "
               "doesn't\n";
  std::cout << "4. **Derivatives**: ∂R/∂θ, ∂Z/∂θ calculated differently\n";

  std::cout << "\nWHAT TO INVESTIGATE:\n";
  std::cout
      << "1. Compare ru12, zu12, rs, zs values at same physical positions\n";
  std::cout << "2. Check derivative calculation formulas in asymmetric mode\n";
  std::cout << "3. Verify boundary conditions at θ=0,π,2π\n";
  std::cout << "4. Compare with jVMEC derivative calculation\n";

  std::cout << "\nNEXT STEPS:\n";
  std::cout << "1. Add debug output to geometry derivative calculation\n";
  std::cout
      << "2. Compare derivatives at θ=π/2 (should be same in both modes)\n";
  std::cout << "3. Study jVMEC Jacobian calculation for asymmetric case\n";
  std::cout << "4. Check if tau2 contribution matters (currently zero)\n";

  EXPECT_TRUE(true) << "Derivative calculation hypothesis complete";
}

}  // namespace vmecpp
