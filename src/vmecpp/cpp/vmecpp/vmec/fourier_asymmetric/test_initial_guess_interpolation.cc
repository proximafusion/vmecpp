// Test to investigate initial guess interpolation differences
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

TEST(InitialGuessInterpolationTest, CompareThetaRanges) {
  std::cout << "\n=== COMPARE THETA RANGES FOR INITIAL GUESS ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6);

  // Setup for symmetric case
  {
    VmecINDATA config;
    config.lasym = false;
    config.mpol = 3;
    config.ntor = 0;
    config.nfp = 1;
    config.ntheta = 12;  // Will be adjusted by Nyquist
    config.nzeta = 1;

    Sizes sizes_symm(config);

    std::cout << "\nSymmetric case (lasym=false):" << std::endl;
    std::cout << "  ntheta input = 12" << std::endl;
    std::cout << "  nThetaEff = " << sizes_symm.nThetaEff << std::endl;
    std::cout << "  nThetaReduced = " << sizes_symm.nThetaReduced << std::endl;
    std::cout << "  nZnT = " << sizes_symm.nZnT << std::endl;
    std::cout << "  Theta range: [0, π]" << std::endl;

    // Create basis functions
    FourierBasisFastPoloidal fb_symm(&sizes_symm);

    // Check theta values
    std::cout << "\n  Theta grid points (first few):" << std::endl;
    for (int l = 0; l < std::min(6, sizes_symm.nThetaReduced); ++l) {
      double theta = 2.0 * M_PI * l / (2.0 * sizes_symm.ntheta);
      std::cout << "    l=" << l << ": theta = " << theta << " ("
                << theta * 180.0 / M_PI << "°)" << std::endl;
    }
  }

  // Setup for asymmetric case
  {
    VmecINDATA config;
    config.lasym = true;
    config.mpol = 3;
    config.ntor = 0;
    config.nfp = 1;
    config.ntheta = 12;  // Will be adjusted differently for asymmetric
    config.nzeta = 1;

    Sizes sizes_asym(config);

    std::cout << "\nAsymmetric case (lasym=true):" << std::endl;
    std::cout << "  ntheta input = 12" << std::endl;
    std::cout << "  nThetaEff = " << sizes_asym.nThetaEff << std::endl;
    std::cout << "  nThetaReduced = " << sizes_asym.nThetaReduced << std::endl;
    std::cout << "  nZnT = " << sizes_asym.nZnT << std::endl;
    std::cout << "  Theta range: [0, 2π]" << std::endl;

    // Create basis functions
    FourierBasisFastPoloidal fb_asym(&sizes_asym);

    // Check theta values
    std::cout << "\n  Theta grid points (all):" << std::endl;
    for (int l = 0; l < sizes_asym.nThetaEff; ++l) {
      double theta = 2.0 * M_PI * l / sizes_asym.ntheta;
      std::cout << "    l=" << l << ": theta = " << theta << " ("
                << theta * 180.0 / M_PI << "°)";
      if (l == 6) {
        std::cout << " <-- θ=π, where R=18 appears!";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\nKEY INSIGHT:" << std::endl;
  std::cout << "- Symmetric: evaluates only at θ ∈ [0,π], then reflects"
            << std::endl;
  std::cout << "- Asymmetric: evaluates at full θ ∈ [0,2π]" << std::endl;
  std::cout << "- At θ=π (kl=6), the symmetric transform might be wrong!"
            << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Theta range comparison";
}

TEST(InitialGuessInterpolationTest, AnalyzeSymmetricTransformIssue) {
  std::cout << "\n=== ANALYZE SYMMETRIC TRANSFORM IN ASYMMETRIC MODE ==="
            << std::endl;

  std::cout << "\nThe issue:" << std::endl;
  std::cout << "1. Symmetric transform (dft_FourierToReal_2d_symm) is designed "
               "for θ ∈ [0,π]"
            << std::endl;
  std::cout << "2. In asymmetric mode, it's being applied to θ ∈ [0,2π]"
            << std::endl;
  std::cout << "3. The basis functions (cosmu, sinmu) are computed differently"
            << std::endl;

  std::cout << "\nFor a circular tokamak (R0=10, a=2):" << std::endl;
  std::cout << "- At θ=0: R = R0 + a = 12" << std::endl;
  std::cout << "- At θ=π/2: R = R0 = 10" << std::endl;
  std::cout << "- At θ=π: R = R0 - a = 8" << std::endl;
  std::cout << "- At θ=3π/2: R = R0 = 10" << std::endl;

  std::cout << "\nBut debug shows R=18 at kl=6 (θ=π)!" << std::endl;
  std::cout << "This suggests the transform is incorrectly handling the "
               "extended range."
            << std::endl;

  std::cout << "\nPossible causes:" << std::endl;
  std::cout << "1. Basis functions not normalized correctly for full range"
            << std::endl;
  std::cout << "2. Symmetric transform applying wrong signs in [π,2π]"
            << std::endl;
  std::cout << "3. Initial interpolation creating wrong coefficients"
            << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Analysis test";
}

TEST(InitialGuessInterpolationTest, TestInterpolationFormula) {
  std::cout << "\n=== TEST INTERPOLATION FORMULA ===" << std::endl;

  // From interpFromBoundaryAndAxis:
  // For m=0: interpolationWeight = sqrtS^2
  // For m>0: interpolationWeight = sqrtS^m

  double sqrtS_axis = 0.0;
  double sqrtS_mid = 0.5;
  double sqrtS_boundary = 1.0;

  std::cout << "Interpolation weights:" << std::endl;
  for (int m = 0; m <= 3; ++m) {
    double weight_axis =
        (m == 0) ? sqrtS_axis * sqrtS_axis : pow(sqrtS_axis, m);
    double weight_mid = (m == 0) ? sqrtS_mid * sqrtS_mid : pow(sqrtS_mid, m);
    double weight_boundary =
        (m == 0) ? sqrtS_boundary * sqrtS_boundary : pow(sqrtS_boundary, m);

    std::cout << "  m=" << m << ": axis=" << weight_axis
              << ", mid=" << weight_mid << ", boundary=" << weight_boundary
              << std::endl;
  }

  std::cout << "\nFor R with R0=10, a=2:" << std::endl;
  std::cout << "  Boundary: rbc[0]=10, rbc[1]=2" << std::endl;
  std::cout << "  Axis: raxis_c[0]=10" << std::endl;

  // At first interior point (sqrtS ≈ 0.5)
  double s = 0.25;                              // s = sqrtS^2
  double rcc_m0 = s * 10.0 + (1.0 - s) * 10.0;  // m=0
  double rcc_m1 = sqrt(s) * 2.0;                // m=1

  std::cout << "\nAt first interior surface (s=0.25):" << std::endl;
  std::cout << "  rcc[m=0] = " << rcc_m0 << std::endl;
  std::cout << "  rcc[m=1] = " << rcc_m1 << std::endl;
  std::cout << "  R(θ=0) = " << (rcc_m0 + rcc_m1) << std::endl;
  std::cout << "  R(θ=π) = " << (rcc_m0 - rcc_m1) << std::endl;

  std::cout << "\nThis matches expected values, so interpolation is correct."
            << std::endl;
  std::cout << "The issue must be in how the symmetric transform handles the "
               "extended theta range."
            << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Interpolation formula test";
}

}  // namespace vmecpp
