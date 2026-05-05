#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

TEST(DebugTransformOutput, AsymmetricTransformMath) {
  std::cout << "\n=== DEBUGGING ASYMMETRIC TRANSFORM OUTPUT ===\n";
  
  // Setup minimal configuration
  Sizes sizes(true, 1, 3, 0, 9, 2);  // lasym=true, nfp=1, mpol=3, ntor=0, ntheta=9, nzeta=2
  
  std::cout << "Configuration:\n";
  std::cout << "  mpol=" << sizes.mpol << ", ntor=" << sizes.ntor << "\n";
  std::cout << "  nThetaReduced=" << sizes.nThetaReduced << " (symmetric [0,pi])\n";
  std::cout << "  nThetaEff=" << sizes.nThetaEff << " (asymmetric [0,2pi])\n";
  std::cout << "  mnmax=" << sizes.mnmax << "\n\n";
  
  // Create coefficient arrays (ns=1 for simplicity)
  const int ns = 1;
  const int coeff_size = sizes.mnmax * ns;
  
  // Symmetric coefficients (baseline)
  std::vector<double> rmncc(coeff_size, 0.0);
  std::vector<double> rmnss(coeff_size, 0.0);
  std::vector<double> zmnsc(coeff_size, 0.0);
  std::vector<double> zmncs(coeff_size, 0.0);
  
  // Asymmetric coefficients (perturbation)
  std::vector<double> rmnsc(coeff_size, 0.0);
  std::vector<double> rmncs(coeff_size, 0.0);
  std::vector<double> zmncc(coeff_size, 0.0);
  std::vector<double> zmnss(coeff_size, 0.0);
  
  // Set up symmetric tokamak: R = R0 + a*cos(theta), Z = a*sin(theta)
  rmncc[0] = 1.0;   // R00 = 1.0 (major radius)
  rmncc[1] = 0.3;   // R10 = 0.3 (ellipticity)
  zmnsc[1] = 0.3;   // Z10 = 0.3 (elongation)
  
  // Add tiny asymmetric perturbation (following jVMEC test case)
  rmnsc[1] = 0.001; // RBS(1,0) = 0.001 (0.1% perturbation)
  
  std::cout << "Fourier Coefficients:\n";
  std::cout << "Symmetric:\n";
  std::cout << "  rmncc[0] (R00) = " << rmncc[0] << "\n";
  std::cout << "  rmncc[1] (R10) = " << rmncc[1] << "\n";
  std::cout << "  zmnsc[1] (Z10) = " << zmnsc[1] << "\n";
  std::cout << "Asymmetric:\n";
  std::cout << "  rmnsc[1] (RBS10) = " << rmnsc[1] << "\n\n";
  
  // Real space arrays
  const int real_size = ns * sizes.nZnT;
  std::vector<double> r_real(real_size, 0.0);
  std::vector<double> z_real(real_size, 0.0);
  std::vector<double> lambda_real(real_size, 0.0);
  std::vector<double> ru_real(real_size, 0.0);
  std::vector<double> zu_real(real_size, 0.0);
  
  // Transform to real space
  FourierToReal3DAsymmFastPoloidal(
      sizes,
      absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs),
      absl::MakeSpan(zmnsc), absl::MakeSpan(zmncs),
      absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real), absl::MakeSpan(lambda_real),
      absl::MakeSpan(ru_real), absl::MakeSpan(zu_real));
  
  std::cout << "Transform Output Analysis:\n";
  std::cout << "Theta range [0, 2pi] with " << sizes.nThetaEff << " points:\n";
  std::cout << std::fixed << std::setprecision(6);
  
  for (int l = 0; l < sizes.nThetaEff; ++l) {
    double theta = 2.0 * M_PI * l / sizes.nThetaEff;
    int idx = l * sizes.nZeta + 0;  // k=0 for ntor=0
    
    if (idx < r_real.size()) {
      std::cout << "  θ=" << std::setw(8) << theta 
                << " R=" << std::setw(10) << r_real[idx]
                << " Z=" << std::setw(10) << z_real[idx] << "\n";
    }
  }
  
  // Check mathematical consistency
  std::cout << "\nMathematical Check:\n";
  std::cout << "Expected symmetric values (without asymmetric perturbation):\n";
  std::cout << "  R(θ=0) = R00 + R10*cos(0) = " << (rmncc[0] + rmncc[1]) << "\n";
  std::cout << "  R(θ=π) = R00 + R10*cos(π) = " << (rmncc[0] - rmncc[1]) << "\n";
  std::cout << "  Z(θ=0) = Z10*sin(0) = " << (zmnsc[1] * sin(0.0)) << "\n";
  std::cout << "  Z(θ=π) = Z10*sin(π) = " << (zmnsc[1] * sin(M_PI)) << "\n";
  
  // Verify asymmetric perturbation
  std::cout << "\nAsymmetric perturbation effect:\n";
  std::cout << "  RBS(1,0) = " << rmnsc[1] << " should add sin(θ) term to R\n";
  std::cout << "  Expected R(θ=π/2) = R00 + RBS10*sin(π/2) = " 
            << (rmncc[0] + rmnsc[1]) << "\n";
  
  // Find θ=π/2 point
  int half_pi_idx = -1;
  for (int l = 0; l < sizes.nThetaEff; ++l) {
    double theta = 2.0 * M_PI * l / sizes.nThetaEff;
    if (std::abs(theta - M_PI/2) < 0.1) {
      half_pi_idx = l * sizes.nZeta + 0;
      break;
    }
  }
  
  if (half_pi_idx >= 0 && half_pi_idx < r_real.size()) {
    std::cout << "  Actual R(θ≈π/2) = " << r_real[half_pi_idx] << "\n";
    double expected = rmncc[0] + rmnsc[1];
    double error = std::abs(r_real[half_pi_idx] - expected);
    std::cout << "  Error = " << error << " (should be small)\n";
    
    if (error > 1e-10) {
      std::cout << "  ❌ ASYMMETRIC TRANSFORM ERROR DETECTED!\n";
    } else {
      std::cout << "  ✓ Asymmetric transform looks correct\n";
    }
  }
}

}  // namespace vmecpp