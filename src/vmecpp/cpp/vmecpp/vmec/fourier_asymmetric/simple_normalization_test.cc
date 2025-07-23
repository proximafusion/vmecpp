// Simple normalization test following strict TDD
#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

TEST(SimpleNormalizationTest, SingleAsymmetricMode) {
  // Minimal test: single asymmetric mode, 2D case
  // Based on failing tests, normalization factor is wrong by factor of √2

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 0;    // 2D case to avoid complexity
  int ntheta = 4;  // Small for easy debugging
  int nzeta = 1;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "Test setup: mnmax=" << sizes.mnmax << ", nZnT=" << sizes.nZnT
            << ", nThetaEff=" << sizes.nThetaEff << std::endl;

  // Create coefficient arrays
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set ONLY asymmetric Z coefficient for m=1, n=0: zmncc = 1.0
  // This should give Z ~ cos(mu) with sqrt(2) normalization
  if (sizes.mnmax >= 2) {
    zmncc[1] = 1.0;  // m=1, n=0 mode
  }

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);
  std::vector<double> ru_real(sizes.nZnT);
  std::vector<double> zu_real(sizes.nZnT);
  std::vector<double> ru_real(real_size);
  std::vector<double> zu_real(real_size);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Check Z values - should be cos(mu) * sqrt(2)
  for (int i = 0; i < sizes.nThetaEff; ++i) {
    double u = 2.0 * M_PI * i / sizes.nThetaEff;
    double expected_z = cos(u) * sqrt(2.0);

    std::cout << "i=" << i << ", u=" << u << ", Z=" << z_real[i]
              << ", expected=" << expected_z
              << ", diff=" << (z_real[i] - expected_z) << std::endl;

    // This will FAIL until normalization is fixed
    EXPECT_NEAR(z_real[i], expected_z, 1e-10);
  }
}

}  // namespace vmecpp
