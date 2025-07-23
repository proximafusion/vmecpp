// Debug normalization issue - factor of √2 error
#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

TEST(DebugNormalizationTest, CompareWithBasisFunctions) {
  // Debug the exact normalization by comparing with FourierBasisFastPoloidal
  // The issue: Expected 1.41421 (√2), actual 2.0 (double √2 application)

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 1;
  int ntheta = 4;  // Small for easy debugging
  int nzeta = 4;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "Debug normalization with mnmax=" << sizes.mnmax
            << ", nThetaEff=" << sizes.nThetaEff << std::endl;

  // Create coefficient arrays - set only m=1, n=1 mode
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Find m=1, n=1 mode
  FourierBasisFastPoloidal fourier_basis(&sizes);
  int mn_target = -1;
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;
    std::cout << "Mode mn=" << mn << ": m=" << m << ", n=" << n << std::endl;
    if (m == 1 && n == 1) {
      mn_target = mn;
    }
  }

  ASSERT_NE(mn_target, -1) << "Could not find m=1, n=1 mode";

  // Set coefficient to 1.0
  rmncc[mn_target] = 1.0;
  std::cout << "Set rmncc[" << mn_target << "] = 1.0 for m=1, n=1" << std::endl;

  // Check what FourierBasisFastPoloidal gives us directly
  std::cout << "\nDirect FourierBasisFastPoloidal values:" << std::endl;
  for (int i = 0; i < sizes.nThetaEff; ++i) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      double u = 2.0 * M_PI * i / sizes.nThetaEff;
      double v = 2.0 * M_PI * k / sizes.nZeta;

      // Check basis function values
      int idx_basis_m1 = 1 * sizes.nThetaReduced + (i % sizes.nThetaReduced);
      if (idx_basis_m1 < fourier_basis.cosmu.size()) {
        double cos_mu = fourier_basis.cosmu[idx_basis_m1];

        int idx_nv = k * (sizes.nnyq2 + 1) + 1;  // n=1
        double cos_nv = fourier_basis.cosnv[idx_nv];

        double basis_product = cos_mu * cos_nv;
        double expected_raw = cos(u - v);  // Raw cos(mu - nv)
        double expected_normalized = cos(u - v) * sqrt(2.0);  // With √2

        std::cout << "  i=" << i << ", k=" << k << ", u=" << u << ", v=" << v
                  << std::endl;
        std::cout << "    cos_mu=" << cos_mu << ", cos_nv=" << cos_nv
                  << ", product=" << basis_product << std::endl;
        std::cout << "    expected_raw=" << expected_raw
                  << ", expected_norm=" << expected_normalized << std::endl;
        std::cout << "    basis/raw ratio=" << (basis_product / expected_raw)
                  << std::endl;
      }
    }
  }

  // Now test our transform
  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);
  std::vector<double> ru_real(sizes.nZnT);
  std::vector<double> zu_real(sizes.nZnT);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real), absl::MakeSpan(ru_real),
      absl::MakeSpan(zu_real));

  std::cout << "\nTransform output vs expected:" << std::endl;
  for (int i = 0; i < sizes.nThetaEff; ++i) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      double u = 2.0 * M_PI * i / sizes.nThetaEff;
      double v = 2.0 * M_PI * k / sizes.nZeta;

      double expected_raw = cos(u - v);
      double expected_sqrt2 = cos(u - v) * sqrt(2.0);

      std::cout << "  idx=" << idx << ", u=" << u << ", v=" << v << std::endl;
      std::cout << "    R_actual=" << r_real[idx] << std::endl;
      std::cout << "    expected_raw=" << expected_raw << std::endl;
      std::cout << "    expected_sqrt2=" << expected_sqrt2 << std::endl;
      std::cout << "    actual/raw=" << (r_real[idx] / expected_raw)
                << std::endl;
      std::cout << "    actual/sqrt2=" << (r_real[idx] / expected_sqrt2)
                << std::endl;
    }
  }
}

}  // namespace vmecpp
