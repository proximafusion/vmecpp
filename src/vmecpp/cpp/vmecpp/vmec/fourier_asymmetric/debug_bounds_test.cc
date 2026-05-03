// Minimal test to debug vector bounds assertion
#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

TEST(DebugBoundsTest, MinimalFailingCase) {
  // Reproduce exact failing case from RoundTripTransform
  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 1;  // 3D case that fails
  int ntheta = 8;
  int nzeta = 8;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "Debug: mnmax=" << sizes.mnmax << ", nZnT=" << sizes.nZnT
            << ", nThetaEff=" << sizes.nThetaEff << ", nZeta=" << sizes.nZeta
            << std::endl;

  // Create arrays with correct size
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set only ONE coefficient to debug safely
  if (sizes.mnmax >= 1) {
    rmncc[0] = 0.01;  // Safe: m=0, n=0 mode
  }

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);
  std::vector<double> ru_real(sizes.nZnT);
  std::vector<double> zu_real(sizes.nZnT);

  std::cout << "About to call FourierToReal3DAsymmFastPoloidal..." << std::endl;

  try {
    FourierToReal3DAsymmFastPoloidal(
        sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
        absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
        absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
        absl::MakeSpan(r_real), absl::MakeSpan(z_real),
        absl::MakeSpan(lambda_real));

    std::cout << "Forward transform completed successfully" << std::endl;

    // Simple check - should work if bounds are correct
    EXPECT_TRUE(r_real[0] > 0)
        << "Expected positive R value from constant term";

  } catch (const std::exception& e) {
    FAIL() << "Exception in forward transform: " << e.what();
  }
}

TEST(DebugBoundsTest, CheckArraySizes) {
  // Debug the exact array sizes and indexing
  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 1;
  int ntheta = 8;
  int nzeta = 8;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "Size analysis:" << std::endl;
  std::cout << "  mpol: " << sizes.mpol << std::endl;
  std::cout << "  ntor: " << sizes.ntor << std::endl;
  std::cout << "  mnmax: " << sizes.mnmax << std::endl;
  std::cout << "  nZnT: " << sizes.nZnT << std::endl;
  std::cout << "  nThetaEff: " << sizes.nThetaEff << std::endl;
  std::cout << "  nThetaReduced: " << sizes.nThetaReduced << std::endl;
  std::cout << "  nZeta: " << sizes.nZeta << std::endl;

  // Check FourierBasisFastPoloidal sizes
  FourierBasisFastPoloidal fourier_basis(&sizes);
  std::cout << "  xm.size(): " << fourier_basis.xm.size() << std::endl;
  std::cout << "  xn.size(): " << fourier_basis.xn.size() << std::endl;
  std::cout << "  sinmu.size(): " << fourier_basis.sinmu.size() << std::endl;
  std::cout << "  cosmu.size(): " << fourier_basis.cosmu.size() << std::endl;

  // Verify expected relationships
  EXPECT_EQ(fourier_basis.xm.size(), sizes.mnmax);
  EXPECT_EQ(fourier_basis.xn.size(), sizes.mnmax);

  // Print mode information to debug indexing
  for (int mn = 0; mn < std::min(5, sizes.mnmax); ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;
    std::cout << "  mn=" << mn << ": m=" << m << ", n=" << n << std::endl;
  }
}

}  // namespace vmecpp
