// Test program to identify NaN issue in asymmetric transforms
#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

int main() {
  // Simple test case
  bool lasym = true;
  int nfp = 1;
  int mpol = 3;
  int ntor = 2;
  int ntheta = 16;
  int nzeta = 16;

  vmecpp::Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);
  vmecpp::FourierBasisFastPoloidal fb(&sizes);

  std::cout << "Sizes: mnmax=" << sizes.mnmax
            << ", nThetaEff=" << sizes.nThetaEff << ", nZeta=" << sizes.nZeta
            << ", nnyq2=" << sizes.nnyq2 << std::endl;

  // Print mode information
  std::cout << "\nMode information:" << std::endl;
  for (int mn = 0; mn < std::min(10, sizes.mnmax); ++mn) {
    std::cout << "mn=" << mn << ", m=" << fb.xm[mn]
              << ", n=" << fb.xn[mn] / sizes.nfp << std::endl;
  }

  // Initialize test Fourier coefficients
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set simple test values
  rmncc[0] = 1.0;  // R00
  if (sizes.mnmax > 1) {
    zmnsc[1] = 0.3;  // Z10
  }

  // Output arrays
  int real_size = sizes.nThetaEff * sizes.nZeta;
  std::vector<double> r_real(real_size);
  std::vector<double> z_real(real_size);
  std::vector<double> lambda_real(real_size);

  std::cout << "\nTesting 2D transform first..." << std::endl;

  // Test 2D transform
  vmecpp::FourierToReal2DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Check for NaN in 2D
  bool found_nan_2d = false;
  for (int i = 0; i < real_size; ++i) {
    if (std::isnan(r_real[i]) || std::isnan(z_real[i])) {
      std::cout << "NaN found in 2D at index " << i << std::endl;
      found_nan_2d = true;
      break;
    }
  }

  if (!found_nan_2d) {
    std::cout << "No NaN in 2D transform." << std::endl;
  }

  // Reset arrays
  std::fill(r_real.begin(), r_real.end(), 0.0);
  std::fill(z_real.begin(), z_real.end(), 0.0);
  std::fill(lambda_real.begin(), lambda_real.end(), 0.0);

  std::cout << "\nNow testing 3D transform..." << std::endl;

  // Call the 3D transform
  vmecpp::FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Check for NaN
  bool found_nan = false;
  for (int i = 0; i < real_size; ++i) {
    if (std::isnan(r_real[i]) || std::isnan(z_real[i])) {
      std::cout << "NaN found at index " << i << std::endl;
      found_nan = true;
      break;
    }
  }

  if (!found_nan) {
    std::cout << "No NaN found. First few values:" << std::endl;
    for (int i = 0; i < std::min(5, real_size); ++i) {
      std::cout << "idx=" << i << ", R=" << r_real[i] << ", Z=" << z_real[i]
                << std::endl;
    }
  }

  return 0;
}
