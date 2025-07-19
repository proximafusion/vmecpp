#include <cmath>
#include <iostream>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

int main() {
  std::cout << "DEBUG: Creating minimal test to reproduce NaN issue"
            << std::endl;

  // Create sizes for 2D asymmetric case
  vmecpp::Sizes sizes(
      true, 1, 2, 0, 16,
      1);  // lasym=true, nfp=1, mpol=2, ntor=0, ntheta=16, nzeta=1

  std::cout << "DEBUG: Created sizes - nThetaEff=" << sizes.nThetaEff
            << ", nZeta=" << sizes.nZeta << ", nZnT=" << sizes.nZnT
            << std::endl;

  // Create input arrays like the test case
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set up coefficients like the failing test
  rmnsc[1] = 0.0670068;
  zmncc[1] = 0.0670068;

  std::cout << "DEBUG: Set input coefficients - rmnsc[1]=" << rmnsc[1]
            << ", zmncc[1]=" << zmncc[1] << std::endl;

  // Create output arrays
  std::vector<double> r_real(sizes.nZnT, 0.0);
  std::vector<double> z_real(sizes.nZnT, 0.0);
  std::vector<double> lambda_real(sizes.nZnT, 0.0);

  std::cout << "DEBUG: Created output arrays of size " << sizes.nZnT
            << std::endl;

  try {
    std::cout << "DEBUG: About to call FourierToReal2DAsymmFastPoloidal"
              << std::endl;

    vmecpp::FourierToReal2DAsymmFastPoloidal(
        sizes, rmncc, rmnss, rmnsc, rmncs, zmnsc, zmncs, zmncc, zmnss,
        absl::Span<double>(r_real.data(), sizes.nZnT),
        absl::Span<double>(z_real.data(), sizes.nZnT),
        absl::Span<double>(lambda_real.data(), sizes.nZnT));

    std::cout << "DEBUG: Transform completed successfully" << std::endl;

    // Check output for NaN/inf
    bool found_non_finite = false;
    for (int i = 0; i < sizes.nZnT; ++i) {
      if (!std::isfinite(r_real[i]) || !std::isfinite(z_real[i])) {
        std::cout << "ERROR: Non-finite at i=" << i << ", R=" << r_real[i]
                  << ", Z=" << z_real[i] << std::endl;
        found_non_finite = true;
      }
    }

    if (!found_non_finite) {
      std::cout << "SUCCESS: All transform outputs are finite" << std::endl;
    }

    std::cout << "DEBUG: About to call SymmetrizeRealSpaceGeometry"
              << std::endl;

    vmecpp::SymmetrizeRealSpaceGeometry(
        sizes, absl::Span<double>(r_real.data(), sizes.nZnT),
        absl::Span<double>(z_real.data(), sizes.nZnT),
        absl::Span<double>(lambda_real.data(), sizes.nZnT));

    std::cout << "DEBUG: Symmetrization completed successfully" << std::endl;

    // Check output again after symmetrization
    found_non_finite = false;
    for (int i = 0; i < sizes.nZnT; ++i) {
      if (!std::isfinite(r_real[i]) || !std::isfinite(z_real[i])) {
        std::cout << "ERROR: Non-finite after symmetrization at i=" << i
                  << ", R=" << r_real[i] << ", Z=" << z_real[i] << std::endl;
        found_non_finite = true;
      }
    }

    if (!found_non_finite) {
      std::cout << "SUCCESS: All symmetrization outputs are finite"
                << std::endl;
    }

    std::cout << "DEBUG: Test completed without crashing" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "ERROR: Exception caught: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cout << "ERROR: Unknown exception caught" << std::endl;
    return 1;
  }

  return 0;
}
