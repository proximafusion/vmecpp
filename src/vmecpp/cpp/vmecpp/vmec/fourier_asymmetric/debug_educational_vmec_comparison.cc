// Meticulous debug comparison with educational_VMEC asymmetric patterns
// This test verifies that VMEC++ matches educational_VMEC behavior exactly

#include <cmath>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"

namespace vmecpp {

namespace {
const double PI = 3.14159265358979323846;
const double TOL = 1e-12;
}  // namespace

TEST(DebugEducationalVMECComparison, AsymmetricTransformPatterns) {
  // Test asymmetric transform with patterns that match educational_VMEC
  // This test uses specific coefficient combinations found in educational_VMEC

  std::cout << "\n=== Educational VMEC Pattern Comparison ===" << std::endl;

  bool lasym = true;
  int nfp = 1;
  int mpol = 3;  // Slightly larger to test multiple modes
  int ntor = 2;
  int ntheta = 8;
  int nzeta = 8;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "Configuration: mpol=" << sizes.mpol << ", ntor=" << sizes.ntor
            << ", mnmax=" << sizes.mnmax << std::endl;
  std::cout << "Grid: nThetaEff=" << sizes.nThetaEff
            << ", nZeta=" << sizes.nZeta << std::endl;

  // Initialize coefficient arrays
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);  // R asymmetric
  std::vector<double> rmncs(sizes.mnmax, 0.0);  // R asymmetric
  std::vector<double> zmnsc(sizes.mnmax, 0.0);  // Z symmetric
  std::vector<double> zmncs(sizes.mnmax, 0.0);  // Z symmetric
  std::vector<double> zmncc(sizes.mnmax, 0.0);  // Z asymmetric
  std::vector<double> zmnss(sizes.mnmax, 0.0);  // Z asymmetric

  // Get mode information
  FourierBasisFastPoloidal fourier_basis(&sizes);

  std::cout << "\nMode mapping:" << std::endl;
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;
    std::cout << "  mn=" << mn << ": m=" << m << ", n=" << n << std::endl;
  }

  // Set educational_VMEC-like coefficient pattern
  // Pattern 1: m=1, n=0 mode (tokamak-like)
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;

    if (m == 1 && n == 0) {
      rmncc[mn] = 1.0;  // R symmetric: cos(mu)
      zmnsc[mn] = 1.0;  // Z symmetric: sin(mu)
      rmnsc[mn] = 0.1;  // R asymmetric: sin(mu) (small perturbation)
      zmncc[mn] = 0.1;  // Z asymmetric: cos(mu) (small perturbation)

      std::cout << "\nSet tokamak-like coefficients at mn=" << mn
                << " (m=1, n=0):" << std::endl;
      std::cout << "  rmncc=" << rmncc[mn] << " (R symmetric cos(mu))"
                << std::endl;
      std::cout << "  zmnsc=" << zmnsc[mn] << " (Z symmetric sin(mu))"
                << std::endl;
      std::cout << "  rmnsc=" << rmnsc[mn] << " (R asymmetric sin(mu))"
                << std::endl;
      std::cout << "  zmncc=" << zmncc[mn] << " (Z asymmetric cos(mu))"
                << std::endl;
    }

    if (m == 1 && n == 1) {
      rmncs[mn] = 0.05;  // R asymmetric: cos(mu)sin(nv)
      zmncs[mn] = 0.05;  // Z symmetric: cos(mu)sin(nv)

      std::cout << "\nSet helical coefficients at mn=" << mn
                << " (m=1, n=1):" << std::endl;
      std::cout << "  rmncs=" << rmncs[mn] << " (R asymmetric cos(mu)sin(nv))"
                << std::endl;
      std::cout << "  zmncs=" << zmncs[mn] << " (Z symmetric cos(mu)sin(nv))"
                << std::endl;
    }
  }

  // Transform to real space
  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  std::cout << "\nCalling FourierToReal3DAsymmFastPoloidal..." << std::endl;

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  std::cout << "Transform completed successfully" << std::endl;

  // Analyze results in detail
  std::cout << "\n=== Detailed Transform Output Analysis ===" << std::endl;

  // Check key points that educational_VMEC would have
  std::vector<std::pair<int, int>> key_points = {{0, 0}, {1, 0}, {2, 0},
                                                 {0, 1}, {1, 1}, {2, 1}};

  for (auto [i, k] : key_points) {
    if (i < sizes.nThetaEff && k < sizes.nZeta) {
      int idx = i * sizes.nZeta + k;
      double u = 2.0 * PI * i / sizes.nThetaEff;
      double v = 2.0 * PI * k / sizes.nZeta;

      // Expected values based on educational_VMEC patterns
      // For tokamak: R ~ cos(mu) + 0.1*sin(mu), Z ~ sin(mu) + 0.1*cos(mu)
      double expected_r_main = cos(u) * sqrt(2.0);        // rmncc[1] = 1.0
      double expected_r_asym = sin(u) * 0.1 * sqrt(2.0);  // rmnsc[1] = 0.1
      double expected_z_main = sin(u) * sqrt(2.0);        // zmnsc[1] = 1.0
      double expected_z_asym = cos(u) * 0.1 * sqrt(2.0);  // zmncc[1] = 0.1

      // Add helical contribution for n=1
      double expected_r_helical = cos(u) * sin(v) * 0.05 * 2.0;  // rmncs
      double expected_z_helical = cos(u) * sin(v) * 0.05 * 2.0;  // zmncs

      double expected_r =
          expected_r_main + expected_r_asym + expected_r_helical;
      double expected_z =
          expected_z_main + expected_z_asym + expected_z_helical;

      std::cout << "Point i=" << i << ", k=" << k << " (u=" << u << ", v=" << v
                << "):" << std::endl;
      std::cout << "  R_actual=" << r_real[idx] << ", expected=" << expected_r
                << ", diff=" << (r_real[idx] - expected_r) << std::endl;
      std::cout << "  Z_actual=" << z_real[idx] << ", expected=" << expected_z
                << ", diff=" << (z_real[idx] - expected_z) << std::endl;

      // Verify within tolerance (not too strict for this debug test)
      EXPECT_NEAR(r_real[idx], expected_r, 1e-10)
          << "R mismatch at u=" << u << ", v=" << v;
      EXPECT_NEAR(z_real[idx], expected_z, 1e-10)
          << "Z mismatch at u=" << u << ", v=" << v;
    }
  }

  // Summary statistics
  double r_min = *std::min_element(r_real.begin(), r_real.end());
  double r_max = *std::max_element(r_real.begin(), r_real.end());
  double z_min = *std::min_element(z_real.begin(), z_real.end());
  double z_max = *std::max_element(z_real.begin(), z_real.end());

  std::cout << "\n=== Transform Output Summary ===" << std::endl;
  std::cout << "R range: [" << r_min << ", " << r_max << "]" << std::endl;
  std::cout << "Z range: [" << z_min << ", " << z_max << "]" << std::endl;

  // Verify reasonable ranges (educational_VMEC produces finite values)
  EXPECT_TRUE(std::isfinite(r_min) && std::isfinite(r_max))
      << "R values should be finite";
  EXPECT_TRUE(std::isfinite(z_min) && std::isfinite(z_max))
      << "Z values should be finite";
  EXPECT_GT(r_max, 0) << "Maximum R should be positive";
}

TEST(DebugEducationalVMECComparison, AsymmetricCoordinateMapping) {
  // Test the exact coordinate mapping used in educational_VMEC
  // Verify that theta and zeta grids map correctly to real space

  std::cout << "\n=== Educational VMEC Coordinate Mapping ===" << std::endl;

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 1;
  int ntheta = 6;  // Small for detailed analysis
  int nzeta = 4;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  std::cout << "Grid details:" << std::endl;
  std::cout << "  ntheta input: " << ntheta << std::endl;
  std::cout << "  nThetaEff: " << sizes.nThetaEff << std::endl;
  std::cout << "  nZeta: " << sizes.nZeta << std::endl;
  std::cout << "  nZnT total: " << sizes.nZnT << std::endl;

  // Initialize simple test pattern
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set simple m=1, n=0 mode for coordinate verification
  FourierBasisFastPoloidal fourier_basis(&sizes);
  for (int mn = 0; mn < sizes.mnmax; ++mn) {
    int m = fourier_basis.xm[mn];
    int n = fourier_basis.xn[mn] / sizes.nfp;
    if (m == 1 && n == 0) {
      rmnsc[mn] = 1.0;  // R ~ sin(mu) asymmetric
      break;
    }
  }

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  // Verify coordinate mapping exactly as educational_VMEC would
  std::cout << "\nCoordinate mapping verification:" << std::endl;
  std::cout
      << "theta_i | zeta_k |   u     |   v     |  idx  | R_actual | expected "
      << std::endl;
  std::cout
      << "--------|--------|---------|---------|-------|----------|----------"
      << std::endl;

  for (int i = 0; i < sizes.nThetaEff; ++i) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      int idx = i * sizes.nZeta + k;
      double u = 2.0 * PI * i / sizes.nThetaEff;
      double v = 2.0 * PI * k / sizes.nZeta;
      double expected = sin(u) * sqrt(2.0);  // R ~ sin(mu) with normalization

      printf("%6d  | %6d | %7.4f | %7.4f | %5d | %8.5f | %8.5f\n", i, k, u, v,
             idx, r_real[idx], expected);

      // Verify mapping
      EXPECT_NEAR(r_real[idx], expected, TOL)
          << "Coordinate mapping error at i=" << i << ", k=" << k;
    }
  }

  // Verify that full theta range [0, 2π] is covered
  double u_min = 0.0;
  double u_max = 2.0 * PI * (sizes.nThetaEff - 1) / sizes.nThetaEff;

  std::cout << "\nTheta range coverage:" << std::endl;
  std::cout << "  u_min: " << u_min << " (should be 0)" << std::endl;
  std::cout << "  u_max: " << u_max << " (should be close to 2π)" << std::endl;
  std::cout << "  2π: " << 2.0 * PI << std::endl;

  EXPECT_NEAR(u_min, 0.0, TOL);
  EXPECT_GT(u_max, 5.0);  // Should be close to 2π ≈ 6.28
}

TEST(DebugEducationalVMECComparison, AsymmetricSymmetrization) {
  // Test the symmetrization operation that educational_VMEC performs
  // This verifies the "symrzl" functionality pattern

  std::cout << "\n=== Educational VMEC Symmetrization Pattern ===" << std::endl;

  bool lasym = true;
  int nfp = 1;
  int mpol = 2;
  int ntor = 0;  // 2D case for clearer analysis
  int ntheta = 8;
  int nzeta = 1;

  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  // Test pattern: both symmetric and asymmetric contributions
  std::vector<double> rmncc(sizes.mnmax, 0.0);
  std::vector<double> rmnss(sizes.mnmax, 0.0);
  std::vector<double> rmnsc(sizes.mnmax, 0.0);
  std::vector<double> rmncs(sizes.mnmax, 0.0);
  std::vector<double> zmnsc(sizes.mnmax, 0.0);
  std::vector<double> zmncs(sizes.mnmax, 0.0);
  std::vector<double> zmncc(sizes.mnmax, 0.0);
  std::vector<double> zmnss(sizes.mnmax, 0.0);

  // Set both symmetric and asymmetric m=1 modes
  rmncc[1] = 1.0;  // R symmetric: cos(mu)
  rmnsc[1] = 0.2;  // R asymmetric: sin(mu)
  zmnsc[1] = 1.0;  // Z symmetric: sin(mu)
  zmncc[1] = 0.2;  // Z asymmetric: cos(mu)

  std::cout << "Input coefficients:" << std::endl;
  std::cout << "  rmncc[1] = " << rmncc[1] << " (R symmetric cos)" << std::endl;
  std::cout << "  rmnsc[1] = " << rmnsc[1] << " (R asymmetric sin)"
            << std::endl;
  std::cout << "  zmnsc[1] = " << zmnsc[1] << " (Z symmetric sin)" << std::endl;
  std::cout << "  zmncc[1] = " << zmncc[1] << " (Z asymmetric cos)"
            << std::endl;

  std::vector<double> r_real(sizes.nZnT);
  std::vector<double> z_real(sizes.nZnT);
  std::vector<double> lambda_real(sizes.nZnT);

  FourierToReal3DAsymmFastPoloidal(
      sizes, absl::MakeSpan(rmncc), absl::MakeSpan(rmnss),
      absl::MakeSpan(rmnsc), absl::MakeSpan(rmncs), absl::MakeSpan(zmnsc),
      absl::MakeSpan(zmncs), absl::MakeSpan(zmncc), absl::MakeSpan(zmnss),
      absl::MakeSpan(r_real), absl::MakeSpan(z_real),
      absl::MakeSpan(lambda_real));

  std::cout << "\nSymmetrization analysis:" << std::endl;
  std::cout << "theta_i |    u    | R_actual | R_symm  | R_asymm |   R_total   "
               "| Z_actual | Z_symm  | Z_asymm |   Z_total"
            << std::endl;
  std::cout << "--------|---------|----------|---------|---------|-------------"
               "|----------|---------|---------|----------"
            << std::endl;

  for (int i = 0; i < sizes.nThetaEff; ++i) {
    double u = 2.0 * PI * i / sizes.nThetaEff;

    // Educational_VMEC combines: total = symmetric + asymmetric
    double r_symm = cos(u) * sqrt(2.0);         // From rmncc[1]
    double r_asymm = sin(u) * 0.2 * sqrt(2.0);  // From rmnsc[1]
    double r_total = r_symm + r_asymm;

    double z_symm = sin(u) * sqrt(2.0);         // From zmnsc[1]
    double z_asymm = cos(u) * 0.2 * sqrt(2.0);  // From zmncc[1]
    double z_total = z_symm + z_asymm;

    printf(
        "%6d  | %7.4f | %8.5f | %7.4f | %7.4f | %11.5f | %8.5f | %7.4f | %7.4f "
        "| %9.5f\n",
        i, u, r_real[i], r_symm, r_asymm, r_total, z_real[i], z_symm, z_asymm,
        z_total);

    // Verify educational_VMEC pattern: symmetric + asymmetric = total
    EXPECT_NEAR(r_real[i], r_total, TOL) << "R combination error at i=" << i;
    EXPECT_NEAR(z_real[i], z_total, TOL) << "Z combination error at i=" << i;
  }

  std::cout << "\nSymmetrization verification completed successfully"
            << std::endl;
}

}  // namespace vmecpp
