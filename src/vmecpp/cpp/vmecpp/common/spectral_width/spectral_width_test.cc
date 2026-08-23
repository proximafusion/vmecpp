// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/spectral_width/spectral_width.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"

namespace vmecpp {

namespace {
constexpr double kTolerance = 1.0e-12;

SurfaceFourierGeometry StellaratorSymmetricSurface(
    const std::vector<double>& rmncc, const std::vector<double>& zmnsc) {
  SurfaceFourierGeometry geometry;
  geometry.rmncc = rmncc;
  geometry.zmnsc = zmnsc;
  return geometry;
}
}  // namespace

TEST(TestSpectralWidth, SingleModeGivesItsOwnPoloidalModeNumber) {
  // With only one poloidal mode present, both sums carry the same coefficient
  // and <M> collapses to that mode number for any p and q.
  const Sizes sizes(/*lasym=*/false, /*nfp=*/1, /*mpol=*/8, /*ntor=*/0,
                    /*ntheta=*/0, /*nzeta=*/0);
  const std::vector<double> unit_scale(sizes.mpol, 1.0);

  for (int m = 1; m < sizes.mpol; ++m) {
    std::vector<double> rmncc(sizes.mpol, 0.0);
    std::vector<double> zmnsc(sizes.mpol, 0.0);
    rmncc[m] = 0.3;
    zmnsc[m] = -0.2;

    const double spectral_width =
        SpectralWidth(StellaratorSymmetricSurface(rmncc, zmnsc), sizes,
                      unit_scale, unit_scale);
    EXPECT_NEAR(spectral_width, m, kTolerance);
  }
}

TEST(TestSpectralWidth, TwoModesAreWeightedByTheirEnergyAndExponents) {
  const Sizes sizes(/*lasym=*/false, /*nfp=*/1, /*mpol=*/8, /*ntor=*/0,
                    /*ntheta=*/0, /*nzeta=*/0);
  const std::vector<double> unit_scale(sizes.mpol, 1.0);

  const double low_amplitude = 0.5;
  const double high_amplitude = 0.1;
  std::vector<double> rmncc(sizes.mpol, 0.0);
  const std::vector<double> zmnsc(sizes.mpol, 0.0);
  rmncc[2] = low_amplitude;
  rmncc[6] = high_amplitude;

  const int p = 4;
  const int q = 1;
  const double expected =
      (low_amplitude * low_amplitude * std::pow(2, p + q) +
       high_amplitude * high_amplitude * std::pow(6, p + q)) /
      (low_amplitude * low_amplitude * std::pow(2, p) +
       high_amplitude * high_amplitude * std::pow(6, p));

  EXPECT_NEAR(SpectralWidth(StellaratorSymmetricSurface(rmncc, zmnsc), sizes,
                            unit_scale, unit_scale, p, q),
              expected, kTolerance);
}

TEST(TestSpectralWidth, BasisScalesWeightTheToroidalModes) {
  // The internal basis carries a factor nscale[n] on each coefficient, so
  // passing the basis arrays weights an n != 0 mode twice as strongly as the
  // same coefficient at n = 0.
  const Sizes sizes(/*lasym=*/false, /*nfp=*/5, /*mpol=*/4, /*ntor=*/2,
                    /*ntheta=*/0, /*nzeta=*/0);
  const FourierBasisFastPoloidal fourier_basis(&sizes);

  const int coefficients_per_surface = sizes.mpol * (sizes.ntor + 1);
  std::vector<double> rmncc(coefficients_per_surface, 0.0);
  std::vector<double> rmnss(coefficients_per_surface, 0.0);
  const std::vector<double> zeros(coefficients_per_surface, 0.0);

  // m = 2 at n = 0, and m = 3 at n = 1.
  rmncc[2 * (sizes.ntor + 1) + 0] = 0.4;
  rmncc[3 * (sizes.ntor + 1) + 1] = 0.4;

  SurfaceFourierGeometry geometry;
  geometry.rmncc = rmncc;
  geometry.rmnss = rmnss;
  geometry.zmnsc = zeros;
  geometry.zmncs = zeros;

  const std::vector<double> unit_scale(std::max(sizes.mpol, sizes.ntor + 1) + 1,
                                       1.0);
  const double unweighted =
      SpectralWidth(geometry, sizes, unit_scale, unit_scale);

  const std::span<const double> mscale(fourier_basis.mscale.data(),
                                       fourier_basis.mscale.size());
  const std::span<const double> nscale(fourier_basis.nscale.data(),
                                       fourier_basis.nscale.size());
  const double weighted = SpectralWidth(geometry, sizes, mscale, nscale);

  // nscale[1]^2 = 2 doubles the weight of the m = 3 mode, which pulls <M> up.
  EXPECT_GT(weighted, unweighted);

  const double amplitude = 0.4 * 0.4;
  const int p = 4;
  const int q = 1;
  const double expected =
      (amplitude * std::pow(2, p + q) + 2.0 * amplitude * std::pow(3, p + q)) /
      (amplitude * std::pow(2, p) + 2.0 * amplitude * std::pow(3, p));
  // The constant mscale[m >= 1]^2 cancels between numerator and denominator.
  EXPECT_NEAR(weighted, expected, kTolerance);
}

TEST(TestSpectralWidth, ModeOneIsUnpackedFromTheInternalRepresentation) {
  // m = 1 is stored as R+ = (rmnss + zmncs) / 2 and R- = (rmnss - zmncs) / 2,
  // so the sums have to see the original rmnss and zmncs again.
  const Sizes sizes(/*lasym=*/false, /*nfp=*/5, /*mpol=*/3, /*ntor=*/1,
                    /*ntheta=*/0, /*nzeta=*/0);
  const int coefficients_per_surface = sizes.mpol * (sizes.ntor + 1);
  const std::vector<double> unit_scale(coefficients_per_surface, 1.0);

  const double rmnss_value = 0.6;
  const double zmncs_value = 0.2;
  const int index = 1 * (sizes.ntor + 1) + 1;

  std::vector<double> rmnss(coefficients_per_surface, 0.0);
  std::vector<double> zmncs(coefficients_per_surface, 0.0);
  const std::vector<double> zeros(coefficients_per_surface, 0.0);
  rmnss[index] = 0.5 * (rmnss_value + zmncs_value);
  zmncs[index] = 0.5 * (rmnss_value - zmncs_value);

  SurfaceFourierGeometry geometry;
  geometry.rmncc = zeros;
  geometry.rmnss = rmnss;
  geometry.zmnsc = zeros;
  geometry.zmncs = zmncs;

  // Only m = 1 carries any amplitude, so <M> has to come out as exactly 1.
  EXPECT_NEAR(SpectralWidth(geometry, sizes, unit_scale, unit_scale), 1.0,
              kTolerance);
}

}  // namespace vmecpp
