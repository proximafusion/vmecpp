// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/bootstrap_current/bootstrap_current.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace vmecpp {

namespace {

constexpr double kMu0 = 4.0e-7 * M_PI;
constexpr double kElementaryCharge = 1.602176634e-19;

// n_e = 0.02 (1 - 0.8 rho) 1e20 m^-3, T_e = 0.7 (1 - 0.8 rho) keV,
// T_i = 0.6 (1 - 0.7 rho) keV, the profiles the SIMSOPT reference values below
// were generated with.
BootstrapProfiles ReferenceProfiles() {
  BootstrapProfiles profiles;
  profiles.ne.resize(2);
  profiles.ne << 0.02, -0.016;
  profiles.te.resize(2);
  profiles.te << 0.7, -0.56;
  profiles.ti.resize(2);
  profiles.ti << 0.6, -0.42;
  profiles.zeff = 1.3;
  return profiles;
}

// |B| = 1 / (1 + epsilon cos theta) with Jacobian weight 1 + epsilon cos theta
void ModelTokamakSurface(double epsilon, int ntheta, std::vector<double>& m_b,
                         std::vector<double>& m_w) {
  m_b.resize(ntheta);
  m_w.resize(ntheta);
  for (int l = 0; l < ntheta; ++l) {
    const double theta = 2.0 * M_PI * l / ntheta;
    m_w[l] = 1.0 + epsilon * std::cos(theta);
    m_b[l] = 1.0 / m_w[l];
  }
}

}  // namespace

TEST(BootstrapCurrent, KineticPressureIsMu0TimesElectronPlusIonPressure) {
  const BootstrapProfiles profiles = ReferenceProfiles();
  const double rho = 0.3;
  const double ne = 0.02 * (1.0 - 0.8 * rho) * 1.0e20;
  const double te = 0.7 * (1.0 - 0.8 * rho) * 1.0e3;
  const double ti = 0.6 * (1.0 - 0.7 * rho) * 1.0e3;
  const double expected = kMu0 * kElementaryCharge * (ne * te + ne / 1.3 * ti);
  EXPECT_NEAR(KineticPressure(profiles, rho), expected, 1.0e-12 * expected);
}

TEST(BootstrapCurrent, TrappedFractionVanishesForUniformField) {
  const std::vector<double> b(64, 1.7);
  const std::vector<double> w(64, -0.3);
  EXPECT_NEAR(TrappedFraction(b, w), 0.0, 1.0e-8);
}

// Reference values from simsopt.mhd.bootstrap.compute_trapped_fraction on the
// same 256-point surface.
TEST(BootstrapCurrent, TrappedFractionMatchesSimsoptOnModelTokamakSurface) {
  struct Case {
    double epsilon;
    double f_t;
    double b2_avg;
    double b_inv_avg;
  };
  const std::array<Case, 3> cases = {{
      {0.01, 0.14595335066669668, 1.0000500037503124, 1.0000499999999999},
      {0.1, 0.44920941064785014, 1.005037815259212, 1.005},
      {0.3, 0.7229707887706107, 1.0482848367219182, 1.045},
  }};
  for (const Case& c : cases) {
    std::vector<double> b;
    std::vector<double> w;
    ModelTokamakSurface(c.epsilon, 256, b, w);
    double b_max = 0.0;
    double b_min = 0.0;
    double b2_avg = 0.0;
    double b_inv_avg = 0.0;
    SurfaceFieldMoments(b, w, b_max, b_min, b2_avg, b_inv_avg);
    EXPECT_NEAR(b_max, 1.0 / (1.0 - c.epsilon), 1.0e-14);
    EXPECT_NEAR(b_min, 1.0 / (1.0 + c.epsilon), 1.0e-14);
    EXPECT_NEAR(b2_avg, c.b2_avg, 1.0e-12);
    EXPECT_NEAR(b_inv_avg, c.b_inv_avg, 1.0e-12);
    EXPECT_NEAR(TrappedFraction(b, w), c.f_t, 1.0e-6) << c.epsilon;
  }
}

// Reference values from simsopt.mhd.bootstrap.j_dot_B_Redl with the profiles
// above, Zeff = 1.3, nfp = 5 and psi_edge = 0.035 / (2 pi) on three synthetic
// surfaces, for quasi-axisymmetry and for helicity_n = 1.
TEST(BootstrapCurrent, RedlClosureMatchesSimsopt) {
  struct Surface {
    double rho;
    double g;
    double i;
    double iota;
    double b_max;
    double b_min;
    double b_inv_avg;
    double f_t;
  };
  const std::array<Surface, 3> surfaces = {{
      {0.3, 0.45, 0.0005, 0.31, 0.56, 0.50, 1.9, 0.20},
      {0.6, 0.45, 0.0015, 0.35, 0.60, 0.46, 1.92, 0.35},
      {0.9, 0.44, 0.0022, 0.42, 0.66, 0.40, 1.95, 0.45},
  }};
  struct Reference {
    double nu_e;
    double nu_i;
    double l31;
    double l32;
    double alpha;
    double j_dot_b;
  };
  const std::array<std::array<Reference, 3>, 2> references = {{
      {{{0.16476946307980295, 0.26340519869780843, 0.18471172658993806,
         -0.04762836261588986, -0.6697287826158598, 17186.12806718609},
        {0.05983295052213887, 0.08272894989218367, 0.34791087303299695,
         -0.10330780218497179, -0.6330433385629823, 19857.863273376985},
        {0.03568350143519251, 0.03507370079890019, 0.45243429096727084,
         -0.12268547505844979, -0.6014168073961933, 12099.075582465464}}},
      {{{0.01089094532083985, 0.01741057816552678, 0.2204425644502428,
         -0.115631970540459, -0.8661942287285692, -1158.9176576071868},
        {0.00450355541564486, 0.00622691020693856, 0.379712226004578,
         -0.15893687977884138, -0.7672841138257421, -1508.7586007798006},
        {0.00327228615781241, 0.003216365575445, 0.47935058702043476,
         -0.1682837971396947, -0.6943070707353931, -1120.0822028179907}}},
  }};
  const double psi_edge = 0.035 / (2.0 * M_PI);
  const int nfp = 5;
  for (int helicity_n = 0; helicity_n <= 1; ++helicity_n) {
    BootstrapProfiles profiles = ReferenceProfiles();
    profiles.helicity_n = helicity_n;
    for (int k = 0; k < 3; ++k) {
      const Surface& s = surfaces[k];
      const Reference& r = references[helicity_n][k];
      BootstrapSurface surface;
      surface.g = s.g;
      surface.i = s.i;
      surface.iota = s.iota;
      surface.f_t = s.f_t;
      surface.b_max = s.b_max;
      surface.b_min = s.b_min;
      surface.b_inv_avg = s.b_inv_avg;
      RedlCoefficients c;
      const double j_dot_b =
          RedlJDotB(profiles, surface, s.rho, psi_edge, nfp, &c);
      const double tol = 1.0e-10;
      EXPECT_NEAR(c.nu_e, r.nu_e, tol * std::abs(r.nu_e)) << helicity_n << k;
      EXPECT_NEAR(c.nu_i, r.nu_i, tol * std::abs(r.nu_i)) << helicity_n << k;
      EXPECT_NEAR(c.l31, r.l31, tol * std::abs(r.l31)) << helicity_n << k;
      EXPECT_NEAR(c.l32, r.l32, tol * std::abs(r.l32)) << helicity_n << k;
      EXPECT_NEAR(c.l34, r.l31, tol * std::abs(r.l31)) << helicity_n << k;
      EXPECT_NEAR(c.alpha, r.alpha, tol * std::abs(r.alpha)) << helicity_n << k;
      EXPECT_NEAR(j_dot_b, r.j_dot_b, tol * std::abs(r.j_dot_b))
          << helicity_n << k;
    }
  }
}

TEST(BootstrapCurrent, RedlClosureVanishesWithoutTrappedParticles) {
  const BootstrapProfiles profiles = ReferenceProfiles();
  BootstrapSurface surface;
  surface.g = 0.45;
  surface.iota = 0.3;
  surface.b_max = 0.5;
  surface.b_min = 0.5;
  surface.b_inv_avg = 2.0;
  surface.f_t = 0.0;
  EXPECT_EQ(RedlJDotB(profiles, surface, 0.5, 0.035 / (2.0 * M_PI), 5), 0.0);
}

namespace {

// f on the stored points of grid, zeta-major.
template <typename Function>
std::vector<double> Sample(const SurfaceGrid& grid, Function f) {
  std::vector<double> values(grid.n_theta_eff * grid.n_zeta);
  for (int k = 0; k < grid.n_zeta; ++k) {
    for (int l = 0; l < grid.n_theta_eff; ++l) {
      const double theta = 2.0 * M_PI * l / grid.n_theta_even;
      const double v = 2.0 * M_PI * k / grid.n_zeta;
      values[k * grid.n_theta_eff + l] = f(theta, v);
    }
  }
  return values;
}

// The interpolant of values at every point of the full grid.
void ExpectInterpolatesGrid(const SurfaceGrid& grid,
                            const std::vector<double>& values) {
  const int m_count = SurfaceGridMMax(grid) + 1;
  const int n_count = 2 * SurfaceGridNMax(grid) + 1;
  std::vector<double> work(SurfaceExtremaWorkSize(grid));
  std::vector<double> c(m_count * n_count);
  std::vector<double> s(m_count * n_count);
  SurfaceInterpolantKernel(values.data(), grid, work.data(), c.data(),
                           s.data());
  for (int k = 0; k < grid.n_zeta; ++k) {
    for (int l = 0; l < grid.n_theta_even; ++l) {
      double f = 0.0;
      double f_t = 0.0;
      double f_v = 0.0;
      double f_tt = 0.0;
      double f_tv = 0.0;
      double f_vv = 0.0;
      EvaluateSurfaceInterpolant(
          c.data(), s.data(), SurfaceGridMMax(grid), SurfaceGridNMax(grid),
          2.0 * M_PI * l / grid.n_theta_even, 2.0 * M_PI * k / grid.n_zeta,
          work.data(), f, f_t, f_v, f_tt, f_tv, f_vv);
      EXPECT_NEAR(f, SurfaceGridValue(values.data(), grid, l, k), 1.0e-13)
          << l << " " << k;
    }
  }
}

}  // namespace

TEST(BootstrapCurrent, InterpolantReproducesTheGridValues) {
  // a full poloidal grid with an even and a stellarator-symmetric grid with an
  // odd number of toroidal points
  const SurfaceGridTables full(12, 12, 10);
  std::vector<double> arbitrary(12 * 10);
  for (std::size_t i = 0; i < arbitrary.size(); ++i) {
    arbitrary[i] = std::sin(0.7 * i) + 0.1 * std::cos(2.3 * i * i);
  }
  ExpectInterpolatesGrid(full.grid(), arbitrary);

  const SurfaceGridTables symmetric(12, 7, 9);
  ExpectInterpolatesGrid(
      symmetric.grid(), Sample(symmetric.grid(), [](double theta, double v) {
        return 1.0 + 0.3 * std::cos(theta) + 0.2 * std::cos(2.0 * theta - v) +
               0.05 * std::cos(theta + 2.0 * v);
      }));
}

// Extrema between grid points: 1 + 0.2 cos(theta - 0.37) + 0.1 cos(v - 1.1)
// has its maximum 1.3 and minimum 0.7 off the grid.
TEST(BootstrapCurrent, SurfaceExtremaLieBetweenGridPoints) {
  const SurfaceGridTables tables(12, 12, 10);
  const std::vector<double> b =
      Sample(tables.grid(), [](double theta, double v) {
        return 1.0 + 0.2 * std::cos(theta - 0.37) + 0.1 * std::cos(v - 1.1);
      });
  std::vector<double> work(SurfaceExtremaWorkSize(tables.grid()));
  double b_max = 0.0;
  double b_min = 0.0;
  SurfaceExtremaKernel(b.data(), tables.grid(), work.data(), b_max, b_min);
  EXPECT_NEAR(b_max, 1.3, 1.0e-13);
  EXPECT_NEAR(b_min, 0.7, 1.0e-13);
  EXPECT_GT(b_max, *std::max_element(b.begin(), b.end()));
  EXPECT_LT(b_min, *std::min_element(b.begin(), b.end()));
}

// An axisymmetric, stellarator-symmetric surface: 1 - 0.2 cos(theta) +
// 0.3 cos(2 theta) has its minimum at cos(theta) = 1/6, between grid points,
// and its maximum 1.5 at theta = pi.
TEST(BootstrapCurrent, SurfaceExtremaOfAnAxisymmetricSurface) {
  const SurfaceGridTables tables(16, 9, 1);
  const std::vector<double> b =
      Sample(tables.grid(), [](double theta, double /*v*/) {
        return 1.0 - 0.2 * std::cos(theta) + 0.3 * std::cos(2.0 * theta);
      });
  std::vector<double> work(SurfaceExtremaWorkSize(tables.grid()));
  double b_max = 0.0;
  double b_min = 0.0;
  SurfaceExtremaKernel(b.data(), tables.grid(), work.data(), b_max, b_min);
  const double c = 1.0 / 6.0;
  EXPECT_NEAR(b_max, 1.5, 1.0e-13);
  EXPECT_NEAR(b_min, 1.0 - 0.2 * c + 0.3 * (2.0 * c * c - 1.0), 1.0e-13);
}

// G = 0.45 - 0.02 s, I = 0.003 s^2, dV/ds = 0.2 + 0.05 s: the parallel current
// of the identity mu0 <J.B> = signgs (G I' - I G') / dVds integrates back to I.
TEST(BootstrapCurrent, IntegrationInvertsTheParallelCurrentIdentity) {
  constexpr int kNumHalf = 200;
  const double delta_s = 1.0 / kNumHalf;
  const int signgs = -1;
  std::vector<double> j_dot_b(kNumHalf);
  std::vector<double> g(kNumHalf);
  std::vector<double> dvds(kNumHalf);
  std::vector<double> expected(kNumHalf);
  for (int j = 0; j < kNumHalf; ++j) {
    const double s = (j + 0.5) * delta_s;
    g[j] = 0.45 - 0.02 * s;
    const double dg = -0.02;
    expected[j] = 0.003 * s * s;
    const double di = 0.006 * s;
    dvds[j] = 0.2 + 0.05 * s;
    j_dot_b[j] = signgs * (g[j] * di - expected[j] * dg) / (kMu0 * dvds[j]);
  }
  std::vector<double> buco(kNumHalf, 0.0);
  IntegrateBootstrapCurrent(j_dot_b, g, dvds, delta_s, signgs, buco);
  for (int j = 0; j < kNumHalf; ++j) {
    EXPECT_NEAR(buco[j], expected[j], 1.0e-6) << j;
  }
}

}  // namespace vmecpp
