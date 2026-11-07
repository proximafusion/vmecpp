// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

namespace {

// K(k) and E(k), the complete elliptic integrals of modulus k, by the
// arithmetic-geometric mean.
void CompleteEllipticIntegrals(double k, double& m_K, double& m_E) {
  double a = 1.0;
  double b = std::sqrt(1.0 - k * k);
  double c = k;
  double weight = 0.5;
  double sum = weight * c * c;
  while (std::abs(c) > 1.0e-16) {
    c = 0.5 * (a - b);
    const double a_next = 0.5 * (a + b);
    b = std::sqrt(a * b);
    a = a_next;
    weight *= 2.0;
    sum += weight * c * c;
  }
  m_K = M_PI / (2.0 * a);
  m_E = m_K * (1.0 - sum);
}

// B_R and B_Z at (r, z) of the circular loop of radius radius at height z0
// about the Z axis, carrying current towards increasing phi.
void CircularLoopField(double radius, double z0, double current, double r,
                       double z, double& m_br, double& m_bz) {
  const double dz = z - z0;
  const double plus = (radius + r) * (radius + r) + dz * dz;
  const double minus = (radius - r) * (radius - r) + dz * dz;
  double K = 0.0;
  double E = 0.0;
  CompleteEllipticIntegrals(std::sqrt(4.0 * radius * r / plus), K, E);
  // mu0 / (2 pi) = 2e-7
  const double scale = 2.0e-7 * current / std::sqrt(plus);
  m_br =
      scale * dz / r * (-K + (radius * radius + r * r + dz * dz) / minus * E);
  m_bz = scale * (K + (radius * radius - r * r - dz * dz) / minus * E);
}

}  // namespace

// The axis current of an axisymmetric plasma has, on a boundary of aspect
// ratio 20, the closed-form field of a circular loop.
TEST(TestExternalMagneticField, AxisymmetricAxisCurrentIsCircularLoop) {
  const bool lasym = false;
  const int nfp = 1;
  const int mpol = 6;
  const int ntor = 0;
  const int ntheta = 24;
  const int nzeta = 1;
  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);

  const double R0 = 10.0;
  const double a = 0.5;
  std::vector<double> rmnc(s.mnmax, 0.0);
  std::vector<double> zmns(s.mnmax, 0.0);
  rmnc[0] = R0;
  rmnc[1] = a;
  zmns[1] = a;
  std::vector<double> rCC(s.mnsize), rSS(s.mnsize), zSC(s.mnsize),
      zCS(s.mnsize);
  std::vector<double> rSC, rCS, zCC, zSS;
  fb.cos_to_cc_ss(rmnc, rCC, rSS, s.ntor, s.mpol);
  fb.sin_to_sc_cs(zmns, zSC, zCS, s.ntor, s.mpol);

  SurfaceGeometry sg(&s, &fb, &tp);
  const int kSignOfJacobian = -1;
  sg.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, kSignOfJacobian,
            /*fullUpdate=*/true);

  const MagneticFieldFunction no_coils =
      [](const Eigen::VectorXd& r, const Eigen::VectorXd& /*phi*/,
         const Eigen::VectorXd& /*z*/, Eigen::VectorXd& m_b_r,
         Eigen::VectorXd& m_b_phi, Eigen::VectorXd& m_b_z) {
        m_b_r.setZero(r.size());
        m_b_phi.setZero(r.size());
        m_b_z.setZero(r.size());
        return absl::OkStatus();
      };
  ExternalMagneticField ef(&s, &tp, &sg, &no_coils);

  // the magnetic axis of the plasma, shifted out and up from the centre
  const std::vector<double> axis_r = {R0 + 0.01};
  const std::vector<double> axis_z = {0.002};
  const double current = 1.0e5;
  ASSERT_TRUE(ef.update(axis_r, axis_z, current).ok());

  // the closed form loses digits to cancellation this close to the filament
  static constexpr double kTolerance = 1.0e-8;
  for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
    double br = 0.0;
    double bz = 0.0;
    CircularLoopField(axis_r[0], axis_z[0], current, sg.r1b[kl], sg.z1b[kl], br,
                      bz);
    const double scale = std::hypot(br, bz);
    EXPECT_NEAR(ef.curtorBr[kl - tp.ztMin], br, kTolerance * scale) << kl;
    EXPECT_NEAR(ef.curtorBz[kl - tp.ztMin], bz, kTolerance * scale) << kl;
    EXPECT_NEAR(ef.curtorBp[kl - tp.ztMin], 0.0, kTolerance * scale) << kl;
  }
}

}  // namespace vmecpp
