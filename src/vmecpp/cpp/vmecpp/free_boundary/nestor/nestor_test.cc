// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/nestor/nestor.h"

#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

// Exterior Neumann problem with a known solution: the external field is that
// of a line of vertical point dipoles along the circular magnetic axis of a
// helically deformed torus, i.e. an interior source with a single-valued
// potential that is harmonic in the whole exterior. The vacuum field that
// NESTOR reconstructs must vanish on the surface, so |B|^2/2 there is the
// solver's error, measured against the external field's own |B_ext|^2/2.
// The field is evaluated exactly on the surface (no mgrid interpolation).
TEST(TestNestor, InteriorDipoleLineGivesZeroFieldOnHelicalTorus) {
  const bool lasym = false;
  const int nfp = 2;
  const int mpol = 8;
  const int ntor = 4;
  const int ntheta = 0;
  const int nzeta = 24;
  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // circular torus R0 = 1, a = 0.3 with a helical (m = 1, n = 1) deformation
  const double R0 = 1.0;
  const double a = 0.3;
  const double eps = 0.05;
  std::vector<double> rmnc(s.mnmax, 0.0);
  std::vector<double> zmns(s.mnmax, 0.0);
  std::vector<int> xm(s.mnmax);
  std::vector<int> xn(s.mnmax);
  {
    int mn = 0;
    for (int n = 0; n < s.ntor + 1; ++n) {
      xm[mn] = 0;
      xn[mn] = n;
      ++mn;
    }
    for (int m = 1; m < s.mpol; ++m) {
      for (int n = -s.ntor; n < s.ntor + 1; ++n) {
        xm[mn] = m;
        xn[mn] = n;
        ++mn;
      }
    }
    ASSERT_EQ(mn, s.mnmax);
    for (mn = 0; mn < s.mnmax; ++mn) {
      if (xm[mn] == 0 && xn[mn] == 0) rmnc[mn] = R0;
      if (xm[mn] == 1 && xn[mn] == 0) {
        rmnc[mn] = a;
        zmns[mn] = a;
      }
      if (xm[mn] == 1 && xn[mn] == 1) {
        rmnc[mn] = eps;
        zmns[mn] = eps;
      }
    }
  }
  std::vector<double> rCC(s.mnsize), rSS(s.mnsize), zSC(s.mnsize),
      zCS(s.mnsize);
  std::vector<double> rSC, rCS, zCC, zSS;
  fb.cos_to_cc_ss(rmnc, rCC, rSS, s.ntor, s.mpol);
  fb.sin_to_sc_cs(zmns, zSC, zCS, s.ntor, s.mpol);

  // field of the dipole line in cylindrical components at (R, phi, Z)
  const int num_dipoles = 96 * nfp;
  const double moment = 1.0 / 96.0;  // total vertical moment 1 A m^2 per period
  const auto dipole_field = [&](double R, double phi, double Z, double& br,
                                double& bp, double& bz) {
    const double x = R * std::cos(phi);
    const double y = R * std::sin(phi);
    double bx = 0.0, by = 0.0;
    bz = 0.0;
    for (int j = 0; j < num_dipoles; ++j) {
      const double phi_j = 2.0 * M_PI * j / num_dipoles;
      const double dx = x - R0 * std::cos(phi_j);
      const double dy = y - R0 * std::sin(phi_j);
      const double dz = Z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double r = std::sqrt(r2);
      const double f = 1.0e-7 / (r2 * r);  // mu0 / (4 pi) / r^3
      const double mdotn = moment * dz / r;
      bx += f * 3.0 * mdotn * dx / r;
      by += f * 3.0 * mdotn * dy / r;
      bz += f * (3.0 * mdotn * dz / r - moment);
    }
    br = bx * std::cos(phi) + by * std::sin(phi);
    bp = -bx * std::sin(phi) + by * std::cos(phi);
  };
  const MagneticFieldFunction magnetic_field =
      [&](const Eigen::VectorXd& r, const Eigen::VectorXd& phi,
          const Eigen::VectorXd& z, Eigen::VectorXd& m_b_r,
          Eigen::VectorXd& m_b_phi, Eigen::VectorXd& m_b_z) {
        for (Eigen::Index i = 0; i < r.size(); ++i) {
          dipole_field(r[i], phi[i], z[i], m_b_r[i], m_b_phi[i], m_b_z[i]);
        }
        return absl::OkStatus();
      };

  // |B_ext|^2/2 at the surface points on NESTOR's grid: kl = l * nZeta + k,
  // theta_l = 2 pi l / nThetaEven, zeta_k = 2 pi k / (nfp nZeta)
  double ref_sum = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int l = kl / s.nZeta;
    const int k = kl % s.nZeta;
    const double theta = 2.0 * M_PI * l / s.nThetaEven;
    const double zeta = 2.0 * M_PI * k / (nfp * s.nZeta);
    double R = 0.0, Z = 0.0;
    for (int mn = 0; mn < s.mnmax; ++mn) {
      const double arg = xm[mn] * theta - xn[mn] * nfp * zeta;
      R += rmnc[mn] * std::cos(arg);
      Z += zmns[mn] * std::sin(arg);
    }
    double br = 0.0, bp = 0.0, bz = 0.0;
    dipole_field(R, zeta, Z, br, bp, bz);
    const double half_b2 = 0.5 * (br * br + bp * bp + bz * bz);
    ref_sum += half_b2 * half_b2;
  }
  const double ref_rms = std::sqrt(ref_sum / s.nZnT);

  TangentialPartitioning tp(s.nZnT);
  const int nf = s.ntor;
  const int mf = s.mpol + 1;
  const int mnpd = (2 * nf + 1) * (mf + 1);
  std::vector<double> matrixShare(mnpd * mnpd);
  std::vector<double> bvecShare(mnpd);
  std::vector<double> bSqVacShare(s.nZnT);
  std::vector<double> vacuum_b_r(s.nZnT), vacuum_b_phi(s.nZnT),
      vacuum_b_z(s.nZnT);
  std::vector<double> reduce_slots(matrixShare.size());
  Eigen::PartialPivLU<Eigen::MatrixXd> lu_decomposition;
  Nestor nestor(&s, &tp, &magnetic_field, matrixShare, bvecShare, bSqVacShare,
                &lu_decomposition, vacuum_b_r, vacuum_b_phi, vacuum_b_z,
                reduce_slots);

  std::vector<double> axis_r(s.nZeta, R0);
  std::vector<double> axis_z(s.nZeta, 0.0);
  double bSubUVac = 0.0;
  double bSubVVac = 0.0;
  const int kSignOfJacobian = -1;
  const auto status = nestor.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS,
                                    kSignOfJacobian, axis_r, axis_z, &bSubUVac,
                                    &bSubVVac, /*netToroidalCurrent=*/0.0,
                                    /*ivacskip=*/0);
  ASSERT_TRUE(status.ok()) << status.status();

  double err_sum = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    err_sum += bSqVacShare[kl] * bSqVacShare[kl];
  }
  const double err_rms = std::sqrt(err_sum / s.nZnT);
  // 4.9e-3 with the cross-term sign defect in the analytic add-back
  EXPECT_LT(err_rms / ref_rms, 1.0e-4)
      << "|B|^2/2 residual relative to |B_ext|^2/2: " << err_rms / ref_rms;
}

namespace {

// Complete elliptic integrals K and E of the given parameter, by the
// arithmetic-geometric mean.
void CompleteEllipticIntegrals(double parameter, double& m_k, double& m_e) {
  double a = 1.0;
  double b = std::sqrt(1.0 - parameter);
  double c = std::sqrt(parameter);
  double weight = 0.5;
  double sum_c2 = weight * c * c;
  while (c > std::numeric_limits<double>::epsilon() * a) {
    c = 0.5 * (a - b);
    const double a_next = 0.5 * (a + b);
    b = std::sqrt(a * b);
    a = a_next;
    weight *= 2.0;
    sum_c2 += weight * c * c;
  }
  m_k = 0.5 * M_PI / a;
  m_e = m_k * (1.0 - sum_c2);
}

// Adds the field of a circular ring of the given radius and current in Z = 0
// at (r, z) to m_b_r and m_b_z.
void AddRingField(double radius, double current, double r, double z,
                  double& m_b_r, double& m_b_z) {
  const double alpha2 = (r - radius) * (r - radius) + z * z;
  const double beta2 = (r + radius) * (r + radius) + z * z;
  const double beta = std::sqrt(beta2);
  double k = 0.0;
  double e = 0.0;
  CompleteEllipticIntegrals(4.0 * radius * r / beta2, k, e);
  // \mu_0 I / \pi
  const double c = 4.0e-7 * current;
  m_b_r += c * z / (2.0 * alpha2 * beta * r) *
           ((radius * radius + r * r + z * z) * e - alpha2 * k);
  m_b_z += c / (2.0 * alpha2 * beta) *
           ((radius * radius - r * r - z * z) * e + alpha2 * k);
}

struct TwoRingGrid {
  bool lasym;
  int nfp;
  int ntor;
  int nzeta;
  int ntheta;
  double tolerance;
};

}  // namespace

// Exterior Neumann problem with a known solution: two circular current rings
// in Z = 0 at R = 0.94 and R = 1.06 with opposite currents, inside a circular
// torus R0 = 1, a = 0.3. Outside the torus the field of the rings is the
// gradient of a single-valued potential, so the vacuum field that NESTOR
// returns on the surface is its error, measured as its rms over the rms of the
// field of the rings.
class TwoRingTest : public ::testing::TestWithParam<TwoRingGrid> {};

TEST_P(TwoRingTest, VacuumFieldVanishesOnTorus) {
  const TwoRingGrid grid = GetParam();
  const int mpol = 8;
  Sizes s(grid.lasym, grid.nfp, mpol, grid.ntor, grid.ntheta, grid.nzeta);
  FourierBasisFastToroidal fb(&s);

  const double R0 = 1.0;
  const double a = 0.3;
  std::vector<double> rmnc(s.mnmax, 0.0);
  std::vector<double> zmns(s.mnmax, 0.0);
  // (m, n) = (0, 0) comes first and (1, 0) after the m = 0 modes and the
  // m = 1 modes of negative n
  rmnc[0] = R0;
  rmnc[2 * s.ntor + 1] = a;
  zmns[2 * s.ntor + 1] = a;
  std::vector<double> rCC(s.mnsize), rSS(s.mnsize), zSC(s.mnsize),
      zCS(s.mnsize);
  fb.cos_to_cc_ss(rmnc, rCC, rSS, s.ntor, s.mpol);
  fb.sin_to_sc_cs(zmns, zSC, zCS, s.ntor, s.mpol);
  const int asym_size = grid.lasym ? s.mnsize : 0;
  std::vector<double> rSC(asym_size, 0.0), rCS(asym_size, 0.0),
      zCC(asym_size, 0.0), zSS(asym_size, 0.0);

  const auto ring_field = [](double r, double z, double& m_b_r, double& m_b_z) {
    m_b_r = 0.0;
    m_b_z = 0.0;
    AddRingField(0.94, 1.0e4, r, z, m_b_r, m_b_z);
    AddRingField(1.06, -1.0e4, r, z, m_b_r, m_b_z);
  };
  const MagneticFieldFunction magnetic_field =
      [&](const Eigen::VectorXd& r, const Eigen::VectorXd& phi,
          const Eigen::VectorXd& z, Eigen::VectorXd& m_b_r,
          Eigen::VectorXd& m_b_phi, Eigen::VectorXd& m_b_z) {
        for (Eigen::Index i = 0; i < r.size(); ++i) {
          ring_field(r[i], z[i], m_b_r[i], m_b_z[i]);
          m_b_phi[i] = 0.0;
        }
        return absl::OkStatus();
      };

  TangentialPartitioning tp(s.nZnT);
  const int nf = s.ntor;
  const int mf = s.mpol + 1;
  const int mnpd = (2 * nf + 1) * (mf + 1) * (grid.lasym ? 2 : 1);
  std::vector<double> matrixShare(mnpd * mnpd);
  std::vector<double> bvecShare(mnpd);
  std::vector<double> bSqVacShare(s.nZnT);
  std::vector<double> vacuum_b_r(s.nZnT), vacuum_b_phi(s.nZnT),
      vacuum_b_z(s.nZnT);
  std::vector<double> reduce_slots(matrixShare.size());
  Eigen::PartialPivLU<Eigen::MatrixXd> lu_decomposition;
  Nestor nestor(&s, &tp, &magnetic_field, matrixShare, bvecShare, bSqVacShare,
                &lu_decomposition, vacuum_b_r, vacuum_b_phi, vacuum_b_z,
                reduce_slots);

  std::vector<double> axis_r(s.nZeta, R0);
  std::vector<double> axis_z(s.nZeta, 0.0);
  double bSubUVac = 0.0;
  double bSubVVac = 0.0;
  const int kSignOfJacobian = -1;
  const auto status = nestor.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS,
                                    kSignOfJacobian, axis_r, axis_z, &bSubUVac,
                                    &bSubVVac, /*netToroidalCurrent=*/0.0,
                                    /*ivacskip=*/0);
  ASSERT_TRUE(status.ok()) << status.status();

  double err_sum = 0.0;
  double ref_sum = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const double theta = 2.0 * M_PI * (kl / s.nZeta) / s.nThetaEven;
    double b_r = 0.0;
    double b_z = 0.0;
    ring_field(R0 + a * std::cos(theta), a * std::sin(theta), b_r, b_z);
    ref_sum += b_r * b_r + b_z * b_z;
    err_sum += vacuum_b_r[kl] * vacuum_b_r[kl] +
               vacuum_b_phi[kl] * vacuum_b_phi[kl] +
               vacuum_b_z[kl] * vacuum_b_z[kl];
  }
  const double error = std::sqrt(err_sum / ref_sum);
  EXPECT_LT(error, grid.tolerance)
      << "rms of the vacuum field over that of the rings";
}

// the axisymmetric grid, whose toroidal integral is a sum over images, three-
// dimensional grids with even and odd nzeta, and the full poloidal grid of
// lasym; then the same on finer grids, where the error that remains is the
// third-order defect of the sums at the source point
INSTANTIATE_TEST_SUITE_P(
    TestNestor, TwoRingTest,
    ::testing::Values(TwoRingGrid{false, 1, 0, 1, 22, 5.0e-4},
                      TwoRingGrid{false, 2, 4, 32, 22, 5.0e-4},
                      TwoRingGrid{false, 2, 4, 31, 22, 5.0e-4},
                      TwoRingGrid{true, 2, 4, 31, 22, 5.0e-4},
                      TwoRingGrid{false, 1, 0, 1, 32, 6.0e-5},
                      TwoRingGrid{false, 2, 4, 32, 44, 3.0e-5},
                      TwoRingGrid{false, 2, 4, 31, 44, 3.0e-5},
                      TwoRingGrid{true, 2, 4, 32, 44, 3.0e-5}));

}  // namespace vmecpp
