// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/nestor/nestor.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

// Exterior Neumann problem with a known solution: the external field is that
// of a line of vertical point dipoles along the circular magnetic axis of a
// helically deformed torus, i.e. an interior source with a single-valued
// potential that is harmonic in the whole exterior. The vacuum field that
// NESTOR reconstructs must vanish on the surface, so |B|^2/2 there is the
// solver's error, measured against the external field's own |B_ext|^2/2.
// The field is supplied exactly on the surface (no mgrid interpolation).
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

  // surface points on NESTOR's grid: kl = l * nZeta + k, theta_l = 2 pi l /
  // nThetaEven, zeta_k = 2 pi k / (nfp nZeta)
  Eigen::VectorXd br = Eigen::VectorXd::Zero(s.nZnT);
  Eigen::VectorXd bp = Eigen::VectorXd::Zero(s.nZnT);
  Eigen::VectorXd bz = Eigen::VectorXd::Zero(s.nZnT);
  const int num_dipoles = 96 * nfp;
  const double moment = 1.0 / 96.0;  // total vertical moment 1 A m^2 per period
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
    const double x = R * std::cos(zeta);
    const double y = R * std::sin(zeta);
    double bx = 0.0, by = 0.0, bzc = 0.0;
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
      bzc += f * (3.0 * mdotn * dz / r - moment);
    }
    br[kl] = bx * std::cos(zeta) + by * std::sin(zeta);
    bp[kl] = -bx * std::sin(zeta) + by * std::cos(zeta);
    bz[kl] = bzc;
    const double half_b2 = 0.5 * (bx * bx + by * by + bzc * bzc);
    ref_sum += half_b2 * half_b2;
  }
  const double ref_rms = std::sqrt(ref_sum / s.nZnT);

  MGridProvider mgrid;
  mgrid.SetFixedMagneticField(br, bp, bz);

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
  Nestor nestor(&s, &tp, &mgrid, matrixShare, bvecShare, bSqVacShare,
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
  EXPECT_LT(err_rms / ref_rms, 1.0e-4) << "|B|^2/2 residual relative to |B_ext|^2/2: " << err_rms / ref_rms;
}

}  // namespace vmecpp
