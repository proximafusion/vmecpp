// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/bootstrap_current/bootstrap_current.h"

#include <cmath>

#include "vmecpp/vmec/bootstrap_current/bootstrap_current_kernel.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

using vmecpp::vmec_algorithm_constants::kVacuumPermeability;

namespace vmecpp {

namespace {

constexpr double kElementaryCharge = 1.602176634e-19;

// the profiles are given in 1e20 m^-3 and keV
constexpr double kDensityUnit = 1.0e20;
constexpr double kTemperatureUnit = 1.0e3;

}  // namespace

SurfaceGridTables::SurfaceGridTables(int n_theta_even, int n_theta_eff,
                                     int n_zeta)
    : n_theta_even_(n_theta_even),
      n_theta_eff_(n_theta_eff),
      n_zeta_(n_zeta),
      cos_mt_((n_theta_even / 2 + 1) * n_theta_even),
      sin_mt_((n_theta_even / 2 + 1) * n_theta_even),
      cos_nv_((n_zeta / 2 + 1) * n_zeta),
      sin_nv_((n_zeta / 2 + 1) * n_zeta) {
  FillSurfaceGridTables(n_theta_even, n_zeta, cos_mt_.data(), sin_mt_.data(),
                        cos_nv_.data(), sin_nv_.data());
}

SurfaceGrid SurfaceGridTables::grid() const {
  SurfaceGrid grid;
  grid.n_theta_even = n_theta_even_;
  grid.n_theta_eff = n_theta_eff_;
  grid.n_zeta = n_zeta_;
  grid.cos_mt = cos_mt_.data();
  grid.sin_mt = sin_mt_.data();
  grid.cos_nv = cos_nv_.data();
  grid.sin_nv = sin_nv_.data();
  return grid;
}

void EvaluatePowerSeries(const Eigen::VectorXd& c, double x, double& m_value,
                         double& m_derivative) {
  double value = 0.0;
  double derivative = 0.0;
  for (Eigen::Index k = c.size() - 1; k >= 0; --k) {
    derivative = derivative * x + value;
    value = value * x + c[k];
  }
  m_value = value;
  m_derivative = derivative;
}

KineticPoint EvaluateKineticPoint(const BootstrapProfiles& profiles,
                                  double rho) {
  KineticPoint kinetic;
  EvaluatePowerSeries(profiles.ne, rho, kinetic.ne, kinetic.dne);
  EvaluatePowerSeries(profiles.te, rho, kinetic.te, kinetic.dte);
  EvaluatePowerSeries(profiles.ti, rho, kinetic.ti, kinetic.dti);
  kinetic.ne *= kDensityUnit;
  kinetic.dne *= kDensityUnit;
  kinetic.te *= kTemperatureUnit;
  kinetic.dte *= kTemperatureUnit;
  kinetic.ti *= kTemperatureUnit;
  kinetic.dti *= kTemperatureUnit;
  return kinetic;
}

double KineticPressure(const BootstrapProfiles& profiles, double rho) {
  const KineticPoint kinetic = EvaluateKineticPoint(profiles, rho);
  const double ni = kinetic.ne / profiles.zeff;
  return kVacuumPermeability * kElementaryCharge *
         (kinetic.ne * kinetic.te + ni * kinetic.ti);
}

void SurfaceFieldMoments(std::span<const double> b, std::span<const double> w,
                         double& m_b_max, double& m_b_min, double& m_b2_avg,
                         double& m_b_inv_avg) {
  double w_sum = 0.0;
  SurfaceFieldMomentsKernel(b.data(), w.data(), static_cast<int>(b.size()),
                            m_b_max, m_b_min, m_b2_avg, m_b_inv_avg, w_sum);
}

double TrappedFraction(std::span<const double> b, std::span<const double> w) {
  double b_max = 0.0;
  double b_min = 0.0;
  double b2_avg = 0.0;
  double b_inv_avg = 0.0;
  double w_sum = 0.0;
  const int n = static_cast<int>(b.size());
  SurfaceFieldMomentsKernel(b.data(), w.data(), n, b_max, b_min, b2_avg,
                            b_inv_avg, w_sum);
  return TrappedFractionKernel(b.data(), w.data(), n, b_max, b2_avg, w_sum);
}

RedlCoefficients ComputeRedlCoefficients(double f_t, double nu_e, double nu_i,
                                         double zeff) {
  return ComputeRedlCoefficientsKernel(f_t, nu_e, nu_i, zeff);
}

double RedlJDotB(const BootstrapProfiles& profiles,
                 const BootstrapSurface& surface, double rho, double psi_edge,
                 int nfp, RedlCoefficients* m_coefficients) {
  const KineticPoint kinetic = EvaluateKineticPoint(profiles, rho);
  return RedlJDotBKernel(kinetic, profiles.zeff, profiles.helicity_n * nfp,
                         surface.g, surface.i, surface.iota, surface.f_t,
                         surface.b_max, surface.b_min, surface.b_inv_avg,
                         psi_edge, m_coefficients);
}

void IntegrateBootstrapCurrent(std::span<const double> j_dot_b,
                               std::span<const double> g,
                               std::span<const double> dvds, double delta_s,
                               int sign_of_jacobian, std::span<double> m_buco) {
  IntegrateBootstrapCurrentKernel(j_dot_b.data(), g.data(), dvds.data(),
                                  static_cast<int>(j_dot_b.size()), delta_s,
                                  sign_of_jacobian, m_buco.data());
}

}  // namespace vmecpp
