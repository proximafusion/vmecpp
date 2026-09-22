// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/bootstrap_current/bootstrap_current.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

using vmecpp::vmec_algorithm_constants::kGaussLegendreAbscissae10;
using vmecpp::vmec_algorithm_constants::kGaussLegendreWeights10;
using vmecpp::vmec_algorithm_constants::kVacuumPermeability;

namespace vmecpp {

namespace {

constexpr double kElementaryCharge = 1.602176634e-19;

// the profiles are given in 1e20 m^-3 and keV
constexpr double kDensityUnit = 1.0e20;
constexpr double kTemperatureUnit = 1.0e3;

}  // namespace

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

double KineticPressure(const BootstrapProfiles& profiles, double rho) {
  double ne = 0.0;
  double dne = 0.0;
  double te = 0.0;
  double dte = 0.0;
  double ti = 0.0;
  double dti = 0.0;
  EvaluatePowerSeries(profiles.ne, rho, ne, dne);
  EvaluatePowerSeries(profiles.te, rho, te, dte);
  EvaluatePowerSeries(profiles.ti, rho, ti, dti);
  const double ni = ne / profiles.zeff;
  return kVacuumPermeability * kElementaryCharge * kDensityUnit *
         kTemperatureUnit * (ne * te + ni * ti);
}

void SurfaceFieldMoments(std::span<const double> b, std::span<const double> w,
                         double& m_b_max, double& m_b_min, double& m_b2_avg,
                         double& m_b_inv_avg) {
  double w_sum = 0.0;
  double b2_sum = 0.0;
  double b_inv_sum = 0.0;
  m_b_max = -std::numeric_limits<double>::infinity();
  m_b_min = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < b.size(); ++k) {
    w_sum += w[k];
    b2_sum += w[k] * b[k] * b[k];
    b_inv_sum += w[k] / b[k];
    m_b_max = std::max(m_b_max, b[k]);
    m_b_min = std::min(m_b_min, b[k]);
  }
  m_b2_avg = b2_sum / w_sum;
  m_b_inv_avg = b_inv_sum / w_sum;
}

double TrappedFraction(std::span<const double> b, std::span<const double> w) {
  double b_max = 0.0;
  double b_min = 0.0;
  double b2_avg = 0.0;
  double b_inv_avg = 0.0;
  SurfaceFieldMoments(b, w, b_max, b_min, b2_avg, b_inv_avg);

  double w_sum = 0.0;
  for (std::size_t k = 0; k < w.size(); ++k) {
    w_sum += w[k];
  }

  // lambda = (1 - t^2) / B_max removes the inverse square root that
  // <sqrt(1 - lambda B)> develops at lambda B_max = 1 for a nearly uniform B.
  constexpr int kPanels = 8;
  const double lambda_max = 1.0 / b_max;
  const double h = 1.0 / kPanels;
  double integral = 0.0;
  for (int panel = 0; panel < kPanels; ++panel) {
    for (std::size_t node = 0; node < kGaussLegendreAbscissae10.size();
         ++node) {
      const double t =
          h * (panel + 0.5 * (1.0 + kGaussLegendreAbscissae10[node]));
      const double lambda = lambda_max * (1.0 - t * t);
      double avg = 0.0;
      for (std::size_t k = 0; k < b.size(); ++k) {
        avg += w[k] * std::sqrt(std::max(0.0, 1.0 - lambda * b[k]));
      }
      avg /= w_sum;
      integral += 0.5 * h * kGaussLegendreWeights10[node] * lambda * 2.0 *
                  lambda_max * t / avg;
    }
  }
  return 1.0 - 0.75 * b2_avg * integral;
}

RedlCoefficients ComputeRedlCoefficients(double f_t, double nu_e, double nu_i,
                                         double zeff) {
  RedlCoefficients c;
  c.nu_e = nu_e;
  c.nu_i = nu_i;
  const double z = zeff;
  const double sqrt_nu_e = std::sqrt(nu_e);
  const double sqrt_nu_i = std::sqrt(nu_i);

  // equation (11)
  const double x31 =
      f_t / (1.0 + 0.67 * (1.0 - 0.7 * f_t) * sqrt_nu_e / (0.56 + 0.44 * z) +
             (0.52 + 0.086 * sqrt_nu_e) * (1.0 + 0.87 * f_t) * nu_e /
                 (1.0 + 1.13 * std::sqrt(z - 1.0)));
  // equation (10)
  const double z_fac = std::pow(z, 1.2) - 0.71;
  c.l31 = (1.0 + 0.15 / z_fac) * x31 - 0.22 / z_fac * x31 * x31 +
          0.01 / z_fac * std::pow(x31, 3) + 0.06 / z_fac * std::pow(x31, 4);

  // equation (14)
  const double x32e =
      f_t /
      (1.0 + 0.23 * (1.0 - 0.96 * f_t) * sqrt_nu_e / std::sqrt(z) +
       0.13 * (1.0 - 0.38 * f_t) * nu_e / (z * z) *
           (std::sqrt(1.0 + 2.0 * std::sqrt(z - 1.0)) +
            f_t * f_t *
                std::sqrt((0.075 + 0.25 * (z - 1.0) * (z - 1.0)) * nu_e)));
  // equation (13)
  const double x32e4 = std::pow(x32e, 4);
  const double f32ee =
      (0.1 + 0.6 * z) * (x32e - x32e4) /
          (z * (0.77 + 0.63 * (1.0 + std::pow(z - 1.0, 1.1)))) +
      0.7 / (1.0 + 0.2 * z) *
          (x32e * x32e - x32e4 - 1.2 * (std::pow(x32e, 3) - x32e4)) +
      1.3 / (1.0 + 0.5 * z) * x32e4;

  // equation (16)
  const double x32ei =
      f_t / (1.0 +
             0.87 * (1.0 + 0.39 * f_t) * sqrt_nu_e /
                 (1.0 + 2.95 * (z - 1.0) * (z - 1.0)) +
             1.53 * (1.0 - 0.37 * f_t) * nu_e * (2.0 + 0.375 * (z - 1.0)));
  // equation (15)
  const double x32ei4 = std::pow(x32ei, 4);
  const double f32ei =
      -(0.4 + 1.93 * z) / (z * (0.8 + 0.6 * z)) * (x32ei - x32ei4) +
      5.5 / (1.5 + 2.0 * z) *
          (x32ei * x32ei - x32ei4 - 0.8 * (std::pow(x32ei, 3) - x32ei4)) -
      1.3 / (1.0 + 0.5 * z) * x32ei4;

  // equations (12) and (19)
  c.l32 = f32ee + f32ei;
  c.l34 = c.l31;

  // equations (20) and (21)
  const double alpha0 =
      -(0.62 + 0.055 * (z - 1.0)) * (1.0 - f_t) /
      ((0.53 + 0.17 * (z - 1.0)) *
       (1.0 - (0.31 - 0.065 * (z - 1.0)) * f_t - 0.25 * f_t * f_t));
  const double nu_i2_ft6 = nu_i * nu_i * std::pow(f_t, 6);
  c.alpha =
      ((alpha0 + 0.7 * z * std::sqrt(f_t * nu_i)) / (1.0 + 0.18 * sqrt_nu_i) -
       0.002 * nu_i2_ft6) /
      (1.0 + 0.004 * nu_i2_ft6);
  return c;
}

double RedlJDotB(const BootstrapProfiles& profiles,
                 const BootstrapSurface& surface, double rho, double psi_edge,
                 int nfp, RedlCoefficients* m_coefficients) {
  double ne = 0.0;
  double dne = 0.0;
  double te = 0.0;
  double dte = 0.0;
  double ti = 0.0;
  double dti = 0.0;
  EvaluatePowerSeries(profiles.ne, rho, ne, dne);
  EvaluatePowerSeries(profiles.te, rho, te, dte);
  EvaluatePowerSeries(profiles.ti, rho, ti, dti);
  ne *= kDensityUnit;
  dne *= kDensityUnit;
  te *= kTemperatureUnit;
  dte *= kTemperatureUnit;
  ti *= kTemperatureUnit;
  dti *= kTemperatureUnit;

  const double zeff = profiles.zeff;
  const double ni = ne / zeff;
  const double epsilon =
      (surface.b_max - surface.b_min) / (surface.b_max + surface.b_min);
  const int helicity_big_n = profiles.helicity_n * nfp;
  const double iota_minus_n = surface.iota - helicity_big_n;

  // no trapped particles, no bootstrap current
  if (!(epsilon > 0.0) || !(surface.f_t > 0.0)) {
    if (m_coefficients != nullptr) {
      *m_coefficients = RedlCoefficients();
    }
    return 0.0;
  }

  // Sauter et al. (1999), equations (18b) to (18e)
  const double ln_lambda_e = 31.3 - std::log(std::sqrt(ne) / te);
  const double ln_lambda_ii =
      30.0 - std::log(std::pow(zeff, 3) * std::sqrt(ni) / std::pow(ti, 1.5));
  const double r_major =
      (surface.g + surface.iota * surface.i) * surface.b_inv_avg;
  const double geometry_factor = std::abs(r_major / iota_minus_n);
  const double epsilon32 = std::pow(epsilon, 1.5);
  const double nu_e = geometry_factor * 6.921e-18 * ne * zeff * ln_lambda_e /
                      (te * te * epsilon32);
  const double nu_i = geometry_factor * 4.90e-18 * ni * std::pow(zeff, 4) *
                      ln_lambda_ii / (ti * ti * epsilon32);

  const RedlCoefficients c =
      ComputeRedlCoefficients(surface.f_t, nu_e, nu_i, zeff);
  if (m_coefficients != nullptr) {
    *m_coefficients = c;
  }

  const double pe = ne * te;
  const double pi = ni * ti;
  const double denominator = psi_edge * iota_minus_n;
  const double dnds_term = -surface.g * kElementaryCharge * (pe + pi) * c.l31 *
                           (dne / ne) / denominator;
  const double dteds_term = -surface.g * kElementaryCharge * pe *
                            (c.l31 + c.l32) * (dte / te) / denominator;
  const double dtids_term = -surface.g * kElementaryCharge * pi *
                            (c.l31 + c.l34 * c.alpha) * (dti / ti) /
                            denominator;
  return dnds_term + dteds_term + dtids_term;
}

void IntegrateBootstrapCurrent(std::span<const double> j_dot_b,
                               std::span<const double> g,
                               std::span<const double> dvds, double delta_s,
                               int sign_of_jacobian, std::span<double> m_buco) {
  double running = 0.0;
  double previous = 0.0;
  for (std::size_t j = 0; j < j_dot_b.size(); ++j) {
    const double integrand = sign_of_jacobian * kVacuumPermeability *
                             j_dot_b[j] * dvds[j] / (g[j] * g[j]);
    if (j == 0) {
      // the first half-grid point sits delta_s / 2 from the axis
      running = integrand * 0.5 * delta_s;
    } else {
      running += 0.5 * (previous + integrand) * delta_s;
    }
    previous = integrand;
    m_buco[j] = g[j] * running;
  }
}

}  // namespace vmecpp
