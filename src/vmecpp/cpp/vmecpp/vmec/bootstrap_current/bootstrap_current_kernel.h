// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_BOOTSTRAP_CURRENT_BOOTSTRAP_CURRENT_KERNEL_H_
#define VMECPP_VMEC_BOOTSTRAP_CURRENT_BOOTSTRAP_CURRENT_KERNEL_H_

// Header-only kernels of the bootstrap closure on one flux surface, shared by
// the solver (bootstrap_current.cc) and the local force composition that the
// exact derivatives differentiate (local_force_composition.h). Raw pointers
// and no allocation, so the composition stays a single differentiable map.

#include <algorithm>
#include <cmath>
#include <limits>

#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

namespace vmecpp {

struct RedlCoefficients {
  double nu_e = 0.0;
  double nu_i = 0.0;
  double l31 = 0.0;
  double l32 = 0.0;
  double l34 = 0.0;
  double alpha = 0.0;
};

// Kinetic values of one surface in SI units: densities in m^-3, temperatures
// in eV, derivatives with respect to the normalized toroidal flux.
struct KineticPoint {
  double ne = 0.0;
  double dne = 0.0;
  double te = 0.0;
  double dte = 0.0;
  double ti = 0.0;
  double dti = 0.0;
};

// B_max, B_min, <B^2>, <1/B> and the weight sum from n surface samples b with
// quadrature weights w (Jacobian times integration weight; a common sign or
// scale of w cancels in the averages).
inline void SurfaceFieldMomentsKernel(const double* b, const double* w, int n,
                                      double& m_b_max, double& m_b_min,
                                      double& m_b2_avg, double& m_b_inv_avg,
                                      double& m_w_sum) {
  double w_sum = 0.0;
  double b2_sum = 0.0;
  double b_inv_sum = 0.0;
  double b_max = -std::numeric_limits<double>::infinity();
  double b_min = std::numeric_limits<double>::infinity();
  for (int k = 0; k < n; ++k) {
    w_sum += w[k];
    b2_sum += w[k] * b[k] * b[k];
    b_inv_sum += w[k] / b[k];
    b_max = std::max(b_max, b[k]);
    b_min = std::min(b_min, b[k]);
  }
  m_b_max = b_max;
  m_b_min = b_min;
  m_b2_avg = b2_sum / w_sum;
  m_b_inv_avg = b_inv_sum / w_sum;
  m_w_sum = w_sum;
}

// f_t = 1 - 3/4 <B^2> \int_0^{1/B_max} \lambda d\lambda / <\sqrt{1 - \lambda
// B}> with \lambda = (1 - t^2) / B_max, which removes the inverse square root
// <\sqrt{1 - \lambda B}> develops at \lambda B_max = 1 for a nearly uniform B.
inline double TrappedFractionKernel(const double* b, const double* w, int n,
                                    double b_max, double b2_avg, double w_sum) {
  constexpr int kPanels = 8;
  const double lambda_max = 1.0 / b_max;
  const double h = 1.0 / kPanels;
  double integral = 0.0;
  for (int panel = 0; panel < kPanels; ++panel) {
    for (int node = 0; node < 10; ++node) {
      const double t =
          h *
          (panel +
           0.5 * (1.0 +
                  vmec_algorithm_constants::kGaussLegendreAbscissae10[node]));
      const double lambda = lambda_max * (1.0 - t * t);
      double avg = 0.0;
      for (int k = 0; k < n; ++k) {
        avg += w[k] * std::sqrt(std::max(0.0, 1.0 - lambda * b[k]));
      }
      avg /= w_sum;
      integral += 0.5 * h *
                  vmec_algorithm_constants::kGaussLegendreWeights10[node] *
                  lambda * 2.0 * lambda_max * t / avg;
    }
  }
  return 1.0 - 0.75 * b2_avg * integral;
}

// Redl et al., Phys. Plasmas 28, 022502 (2021), equations (10) to (21).
inline RedlCoefficients ComputeRedlCoefficientsKernel(double f_t, double nu_e,
                                                      double nu_i,
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
          0.01 / z_fac * (x31 * x31 * x31) +
          0.06 / z_fac * (x31 * x31 * x31 * x31);

  // equation (14)
  const double x32e =
      f_t /
      (1.0 + 0.23 * (1.0 - 0.96 * f_t) * sqrt_nu_e / std::sqrt(z) +
       0.13 * (1.0 - 0.38 * f_t) * nu_e / (z * z) *
           (std::sqrt(1.0 + 2.0 * std::sqrt(z - 1.0)) +
            f_t * f_t *
                std::sqrt((0.075 + 0.25 * (z - 1.0) * (z - 1.0)) * nu_e)));
  // equation (13)
  const double x32e2 = x32e * x32e;
  const double x32e3 = x32e2 * x32e;
  const double x32e4 = x32e2 * x32e2;
  const double f32ee =
      (0.1 + 0.6 * z) * (x32e - x32e4) /
          (z * (0.77 + 0.63 * (1.0 + std::pow(z - 1.0, 1.1)))) +
      0.7 / (1.0 + 0.2 * z) * (x32e2 - x32e4 - 1.2 * (x32e3 - x32e4)) +
      1.3 / (1.0 + 0.5 * z) * x32e4;

  // equation (16)
  const double x32ei =
      f_t / (1.0 +
             0.87 * (1.0 + 0.39 * f_t) * sqrt_nu_e /
                 (1.0 + 2.95 * (z - 1.0) * (z - 1.0)) +
             1.53 * (1.0 - 0.37 * f_t) * nu_e * (2.0 + 0.375 * (z - 1.0)));
  // equation (15)
  const double x32ei2 = x32ei * x32ei;
  const double x32ei3 = x32ei2 * x32ei;
  const double x32ei4 = x32ei2 * x32ei2;
  const double f32ei =
      -(0.4 + 1.93 * z) / (z * (0.8 + 0.6 * z)) * (x32ei - x32ei4) +
      5.5 / (1.5 + 2.0 * z) * (x32ei2 - x32ei4 - 0.8 * (x32ei3 - x32ei4)) -
      1.3 / (1.0 + 0.5 * z) * x32ei4;

  // equations (12) and (19)
  c.l32 = f32ee + f32ei;
  c.l34 = c.l31;

  // equations (20) and (21)
  const double alpha0 =
      -(0.62 + 0.055 * (z - 1.0)) * (1.0 - f_t) /
      ((0.53 + 0.17 * (z - 1.0)) *
       (1.0 - (0.31 - 0.065 * (z - 1.0)) * f_t - 0.25 * f_t * f_t));
  const double f_t6 = f_t * f_t * f_t * f_t * f_t * f_t;
  const double nu_i2_ft6 = nu_i * nu_i * f_t6;
  c.alpha =
      ((alpha0 + 0.7 * z * std::sqrt(f_t * nu_i)) / (1.0 + 0.18 * sqrt_nu_i) -
       0.002 * nu_i2_ft6) /
      (1.0 + 0.004 * nu_i2_ft6);
  return c;
}

// <J . B> in A T / m^2 on one surface, the Redl closure as SIMSOPT's
// j_dot_B_Redl evaluates it on a VMEC equilibrium: g = bvco, i = buco,
// iota = iotas, psi_edge = -phi_edge / (2 pi) with phi_edge the enclosed
// toroidal flux of the wout file, R = (G + \iota I) <1/B>,
// \epsilon = (B_max - B_min) / (B_max + B_min), and the helicity enters as
// \iota - N with N = helicity_n nfp. Zero without trapped particles.
inline double RedlJDotBKernel(const KineticPoint& kinetic, double zeff,
                              int helicity_big_n, double g, double i,
                              double iota, double f_t, double b_max,
                              double b_min, double b_inv_avg, double psi_edge,
                              RedlCoefficients* m_coefficients) {
  constexpr double kElementaryCharge = 1.602176634e-19;
  const double ne = kinetic.ne;
  const double te = kinetic.te;
  const double ti = kinetic.ti;
  const double ni = ne / zeff;
  const double epsilon = (b_max - b_min) / (b_max + b_min);
  const double iota_minus_n = iota - helicity_big_n;

  if (!(epsilon > 0.0) || !(f_t > 0.0)) {
    if (m_coefficients != nullptr) {
      *m_coefficients = RedlCoefficients();
    }
    return 0.0;
  }

  // Sauter et al. (1999), equations (18b) to (18e)
  const double ln_lambda_e = 31.3 - std::log(std::sqrt(ne) / te);
  const double ln_lambda_ii =
      30.0 - std::log(zeff * zeff * zeff * std::sqrt(ni) / std::pow(ti, 1.5));
  const double r_major = (g + iota * i) * b_inv_avg;
  const double geometry_factor = std::abs(r_major / iota_minus_n);
  const double epsilon32 = std::pow(epsilon, 1.5);
  const double nu_e = geometry_factor * 6.921e-18 * ne * zeff * ln_lambda_e /
                      (te * te * epsilon32);
  const double nu_i = geometry_factor * 4.90e-18 * ni * zeff * zeff * zeff *
                      zeff * ln_lambda_ii / (ti * ti * epsilon32);

  const RedlCoefficients c =
      ComputeRedlCoefficientsKernel(f_t, nu_e, nu_i, zeff);
  if (m_coefficients != nullptr) {
    *m_coefficients = c;
  }

  const double pe = ne * te;
  const double pi = ni * ti;
  const double denominator = psi_edge * iota_minus_n;
  const double dnds_term = -g * kElementaryCharge * (pe + pi) * c.l31 *
                           (kinetic.dne / ne) / denominator;
  const double dteds_term = -g * kElementaryCharge * pe * (c.l31 + c.l32) *
                            (kinetic.dte / te) / denominator;
  const double dtids_term = -g * kElementaryCharge * pi *
                            (c.l31 + c.l34 * c.alpha) * (kinetic.dti / ti) /
                            denominator;
  return dnds_term + dteds_term + dtids_term;
}

// Integrates <J . B> on the half grid, s_j = (j + 1/2) delta_s, into the
// enclosed toroidal current <B_theta>(s_j) that currH prescribes, using
// \mu_0 <J . B> = signgs (G I' - I G') / dVds, i.e.
// (I / G)' = signgs \mu_0 <J . B> dVds / G^2 with I(0) = 0.
inline void IntegrateBootstrapCurrentKernel(const double* j_dot_b,
                                            const double* g, const double* dvds,
                                            int n, double delta_s,
                                            int sign_of_jacobian,
                                            double* m_buco) {
  double running = 0.0;
  double previous = 0.0;
  for (int j = 0; j < n; ++j) {
    const double integrand = sign_of_jacobian *
                             vmec_algorithm_constants::kVacuumPermeability *
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

// The angular grid of one flux surface. The poloidal grid has n_theta_even
// points theta_l = 2 \pi l / n_theta_even, of which the first n_theta_eff are
// stored: all of them with lasym, and theta in [0, \pi] in a
// stellarator-symmetric run, whose values on (\pi, 2 \pi) follow from
// f(\theta, v) = f(-\theta, -v). The toroidal grid has n_zeta points
// v_k = 2 \pi k / n_zeta over one field period, v = nfp \zeta. Values are
// stored zeta-major at k * n_theta_eff + l. The tables hold cos(m \theta_l),
// sin(m \theta_l) at m * n_theta_even + l for m <= n_theta_even / 2, and
// cos(n v_k), sin(n v_k) at n * n_zeta + k for n <= n_zeta / 2.
struct SurfaceGrid {
  int n_theta_even = 0;
  int n_theta_eff = 0;
  int n_zeta = 0;
  const double* cos_mt = nullptr;
  const double* sin_mt = nullptr;
  const double* cos_nv = nullptr;
  const double* sin_nv = nullptr;
};

inline int SurfaceGridMMax(const SurfaceGrid& grid) {
  return grid.n_theta_even / 2;
}
inline int SurfaceGridNMax(const SurfaceGrid& grid) { return grid.n_zeta / 2; }

// Fills the tables of SurfaceGrid; the arrays hold
// (n_theta_even / 2 + 1) * n_theta_even and (n_zeta / 2 + 1) * n_zeta doubles.
inline void FillSurfaceGridTables(int n_theta_even, int n_zeta,
                                  double* m_cos_mt, double* m_sin_mt,
                                  double* m_cos_nv, double* m_sin_nv) {
  constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
  for (int m = 0; m <= n_theta_even / 2; ++m) {
    for (int l = 0; l < n_theta_even; ++l) {
      const double angle = kTwoPi * ((m * l) % n_theta_even) / n_theta_even;
      m_cos_mt[m * n_theta_even + l] = std::cos(angle);
      m_sin_mt[m * n_theta_even + l] = std::sin(angle);
    }
  }
  for (int n = 0; n <= n_zeta / 2; ++n) {
    for (int k = 0; k < n_zeta; ++k) {
      const double angle = kTwoPi * ((n * k) % n_zeta) / n_zeta;
      m_cos_nv[n * n_zeta + k] = std::cos(angle);
      m_sin_nv[n * n_zeta + k] = std::sin(angle);
    }
  }
}

// The value at the full-grid point (l, k), l < n_theta_even, reflected into
// the stored half for a stellarator-symmetric grid.
inline double SurfaceGridValue(const double* f, const SurfaceGrid& grid, int l,
                               int k) {
  if (l < grid.n_theta_eff) {
    return f[k * grid.n_theta_eff + l];
  }
  const int l_mirror = grid.n_theta_even - l;
  const int k_mirror = (grid.n_zeta - k) % grid.n_zeta;
  return f[k_mirror * grid.n_theta_eff + l_mirror];
}

// Coefficients of the trigonometric interpolant of the grid values f,
// f(\theta, v) = \sum_{m=0}^{M} \sum_{n=-N}^{N}
//   [c_{mn} cos(m \theta - n v) + s_{mn} sin(m \theta - n v)],
// M = n_theta_even / 2, N = n_zeta / 2, which reproduces f at every grid
// point. The coefficients are stored at m * (2 N + 1) + n + N, and those with
// m = 0, n < 0 are zero. work holds 2 (M + 1) n_zeta doubles.
inline void SurfaceInterpolantKernel(const double* f, const SurfaceGrid& grid,
                                     double* work, double* m_c, double* m_s) {
  const int m_max = SurfaceGridMMax(grid);
  const int n_max = SurfaceGridNMax(grid);
  const int n_count = 2 * n_max + 1;
  const int nt = grid.n_theta_even;
  const int nz = grid.n_zeta;
  double* partial_cos = work;
  double* partial_sin = work + (m_max + 1) * nz;
  for (int m = 0; m <= m_max; ++m) {
    for (int k = 0; k < nz; ++k) {
      double a = 0.0;
      double b = 0.0;
      for (int l = 0; l < nt; ++l) {
        const double value = SurfaceGridValue(f, grid, l, k);
        a += value * grid.cos_mt[m * nt + l];
        b += value * grid.sin_mt[m * nt + l];
      }
      partial_cos[m * nz + k] = a;
      partial_sin[m * nz + k] = b;
    }
  }
  const double norm = 1.0 / (static_cast<double>(nt) * nz);
  for (int m = 0; m <= m_max; ++m) {
    // the Nyquist modes of an even grid enter with half weight
    const double weight_m = (m == nt / 2) ? 0.5 : 1.0;
    for (int n = -n_max; n <= n_max; ++n) {
      const int index = m * n_count + n + n_max;
      if (m == 0 && n < 0) {
        m_c[index] = 0.0;
        m_s[index] = 0.0;
        continue;
      }
      const int n_abs = n < 0 ? -n : n;
      const double sign_n = n < 0 ? -1.0 : 1.0;
      const double weight_n = (nz % 2 == 0 && n_abs == nz / 2) ? 0.5 : 1.0;
      // (m, n) stands for itself and (-m, -n), except the constant
      const double pair = (m == 0 && n == 0) ? 1.0 : 2.0;
      double sum_cos = 0.0;
      double sum_sin = 0.0;
      for (int k = 0; k < nz; ++k) {
        const double cos_nv = grid.cos_nv[n_abs * nz + k];
        const double sin_nv = sign_n * grid.sin_nv[n_abs * nz + k];
        const double a = partial_cos[m * nz + k];
        const double b = partial_sin[m * nz + k];
        sum_cos += a * cos_nv + b * sin_nv;
        sum_sin += b * cos_nv - a * sin_nv;
      }
      const double scale = pair * weight_m * weight_n * norm;
      m_c[index] = scale * sum_cos;
      m_s[index] = scale * sum_sin;
    }
  }
}

// Value, gradient and Hessian at (\theta, v) of the interpolant with the
// coefficients of SurfaceInterpolantKernel. work holds 2 (m_max + 1) +
// 2 (n_max + 1) doubles.
inline void EvaluateSurfaceInterpolant(const double* c, const double* s,
                                       int m_max, int n_max, double theta,
                                       double v, double* work, double& m_f,
                                       double& m_f_t, double& m_f_v,
                                       double& m_f_tt, double& m_f_tv,
                                       double& m_f_vv) {
  double* cos_m = work;
  double* sin_m = work + (m_max + 1);
  double* cos_n = work + 2 * (m_max + 1);
  double* sin_n = cos_n + (n_max + 1);
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  const double cos_v = std::cos(v);
  const double sin_v = std::sin(v);
  cos_m[0] = 1.0;
  sin_m[0] = 0.0;
  for (int m = 1; m <= m_max; ++m) {
    cos_m[m] = cos_m[m - 1] * cos_theta - sin_m[m - 1] * sin_theta;
    sin_m[m] = sin_m[m - 1] * cos_theta + cos_m[m - 1] * sin_theta;
  }
  cos_n[0] = 1.0;
  sin_n[0] = 0.0;
  for (int n = 1; n <= n_max; ++n) {
    cos_n[n] = cos_n[n - 1] * cos_v - sin_n[n - 1] * sin_v;
    sin_n[n] = sin_n[n - 1] * cos_v + cos_n[n - 1] * sin_v;
  }
  const int n_count = 2 * n_max + 1;
  double f = 0.0;
  double f_t = 0.0;
  double f_v = 0.0;
  double f_tt = 0.0;
  double f_tv = 0.0;
  double f_vv = 0.0;
  for (int m = 0; m <= m_max; ++m) {
    for (int n = (m == 0 ? 0 : -n_max); n <= n_max; ++n) {
      const int index = m * n_count + n + n_max;
      const int n_abs = n < 0 ? -n : n;
      const double sign_n = n < 0 ? -1.0 : 1.0;
      const double cos_phase =
          cos_m[m] * cos_n[n_abs] + sign_n * sin_m[m] * sin_n[n_abs];
      const double sin_phase =
          sin_m[m] * cos_n[n_abs] - sign_n * cos_m[m] * sin_n[n_abs];
      const double even = c[index] * cos_phase + s[index] * sin_phase;
      const double odd = -c[index] * sin_phase + s[index] * cos_phase;
      f += even;
      f_t += m * odd;
      f_v -= n * odd;
      f_tt -= m * m * even;
      f_tv += m * n * even;
      f_vv -= n * n * even;
    }
  }
  m_f = f;
  m_f_t = f_t;
  m_f_v = f_v;
  m_f_tt = f_tt;
  m_f_tv = f_tv;
  m_f_vv = f_vv;
}

// The extremum of the interpolant next to the extremum of the grid values f:
// the maximum for sense = +1, the minimum for sense = -1. Newton iterations
// from the grid point, steps limited to half a grid cell and halved until the
// interpolant does not decrease (increase), so the result is at least the grid
// maximum (at most the grid minimum). At a nondegenerate extremum the value is
// a smooth function of f.
inline double SurfaceExtremumKernel(const double* f, const SurfaceGrid& grid,
                                    const double* c, const double* s,
                                    double* work, double sense) {
  constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
  constexpr int kNewtonIterations = 8;
  constexpr int kHalvings = 12;
  const int m_max = SurfaceGridMMax(grid);
  const int n_max = SurfaceGridNMax(grid);
  const int stored = grid.n_theta_eff * grid.n_zeta;
  int best = 0;
  for (int kl = 1; kl < stored; ++kl) {
    if (sense * f[kl] > sense * f[best]) {
      best = kl;
    }
  }
  const double h_theta = kTwoPi / grid.n_theta_even;
  const double h_v = kTwoPi / grid.n_zeta;
  double theta = h_theta * (best % grid.n_theta_eff);
  double v = h_v * (best / grid.n_theta_eff);
  double value = 0.0;
  double f_t = 0.0;
  double f_v = 0.0;
  double f_tt = 0.0;
  double f_tv = 0.0;
  double f_vv = 0.0;
  EvaluateSurfaceInterpolant(c, s, m_max, n_max, theta, v, work, value, f_t,
                             f_v, f_tt, f_tv, f_vv);
  for (int iteration = 0; iteration < kNewtonIterations; ++iteration) {
    // gradient and Hessian of sense * f, maximized
    const double g_t = sense * f_t;
    const double g_v = sense * f_v;
    const double h_tt = sense * f_tt;
    const double h_tv = sense * f_tv;
    const double h_vv = sense * f_vv;
    double d_theta = 0.0;
    double d_v = 0.0;
    if (grid.n_zeta == 1) {
      d_theta = h_tt < 0.0 ? -g_t / h_tt : (g_t >= 0.0 ? h_theta : -h_theta);
    } else {
      const double det = h_tt * h_vv - h_tv * h_tv;
      if (h_tt < 0.0 && det > 0.0) {
        d_theta = -(h_vv * g_t - h_tv * g_v) / det;
        d_v = -(h_tt * g_v - h_tv * g_t) / det;
      } else {
        const double norm =
            std::max(std::abs(g_t) / h_theta, std::abs(g_v) / h_v);
        const double scale = norm > 0.0 ? 0.5 / norm : 0.0;
        d_theta = scale * g_t;
        d_v = scale * g_v;
      }
    }
    d_theta = std::clamp(d_theta, -0.5 * h_theta, 0.5 * h_theta);
    d_v = std::clamp(d_v, -0.5 * h_v, 0.5 * h_v);
    for (int halving = 0; halving < kHalvings; ++halving) {
      double trial = 0.0;
      double t_t = 0.0;
      double t_v = 0.0;
      double t_tt = 0.0;
      double t_tv = 0.0;
      double t_vv = 0.0;
      EvaluateSurfaceInterpolant(c, s, m_max, n_max, theta + d_theta, v + d_v,
                                 work, trial, t_t, t_v, t_tt, t_tv, t_vv);
      if (sense * trial >= sense * value) {
        theta += d_theta;
        v += d_v;
        value = trial;
        f_t = t_t;
        f_v = t_v;
        f_tt = t_tt;
        f_tv = t_tv;
        f_vv = t_vv;
        break;
      }
      d_theta *= 0.5;
      d_v *= 0.5;
    }
  }
  return value;
}

// Doubles of work SurfaceExtremaKernel slices.
inline int SurfaceExtremaWorkSize(const SurfaceGrid& grid) {
  const int m_count = SurfaceGridMMax(grid) + 1;
  const int n_half = SurfaceGridNMax(grid) + 1;
  const int n_count = 2 * SurfaceGridNMax(grid) + 1;
  return 2 * m_count * grid.n_zeta + 2 * m_count * n_count + 2 * m_count +
         2 * n_half;
}

// B_max and B_min of a flux surface as the extrema of the interpolant of the
// grid values b, which unlike the grid extrema are smooth in b.
inline void SurfaceExtremaKernel(const double* b, const SurfaceGrid& grid,
                                 double* work, double& m_b_max,
                                 double& m_b_min) {
  const int m_count = SurfaceGridMMax(grid) + 1;
  const int n_count = 2 * SurfaceGridNMax(grid) + 1;
  double* transform_work = work;
  double* c = transform_work + 2 * m_count * grid.n_zeta;
  double* s = c + m_count * n_count;
  double* eval_work = s + m_count * n_count;
  SurfaceInterpolantKernel(b, grid, transform_work, c, s);
  m_b_max = SurfaceExtremumKernel(b, grid, c, s, eval_work, 1.0);
  m_b_min = SurfaceExtremumKernel(b, grid, c, s, eval_work, -1.0);
}

// The Redl closure on one half surface from the half-grid fields on grid:
// b and w are |B| and gsqrt * wInt, g = <B_zeta>, i the enclosed current
// currH and iota = chip / phip in the signs of the run, which enter the
// closure in the Fortran convention signgs = -1 of psi_edge_ref and
// helicity_big_n. work holds SurfaceExtremaWorkSize(grid) doubles.
inline double RedlSurfaceJDotB(const double* b, const double* w,
                               const SurfaceGrid& grid, double* work,
                               const KineticPoint& kinetic, double zeff,
                               int helicity_big_n, int sign_of_jacobian,
                               double g, double i, double iota,
                               double psi_edge_ref) {
  const int n = grid.n_theta_eff * grid.n_zeta;
  double b_max = 0.0;
  double b_min = 0.0;
  double b2_avg = 0.0;
  double b_inv_avg = 0.0;
  double w_sum = 0.0;
  SurfaceFieldMomentsKernel(b, w, n, b_max, b_min, b2_avg, b_inv_avg, w_sum);
  SurfaceExtremaKernel(b, grid, work, b_max, b_min);
  const double f_t = TrappedFractionKernel(b, w, n, b_max, b2_avg, w_sum);
  const double sign_to_reference = -sign_of_jacobian;
  return RedlJDotBKernel(kinetic, zeff, helicity_big_n, g,
                         sign_to_reference * i, sign_to_reference * iota, f_t,
                         b_max, b_min, b_inv_avg, psi_edge_ref, nullptr);
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_BOOTSTRAP_CURRENT_BOOTSTRAP_CURRENT_KERNEL_H_
