// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_BOOTSTRAP_CURRENT_BOOTSTRAP_CURRENT_H_
#define VMECPP_VMEC_BOOTSTRAP_CURRENT_BOOTSTRAP_CURRENT_H_

#include <Eigen/Dense>
#include <span>
#include <vector>

#include "vmecpp/vmec/bootstrap_current/bootstrap_current_kernel.h"

namespace vmecpp {

// Kinetic profiles of the bootstrap closure, power series in the normalized
// toroidal flux.
struct BootstrapProfiles {
  // electron density in 1e20 m^-3
  Eigen::VectorXd ne;
  // electron temperature in keV
  Eigen::VectorXd te;
  // ion temperature in keV
  Eigen::VectorXd ti;
  // effective ion charge; the ion density is n_e / zeff
  double zeff = 1.0;
  // helicity of the quasi-symmetry, N = helicity_n * nfp; 0 for
  // quasi-axisymmetry and tokamaks
  int helicity_n = 0;
};

// Flux-surface quantities the closure reads on one half-grid surface, in the
// conventions of the wout file: g = bvco, i = buco, iota = iotas.
struct BootstrapSurface {
  double g = 0.0;
  double i = 0.0;
  double iota = 0.0;
  // effective trapped fraction
  double f_t = 0.0;
  double b_max = 0.0;
  double b_min = 0.0;
  // flux-surface averages <B^2> and <1/B>
  double b2_avg = 0.0;
  double b_inv_avg = 0.0;
};

// Owns the tables of a SurfaceGrid.
class SurfaceGridTables {
 public:
  SurfaceGridTables() = default;
  SurfaceGridTables(int n_theta_even, int n_theta_eff, int n_zeta);

  SurfaceGrid grid() const;

 private:
  int n_theta_even_ = 0;
  int n_theta_eff_ = 0;
  int n_zeta_ = 0;
  std::vector<double> cos_mt_;
  std::vector<double> sin_mt_;
  std::vector<double> cos_nv_;
  std::vector<double> sin_nv_;
};

// Value and derivative of \sum_k c_k x^k at x.
void EvaluatePowerSeries(const Eigen::VectorXd& c, double x, double& m_value,
                         double& m_derivative);

// The kinetic profiles at the normalized toroidal flux rho in SI units.
KineticPoint EvaluateKineticPoint(const BootstrapProfiles& profiles,
                                  double rho);

// \mu_0 p at the normalized toroidal flux rho with
// p = e (n_e T_e + n_i T_i), n_i = n_e / zeff, in the pressure units of the
// mass profile.
double KineticPressure(const BootstrapProfiles& profiles, double rho);

// B_max, B_min, <B^2> and <1/B> from the surface samples b with the quadrature
// weights w (Jacobian times integration weight; a common sign or scale of w
// cancels).
void SurfaceFieldMoments(std::span<const double> b, std::span<const double> w,
                         double& m_b_max, double& m_b_min, double& m_b2_avg,
                         double& m_b_inv_avg);

// f_t = 1 - 3/4 <B^2> \int_0^{1/B_max} \lambda d\lambda / <\sqrt{1 - \lambda
// B}>
double TrappedFraction(std::span<const double> b, std::span<const double> w);

// Redl et al., Phys. Plasmas 28, 022502 (2021), equations (10) to (21).
RedlCoefficients ComputeRedlCoefficients(double f_t, double nu_e, double nu_i,
                                         double zeff);

// <J . B> in A T / m^2 on one surface at the normalized toroidal flux rho,
// the Redl closure as SIMSOPT's j_dot_B_Redl evaluates it on a VMEC
// equilibrium: psi_edge = -phi_edge / (2 \pi) with phi_edge the enclosed
// toroidal flux of the wout file, R = (G + \iota I) <1/B>,
// \epsilon = (B_max - B_min) / (B_max + B_min), and the quasi-symmetry helicity
// enters as \iota - N with N = helicity_n nfp.
double RedlJDotB(const BootstrapProfiles& profiles,
                 const BootstrapSurface& surface, double rho, double psi_edge,
                 int nfp, RedlCoefficients* m_coefficients = nullptr);

// Integrates <J . B> on the half grid, s_j = (j + 1/2) delta_s, into the
// enclosed toroidal current <B_theta>(s_j) that currH prescribes, using
// \mu_0 <J . B> = signgs (G I' - I G') / dVds, i.e.
// (I / G)' = signgs \mu_0 <J . B> dVds / G^2 with I(0) = 0.
void IntegrateBootstrapCurrent(std::span<const double> j_dot_b,
                               std::span<const double> g,
                               std::span<const double> dvds, double delta_s,
                               int sign_of_jacobian, std::span<double> m_buco);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_BOOTSTRAP_CURRENT_BOOTSTRAP_CURRENT_H_
