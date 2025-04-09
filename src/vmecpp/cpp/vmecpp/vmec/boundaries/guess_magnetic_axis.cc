// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/boundaries/guess_magnetic_axis.h"

#include <algorithm>  // min_element, max_element
#include <vector>

namespace vmecpp {

RecomputeAxisWorkspace RecomputeMagneticAxisToFixJacobianSign(
    int number_of_flux_surfaces, int sign_of_jacobian, const Sizes& s,
    const FourierBasisFastPoloidal& t, const std::vector<double>& rbcc,
    const std::vector<double>& rbss, const std::vector<double>& rbsc,
    const std::vector<double>& rbcs, const std::vector<double>& zbsc,
    const std::vector<double>& zbcs, const std::vector<double>& zbcc,
    const std::vector<double>& zbss, const std::vector<double>& raxis_c,
    const std::vector<double>& raxis_s, const std::vector<double>& zaxis_s,
    const std::vector<double>& zaxis_c) {
  RecomputeAxisWorkspace w;

  w.r_axis.resize(s.nZeta);
  w.z_axis.resize(s.nZeta);

  w.r_lcfs.resize(s.nZeta);
  w.z_lcfs.resize(s.nZeta);
  w.d_r_d_theta_lcfs.resize(s.nZeta);
  w.d_z_d_theta_lcfs.resize(s.nZeta);
  for (int k = 0; k < s.nZeta; ++k) {
    w.r_lcfs[k].resize(s.nThetaEven);
    w.z_lcfs[k].resize(s.nThetaEven);
    w.d_r_d_theta_lcfs[k].resize(s.nThetaEven);
    w.d_z_d_theta_lcfs[k].resize(s.nThetaEven);
  }  // k

  w.r_half.resize(s.nZeta);
  w.z_half.resize(s.nZeta);
  w.d_r_d_theta_half.resize(s.nZeta);
  w.d_z_d_theta_half.resize(s.nZeta);
  w.d_r_d_s_half.resize(s.nZeta);
  w.d_z_d_s_half.resize(s.nZeta);
  for (int k = 0; k < s.nZeta; ++k) {
    w.r_half[k].resize(s.nThetaEven);
    w.z_half[k].resize(s.nThetaEven);
    w.d_r_d_theta_half[k].resize(s.nThetaEven);
    w.d_z_d_theta_half[k].resize(s.nThetaEven);
    w.d_r_d_s_half[k].resize(s.nThetaEven);
    w.d_z_d_s_half[k].resize(s.nThetaEven);
  }  // k

  w.tau0.resize(s.nZeta);
  w.tau.resize(s.nZeta);
  for (int k = 0; k < s.nZeta; ++k) {
    w.tau0[k].resize(s.nThetaEven);
    w.tau[k].resize(s.nThetaEven);
  }  // k

  w.new_r_axis.resize(s.nZeta);
  w.new_z_axis.resize(s.nZeta);

  w.new_raxis_c.resize(s.ntor + 1);
  w.new_zaxis_s.resize(s.ntor + 1);
  if (s.lasym) {
    w.new_raxis_s.resize(s.ntor + 1);
    w.new_zaxis_c.resize(s.ntor + 1);
  }

  // This duplicates some functionality of VMEC,
  // where existing data is re-used in educational_VMEC,
  // for the sake of this being a stand-alone method independent from
  // IdealMHDModel. This is done because other methods to come up with an
  // initial guess for the magnetic axis geometry are known (e.g. the method by
  // Zeno Tecchiolli) and we need to allow implementing them easily in the
  // future in this context.

  // grid resolution in R and Z
  static constexpr int kNumberOfGridPoints = 61;

  // radial index of a flux surface at ~mid-radius
  // -1 wrt. Fortran VMEC since we have 0-based indices in C/C++
  const int ns12 = (number_of_flux_surfaces + 1) / 2 - 1;

  // radial step size between boundary (ns - 1) and ~mid-radius (ns12)
  const double delta_s =
      (number_of_flux_surfaces - 1 - ns12) / (number_of_flux_surfaces - 1.0);

  // sqrt(s_j) at j = ns12, where s is the normalized radial coordinate
  const double sqrtSF12 = std::sqrt(ns12 / (number_of_flux_surfaces - 1.0));

  // start: interpolate Fourier coefficients between initial guess for axis and
  // boundary

  std::vector<std::vector<double> > rcc_half(s.mpol);
  std::vector<std::vector<double> > rss_half;
  std::vector<std::vector<double> > rsc_half;
  std::vector<std::vector<double> > rcs_half;
  std::vector<std::vector<double> > zsc_half(s.mpol);
  std::vector<std::vector<double> > zcs_half;
  std::vector<std::vector<double> > zcc_half;
  std::vector<std::vector<double> > zss_half;
  if (s.lthreed) {
    rss_half.resize(s.mpol);
    zcs_half.resize(s.mpol);
  }
  if (s.lasym) {
    rsc_half.resize(s.mpol);
    zcc_half.resize(s.mpol);
    if (s.lthreed) {
      rcs_half.resize(s.mpol);
      zss_half.resize(s.mpol);
    }
  }

  // undo m=1 constraint
  const double scalingFactor = 1.0;
  std::vector<std::vector<double> > rss_boundary;  // lthreed
  std::vector<std::vector<double> > zcs_boundary;  // lthreed
  std::vector<std::vector<double> > rsc_boundary;  // lasym
  std::vector<std::vector<double> > zcc_boundary;  // lasym
  if (s.lthreed) {
    rss_boundary.resize(s.mpol);
    zcs_boundary.resize(s.mpol);
    for (int m = 0; m < s.mpol; ++m) {
      rss_boundary[m].resize(s.ntor + 1);
      zcs_boundary[m].resize(s.ntor + 1);
      for (int n = 0; n <= s.ntor; ++n) {
        int idx_mn = m * (s.ntor + 1) + n;
        if (m == 1) {
          rss_boundary[m][n] = (rbss[idx_mn] + zbcs[idx_mn]) * scalingFactor;
          zcs_boundary[m][n] = (rbss[idx_mn] - zbcs[idx_mn]) * scalingFactor;
        } else {
          rss_boundary[m][n] = rbss[idx_mn];
          zcs_boundary[m][n] = zbcs[idx_mn];
        }
      }  // n
    }  // m
  }
  if (s.lasym) {
    rsc_boundary.resize(s.mpol);
    zcc_boundary.resize(s.mpol);
    for (int m = 0; m < s.mpol; ++m) {
      rsc_boundary[m].resize(s.ntor + 1);
      zcc_boundary[m].resize(s.ntor + 1);
      for (int n = 0; n <= s.ntor; ++n) {
        int idx_mn = m * (s.ntor + 1) + n;
        if (m == 1) {
          rsc_boundary[m][n] = (rbsc[idx_mn] + zbcc[idx_mn]) * scalingFactor;
          zcc_boundary[m][n] = (rbsc[idx_mn] - zbcc[idx_mn]) * scalingFactor;
        } else {
          rsc_boundary[m][n] = rbsc[idx_mn];
          zcc_boundary[m][n] = zbcc[idx_mn];
        }
      }  // n
    }  // m
  }

  for (int m = 0; m < s.mpol; ++m) {
    rcc_half[m].resize(s.ntor + 1);
    zsc_half[m].resize(s.ntor + 1);
    if (s.lthreed) {
      rss_half[m].resize(s.ntor + 1);
      zcs_half[m].resize(s.ntor + 1);
    }
    if (s.lasym) {
      rsc_half[m].resize(s.ntor + 1);
      zcc_half[m].resize(s.ntor + 1);
      if (s.lthreed) {
        rcs_half[m].resize(s.ntor + 1);
        zss_half[m].resize(s.ntor + 1);
      }
    }

    for (int n = 0; n <= s.ntor; ++n) {
      const int idx_mn = m * (s.ntor + 1) + n;

      if (m == 0) {
        // m = 0: linear interpolation in s between axis and boundary
        const double interpolation_weight = sqrtSF12 * sqrtSF12;

        rcc_half[m][n] = (interpolation_weight * rbcc[idx_mn] +
                          (1.0 - interpolation_weight) * raxis_c[n]);
        // zsc has no m=0-contributions from the axis
        zsc_half[m][n] = interpolation_weight * zbsc[idx_mn];
        if (s.lthreed) {
          // rss has no m=0-contributions from the axis
          rss_half[m][n] = interpolation_weight * rss_boundary[m][n];
          zcs_half[m][n] = (interpolation_weight * zcs_boundary[m][n] -
                            (1.0 - interpolation_weight) * zaxis_s[n]);
        }
        if (s.lasym) {
          // rsc has no m=0-contributions from the axis
          rsc_half[m][n] = interpolation_weight * rsc_boundary[m][n];
          zcc_half[m][n] = (interpolation_weight * zcc_boundary[m][n] +
                            (1.0 - interpolation_weight) * zaxis_c[n]);
          if (s.lthreed) {
            rcs_half[m][n] = (interpolation_weight * rbcs[idx_mn] -
                              (1.0 - interpolation_weight) * raxis_s[n]);
            // zss has no m=0-contributions from the axis
            zss_half[m][n] = interpolation_weight * zbss[idx_mn];
          }
        }
      } else {
        // m > 0: scale boundary into volume by s^(m/2) == sqrt(s)^m == rho^m
        const double interpolation_weight = std::pow(sqrtSF12, m);

        rcc_half[m][n] = interpolation_weight * rbcc[idx_mn];
        zsc_half[m][n] = interpolation_weight * zbsc[idx_mn];
        if (s.lthreed) {
          rss_half[m][n] = interpolation_weight * rss_boundary[m][n];
          zcs_half[m][n] = interpolation_weight * zcs_boundary[m][n];
        }
        if (s.lasym) {
          rsc_half[m][n] = interpolation_weight * rsc_boundary[m][n];
          zcc_half[m][n] = interpolation_weight * zcc_boundary[m][n];
          if (s.lthreed) {
            rcs_half[m][n] = interpolation_weight * rbcs[idx_mn];
            zss_half[m][n] = interpolation_weight * zbss[idx_mn];
          }
        }
      }
    }  // n
  }  // m

  // end: interpolate Fourier coefficients between initial guess for axis and
  // boundary

  // inverse Fourier transforms:
  // evaluate R, Z, dR/dTheta and dZ/dTheta at radial locations ns12 and ns-1
  for (int k = 0; k < s.nZeta; ++k) {
    for (int l = 0; l < s.nThetaReduced; ++l) {
      // set target storage to zero
      w.r_lcfs[k][l] = 0.0;
      w.z_lcfs[k][l] = 0.0;
      w.d_r_d_theta_lcfs[k][l] = 0.0;
      w.d_z_d_theta_lcfs[k][l] = 0.0;

      w.r_half[k][l] = 0.0;
      w.z_half[k][l] = 0.0;
      w.d_r_d_theta_half[k][l] = 0.0;
      w.d_z_d_theta_half[k][l] = 0.0;

      for (int m = 0; m < s.mpol; ++m) {
        for (int n = 0; n <= s.ntor; ++n) {
          int idx_ml = m * s.nThetaReduced + l;
          int idx_kn = k * (s.nnyq2 + 1) + n;
          int idx_mn = m * (s.ntor + 1) + n;

          const double basis_norm = 1.0 / (t.mscale[m] * t.nscale[n]);

          w.r_lcfs[k][l] +=
              basis_norm * rbcc[idx_mn] * t.cosmu[idx_ml] * t.cosnv[idx_kn];
          w.z_lcfs[k][l] +=
              basis_norm * zbsc[idx_mn] * t.sinmu[idx_ml] * t.cosnv[idx_kn];
          w.d_r_d_theta_lcfs[k][l] +=
              basis_norm * rbcc[idx_mn] * t.sinmum[idx_ml] * t.cosnv[idx_kn];
          w.d_z_d_theta_lcfs[k][l] +=
              basis_norm * zbsc[idx_mn] * t.cosmum[idx_ml] * t.cosnv[idx_kn];
          w.r_half[k][l] +=
              basis_norm * rcc_half[m][n] * t.cosmu[idx_ml] * t.cosnv[idx_kn];
          w.z_half[k][l] +=
              basis_norm * zsc_half[m][n] * t.sinmu[idx_ml] * t.cosnv[idx_kn];
          w.d_r_d_theta_half[k][l] +=
              basis_norm * rcc_half[m][n] * t.sinmum[idx_ml] * t.cosnv[idx_kn];
          w.d_z_d_theta_half[k][l] +=
              basis_norm * zsc_half[m][n] * t.cosmum[idx_ml] * t.cosnv[idx_kn];
          if (s.lthreed) {
            w.r_lcfs[k][l] += basis_norm * rss_boundary[m][n] *
                              t.sinmu[idx_ml] * t.sinnv[idx_kn];
            w.z_lcfs[k][l] += basis_norm * zcs_boundary[m][n] *
                              t.cosmu[idx_ml] * t.sinnv[idx_kn];
            w.d_r_d_theta_lcfs[k][l] += basis_norm * rss_boundary[m][n] *
                                        t.cosmum[idx_ml] * t.sinnv[idx_kn];
            w.d_z_d_theta_lcfs[k][l] += basis_norm * zcs_boundary[m][n] *
                                        t.sinmum[idx_ml] * t.sinnv[idx_kn];

            w.r_half[k][l] +=
                basis_norm * rss_half[m][n] * t.sinmu[idx_ml] * t.sinnv[idx_kn];
            w.z_half[k][l] +=
                basis_norm * zcs_half[m][n] * t.cosmu[idx_ml] * t.sinnv[idx_kn];
            w.d_r_d_theta_half[k][l] += basis_norm * rss_half[m][n] *
                                        t.cosmum[idx_ml] * t.sinnv[idx_kn];
            w.d_z_d_theta_half[k][l] += basis_norm * zcs_half[m][n] *
                                        t.sinmum[idx_ml] * t.sinnv[idx_kn];
          }
          if (s.lasym) {
            w.r_lcfs[k][l] += basis_norm * rsc_boundary[m][n] *
                              t.sinmu[idx_ml] * t.cosnv[idx_kn];
            w.z_lcfs[k][l] += basis_norm * zcc_boundary[m][n] *
                              t.cosmu[idx_ml] * t.cosnv[idx_kn];
            w.d_r_d_theta_lcfs[k][l] += basis_norm * rsc_boundary[m][n] *
                                        t.cosmum[idx_ml] * t.cosnv[idx_kn];
            w.d_z_d_theta_lcfs[k][l] += basis_norm * zcc_boundary[m][n] *
                                        t.sinmum[idx_ml] * t.cosnv[idx_kn];

            w.r_half[k][l] +=
                basis_norm * rsc_half[m][n] * t.sinmu[idx_ml] * t.cosnv[idx_kn];
            w.z_half[k][l] +=
                basis_norm * zcc_half[m][n] * t.cosmu[idx_ml] * t.cosnv[idx_kn];
            w.d_r_d_theta_half[k][l] += basis_norm * rsc_half[m][n] *
                                        t.cosmum[idx_ml] * t.cosnv[idx_kn];
            w.d_z_d_theta_half[k][l] += basis_norm * zcc_half[m][n] *
                                        t.sinmum[idx_ml] * t.cosnv[idx_kn];
            if (s.lthreed) {
              w.r_lcfs[k][l] +=
                  basis_norm * rbcs[idx_mn] * t.cosmu[idx_ml] * t.sinnv[idx_kn];
              w.z_lcfs[k][l] +=
                  basis_norm * zbss[idx_mn] * t.sinmu[idx_ml] * t.sinnv[idx_kn];
              w.d_r_d_theta_lcfs[k][l] += basis_norm * rbcs[idx_mn] *
                                          t.sinmum[idx_ml] * t.sinnv[idx_kn];
              w.d_z_d_theta_lcfs[k][l] += basis_norm * zbss[idx_mn] *
                                          t.cosmum[idx_ml] * t.sinnv[idx_kn];

              w.r_half[k][l] += basis_norm * rcs_half[m][n] * t.cosmu[idx_ml] *
                                t.sinnv[idx_kn];
              w.z_half[k][l] += basis_norm * zss_half[m][n] * t.sinmu[idx_ml] *
                                t.sinnv[idx_kn];
              w.d_r_d_theta_half[k][l] += basis_norm * rcs_half[m][n] *
                                          t.sinmum[idx_ml] * t.sinnv[idx_kn];
              w.d_z_d_theta_half[k][l] += basis_norm * zss_half[m][n] *
                                          t.cosmum[idx_ml] * t.sinnv[idx_kn];
            }
          }
        }  // n
      }  // m

      // weird averaging of dX/dTheta in guess_axis
      w.d_r_d_theta_half[k][l] =
          (w.d_r_d_theta_lcfs[k][l] + w.d_r_d_theta_half[k][l]) / 2.0;
      w.d_z_d_theta_half[k][l] =
          (w.d_z_d_theta_lcfs[k][l] + w.d_z_d_theta_half[k][l]) / 2.0;
    }  // l
  }  // k

  // flip-mirror geometry into non-stellarator-symmetric half in case of
  // stellarator symmetry
  if (!s.lasym) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int k_reversed = (s.nZeta - k) % s.nZeta;

      for (int l = 1; l < s.nThetaReduced - 1; ++l) {
        const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;

        w.r_lcfs[k_reversed][l_reversed] = w.r_lcfs[k][l];
        w.z_lcfs[k_reversed][l_reversed] = -w.z_lcfs[k][l];
        w.d_r_d_theta_lcfs[k_reversed][l_reversed] = -w.d_r_d_theta_lcfs[k][l];
        w.d_z_d_theta_lcfs[k_reversed][l_reversed] = w.d_z_d_theta_lcfs[k][l];

        w.r_half[k_reversed][l_reversed] = w.r_half[k][l];
        w.z_half[k_reversed][l_reversed] = -w.z_half[k][l];
        w.d_r_d_theta_half[k_reversed][l_reversed] = -w.d_r_d_theta_half[k][l];
        w.d_z_d_theta_half[k_reversed][l_reversed] = w.d_z_d_theta_half[k][l];
      }  // l
    }  // k
  }  // !lasym

  // inverse Fourier transform for current axis geometry
  // axis has m = 0
  const int m0 = 0;
  for (int k = 0; k < s.nZeta; ++k) {
    // set target storage to zero
    w.r_axis[k] = 0.0;
    w.z_axis[k] = 0.0;

    // accumulate all contributions to the Fourier integrals
    for (int n = 0; n <= s.ntor; ++n) {
      int idx_kn = k * (s.nnyq2 + 1) + n;

      const double basis_norm = 1.0 / (t.mscale[m0] * t.nscale[n]);

      w.r_axis[k] += raxis_c[n] * t.cosnv[idx_kn] * basis_norm;
      w.z_axis[k] -= zaxis_s[n] * t.sinnv[idx_kn] * basis_norm;
      if (s.lasym) {
        w.r_axis[k] -= raxis_s[n] * t.sinnv[idx_kn] * basis_norm;
        w.z_axis[k] += zaxis_c[n] * t.cosnv[idx_kn] * basis_norm;
      }
    }  // n
  }  // k

  // main loop in which, for each poloidal cutplane,
  // the new axis position is estimated
  for (int k = 0; k < s.nZeta / 2 + 1; ++k) {
    // compute grid extent
    const double min_r =
        *std::min_element(w.r_lcfs[k].begin(), w.r_lcfs[k].end());
    const double max_r =
        *std::max_element(w.r_lcfs[k].begin(), w.r_lcfs[k].end());
    const double min_z =
        *std::min_element(w.z_lcfs[k].begin(), w.z_lcfs[k].end());
    const double max_z =
        *std::max_element(w.z_lcfs[k].begin(), w.z_lcfs[k].end());

    // grid step sizes
    const double delta_r = (max_r - min_r) / (kNumberOfGridPoints - 1.0);
    const double delta_z = (max_z - min_z) / (kNumberOfGridPoints - 1.0);

    // compute initial guess for new axis: center of grid in each toroidal plane
    w.new_r_axis[k] = (max_r + min_r) / 2.0;
    w.new_z_axis[k] = (max_z + min_z) / 2.0;

    // Compute the static part of the Jacobian (tau0)
    for (int l = 0; l < s.nThetaEven; ++l) {
      w.d_r_d_s_half[k][l] =
          (w.r_lcfs[k][l] - w.r_half[k][l]) / delta_s + w.r_axis[k];
      w.d_z_d_s_half[k][l] =
          (w.z_lcfs[k][l] - w.z_half[k][l]) / delta_s + w.z_axis[k];
      w.tau0[k][l] = w.d_r_d_theta_half[k][l] * w.d_z_d_s_half[k][l] -
                     w.d_z_d_theta_half[k][l] * w.d_r_d_s_half[k][l];
    }  // l

    double min_tau = 0.0;

    for (int index_z = 0; index_z < kNumberOfGridPoints; ++index_z) {
      double z_grid = min_z + index_z * delta_z;

      // early exit in some cases(?)
      if (!s.lasym && (k == 0 || k == s.nZeta / 2)) {
        z_grid = 0.0;
        if (index_z > 0) {
          break;
        }
      }

      for (int index_r = 0; index_r < kNumberOfGridPoints; ++index_r) {
        double r_grid = min_r + index_r * delta_r;

        // Find position of magnetic axis that maximizes the minimum Jacobian
        // value.

        for (int l = 0; l < s.nThetaEven; ++l) {
          w.tau[k][l] = sign_of_jacobian *
                        (w.tau0[k][l] - w.d_r_d_theta_half[k][l] * z_grid +
                         w.d_z_d_theta_half[k][l] * r_grid);
        }  // l

        double min_tau_temp =
            *std::min_element(w.tau[k].begin(), w.tau[k].end());

        if (min_tau_temp > min_tau) {
          min_tau = min_tau_temp;
          w.new_r_axis[k] = r_grid;
          w.new_z_axis[k] = z_grid;
        } else if (min_tau_temp == min_tau) {
          // If up-down symmetric and lasym=T, need this to pick z = 0
          if (std::abs(w.new_z_axis[k]) > std::abs(z_grid)) {
            w.new_z_axis[k] = z_grid;
          }
        }
      }  // index_r
    }  // index_z
  }  // k

  // flip-mirror stellarator-symmetric half in case of symmetric run
  // in order to always have a full toroidal module for the Fourier transform
  // below
  if (!s.lasym) {
    for (int k = 1; k < s.nZeta / 2; ++k) {
      const int k_reversed = (s.nZeta - k) % s.nZeta;
      w.new_r_axis[k_reversed] = w.new_r_axis[k];
      w.new_z_axis[k_reversed] = -w.new_z_axis[k];
    }  // k
  }  // !lasym

  // Fourier-transform the axis guess
  const double delta_v = 2.0 / s.nZeta;
  for (int k = 0; k < s.nZeta; ++k) {
    for (int n = 0; n <= s.ntor; ++n) {
      // accumulate all contributions to the toroidal Fourier integral
      int idx_kn = k * (s.nnyq2 + 1) + n;

      w.new_raxis_c[n] +=
          delta_v * t.cosnv[idx_kn] * w.new_r_axis[k] / t.nscale[n];
      w.new_zaxis_s[n] -=
          delta_v * t.sinnv[idx_kn] * w.new_z_axis[k] / t.nscale[n];
      if (s.lasym) {
        w.new_raxis_s[n] -=
            delta_v * t.sinnv[idx_kn] * w.new_r_axis[k] / t.nscale[n];
        w.new_zaxis_c[n] +=
            delta_v * t.cosnv[idx_kn] * w.new_z_axis[k] / t.nscale[n];
      }
    }  // n
  }  // k

  // fixup Fourier basis scaling for cos(0) and cos(Nyquist) entries
  w.new_raxis_c[0] /= 2.0;
  // need to check explicitly for ntor > 0, because otherwise for ntor = 0,
  // nZeta / 2 also is zero (integer division doing the rounding-down),
  // which will lead to the n=0 - component being divided by 2 twice!
  if (s.ntor > 0 && s.ntor >= s.nZeta / 2) {
    w.new_raxis_c[s.nZeta / 2] /= 2.0;
  }
  if (s.lasym) {
    w.new_zaxis_c[0] /= 2.0;
    if (s.ntor > 0 && s.ntor >= s.nZeta / 2) {
      w.new_zaxis_c[s.nZeta / 2] /= 2.0;
    }
  }

  return w;
}  // NOLINT(readability/fn_size)

}  // namespace vmecpp
