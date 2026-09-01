// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/boozer/boozer.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace vmecpp {

namespace {

// Fourier coefficients of one surface, with the mode numbers of the set they
// belong to.
struct SurfaceSpectrum {
  const Eigen::VectorXi* xm = nullptr;
  const Eigen::VectorXi* xn = nullptr;
  Eigen::VectorXd coefficients;
};

// Evaluates a cosine or sine series and its angle derivatives on a grid point.
struct PointValue {
  double value = 0.0;
  double d_theta = 0.0;
  double d_zeta = 0.0;
};

PointValue EvaluateCosine(const SurfaceSpectrum& f, double theta, double zeta) {
  PointValue out;
  for (Eigen::Index mn = 0; mn < f.coefficients.size(); ++mn) {
    const double m = (*f.xm)[mn];
    const double n = (*f.xn)[mn];
    const double argument = m * theta - n * zeta;
    const double c = std::cos(argument);
    const double s = std::sin(argument);
    out.value += f.coefficients[mn] * c;
    out.d_theta -= f.coefficients[mn] * m * s;
    out.d_zeta += f.coefficients[mn] * n * s;
  }
  return out;
}

PointValue EvaluateSine(const SurfaceSpectrum& f, double theta, double zeta) {
  PointValue out;
  for (Eigen::Index mn = 0; mn < f.coefficients.size(); ++mn) {
    const double m = (*f.xm)[mn];
    const double n = (*f.xn)[mn];
    const double argument = m * theta - n * zeta;
    const double c = std::cos(argument);
    const double s = std::sin(argument);
    out.value += f.coefficients[mn] * s;
    out.d_theta += f.coefficients[mn] * m * c;
    out.d_zeta -= f.coefficients[mn] * n * c;
  }
  return out;
}

}  // namespace

absl::StatusOr<BoozerCoordinates> BoozerTransform(
    const WOutFileContents& wout, int mboz, int nboz,
    const std::vector<int>& surfaces) {
  if (wout.lasym) {
    return absl::UnimplementedError(
        "BoozerTransform: non-stellarator-symmetric equilibria are not "
        "supported yet");
  }
  if (mboz < 1 || nboz < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BoozerTransform: need mboz >= 1 and nboz >= 0, got %d and %d", mboz,
        nboz));
  }
  const int ns = wout.ns;
  std::vector<int> columns = surfaces;
  if (columns.empty()) {
    for (int js = 1; js < ns; ++js) {
      columns.push_back(js);
    }
  }
  for (const int js : columns) {
    if (js < 1 || js >= ns) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "BoozerTransform: half-grid column %d outside 1..%d", js, ns - 1));
    }
  }
  const int nfp = wout.nfp;
  const int num_surfaces = static_cast<int>(columns.size());

  BoozerCoordinates b;
  b.nfp = nfp;
  b.mboz = mboz;
  b.nboz = nboz;
  const int mnboz = (nboz + 1) + (mboz - 1) * (2 * nboz + 1);
  b.xm_b.resize(mnboz);
  b.xn_b.resize(mnboz);
  {
    int k = 0;
    for (int n = 0; n <= nboz; ++n) {
      b.xm_b[k] = 0;
      b.xn_b[k] = n * nfp;
      ++k;
    }
    for (int m = 1; m < mboz; ++m) {
      for (int n = -nboz; n <= nboz; ++n) {
        b.xm_b[k] = m;
        b.xn_b[k] = n * nfp;
        ++k;
      }
    }
  }
  b.surfaces = Eigen::Map<const Eigen::VectorXi>(columns.data(), num_surfaces);
  b.iota_b.resize(num_surfaces);
  b.g_b.resize(num_surfaces);
  b.i_b.resize(num_surfaces);
  b.jacobian_spread.resize(num_surfaces);
  b.bmnc_b.setZero(mnboz, num_surfaces);
  b.rmnc_b.setZero(mnboz, num_surfaces);
  b.zmns_b.setZero(mnboz, num_surfaces);
  b.numns_b.setZero(mnboz, num_surfaces);
  b.gmnc_b.setZero(mnboz, num_surfaces);

  // lambda lives on the VMEC mode set; the covariant field and |B| on the
  // Nyquist-extended one. Map the VMEC modes into the extended set so the
  // transformation function nu can be assembled mode by mode.
  const int mnmax = wout.mnmax;
  const int mnmax_nyq = wout.mnmax_nyq;
  std::map<std::pair<int, int>, int> nyq_index;
  for (int mn = 0; mn < mnmax_nyq; ++mn) {
    nyq_index[{wout.xm_nyq[mn], wout.xn_nyq[mn]}] = mn;
  }
  std::vector<int> vmec_to_nyq(mnmax, -1);
  for (int mn = 0; mn < mnmax; ++mn) {
    const auto found = nyq_index.find({wout.xm[mn], wout.xn[mn]});
    if (found == nyq_index.end()) {
      return absl::InternalError(
          "BoozerTransform: a VMEC mode is missing from the Nyquist set");
    }
    vmec_to_nyq[mn] = found->second;
  }
  int m_nyq_max = 0;
  int n_nyq_max = 0;
  for (int mn = 0; mn < mnmax_nyq; ++mn) {
    m_nyq_max = std::max(m_nyq_max, wout.xm_nyq[mn]);
    n_nyq_max = std::max(n_nyq_max, std::abs(wout.xn_nyq[mn]) / nfp);
  }

  // The quadrature grid resolves the products of the surface quantities, the
  // angle map and the Boozer harmonics.
  const int ntheta = 2 * (2 * mboz + m_nyq_max + 1);
  const int nzeta =
      (nboz == 0 && n_nyq_max == 0) ? 1 : 2 * (2 * nboz + n_nyq_max + 1);
  const double dtheta = 2.0 * M_PI / ntheta;
  const double dzeta = 2.0 * M_PI / (nfp * nzeta);

  SurfaceSpectrum r_half{&wout.xm, &wout.xn, Eigen::VectorXd(mnmax)};
  SurfaceSpectrum z_half{&wout.xm, &wout.xn, Eigen::VectorXd(mnmax)};
  SurfaceSpectrum lambda{&wout.xm, &wout.xn, Eigen::VectorXd(mnmax)};
  SurfaceSpectrum bmag{&wout.xm_nyq, &wout.xn_nyq, Eigen::VectorXd(mnmax_nyq)};
  SurfaceSpectrum jac{&wout.xm_nyq, &wout.xn_nyq, Eigen::VectorXd(mnmax_nyq)};
  SurfaceSpectrum nu{&wout.xm_nyq, &wout.xn_nyq, Eigen::VectorXd(mnmax_nyq)};

  for (int surface = 0; surface < num_surfaces; ++surface) {
    const int js = columns[surface];

    // Geometry on the half grid, following booz_xform: even-m coefficients
    // are averaged between the neighboring full-grid surfaces, odd-m
    // coefficients are averaged as coefficient / sqrt(s) and rescaled with the
    // half-grid sqrt(s). On the first half-grid surface the m = 1 quotient is
    // extrapolated to the axis from the two innermost surfaces, and the other
    // odd-m quotients take the axis coefficient itself.
    const double s_half = (js - 0.5) / (ns - 1.0);
    const double sqrt_s_half = std::sqrt(s_half);
    const double sqrt_s_outer = std::sqrt(js / (ns - 1.0));
    const double sqrt_s_inner =
        js > 1 ? std::sqrt((js - 1.0) / (ns - 1.0)) : 1.0;
    for (int mn = 0; mn < mnmax; ++mn) {
      const int m = wout.xm[mn];
      double r = 0.0;
      double z = 0.0;
      if (m % 2 == 0) {
        r = 0.5 * (wout.rmnc(mn, js - 1) + wout.rmnc(mn, js));
        z = 0.5 * (wout.zmns(mn, js - 1) + wout.zmns(mn, js));
      } else if (js == 1 && m == 1 && ns > 2) {
        const double sqrt_s_2 = std::sqrt(2.0 / (ns - 1.0));
        r = (1.5 * wout.rmnc(mn, 1) / sqrt_s_outer -
             0.5 * wout.rmnc(mn, 2) / sqrt_s_2) *
            sqrt_s_half;
        z = (1.5 * wout.zmns(mn, 1) / sqrt_s_outer -
             0.5 * wout.zmns(mn, 2) / sqrt_s_2) *
            sqrt_s_half;
      } else {
        r = 0.5 *
            (wout.rmnc(mn, js - 1) / sqrt_s_inner +
             wout.rmnc(mn, js) / sqrt_s_outer) *
            sqrt_s_half;
        z = 0.5 *
            (wout.zmns(mn, js - 1) / sqrt_s_inner +
             wout.zmns(mn, js) / sqrt_s_outer) *
            sqrt_s_half;
      }
      r_half.coefficients[mn] = r;
      z_half.coefficients[mn] = z;
      lambda.coefficients[mn] = wout.lmns(mn, js);
    }
    for (int mn = 0; mn < mnmax_nyq; ++mn) {
      bmag.coefficients[mn] = wout.bmnc(mn, js);
      jac.coefficients[mn] = wout.gmnc(mn, js);
    }
    const double iota = wout.iotas[js];
    const double current_i = wout.buco[js];
    const double current_g = wout.bvco[js];
    const double denominator = current_g + iota * current_i;
    if (denominator == 0.0) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "BoozerTransform: G + iota I vanishes on half-grid surface %d", js));
    }

    // w from the periodic parts of B_theta and B_zeta; nu from w and lambda
    for (int mn = 0; mn < mnmax_nyq; ++mn) {
      const int m = wout.xm_nyq[mn];
      const int n = wout.xn_nyq[mn];
      double w = 0.0;
      if (m != 0) {
        w = wout.bsubumnc(mn, js) / m;
      } else if (n != 0) {
        w = -wout.bsubvmnc(mn, js) / n;
      }
      nu.coefficients[mn] = w / denominator;
    }
    for (int mn = 0; mn < mnmax; ++mn) {
      nu.coefficients[vmec_to_nyq[mn]] -=
          current_i * lambda.coefficients[mn] / denominator;
    }

    b.iota_b[surface] = iota;
    b.g_b[surface] = current_g;
    b.i_b[surface] = current_i;

    // direct quadrature over the VMEC angles
    double spread_sum = 0.0;
    double spread_sum_squares = 0.0;
    for (int i = 0; i < ntheta; ++i) {
      const double theta = i * dtheta;
      for (int j = 0; j < nzeta; ++j) {
        const double zeta = j * dzeta;
        const PointValue lam = EvaluateSine(lambda, theta, zeta);
        const PointValue p = EvaluateSine(nu, theta, zeta);
        const double bvalue = EvaluateCosine(bmag, theta, zeta).value;
        const double rvalue = EvaluateCosine(r_half, theta, zeta).value;
        const double zvalue = EvaluateSine(z_half, theta, zeta).value;
        const double gvalue = EvaluateCosine(jac, theta, zeta).value;

        const double theta_b = theta + lam.value + iota * p.value;
        const double zeta_b = zeta + p.value;
        const double map_jacobian =
            (1.0 + lam.d_theta + iota * p.d_theta) * (1.0 + p.d_zeta) -
            (lam.d_zeta + iota * p.d_zeta) * p.d_theta;
        // sqrt(g_B) |B|^2 is a flux function in Boozer coordinates
        const double invariant = gvalue / map_jacobian * bvalue * bvalue;
        spread_sum += invariant;
        spread_sum_squares += invariant * invariant;

        // the Boozer Jacobian with the toroidal flux per radian as the radial
        // coordinate, (G + iota I) / |B|^2, as booz_xform reports it
        const double boozer_jacobian = denominator / (bvalue * bvalue);
        for (int k = 0; k < mnboz; ++k) {
          const double argument = b.xm_b[k] * theta_b - b.xn_b[k] * zeta_b;
          const double c = std::cos(argument) * map_jacobian;
          const double s = std::sin(argument) * map_jacobian;
          b.bmnc_b(k, surface) += bvalue * c;
          b.rmnc_b(k, surface) += rvalue * c;
          b.zmns_b(k, surface) += zvalue * s;
          b.numns_b(k, surface) += p.value * s;
          b.gmnc_b(k, surface) += boozer_jacobian * c;
        }
      }
    }
    const double points = static_cast<double>(ntheta) * nzeta;
    for (int k = 0; k < mnboz; ++k) {
      const double factor =
          (b.xm_b[k] == 0 && b.xn_b[k] == 0) ? 1.0 / points : 2.0 / points;
      b.bmnc_b(k, surface) *= factor;
      b.rmnc_b(k, surface) *= factor;
      b.zmns_b(k, surface) *= factor;
      b.numns_b(k, surface) *= factor;
      b.gmnc_b(k, surface) *= factor;
    }
    const double mean = spread_sum / points;
    const double variance =
        std::max(spread_sum_squares / points - mean * mean, 0.0);
    b.jacobian_spread[surface] =
        mean != 0.0 ? std::sqrt(variance) / std::abs(mean) : 0.0;
  }

  return b;
}

}  // namespace vmecpp
