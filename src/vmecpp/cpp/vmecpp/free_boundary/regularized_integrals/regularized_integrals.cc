// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/regularized_integrals/regularized_integrals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"

namespace vmecpp {

namespace {

// number of the powers 0, ..., SurfaceGeometry::kTaylorDegree of a step
constexpr int kPowers = SurfaceGeometry::kTaylorDegree + 1;

// number of the even harmonics 0, 2, ..., 8 of the polar angle in the
// expansion of the remainders about the source point
constexpr int kHarmonics = 5;

// Polynomial in the steps (s, t) of theta and of the zeta of one field period,
// of total degree up to kTaylorDegree: element [i][j] multiplies s^i t^j.
using TaylorPolynomial = std::array<std::array<double, kPowers>, kPowers>;

// the powers of a step
using StepPowers = std::array<double, kPowers>;

using Vector2 = std::array<double, 2>;

TaylorPolynomial Multiply(const TaylorPolynomial& a,
                          const TaylorPolynomial& b) {
  TaylorPolynomial c = {};
  for (int i = 0; i < kPowers; ++i) {
    for (int j = 0; i + j < kPowers; ++j) {
      if (a[i][j] == 0.0) {
        continue;
      }
      for (int k = 0; i + j + k < kPowers; ++k) {
        for (int l = 0; i + j + k + l < kPowers; ++l) {
          c[i + k][j + l] += a[i][j] * b[k][l];
        }
      }
    }
  }
  return c;
}

// the homogeneous part of the given degree, from the powers of s and t
double HomogeneousPart(const TaylorPolynomial& a, int degree,
                       const StepPowers& s_powers, const StepPowers& t_powers) {
  double sum = 0.0;
  for (int i = 0; i <= degree; ++i) {
    sum += a[i][degree - i] * s_powers[i] * t_powers[degree - i];
  }
  return sum;
}

double Dot(const Vector2& a, const Vector2& b) {
  return a[0] * b[0] + a[1] * b[1];
}

// A Z_k for k = 2 q, q = 0, ..., 4: the regularized sum of
// A |x|^{1-k} (x_1 + i x_2)^k over the lattice spanned by e1 and e2, of cell
// area A, less its integral. Z_k is the Epstein zeta function with the
// harmonic weight at s = (k - 1)/2, split by Ewald's method between the lattice
// and its dual; the terms fall off as exp(-\pi |x|^2) in the lattice of unit
// cell area, and x and -x contribute alike.
std::array<std::complex<double>, kHarmonics> HarmonicLatticeSums(
    const Vector2& e1, const Vector2& e2) {
  // squared radius of the points summed in the lattice of unit cell area and
  // in its dual, where the factor exp(-\pi |x|^2) of the terms is exp(-7 \pi)
  static constexpr double kReach2 = 7.0;
  const double scale = std::sqrt(std::abs(e1[0] * e2[1] - e1[1] * e2[0]));
  Vector2 f1 = {e1[0] / scale, e1[1] / scale};
  Vector2 f2 = {e2[0] / scale, e2[1] / scale};
  // Lagrange-Gauss reduction of the basis
  for (int iteration = 0; iteration < 100; ++iteration) {
    if (Dot(f1, f1) > Dot(f2, f2)) {
      std::swap(f1, f2);
      continue;
    }
    const double mu = std::round(Dot(f1, f2) / Dot(f1, f1));
    if (mu == 0.0) {
      break;
    }
    f2 = {f2[0] - mu * f1[0], f2[1] - mu * f1[1]};
  }
  const double det = f1[0] * f2[1] - f1[1] * f2[0];
  const std::array<Vector2, 2> dual = {Vector2{f2[1] / det, -f2[0] / det},
                                       Vector2{-f1[1] / det, f1[0] / det}};
  const double area = std::abs(det);

  std::array<std::complex<double>, kHarmonics> direct = {};
  std::array<std::complex<double>, kHarmonics> reciprocal = {};
  for (int side = 0; side < 2; ++side) {
    const Vector2& b1 = (side == 0) ? f1 : dual[0];
    const Vector2& b2 = (side == 0) ? f2 : dual[1];
    const double g11 = Dot(b1, b1);
    const double g22 = Dot(b2, b2);
    const double g12 = Dot(b1, b2);
    const double gram = g11 * g22 - g12 * g12;
    const int ni = static_cast<int>(std::ceil(std::sqrt(kReach2 * g22 / gram)));
    const int nj = static_cast<int>(std::ceil(std::sqrt(kReach2 * g11 / gram)));
    for (int i = 0; i <= ni; ++i) {
      for (int j = -nj; j <= nj; ++j) {
        if (i == 0 && j <= 0) {
          continue;
        }
        const double wx = i * b1[0] + j * b2[0];
        const double wy = i * b1[1] + j * b2[1];
        const double r2 = wx * wx + wy * wy;
        if (r2 > kReach2) {
          continue;
        }
        const double x = std::numbers::pi * r2;
        const double x_inv = 1.0 / x;
        // Gamma(m - 1/2, x) for m = 0, ..., 6 by erfc and the recurrence in
        // the order
        std::array<double, kHarmonics + 2> gamma = {};
        const double ex = std::exp(-x);
        const double sx = std::sqrt(x);
        gamma[1] = std::sqrt(std::numbers::pi) * std::erfc(sx);
        gamma[0] = 2.0 * (ex / sx - gamma[1]);
        double order = 0.5;
        double x_power = sx;
        for (int m = 2; m < kHarmonics + 2; ++m) {
          gamma[m] = order * gamma[m - 1] + x_power * ex;
          order += 1.0;
          x_power *= x;
        }
        const std::complex<double> w(wx, wy);
        const std::complex<double> w2 = w * w;
        std::complex<double> w_power = 1.0;
        if (side == 0) {
          // P_k x^{-s} Gamma(s, x)
          double factor = sx;
          for (int q = 0; q < kHarmonics; ++q) {
            direct[q] += w_power * (factor * gamma[q]);
            w_power *= w2;
            factor *= x_inv;
          }
        } else {
          // P_k x^{-(k + 3)/2} Gamma((k + 3)/2, x)
          double factor = sx * x_inv * x_inv;
          for (int q = 0; q < kHarmonics; ++q) {
            reciprocal[q] += w_power * (factor * gamma[q + 2]);
            w_power *= w2;
            factor *= x_inv;
          }
        }
      }  // j
    }  // i
  }  // side

  std::array<std::complex<double>, kHarmonics> sums = {};
  // pi^s / Gamma(s) at s = -1/2, 1/2, ..., 7/2
  double pi_over_gamma = -0.5 / std::numbers::pi;
  for (int q = 0; q < kHarmonics; ++q) {
    // (-i)^k = (-1)^q; the sums ran over half of each lattice
    const double sign = (q % 2 == 0) ? 1.0 : -1.0;
    std::complex<double> total =
        2.0 * (direct[q] + sign * reciprocal[q] / area);
    if (q == 0) {
      total += 2.0 - 2.0 / (3.0 * area);
    }
    sums[q] = area * pi_over_gamma * total * scale * scale * scale;
    pi_over_gamma *= std::numbers::pi / (q - 0.5);
  }
  return sums;
}

// The Cartesian position and normal of a source point, the cosine and sine of
// its toroidal angle and the rate of that angle in zeta, and its metric and
// second form in the steps of theta and of the zeta of one field period.
struct SourcePoint {
  double x;
  double y;
  double z;
  double normal_x;
  double normal_y;
  double normal_z;
  double cos_phi;
  double sin_phi;
  double phi_rate;
  double guu;
  double guv;
  double gvv;
  double auu;
  double auv;
  double avv;
};

// Defects of the punctured trapezoidal sums at a source point, which are minus
// the regularized lattice sums, in the units of the integral over the steps
// (s, t): of G - ga1 and of dG/dn - ga1 ga2, each with the weights 1, s and t.
struct SingularDefects {
  std::array<double, 3> single_layer;
  std::array<double, 3> double_layer;
};

// About the source point the remainders G - ga1 and dG/dn - ga1 ga2 expand as
// f0 + f1 + ... in (s, t), with f0 odd and homogeneous of degree 0 and f1 even
// and homogeneous of degree 1. The punctured trapezoidal sum of a remainder
// times a smooth function b differs from the integral by
// L[f1] b + L[f0 s] db/dtheta + L[f0 t] db/dzeta at third order in the grid
// step, where L is the regularized lattice sum. In coordinates in which the
// metric is the identity the six functions are |x| times even harmonics of
// the polar angle up to order 8, so L follows from HarmonicLatticeSums.
SingularDefects ComputeSingularDefects(const TaylorPolynomial& r,
                                       const TaylorPolynomial& z,
                                       const SourcePoint& source,
                                       double poloidal_step,
                                       double toroidal_step) {
  static constexpr std::array<double, kPowers> kFactorial = {1.0, 1.0, 2.0, 6.0,
                                                             24.0};
  // the six functions are trigonometric polynomials of degree 8 with period
  // pi in the polar angle
  static constexpr int kAngles = 10;

  // Cartesian position relative to the source and its normal component, as
  // polynomials in (s, t); phase holds cos(phi + j pi/2)
  const std::array<double, 4> phase = {source.cos_phi, -source.sin_phi,
                                       -source.cos_phi, source.sin_phi};
  TaylorPolynomial cos_phi = {};
  TaylorPolynomial sin_phi = {};
  double rate = 1.0;
  for (int j = 0; j < kPowers; ++j) {
    cos_phi[0][j] = rate / kFactorial[j] * phase[j % 4];
    sin_phi[0][j] = rate / kFactorial[j] * phase[(j + 3) % 4];
    rate *= source.phi_rate;
  }
  TaylorPolynomial dx = Multiply(r, cos_phi);
  TaylorPolynomial dy = Multiply(r, sin_phi);
  TaylorPolynomial dz = z;
  dx[0][0] -= source.x;
  dy[0][0] -= source.y;
  dz[0][0] -= source.z;
  const TaylorPolynomial dx2 = Multiply(dx, dx);
  const TaylorPolynomial dy2 = Multiply(dy, dy);
  const TaylorPolynomial dz2 = Multiply(dz, dz);
  TaylorPolynomial distance2 = {};
  TaylorPolynomial normal_distance = {};
  for (int i = 0; i < kPowers; ++i) {
    for (int j = 0; i + j < kPowers; ++j) {
      distance2[i][j] = dx2[i][j] + dy2[i][j] + dz2[i][j];
      normal_distance[i][j] = source.normal_x * dx[i][j] +
                              source.normal_y * dy[i][j] +
                              source.normal_z * dz[i][j];
    }
  }

  // (ca s + cb t, cc t) are coordinates in which the metric is the identity
  const double ca = std::sqrt(source.guu);
  const double cb = 0.5 * source.guv / ca;
  const double cc = std::sqrt(source.gvv - cb * cb);

  // harmonics 0, 2, ..., 8 of the six functions on the unit circle of those
  // coordinates
  std::array<std::array<std::complex<double>, kHarmonics>, 6> harmonics = {};
  for (int n = 0; n < kAngles; ++n) {
    const double angle = std::numbers::pi * n / kAngles;
    const double t = std::sin(angle) / cc;
    const double s = (std::cos(angle) - cb * t) / ca;
    StepPowers s_powers = {1.0, s, s * s, s * s * s, s * s * s * s};
    StepPowers t_powers = {1.0, t, t * t, t * t * t, t * t * t * t};
    const double i2 = HomogeneousPart(distance2, 2, s_powers, t_powers);
    const double c3 = HomogeneousPart(distance2, 3, s_powers, t_powers);
    const double d4 = HomogeneousPart(distance2, 4, s_powers, t_powers);
    const double a2 = HomogeneousPart(normal_distance, 2, s_powers, t_powers);
    const double a3 = HomogeneousPart(normal_distance, 3, s_powers, t_powers);
    const double a4 = HomogeneousPart(normal_distance, 4, s_powers, t_powers);
    // quartic parts of the quadratic forms of the approximation in
    // 2 tan(s/2) and 2 tan(t/2)
    const double cross = s_powers[3] * t + s * t_powers[3];
    const double t4 =
        (source.guu * s_powers[4] + source.gvv * t_powers[4]) / 6.0 +
        source.guv * cross / 12.0;
    const double s4 =
        (source.auu * s_powers[4] + source.avv * t_powers[4]) / 6.0 +
        source.auv * cross / 12.0;
    const double i32 = 1.0 / (i2 * std::sqrt(i2));
    const double i52 = i32 / i2;
    const double i72 = i52 / i2;
    const double f0g = -0.5 * c3 * i32;
    const double f1g = -0.5 * (d4 - t4) * i32 + 0.375 * c3 * c3 * i52;
    const double f0k = a3 * i32 - 1.5 * a2 * c3 * i52;
    const double f1k = (a4 - s4) * i32 -
                       1.5 * (a3 * c3 + a2 * (d4 - t4)) * i52 +
                       1.875 * a2 * c3 * c3 * i72;
    const std::array<double, 6> values = {f1g, f0g * s, f0g * t,
                                          f1k, f0k * s, f0k * t};
    const std::complex<double> step = std::polar(1.0, -2.0 * angle);
    std::complex<double> weight = 1.0 / kAngles;
    for (int k = 0; k < kHarmonics; ++k) {
      for (int q = 0; q < 6; ++q) {
        harmonics[q][k] += values[q] * weight;
      }
      weight *= step;
    }
  }

  const std::array<std::complex<double>, kHarmonics> sums =
      HarmonicLatticeSums(Vector2{ca * poloidal_step, 0.0},
                          Vector2{cb * toroidal_step, cc * toroidal_step});
  SingularDefects defects = {};
  for (int q = 0; q < 6; ++q) {
    double sum = (harmonics[q][0] * sums[0]).real();
    for (int k = 1; k < kHarmonics; ++k) {
      sum += 2.0 * (harmonics[q][k] * sums[k]).real();
    }
    const double defect = -sum / (ca * cc);
    if (q < 3) {
      defects.single_layer[q] = defect;
    } else {
      defects.double_layer[q - 3] = defect;
    }
  }
  return defects;
}

// The Taylor polynomial of a column of SurfaceGeometry::rtay or ztay.
TaylorPolynomial TaylorPolynomialAt(const Eigen::MatrixXd& coefficients,
                                    int column) {
  TaylorPolynomial polynomial = {};
  for (int i = 0; i <= SurfaceGeometry::kTaylorDegree; ++i) {
    for (int j = 0; i + j <= SurfaceGeometry::kTaylorDegree; ++j) {
      polynomial[i][j] =
          coefficients(SurfaceGeometry::TaylorIndex(i, j), column);
    }
  }
  return polynomial;
}

// Weights on the source node, its two poloidal and its two toroidal neighbours
// that give d_1 b + d_s db/dtheta + d_t db/dzeta for the defects d of the
// weights 1, s and t.
std::array<double, 5> StencilWeights(const std::array<double, 3>& defects,
                                     double poloidal_step,
                                     double toroidal_step) {
  const double area = poloidal_step * toroidal_step;
  const double d_theta = defects[1] / (2.0 * poloidal_step * area);
  const double d_zeta = defects[2] / (2.0 * toroidal_step * area);
  return {defects[0] / area, d_theta, -d_theta, d_zeta, -d_zeta};
}

}  // namespace

RegularizedIntegrals::RegularizedIntegrals(const Sizes* s,
                                           const TangentialPartitioning* tp,
                                           const SurfaceGeometry* sg)
    : s_(*s), tp_(*tp), sg_(*sg) {
  gsave.resize(s_.nThetaEven * s_.nZeta);
  dsave.resize(s_.nThetaEven * s_.nZeta);

  tanu.resize(s_.nThetaEven);
  tanv.resize(s_.nZeta);

  // thread-local tangential grid point range
  int numLocal = tp_.ztMax - tp_.ztMin;
  greenp.resize(numLocal * s_.nThetaEven * s_.nZeta);

  gstore.resize(s_.nThetaEven * s_.nZeta);

  // --------------

  computeConstants();
}

void RegularizedIntegrals::computeConstants() {
  const double epsTan = 1.0e-15;
  const double bigNo = 1.0e50;

  for (int l = 0; l < s_.nThetaEven; ++l) {
    const double argu = M_PI / s_.nThetaEven * l;
    if (std::abs(argu - 0.5 * M_PI) < epsTan) {
      // mask singularities at pi/2
      tanu[l] = bigNo;
    } else {
      tanu[l] = 2.0 * std::tan(argu);
    }
  }  // l

  for (int k = 0; k < s_.nZeta; ++k) {
    const double argv = M_PI / s_.nZeta * k;
    if (std::abs(argv - 0.5 * M_PI) < epsTan) {
      // mask singularity at pi/2
      tanv[k] = bigNo;
    } else {
      tanv[k] = 2.0 * std::tan(argv);
    }
  }  // k

  // For an axisymmetric (nZeta == 1) plasma the toroidal direction is resolved
  // by nvper_ toroidal images rather than the surface grid; precompute the
  // analytic-approximation toroidal-angle factor at each image angle.
  nvper_ = (s_.nZeta == 1) ? kAxisymmetricToroidalImages : s_.nfp;
  if (s_.nZeta == 1) {
    tanv_per_.resize(nvper_);
    for (int p = 0; p < nvper_; ++p) {
      const double argv = M_PI * p / nvper_;
      if (std::abs(argv - 0.5 * M_PI) < epsTan) {
        // mask singularity at pi/2
        tanv_per_[p] = bigNo;
      } else {
        tanv_per_[p] = 2.0 * std::tan(argv);
      }
    }  // p
  }
}

void RegularizedIntegrals::update(const Eigen::VectorXd& bDotN) {
  if (s_.nZeta == 1) {
    // Axisymmetric plasma: the surface grid has a single toroidal plane, so the
    // toroidal integral is performed over nvper_ toroidal images instead.
    updateAxisymmetric(bDotN);
    return;
  }

  // thread-local tangential grid point range
  const int theta_by_nzeta = s_.nThetaEven * s_.nZeta;
  const double twopidivnfp = 2.0 * M_PI / s_.nfp;
  // grid steps in u and in the toroidal angle v of one field period
  const double poloidal_step = 2.0 * M_PI / s_.nThetaEven;
  const double toroidal_step = 2.0 * M_PI / s_.nZeta;

  greenp.setZero();
  gstore.setZero();

  // storage for intermediate results
  std::vector<double> ga1_buf(s_.nZeta);
  std::vector<double> ga2_buf(s_.nZeta);
  std::vector<double> htemp_buf(s_.nZeta);
  std::vector<double> ftemp_buf(s_.nZeta);

  for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
    const int ip_idx_base = (klp - tp_.ztMin) * theta_by_nzeta;
    int lp = klp / s_.nZeta;
    int kp = klp % s_.nZeta;

    double bexni = bDotN[klp - tp_.ztMin] * s_.wInt[lp];

    double xp = sg_.rcosuv[klp];
    double yp = sg_.rsinuv[klp];

    // index of toridal field period; 0, 1, ..., (nfp-1)
    int p = 0;

    // xper, yper are the Cartesian coordinates of the source point rotated
    // into field period p. sxsave, sysave are the Cartesian x and y components
    // of the surface normal there: rcosuv and rsinuv carry r1b times the
    // cosine and sine of the toroidal angle, so dividing by r1b turns
    // (snr, snv) into the Cartesian pair rotated by that angle.
    double xper = xp * sg_.cos_per[p] - yp * sg_.sin_per[p];
    double yper = xp * sg_.sin_per[p] + yp * sg_.cos_per[p];

    double sxsave =
        (sg_.snr[klp - tp_.ztMin] * xper - sg_.snv[klp - tp_.ztMin] * yper) /
        sg_.r1b[klp];
    double sysave =
        (sg_.snr[klp - tp_.ztMin] * yper + sg_.snv[klp - tp_.ztMin] * xper) /
        sg_.r1b[klp];

    for (int kl = 0; kl < theta_by_nzeta; ++kl) {
      gsave[kl] = sg_.rzb2[klp] + sg_.rzb2[kl] - 2 * sg_.z1b[kl] * sg_.z1b[klp];
    }

    for (int kl = 0; kl < theta_by_nzeta; ++kl) {
      dsave[kl] =
          sg_.drv[klp - tp_.ztMin] + sg_.z1b[kl] * sg_.snz[klp - tp_.ztMin];
    }

    auto do_loop = [&](int kl_start, int kl_end) {
      // NOTE: kl is incremented by the loop body
      for (int kl = kl_start; kl < kl_end;) {
        // we can compute these once at the beginning
        const int l = kl / s_.nZeta;
        const int delta_l = (l - lp + s_.nThetaEven) % s_.nThetaEven;

        // In the following, loops over k are split to help auto-vectorization
        // kl is increased together with k and reset between loops as needed.
        const int k_start = kl % s_.nZeta;
        const int k_end = std::min(s_.nZeta, kl_end - l * s_.nZeta);
        const int kl_backup = kl;

        for (int k = k_start; k < k_end; ++k, ++kl) {
          const int delta_k = (k - kp + s_.nZeta) % s_.nZeta;
          ga1_buf[k] =
              sg_.guu[klp - tp_.ztMin] * tanu[delta_l] * tanu[delta_l] +
              sg_.guv[klp - tp_.ztMin] * tanu[delta_l] * tanv[delta_k] +
              sg_.gvv[klp - tp_.ztMin] * tanv[delta_k] * tanv[delta_k];

          ga2_buf[k] =
              sg_.auu[klp - tp_.ztMin] * tanu[delta_l] * tanu[delta_l] +
              sg_.auv[klp - tp_.ztMin] * tanu[delta_l] * tanv[delta_k] +
              sg_.avv[klp - tp_.ztMin] * tanv[delta_k] * tanv[delta_k];
        }

        for (int k = k_start; k < k_end; ++k, ++kl) {
          ga2_buf[k] /= ga1_buf[k];
          ga1_buf[k] = 1.0 / sqrt(ga1_buf[k]);
        }

        kl = kl_backup;
        for (int k = k_start; k < k_end; ++k, ++kl) {
          ftemp_buf[k] =
              1.0 /
              (gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
          htemp_buf[k] = sqrt(ftemp_buf[k]);
        }

        kl = kl_backup;
        for (int k = k_start; k < k_end; ++k, ++kl) {
          const int ip = ip_idx_base + kl;

          // 2 pi / nfp is the toroidal step of the sum over the nfp
          // field-period images, which turns that sum into the toroidal
          // integral over the whole machine.
          greenp[ip] +=
              twopidivnfp * (htemp_buf[k] * ftemp_buf[k] *
                                 (sg_.rcosuv[kl] * sxsave +
                                  sg_.rsinuv[kl] * sysave + dsave[kl]) -
                             ga1_buf[k] * ga2_buf[k]);
          const double g = twopidivnfp * (htemp_buf[k] - ga1_buf[k]);
          gstore[kl] += bexni * g;
        }
      }  // kl
    };

    // first field period
    // 0 < kl < klp
    do_loop(/*kl_start*/ 0, /*kl_end*/ klp);
    // klp < kl < theta_by_nzeta
    do_loop(/*kl_start*/ klp + 1, /*kl_end*/ theta_by_nzeta);

    // The analytic approximation vanishes with a kink on the lines
    // u - u' = \pi and v - v' = \pi, where the Green's function is smooth: 2
    // tan(\Delta u/2) grows as 4/|\Delta u - \pi|, so 1/\sqrt{ga1} has the
    // slope 1/(4 \sqrt{guu}) on either side of the u line, ga2/\sqrt{ga1} has
    // auu/(4 guu^{3/2}), and the v line has the same with gvv and avv. The
    // trapezoidal sum misses h^2 s/6 at a kink of slope s on a node and
    // counts h^2 s/12 too much at a kink midway between two nodes, so the
    // approximation takes h s/6 on the kink nodes, the v value where the two
    // lines cross, and h s/24 less on the two nodes next to a midway v kink.
    {
      const int klpRel = klp - tp_.ztMin;
      const double kink_u =
          twopidivnfp * poloidal_step / (24.0 * std::sqrt(sg_.guu[klpRel]));
      const double kink_du = kink_u * sg_.auu[klpRel] / sg_.guu[klpRel];
      const double kink_v =
          twopidivnfp * toroidal_step / (24.0 * std::sqrt(sg_.gvv[klpRel]));
      const double kink_dv = kink_v * sg_.avv[klpRel] / sg_.gvv[klpRel];
      const int l_kink = (lp + s_.nThetaEven / 2) % s_.nThetaEven;
      for (int k = 0; k < s_.nZeta; ++k) {
        const int kl = l_kink * s_.nZeta + k;
        greenp[ip_idx_base + kl] -= kink_du;
        gstore[kl] -= bexni * kink_u;
      }
      if (s_.nZeta % 2 == 0) {
        const int k_kink = (kp + s_.nZeta / 2) % s_.nZeta;
        const int kl_cross = l_kink * s_.nZeta + k_kink;
        greenp[ip_idx_base + kl_cross] += kink_du;
        gstore[kl_cross] += bexni * kink_u;
        for (int l = 0; l < s_.nThetaEven; ++l) {
          const int kl = l * s_.nZeta + k_kink;
          greenp[ip_idx_base + kl] -= kink_dv;
          gstore[kl] -= bexni * kink_v;
        }
      } else {
        for (const int dk : {(s_.nZeta - 1) / 2, (s_.nZeta + 1) / 2}) {
          const int k_mid = (kp + dk) % s_.nZeta;
          for (int l = 0; l < s_.nThetaEven; ++l) {
            if (l == l_kink) {
              continue;
            }
            const int kl = l * s_.nZeta + k_mid;
            greenp[ip_idx_base + kl] += 0.25 * kink_dv;
            gstore[kl] += bexni * 0.25 * kink_v;
          }
        }
      }
    }

    // The sum skips the source point, where the remainders tend to a
    // direction-dependent limit and rise linearly; their defects for the
    // weights 1, theta - theta' and zeta - zeta' go onto the source node and
    // its four neighbours.
    {
      const int klpRel = klp - tp_.ztMin;
      const SourcePoint source = {xp,
                                  yp,
                                  sg_.z1b[klp],
                                  sxsave,
                                  sysave,
                                  sg_.snz[klpRel],
                                  sg_.cos_phi[kp],
                                  sg_.sin_phi[kp],
                                  1.0 / s_.nfp,
                                  sg_.guu[klpRel],
                                  sg_.guv[klpRel],
                                  sg_.gvv[klpRel],
                                  sg_.auu[klpRel],
                                  sg_.auv[klpRel],
                                  sg_.avv[klpRel]};
      const SingularDefects defects =
          ComputeSingularDefects(TaylorPolynomialAt(sg_.rtay, klpRel),
                                 TaylorPolynomialAt(sg_.ztay, klpRel), source,
                                 poloidal_step, toroidal_step);
      const std::array<double, 5> single_layer =
          StencilWeights(defects.single_layer, poloidal_step, toroidal_step);
      const std::array<double, 5> double_layer =
          StencilWeights(defects.double_layer, poloidal_step, toroidal_step);
      const int l_next = (lp + 1) % s_.nThetaEven;
      const int l_prev = (lp + s_.nThetaEven - 1) % s_.nThetaEven;
      const int k_next = (kp + 1) % s_.nZeta;
      const int k_prev = (kp + s_.nZeta - 1) % s_.nZeta;
      const std::array<int, 5> nodes = {
          lp * s_.nZeta + kp, l_next * s_.nZeta + kp, l_prev * s_.nZeta + kp,
          lp * s_.nZeta + k_next, lp * s_.nZeta + k_prev};
      for (int q = 0; q < 5; ++q) {
        greenp[ip_idx_base + nodes[q]] += twopidivnfp * double_layer[q];
        gstore[nodes[q]] += bexni * twopidivnfp * single_layer[q];
      }
    }

    // all following field periods
    for (int p = 1; p < s_.nfp; ++p) {
      double xper = xp * sg_.cos_per[p] - yp * sg_.sin_per[p];
      double yper = xp * sg_.sin_per[p] + yp * sg_.cos_per[p];
      double sxsave =
          (sg_.snr[klp - tp_.ztMin] * xper - sg_.snv[klp - tp_.ztMin] * yper) /
          sg_.r1b[klp];
      double sysave =
          (sg_.snr[klp - tp_.ztMin] * yper + sg_.snv[klp - tp_.ztMin] * xper) /
          sg_.r1b[klp];

      for (int kl = 0; kl < theta_by_nzeta; ++kl) {
        double ftemp =
            1.0 /
            (gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
        double htemp = sqrt(ftemp);

        // 2 pi / nfp is the toroidal step of the sum over the nfp
        // field-period images, which turns that sum into the toroidal integral
        // over the whole machine.
        greenp[ip_idx_base + kl] +=
            twopidivnfp * htemp * ftemp *
            (sg_.rcosuv[kl] * sxsave + sg_.rsinuv[kl] * sysave + dsave[kl]);
        const double g = twopidivnfp * htemp;
        gstore[kl] += bexni * g;
      }  // kl
    }  // p
  }  // klp
}

void RegularizedIntegrals::updateAxisymmetric(const Eigen::VectorXd& bDotN) {
  // Axisymmetric (nZeta == 1) Green's-function regularization. The single
  // toroidal plane of the surface grid does not resolve the toroidal angle, so
  // the toroidal integral of the regularized Green's function is performed by
  // summing over nvper_ equally-spaced toroidal images of the evaluation point
  // (educational_VMEC greenf with nvper = 64 for the tokamak). The analytic
  // approximation is subtracted at every image; its closed-form toroidal
  // integral is added back in SingularIntegrals.
  const int numLocal = tp_.ztMax - tp_.ztMin;
  const int nThetaEven = s_.nThetaEven;  // == theta_by_nzeta since nZeta == 1

  absl::c_fill_n(greenp, numLocal * nThetaEven, 0);
  absl::c_fill_n(gstore, nThetaEven, 0);

  // 2 pi / nvper_ is the toroidal step of the sum over the nvper_ images,
  // which turns that sum into the toroidal integral over the whole machine.
  const double toroidal_measure = 2.0 * M_PI / nvper_;
  const double poloidal_step = 2.0 * M_PI / nThetaEven;

  for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
    const int ip_idx_base = (klp - tp_.ztMin) * nThetaEven;
    const int klpRel = klp - tp_.ztMin;
    const int lp = klp;  // k == 0 since nZeta == 1

    const double bexni = bDotN[klpRel] * s_.wInt[lp];

    // source slice at zeta = 0: rcosuv == r1b, rsinuv == 0
    const double xp = sg_.rcosuv[klp];
    const double yp = sg_.rsinuv[klp];

    for (int kl = 0; kl < nThetaEven; ++kl) {
      gsave[kl] = sg_.rzb2[klp] + sg_.rzb2[kl] - 2 * sg_.z1b[kl] * sg_.z1b[klp];
      dsave[kl] = sg_.drv[klpRel] + sg_.z1b[kl] * sg_.snz[klpRel];
    }

    // integrate over the toroidal direction by summing the rotated images of
    // the evaluation point around the whole torus
    for (int p = 0; p < nvper_; ++p) {
      const double cosper = std::cos(toroidal_measure * p);
      const double sinper = std::sin(toroidal_measure * p);

      const double xper = xp * cosper - yp * sinper;
      const double yper = xp * sinper + yp * cosper;

      const double sxsave =
          (sg_.snr[klpRel] * xper - sg_.snv[klpRel] * yper) / sg_.r1b[klp];
      const double sysave =
          (sg_.snr[klpRel] * yper + sg_.snv[klpRel] * xper) / sg_.r1b[klp];

      const double tanv_p = tanv_per_[p];

      for (int kl = 0; kl < nThetaEven; ++kl) {
        // The exact singularity (image coincides with the source point) is
        // handled analytically in SingularIntegrals; skip it here.
        if (p == 0 && kl == klp) {
          continue;
        }

        const int delta_l = (kl - lp + nThetaEven) % nThetaEven;

        double ga1 = sg_.guu[klpRel] * tanu[delta_l] * tanu[delta_l] +
                     sg_.guv[klpRel] * tanu[delta_l] * tanv_p +
                     sg_.gvv[klpRel] * tanv_p * tanv_p;
        double ga2 = sg_.auu[klpRel] * tanu[delta_l] * tanu[delta_l] +
                     sg_.auv[klpRel] * tanu[delta_l] * tanv_p +
                     sg_.avv[klpRel] * tanv_p * tanv_p;
        ga2 /= ga1;
        ga1 = 1.0 / std::sqrt(ga1);

        const double ftemp =
            1.0 /
            (gsave[kl] - 2 * (xper * sg_.rcosuv[kl] + yper * sg_.rsinuv[kl]));
        const double htemp = std::sqrt(ftemp);

        greenp[ip_idx_base + kl] +=
            toroidal_measure * (htemp * ftemp *
                                    (sg_.rcosuv[kl] * sxsave +
                                     sg_.rsinuv[kl] * sysave + dsave[kl]) -
                                ga1 * ga2);
        gstore[kl] += bexni * toroidal_measure * (htemp - ga1);
      }  // kl
    }  // p

    // The kinks of the analytic approximation as in update(), with the image
    // spacing as the toroidal step: the u line crosses every image but the
    // one at v - v' = \pi, which carries the v line.
    const double kink_u =
        toroidal_measure * poloidal_step / (24.0 * std::sqrt(sg_.guu[klpRel]));
    const double kink_du = kink_u * sg_.auu[klpRel] / sg_.guu[klpRel];
    const double kink_v = toroidal_measure * toroidal_measure /
                          (24.0 * std::sqrt(sg_.gvv[klpRel]));
    const double kink_dv = kink_v * sg_.avv[klpRel] / sg_.gvv[klpRel];
    const int l_kink = (lp + nThetaEven / 2) % nThetaEven;
    greenp[ip_idx_base + l_kink] -= (nvper_ - 1) * kink_du;
    gstore[l_kink] -= bexni * (nvper_ - 1) * kink_u;
    for (int kl = 0; kl < nThetaEven; ++kl) {
      greenp[ip_idx_base + kl] -= kink_dv;
      gstore[kl] -= bexni * kink_v;
    }

    // The defects of the source point as in update(), with the image spacing
    // as the step in the toroidal angle, along which the modes do not vary.
    {
      const SourcePoint source = {
          xp,
          yp,
          sg_.z1b[klp],
          (sg_.snr[klpRel] * xp - sg_.snv[klpRel] * yp) / sg_.r1b[klp],
          (sg_.snr[klpRel] * yp + sg_.snv[klpRel] * xp) / sg_.r1b[klp],
          sg_.snz[klpRel],
          1.0,
          0.0,
          1.0,
          sg_.guu[klpRel],
          sg_.guv[klpRel],
          sg_.gvv[klpRel],
          sg_.auu[klpRel],
          sg_.auv[klpRel],
          sg_.avv[klpRel]};
      const SingularDefects defects =
          ComputeSingularDefects(TaylorPolynomialAt(sg_.rtay, klpRel),
                                 TaylorPolynomialAt(sg_.ztay, klpRel), source,
                                 poloidal_step, toroidal_measure);
      const std::array<double, 5> single_layer =
          StencilWeights(defects.single_layer, poloidal_step, toroidal_measure);
      const std::array<double, 5> double_layer =
          StencilWeights(defects.double_layer, poloidal_step, toroidal_measure);
      const std::array<int, 3> nodes = {lp, (lp + 1) % nThetaEven,
                                        (lp + nThetaEven - 1) % nThetaEven};
      for (int q = 0; q < 3; ++q) {
        greenp[ip_idx_base + nodes[q]] += toroidal_measure * double_layer[q];
        gstore[nodes[q]] += bexni * toroidal_measure * single_layer[q];
      }
    }
  }  // klp
}

}  // namespace vmecpp
