// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/geometry/geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace vmecpp {
namespace {

struct RadialWeights {
  int inner;
  int outer;
  double inner_weight;
  double outer_weight;
  double derivative_scale;
};

RadialWeights GetRadialWeights(int ns, double s) {
  if (ns < 2) {
    throw std::invalid_argument("geometry requires at least two radial points");
  }
  if (s < 0.0 || s > 1.0) {
    throw std::out_of_range("s must be in [0, 1]");
  }
  const double scaled = s * (ns - 1);
  const int inner = std::min(static_cast<int>(scaled), ns - 2);
  const double outer_weight = scaled - inner;
  return {.inner = inner,
          .outer = inner + 1,
          .inner_weight = 1.0 - outer_weight,
          .outer_weight = outer_weight,
          .derivative_scale = static_cast<double>(ns - 1)};
}

int CoefficientIndex(const GeometryDimensions& dimensions, int radial_index,
                     int m, int n) {
  return (radial_index * dimensions.mpol + m) * (dimensions.ntor + 1) + n;
}

void CheckSize(const std::vector<double>& values, int expected,
               const char* name, bool may_be_empty = false) {
  if ((may_be_empty && values.empty()) ||
      values.size() == static_cast<std::size_t>(expected)) {
    return;
  }
  throw std::invalid_argument(std::string(name) + " has wrong size");
}

void Validate(const Geometry& geometry) {
  const GeometryDimensions& d = geometry.dimensions;
  if (d.ns < 2 || d.mpol < 1 || d.ntor < 0 || d.nfp < 1) {
    throw std::invalid_argument("invalid geometry dimensions");
  }
  CheckSize(geometry.toroidal_flux, d.ns, "toroidal_flux");
  CheckSize(geometry.poloidal_flux, d.ns, "poloidal_flux");
  const int size = d.ns * d.mpol * (d.ntor + 1);
  CheckSize(geometry.coefficients.r_cc, size, "r_cc");
  CheckSize(geometry.coefficients.z_sc, size, "z_sc");
  CheckSize(geometry.coefficients.lambda_sc, size, "lambda_sc");
  CheckSize(geometry.coefficients.r_ss, size, "r_ss", true);
  CheckSize(geometry.coefficients.r_sc, size, "r_sc", true);
  CheckSize(geometry.coefficients.r_cs, size, "r_cs", true);
  CheckSize(geometry.coefficients.z_cs, size, "z_cs", true);
  CheckSize(geometry.coefficients.z_cc, size, "z_cc", true);
  CheckSize(geometry.coefficients.z_ss, size, "z_ss", true);
  CheckSize(geometry.coefficients.lambda_cs, size, "lambda_cs", true);
  CheckSize(geometry.coefficients.lambda_cc, size, "lambda_cc", true);
  CheckSize(geometry.coefficients.lambda_ss, size, "lambda_ss", true);
}

std::array<double, 3> TrigDerivatives(bool sine, int mode, double angle) {
  const double argument = mode * angle;
  if (sine) {
    return {std::sin(argument), mode * std::cos(argument),
            -mode * mode * std::sin(argument)};
  }
  return {std::cos(argument), -mode * std::sin(argument),
          -mode * mode * std::cos(argument)};
}

GeometryJet EvaluateProfile(const std::vector<double>& profile,
                            const RadialWeights& radial) {
  return {
      radial.inner_weight * profile[radial.inner] +
          radial.outer_weight * profile[radial.outer],
      radial.derivative_scale * (profile[radial.outer] - profile[radial.inner]),
      0.0, 0.0};
}

struct BasisJet {
  double value;
  double ds;
  double dtheta;
  double dzeta;
  double ds_dtheta;
  double ds_dzeta;
  double dtheta2;
  double dtheta_dzeta;
  double dzeta2;
};

BasisJet MakeBasisJet(const std::array<double, 3>& poloidal,
                      const std::array<double, 3>& toroidal,
                      double coefficient_inner, double coefficient_outer,
                      const RadialWeights& radial) {
  const double coefficient = radial.inner_weight * coefficient_inner +
                             radial.outer_weight * coefficient_outer;
  const double coefficient_s =
      radial.derivative_scale * (coefficient_outer - coefficient_inner);
  return {.value = coefficient * poloidal[0] * toroidal[0],
          .ds = coefficient_s * poloidal[0] * toroidal[0],
          .dtheta = coefficient * poloidal[1] * toroidal[0],
          .dzeta = coefficient * poloidal[0] * toroidal[1],
          .ds_dtheta = coefficient_s * poloidal[1] * toroidal[0],
          .ds_dzeta = coefficient_s * poloidal[0] * toroidal[1],
          .dtheta2 = coefficient * poloidal[2] * toroidal[0],
          .dtheta_dzeta = coefficient * poloidal[1] * toroidal[1],
          .dzeta2 = coefficient * poloidal[0] * toroidal[2]};
}

void AddBasis(const std::vector<double>& coefficients, bool sine_m, bool sine_n,
              int m, int n, double theta, double zeta,
              const GeometryDimensions& dimensions, const RadialWeights& radial,
              GeometryJet& m_value) {
  if (coefficients.empty()) return;
  const auto poloidal = TrigDerivatives(sine_m, m, theta);
  const auto toroidal = TrigDerivatives(sine_n, n * dimensions.nfp, zeta);
  const BasisJet basis = MakeBasisJet(
      poloidal, toroidal,
      coefficients[CoefficientIndex(dimensions, radial.inner, m, n)],
      coefficients[CoefficientIndex(dimensions, radial.outer, m, n)], radial);
  m_value[0] += basis.value;
  m_value[1] += basis.ds;
  m_value[2] += basis.dtheta;
  m_value[3] += basis.dzeta;
}

void AddCoefficientVjp(std::vector<double>& m_coefficients, bool sine_m,
                       bool sine_n, int m, int n, double theta, double zeta,
                       const GeometryDimensions& dimensions,
                       const RadialWeights& radial,
                       const GeometryJet& cotangent,
                       std::array<double, 3>& m_coordinate_bar) {
  if (m_coefficients.empty()) return;
  const auto poloidal = TrigDerivatives(sine_m, m, theta);
  const auto toroidal = TrigDerivatives(sine_n, n * dimensions.nfp, zeta);
  const int inner_index = CoefficientIndex(dimensions, radial.inner, m, n);
  const int outer_index = CoefficientIndex(dimensions, radial.outer, m, n);
  const double inner = m_coefficients[inner_index];
  const double outer = m_coefficients[outer_index];
  const BasisJet basis = MakeBasisJet(poloidal, toroidal, inner, outer, radial);

  const double angular_bar = cotangent[0] * poloidal[0] * toroidal[0] +
                             cotangent[2] * poloidal[1] * toroidal[0] +
                             cotangent[3] * poloidal[0] * toroidal[1];
  const double radial_bar =
      cotangent[1] * radial.derivative_scale * poloidal[0] * toroidal[0];
  m_coefficients[inner_index] = radial.inner_weight * angular_bar - radial_bar;
  m_coefficients[outer_index] = radial.outer_weight * angular_bar + radial_bar;

  m_coordinate_bar[0] += cotangent[0] * basis.ds +
                         cotangent[2] * basis.ds_dtheta +
                         cotangent[3] * basis.ds_dzeta;
  m_coordinate_bar[1] +=
      cotangent[0] * basis.dtheta + cotangent[1] * basis.ds_dtheta +
      cotangent[2] * basis.dtheta2 + cotangent[3] * basis.dtheta_dzeta;
  m_coordinate_bar[2] +=
      cotangent[0] * basis.dzeta + cotangent[1] * basis.ds_dzeta +
      cotangent[2] * basis.dtheta_dzeta + cotangent[3] * basis.dzeta2;
}

void AddQuantity(const GeometryCoefficients& coefficients, char quantity, int m,
                 int n, double theta, double zeta,
                 const GeometryDimensions& dimensions,
                 const RadialWeights& radial, GeometryJet& m_value) {
  const std::vector<double>* cc;
  const std::vector<double>* ss;
  const std::vector<double>* sc;
  const std::vector<double>* cs;
  if (quantity == 'r') {
    cc = &coefficients.r_cc;
    ss = &coefficients.r_ss;
    sc = &coefficients.r_sc;
    cs = &coefficients.r_cs;
  } else if (quantity == 'z') {
    cc = &coefficients.z_cc;
    ss = &coefficients.z_ss;
    sc = &coefficients.z_sc;
    cs = &coefficients.z_cs;
  } else {
    cc = &coefficients.lambda_cc;
    ss = &coefficients.lambda_ss;
    sc = &coefficients.lambda_sc;
    cs = &coefficients.lambda_cs;
  }
  AddBasis(*cc, false, false, m, n, theta, zeta, dimensions, radial, m_value);
  AddBasis(*ss, true, true, m, n, theta, zeta, dimensions, radial, m_value);
  AddBasis(*sc, true, false, m, n, theta, zeta, dimensions, radial, m_value);
  AddBasis(*cs, false, true, m, n, theta, zeta, dimensions, radial, m_value);
}

void ZeroLike(std::vector<double>& m_values) {
  std::fill(m_values.begin(), m_values.end(), 0.0);
}

void AddProfileVjp(const std::vector<double>& profile,
                   const RadialWeights& radial, const GeometryJet& cotangent,
                   std::vector<double>& m_profile_bar, double& m_s_bar) {
  m_profile_bar[radial.inner] += radial.inner_weight * cotangent[0] -
                                 radial.derivative_scale * cotangent[1];
  m_profile_bar[radial.outer] += radial.outer_weight * cotangent[0] +
                                 radial.derivative_scale * cotangent[1];
  m_s_bar += radial.derivative_scale *
             (profile[radial.outer] - profile[radial.inner]) * cotangent[0];
}

}  // namespace

GeometryPoint EvaluateGeometry(const Geometry& geometry, double s, double theta,
                               double zeta) {
  Validate(geometry);
  const RadialWeights radial = GetRadialWeights(geometry.dimensions.ns, s);
  GeometryPoint result{};
  result.toroidal_flux = EvaluateProfile(geometry.toroidal_flux, radial);
  result.poloidal_flux = EvaluateProfile(geometry.poloidal_flux, radial);
  for (int m = 0; m < geometry.dimensions.mpol; ++m) {
    for (int n = 0; n <= geometry.dimensions.ntor; ++n) {
      AddQuantity(geometry.coefficients, 'r', m, n, theta, zeta,
                  geometry.dimensions, radial, result.r);
      AddQuantity(geometry.coefficients, 'z', m, n, theta, zeta,
                  geometry.dimensions, radial, result.z);
      AddQuantity(geometry.coefficients, 'l', m, n, theta, zeta,
                  geometry.dimensions, radial, result.lambda);
    }
  }
  return result;
}

GeometryVjp EvaluateGeometryVjp(const Geometry& geometry, double s,
                                double theta, double zeta,
                                const GeometryPoint& cotangent) {
  Validate(geometry);
  const RadialWeights radial = GetRadialWeights(geometry.dimensions.ns, s);
  GeometryVjp result{.geometry = geometry, .coordinates = {0.0, 0.0, 0.0}};
  ZeroLike(result.geometry.toroidal_flux);
  ZeroLike(result.geometry.poloidal_flux);
  AddProfileVjp(geometry.toroidal_flux, radial, cotangent.toroidal_flux,
                result.geometry.toroidal_flux, result.coordinates[0]);
  AddProfileVjp(geometry.poloidal_flux, radial, cotangent.poloidal_flux,
                result.geometry.poloidal_flux, result.coordinates[0]);
  ZeroLike(result.geometry.coefficients.r_cc);
  ZeroLike(result.geometry.coefficients.r_ss);
  ZeroLike(result.geometry.coefficients.r_sc);
  ZeroLike(result.geometry.coefficients.r_cs);
  ZeroLike(result.geometry.coefficients.z_sc);
  ZeroLike(result.geometry.coefficients.z_cs);
  ZeroLike(result.geometry.coefficients.z_cc);
  ZeroLike(result.geometry.coefficients.z_ss);
  ZeroLike(result.geometry.coefficients.lambda_sc);
  ZeroLike(result.geometry.coefficients.lambda_cs);
  ZeroLike(result.geometry.coefficients.lambda_cc);
  ZeroLike(result.geometry.coefficients.lambda_ss);

  // AddCoefficientVjp overwrites the two active radial entries, so use a
  // temporary and accumulate those entries into the result.
  auto accumulate_quantity = [&](const std::vector<double>& primal,
                                 std::vector<double>& bar, bool sine_m,
                                 bool sine_n, int m, int n,
                                 const GeometryJet& seed) {
    if (primal.empty()) return;
    std::vector<double> contribution = primal;
    AddCoefficientVjp(contribution, sine_m, sine_n, m, n, theta, zeta,
                      geometry.dimensions, radial, seed, result.coordinates);
    const int inner = CoefficientIndex(geometry.dimensions, radial.inner, m, n);
    const int outer = CoefficientIndex(geometry.dimensions, radial.outer, m, n);
    bar[inner] += contribution[inner];
    bar[outer] += contribution[outer];
  };
  for (int m = 0; m < geometry.dimensions.mpol; ++m) {
    for (int n = 0; n <= geometry.dimensions.ntor; ++n) {
#define VMECPP_ACCUMULATE(PRIMAL, BAR, SM, SN, SEED) \
  accumulate_quantity(PRIMAL, BAR, SM, SN, m, n, SEED)
      VMECPP_ACCUMULATE(geometry.coefficients.r_cc,
                        result.geometry.coefficients.r_cc, false, false,
                        cotangent.r);
      VMECPP_ACCUMULATE(geometry.coefficients.r_ss,
                        result.geometry.coefficients.r_ss, true, true,
                        cotangent.r);
      VMECPP_ACCUMULATE(geometry.coefficients.r_sc,
                        result.geometry.coefficients.r_sc, true, false,
                        cotangent.r);
      VMECPP_ACCUMULATE(geometry.coefficients.r_cs,
                        result.geometry.coefficients.r_cs, false, true,
                        cotangent.r);
      VMECPP_ACCUMULATE(geometry.coefficients.z_cc,
                        result.geometry.coefficients.z_cc, false, false,
                        cotangent.z);
      VMECPP_ACCUMULATE(geometry.coefficients.z_ss,
                        result.geometry.coefficients.z_ss, true, true,
                        cotangent.z);
      VMECPP_ACCUMULATE(geometry.coefficients.z_sc,
                        result.geometry.coefficients.z_sc, true, false,
                        cotangent.z);
      VMECPP_ACCUMULATE(geometry.coefficients.z_cs,
                        result.geometry.coefficients.z_cs, false, true,
                        cotangent.z);
      VMECPP_ACCUMULATE(geometry.coefficients.lambda_cc,
                        result.geometry.coefficients.lambda_cc, false, false,
                        cotangent.lambda);
      VMECPP_ACCUMULATE(geometry.coefficients.lambda_ss,
                        result.geometry.coefficients.lambda_ss, true, true,
                        cotangent.lambda);
      VMECPP_ACCUMULATE(geometry.coefficients.lambda_sc,
                        result.geometry.coefficients.lambda_sc, true, false,
                        cotangent.lambda);
      VMECPP_ACCUMULATE(geometry.coefficients.lambda_cs,
                        result.geometry.coefficients.lambda_cs, false, true,
                        cotangent.lambda);
#undef VMECPP_ACCUMULATE
    }
  }
  return result;
}

}  // namespace vmecpp
