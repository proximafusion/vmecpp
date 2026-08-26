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
  std::array<int, 4> indices;
  std::array<double, 4> value;
  std::array<double, 4> first;
  std::array<double, 4> second;
  int count;
};

RadialWeights GetRadialWeights(int ns, double s) {
  if (ns < 2) {
    throw std::invalid_argument("geometry requires at least two radial points");
  }
  if (s < 0.0 || s > 1.0) {
    throw std::out_of_range("s must be in [0, 1]");
  }
  const double scale = ns - 1;
  const double scaled = s * scale;
  if (ns < 4) {
    const int inner = std::min(static_cast<int>(scaled), ns - 2);
    const double outer_weight = scaled - inner;
    return {.indices = {inner, inner + 1, 0, 0},
            .value = {1.0 - outer_weight, outer_weight, 0.0, 0.0},
            .first = {-scale, scale, 0.0, 0.0},
            .second = {0.0, 0.0, 0.0, 0.0},
            .count = 2};
  }
  const int start = std::clamp(static_cast<int>(scaled) - 1, 0, ns - 4);
  const double x = scaled - start;
  RadialWeights result{.indices = {start, start + 1, start + 2, start + 3},
                       .value = {},
                       .first = {},
                       .second = {},
                       .count = 4};
  for (int i = 0; i < 4; ++i) {
    double denominator = 1.0;
    for (int j = 0; j < 4; ++j) {
      if (j != i) denominator *= i - j;
    }
    double value = 1.0;
    for (int j = 0; j < 4; ++j) {
      if (j != i) value *= x - j;
    }
    double first = 0.0;
    double second = 0.0;
    for (int omitted = 0; omitted < 4; ++omitted) {
      if (omitted == i) continue;
      double term = 1.0;
      for (int j = 0; j < 4; ++j) {
        if (j != i && j != omitted) term *= x - j;
      }
      first += term;
      for (int omitted_second = 0; omitted_second < 4; ++omitted_second) {
        if (omitted_second == i || omitted_second == omitted) continue;
        double second_term = 1.0;
        for (int j = 0; j < 4; ++j) {
          if (j != i && j != omitted && j != omitted_second) {
            second_term *= x - j;
          }
        }
        second += second_term;
      }
    }
    result.value[i] = value / denominator;
    result.first[i] = scale * first / denominator;
    result.second[i] = scale * scale * second / denominator;
  }
  return result;
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
  GeometryJet result{};
  for (int i = 0; i < radial.count; ++i) {
    result[0] += radial.value[i] * profile[radial.indices[i]];
    result[1] += radial.first[i] * profile[radial.indices[i]];
    result[4] += radial.second[i] * profile[radial.indices[i]];
  }
  return result;
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
  double ds2;
};

BasisJet MakeBasisJet(const std::array<double, 3>& poloidal,
                      const std::array<double, 3>& toroidal,
                      const std::array<double, 4>& coefficients,
                      const RadialWeights& radial) {
  double coefficient = 0.0;
  double coefficient_s = 0.0;
  double coefficient_ss = 0.0;
  for (int i = 0; i < radial.count; ++i) {
    coefficient += radial.value[i] * coefficients[i];
    coefficient_s += radial.first[i] * coefficients[i];
    coefficient_ss += radial.second[i] * coefficients[i];
  }
  return {.value = coefficient * poloidal[0] * toroidal[0],
          .ds = coefficient_s * poloidal[0] * toroidal[0],
          .dtheta = coefficient * poloidal[1] * toroidal[0],
          .dzeta = coefficient * poloidal[0] * toroidal[1],
          .ds_dtheta = coefficient_s * poloidal[1] * toroidal[0],
          .ds_dzeta = coefficient_s * poloidal[0] * toroidal[1],
          .dtheta2 = coefficient * poloidal[2] * toroidal[0],
          .dtheta_dzeta = coefficient * poloidal[1] * toroidal[1],
          .dzeta2 = coefficient * poloidal[0] * toroidal[2],
          .ds2 = coefficient_ss * poloidal[0] * toroidal[0]};
}

void AddBasis(const std::vector<double>& coefficients, bool sine_m, bool sine_n,
              int m, int n, double theta, double zeta,
              const GeometryDimensions& dimensions, const RadialWeights& radial,
              GeometryJet& m_value) {
  if (coefficients.empty()) return;
  const auto poloidal = TrigDerivatives(sine_m, m, theta);
  const auto toroidal = TrigDerivatives(sine_n, n * dimensions.nfp, zeta);
  std::array<double, 4> radial_coefficients{};
  for (int i = 0; i < radial.count; ++i) {
    radial_coefficients[i] =
        coefficients[CoefficientIndex(dimensions, radial.indices[i], m, n)];
  }
  const BasisJet basis =
      MakeBasisJet(poloidal, toroidal, radial_coefficients, radial);
  m_value[0] += basis.value;
  m_value[1] += basis.ds;
  m_value[2] += basis.dtheta;
  m_value[3] += basis.dzeta;
  m_value[4] += basis.ds2;
  m_value[5] += basis.ds_dtheta;
  m_value[6] += basis.ds_dzeta;
  m_value[7] += basis.dtheta2;
  m_value[8] += basis.dtheta_dzeta;
  m_value[9] += basis.dzeta2;
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
  std::array<int, 4> indices{};
  std::array<double, 4> radial_coefficients{};
  for (int i = 0; i < radial.count; ++i) {
    indices[i] = CoefficientIndex(dimensions, radial.indices[i], m, n);
    radial_coefficients[i] = m_coefficients[indices[i]];
  }
  const BasisJet basis =
      MakeBasisJet(poloidal, toroidal, radial_coefficients, radial);

  const double angular_bar = cotangent[0] * poloidal[0] * toroidal[0] +
                             cotangent[2] * poloidal[1] * toroidal[0] +
                             cotangent[3] * poloidal[0] * toroidal[1] +
                             cotangent[4] * poloidal[0] * toroidal[0] +
                             cotangent[5] * poloidal[1] * toroidal[0] +
                             cotangent[6] * poloidal[0] * toroidal[1] +
                             cotangent[7] * poloidal[2] * toroidal[0] +
                             cotangent[8] * poloidal[1] * toroidal[1] +
                             cotangent[9] * poloidal[0] * toroidal[2];
  const double radial_bar = cotangent[1] * poloidal[0] * toroidal[0] +
                            cotangent[5] * poloidal[1] * toroidal[0] +
                            cotangent[6] * poloidal[0] * toroidal[1];
  for (int i = 0; i < radial.count; ++i) {
    m_coefficients[indices[i]] =
        radial.value[i] * angular_bar + radial.first[i] * radial_bar +
        radial.second[i] * cotangent[4] * poloidal[0] * toroidal[0];
  }

  m_coordinate_bar[0] += cotangent[0] * basis.ds + cotangent[1] * basis.ds2 +
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
  for (int i = 0; i < radial.count; ++i) {
    m_profile_bar[radial.indices[i]] +=
        radial.value[i] * cotangent[0] + radial.first[i] * cotangent[1];
    m_s_bar += profile[radial.indices[i]] * (radial.first[i] * cotangent[0] +
                                             radial.second[i] * cotangent[1]);
  }
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
  for (const GeometryJet& seed :
       {cotangent.r, cotangent.z, cotangent.lambda, cotangent.toroidal_flux,
        cotangent.poloidal_flux}) {
    for (int i = 4; i < kGeometryJetSize; ++i) {
      if (seed[i] != 0.0) {
        throw std::invalid_argument(
            "EvaluateGeometryVjp supports value and first-derivative seeds "
            "only");
      }
    }
  }
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

  // AddCoefficientVjp overwrites the active radial entries, so use a
  // temporary and accumulate them into the result.
  auto accumulate_quantity = [&](const std::vector<double>& primal,
                                 std::vector<double>& bar, bool sine_m,
                                 bool sine_n, int m, int n,
                                 const GeometryJet& seed) {
    if (primal.empty()) return;
    std::vector<double> contribution = primal;
    AddCoefficientVjp(contribution, sine_m, sine_n, m, n, theta, zeta,
                      geometry.dimensions, radial, seed, result.coordinates);
    for (int i = 0; i < radial.count; ++i) {
      const int index =
          CoefficientIndex(geometry.dimensions, radial.indices[i], m, n);
      bar[index] += contribution[index];
    }
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
