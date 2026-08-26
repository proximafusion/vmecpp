// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_GEOMETRY_GEOMETRY_H_
#define VMECPP_VMEC_GEOMETRY_GEOMETRY_H_

#include <array>
#include <vector>

namespace vmecpp {

struct GeometryDimensions {
  int ns;
  int mpol;
  int ntor;
  int nfp;
};

struct GeometryCoefficients {
  std::vector<double> r_cc;
  std::vector<double> r_ss;
  std::vector<double> r_sc;
  std::vector<double> r_cs;
  std::vector<double> z_sc;
  std::vector<double> z_cs;
  std::vector<double> z_cc;
  std::vector<double> z_ss;
  std::vector<double> lambda_sc;
  std::vector<double> lambda_cs;
  std::vector<double> lambda_cc;
  std::vector<double> lambda_ss;
};

// Minimal equilibrium geometry independent of any output-file schema.
struct Geometry {
  GeometryDimensions dimensions;
  std::vector<double> toroidal_flux;
  std::vector<double> poloidal_flux;
  GeometryCoefficients coefficients;
};

// Value and derivatives with respect to (s, theta, zeta).
//
// The entries are ordered as value, ds, dtheta, dzeta, dss, ds_dtheta,
// ds_dzeta, dtheta2, dtheta_dzeta, and dzeta2.  Keeping the jet in a fixed
// array makes the C ABI and the JAX representation identical.
inline constexpr int kGeometryJetSize = 10;
using GeometryJet = std::array<double, kGeometryJetSize>;

struct GeometryPoint {
  GeometryJet r;
  GeometryJet z;
  GeometryJet lambda;
  GeometryJet toroidal_flux;
  GeometryJet poloidal_flux;
};

struct GeometryVjp {
  Geometry geometry;
  std::array<double, 3> coordinates;
};

GeometryPoint EvaluateGeometry(const Geometry& geometry, double s, double theta,
                               double zeta);

// Transpose of EvaluateGeometry. The cotangent may seed values and first
// derivatives. The returned coordinate order is (s, theta, zeta).
GeometryVjp EvaluateGeometryVjp(const Geometry& geometry, double s,
                                double theta, double zeta,
                                const GeometryPoint& cotangent);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_GEOMETRY_GEOMETRY_H_
