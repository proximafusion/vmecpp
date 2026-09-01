// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_BOOZER_BOOZER_H_
#define VMECPP_VMEC_BOOZER_BOOZER_H_

#include <Eigen/Dense>
#include <vector>

#include "absl/status/statusor.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"

namespace vmecpp {

// Boozer coordinates of the flux surfaces of a converged stellarator-symmetric
// equilibrium: the angle transformation and the Fourier spectra of |B|, R, Z,
// the transformation function and the Jacobian in the Boozer angles.
//
// With VMEC's straight-field-line angle theta* = theta + lambda, the Boozer
// angles are theta_B = theta + lambda + iota * nu and zeta_B = zeta + nu, with
// nu the flux-surface function that turns the covariant B_theta and B_zeta into
// the flux functions I and G: nu = (w - I lambda) / (G + iota I), where w is
// the periodic part of the magnetic scalar potential on the surface,
// d w / d theta = B_theta - I and d w / d zeta = B_zeta - G. Each spectrum is
// the direct quadrature over the VMEC angles with the Jacobian of the angle
// map, so no inverse of the map is needed. Mode numbers follow the wout
// convention: coefficients of cos(m theta_B - n zeta_B) or the sine, with n
// carrying the field-period factor.
struct BoozerCoordinates {
  int nfp = 0;
  int mboz = 0;
  int nboz = 0;

  // [mnboz] Boozer mode numbers: m = 0 with n = 0..nboz*nfp, then
  // m = 1..mboz-1 with n = -nboz*nfp..nboz*nfp
  Eigen::VectorXi xm_b;
  Eigen::VectorXi xn_b;

  // [num_surfaces] column of the wout half-grid arrays of each surface
  Eigen::VectorXi surfaces;

  // [num_surfaces] rotational transform and the Boozer currents G = B_zeta
  // and I = B_theta of each surface
  Eigen::VectorXd iota_b;
  Eigen::VectorXd g_b;
  Eigen::VectorXd i_b;

  // [num_surfaces] relative spread of sqrt(g_B) |B|^2 over the surface, which
  // is a flux function in exact Boozer coordinates; the size of the
  // discretization error of the transformation
  Eigen::VectorXd jacobian_spread;

  // [mnboz, num_surfaces] spectra in the Boozer angles
  RowMatrixXd bmnc_b;   // |B|, cosine
  RowMatrixXd rmnc_b;   // R, cosine
  RowMatrixXd zmns_b;   // Z, sine
  RowMatrixXd numns_b;  // nu, sine
  // The Boozer Jacobian (G + iota I) / |B|^2, whose radial coordinate is the
  // toroidal flux per radian, expanded in the Boozer angles as booz_xform
  // reports it; cosine
  RowMatrixXd gmnc_b;
};

// Transforms the half-grid surfaces given by their wout columns (all of them,
// 1..ns-1, when `surfaces` is empty) to Boozer coordinates with mboz poloidal
// modes and nboz toroidal modes per field period.
absl::StatusOr<BoozerCoordinates> BoozerTransform(
    const WOutFileContents& wout, int mboz, int nboz,
    const std::vector<int>& surfaces = {});

}  // namespace vmecpp

#endif  // VMECPP_VMEC_BOOZER_BOOZER_H_
