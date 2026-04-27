// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_BOUNDARIES_GUESS_MAGNETIC_AXIS_H_
#define VMECPP_VMEC_BOUNDARIES_GUESS_MAGNETIC_AXIS_H_

#include <Eigen/Dense>
#include <vector>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/real_type.h"

namespace vmecpp {

// workspace for RecomputeMagneticAxisToFixJacobianSign
struct RecomputeAxisWorkspace {
  // geometry of current (= from initial guess) axis: R, Z
  std::vector<real_t> r_axis;
  std::vector<real_t> z_axis;

  // geometry of boundary: R, Z, dR/dTheta, dR/dZeta
  std::vector<std::vector<real_t> > r_lcfs;
  std::vector<std::vector<real_t> > z_lcfs;
  std::vector<std::vector<real_t> > d_r_d_theta_lcfs;
  std::vector<std::vector<real_t> > d_z_d_theta_lcfs;

  // geometry at ~mid-radius: R, Z, dR/dTheta, dR/dZeta, dR/ds, dZ/ds
  std::vector<std::vector<real_t> > r_half;
  std::vector<std::vector<real_t> > z_half;
  std::vector<std::vector<real_t> > d_r_d_theta_half;
  std::vector<std::vector<real_t> > d_z_d_theta_half;
  std::vector<std::vector<real_t> > d_r_d_s_half;
  std::vector<std::vector<real_t> > d_z_d_s_half;

  // cylindrical part of the Jacobian
  // tau0 is the static part,
  // tau is the full thing
  std::vector<std::vector<real_t> > tau0;
  std::vector<std::vector<real_t> > tau;

  // [nZeta] realspace geometry of the guess for the magnetic axis
  std::vector<real_t> new_r_axis;
  std::vector<real_t> new_z_axis;

  // [ntor + 1] Fourier coefficients of new axis
  std::vector<real_t> new_raxis_c;
  std::vector<real_t> new_raxis_s;
  std::vector<real_t> new_zaxis_s;
  std::vector<real_t> new_zaxis_c;
};

// This object is initialized with an initial guess for the magnetic axis
// geometry via setupFromIndata() that is provided by the user. This method
// replaces the axis geometry in this object with a new guess for the magnetic
// axis geometry, if the current axis geometry leads to initial sign change of
// Jacobian somewhere in the plasma volume. It does so by performing a grid
// search along R and Z in each phi plane for locations of the magnetic axis
// which yield a value for the Jacobian with the correct sign
// (sign_of_jacobian). Then, the axis position so that the minimum value of the
// Jacobian (in each poloidal cross section) is maximized. The computation is
// done assuming the given number_of_flux_surfaces.
RecomputeAxisWorkspace RecomputeMagneticAxisToFixJacobianSign(
    int number_of_flux_surfaces, int sign_of_jacobian, const Sizes& s,
    const FourierBasisFastPoloidal& t,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& rbcc,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& rbss,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& rbsc,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& rbcs,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& zbsc,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& zbcs,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& zbcc,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& zbss,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& raxis_c,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& raxis_s,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& zaxis_s,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& zaxis_c);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_BOUNDARIES_GUESS_MAGNETIC_AXIS_H_
