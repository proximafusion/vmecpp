// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_BOUNDARIES_BOUNDARIES_H_
#define VMECPP_VMEC_BOUNDARIES_BOUNDARIES_H_

#include <Eigen/Dense>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// Boundary spectral width above which setupFromIndata warns that a
// fixed-boundary run may converge poorly or resolve the physics incorrectly.
// Every input bundled with VMEC++ stays below this.
inline constexpr double kSpectrallyDenseBoundaryThreshold = 2.0;

/** radial boundaries: magnetic axis and last closed flux surface */
class Boundaries {
 public:
  Boundaries(const Sizes* s, const FourierBasisFastPoloidal* t,
             int sign_of_jacobian);

  bool setupFromIndata(const VmecINDATA& id, bool verbose = true);

  // Set this object up as the inner boundary of a plasma without a magnetic
  // axis, from id.rbc_inner, id.zbs_inner and, with lasym, id.rbs_inner and
  // id.zbc_inner, with the poloidal angle shift that boundary.setupFromIndata
  // gave the boundary and its orientation: have_to_flip_theta is the value
  // that call returned. The axis coefficients are zero.
  void setupInnerFromIndata(const VmecINDATA& id, const Boundaries& boundary,
                            bool have_to_flip_theta);

  void ensureM1Constrained(double scaling_factor);

  // Spectral width <M> of the boundary, the same quantity VMEC++ reports for
  // each flux surface in the <M> column of the iteration table. Only meaningful
  // once setupFromIndata has filled the coefficients below.
  double ComputeSpectralWidth(int p = 4, int q = 1) const;

  // This object is initialized with an initial guess for the magnetic axis
  // geometry via setupFromIndata() that is provided by the user. This method
  // replaces the axis geometry in this object with a new guess for the magnetic
  // axis geometry, if the current axis geometry leads to initial sign change of
  // Jacobian somewhere in the plasma volume. It does so by performing a grid
  // search along R and Z in each phi plane for locations of the magnetic axis
  // which yield a value for the Jacobian with the correct sign
  // (sign_of_jacobian). Then, the axis position so that the minimum value of
  // the Jacobian (in each poloidal cross section) is maximized. The computation
  // is done assuming the given number_of_flux_surfaces.
  void RecomputeMagneticAxisToFixJacobianSign(int number_of_flux_surfaces,
                                              int sign_of_jacobian);

  Eigen::VectorXd raxis_c;
  Eigen::VectorXd zaxis_s;
  Eigen::VectorXd raxis_s;
  Eigen::VectorXd zaxis_c;

  Eigen::VectorXd rbcc;
  Eigen::VectorXd rbss;
  Eigen::VectorXd rbsc;
  Eigen::VectorXd rbcs;

  Eigen::VectorXd zbsc;
  Eigen::VectorXd zbcs;
  Eigen::VectorXd zbcc;
  Eigen::VectorXd zbss;

 private:
  const Sizes& s_;
  const FourierBasisFastPoloidal& t_;

  int sign_of_jacobian_;

  // shift of the poloidal angle applied to the input boundary, which puts an
  // asymmetric boundary into the m = 1 gauge; 0 without lasym
  double theta_shift_ = 0.0;

  // Parse the input boundary Fourier coefficients into the VMEC-internal array
  // format:
  //
  // <pre>
  // R =   RBCC*COS(M*U)*COS(N*V) + RBSS*SIN(M*U)*SIN(N*V)
  //     + RBSC*SIN(M*U)*COS(N*V) + RBCS*COS(M*U)*SIN(N*V)
  // Z =   ZBSC*SIN(M*U)*COS(N*V) + ZBCS*COS(M*U)*SIN(N*V)
  //     + ZBCC*COS(M*U)*COS(N*V) + ZBSS*SIN(M*U)*SIN(N*V)
  // </pre>
  //
  // id: contents of INDATA namelist (VMEC input)
  void parseToInternalArrays(const VmecINDATA& id, bool verbose = true);

  // Sort the boundary coefficients rbc, zbs and, with lasym, rbs, zbc in the
  // layout of the input, shifted in the poloidal angle by delta, into the
  // internal arrays above.
  void sortIntoInternalArrays(const RowMatrixXd& rbc_in,
                              const RowMatrixXd& zbs_in,
                              const std::optional<RowMatrixXd>& rbs_in,
                              const std::optional<RowMatrixXd>& zbc_in,
                              double delta);

  // Check if the sign of the Jacobian of the given input boundary coefficient
  // matches the expected internal sign of the Jacobian.
  // Returns true if we need to flip the poloidal angle direction in the input
  // coefficients; false otherwise.
  bool checkSignOfJacobian();

  // Reverse the implied sign of the poloidal angle theta
  // in the given input boundary Fourier coefficients.
  void flipTheta();
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_BOUNDARIES_BOUNDARIES_H_
