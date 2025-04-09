// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_BOUNDARIES_BOUNDARIES_H_
#define VMECPP_VMEC_BOUNDARIES_BOUNDARIES_H_

#include <vector>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

/** radial boundaries: magnetic axis and last closed flux surface */
class Boundaries {
 public:
  Boundaries(const Sizes* s, const FourierBasisFastPoloidal* t,
             int sign_of_jacobian);

  bool setupFromIndata(const VmecINDATA& id, bool verbose = true);
  void ensureM1Constrained(double scaling_factor);

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

  std::vector<double> raxis_c;
  std::vector<double> zaxis_s;
  std::vector<double> raxis_s;
  std::vector<double> zaxis_c;

  std::vector<double> rbcc;
  std::vector<double> rbss;
  std::vector<double> rbsc;
  std::vector<double> rbcs;

  std::vector<double> zbsc;
  std::vector<double> zbcs;
  std::vector<double> zbcc;
  std::vector<double> zbss;

 private:
  const Sizes& s_;
  const FourierBasisFastPoloidal& t_;

  int sign_of_jacobian_;

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
