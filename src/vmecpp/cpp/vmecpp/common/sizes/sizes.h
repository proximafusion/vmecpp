// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_SIZES_SIZES_H_
#define VMECPP_COMMON_SIZES_SIZES_H_

#include <vector>

#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

class Sizes {
 public:
  // The array sizes etc. in VMEC are derived from a few key parameters
  // specified in the input file. These parameters are set here from the INDATA
  // object.
  explicit Sizes(const VmecINDATA& id);

  Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta);

  // inputs from INDATA

  // flag to indicate non-symmetric case
  bool lasym;

  // number of toroidal field periods
  int nfp;

  // number of poloidal Fourier harmoncis
  int mpol;

  // number of toroidal Fourier harmoncis
  int ntor;

  // number of poloidal grid points
  int ntheta;

  // number of toroidal grid points
  int nZeta;

  // derived

  // flag to indicate 3D case
  bool lthreed;

  // number of Fourier basis components
  int num_basis;

  // number of poloidal grid points
  int nThetaEven;
  int nThetaReduced;
  int nThetaEff;

  // number of grid points on the surface
  int nZnT;

  // [nThetaEff] poloidal integration weights
  std::vector<double> wInt;

  // number of Fourier coefficients in one of the 2D arrays (cc, ss, ...)
  int mnsize;

  // number of Fourier coefficients in linearly-indexed array (xm, xn)
  int mnmax;

  // --------- Nyquist sizes

  // max poloidal mode number to hold full information on realspace grid
  // (nThetaEven) for computing quantities in FourierBasisFastPoloidal
  int mnyq2;

  // max toroidal mode number to hold full information on realspace grid (nZeta)
  // for computing quantities in FourierBasisFastPoloidal
  int nnyq2;

  // max poloidal mode number to hold full information on realspace grid
  // (nThetaEven)
  int mnyq;

  // max toroidal mode number to hold full information on realspace grid (nZeta)
  int nnyq;

  // number of Fourier coefficients in linearly-indexed array (xm_nyq, xn_nyq)
  int mnmax_nyq;

 private:
  void computeDerivedSizes();
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_SIZES_SIZES_H_
