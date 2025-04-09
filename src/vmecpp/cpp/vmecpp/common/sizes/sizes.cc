// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/sizes/sizes.h"

#include <algorithm>
#include <iostream>
#include <string>  // std::to_string

#include "absl/log/check.h"
#include "absl/strings/str_format.h"

namespace vmecpp {

Sizes::Sizes(const VmecINDATA& id)
    : Sizes(id.lasym, id.nfp, id.mpol, id.ntor, id.ntheta, id.nzeta) {}

Sizes::Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta)
    : lasym(lasym),
      nfp(nfp),
      mpol(mpol),
      ntor(ntor),
      ntheta(ntheta),
      nZeta(nzeta) {
  computeDerivedSizes();
}

// Assuming that the key parameters defining the array sizes etc. have been set,
// compute the derived sizes like actual array sizes etc.
void Sizes::computeDerivedSizes() {
  // lasym
  // nothing to check here: lasym can be true or false and both are valid...

  // debug checks: VmecINDATA should already have reported this
  // nfp
  CHECK_GE(nfp, 1) << "input variable 'nfp' needs to be >= 1, but is " << nfp;

  // mpol
  CHECK_GE(mpol, 1) << "input variable 'mpol' needs to be >= 1, but is "
                    << mpol;

  // ntor
  CHECK_GE(ntor, 0) << "input variable 'ntor' needs to be >= 0, but is "
                    << ntor;

  // ntheta
  if (ntheta < 2 * mpol + 6) {
    ntheta = 2 * mpol + 6;
#ifdef DEBUG
    // NOTE: not suppressing this by `verbose` flag (vmec.cc:Vmec), since only
    // enabled when a `DEBUG` build is requested
    std::cout << absl::StrFormat(
        "adjusting 'ntheta' to %d in order to satisfy Nyquist criterion\n",
        ntheta);
#endif
  }

  // nzeta
  if (ntor == 0 && nZeta < 1) {
    // Tokamak (ntor=0) needs (at least) nzeta=1
    // I think this implies that (in principle, not reasonable) one could do an
    // axisymmetric run with nzeta > 1 ...
    nZeta = 1;
  }

  if (ntor > 0) {
    // 3D/Stellarator case needs Nyquist criterion fulfilled for nzeta wrt. ntor
    if (nZeta < 2 * ntor + 4) {
      nZeta = 2 * ntor + 4;
#ifdef DEBUG
      // NOTE: not suppressing this by `verbose` flag (vmec.cc:Vmec), since only
      // enabled when a `DEBUG` build is requested
      std::cout << absl::StrFormat(
          "adjusting 'nzeta' to %d in order to satisfy Nyquist criterion\n",
          nZeta);
#endif
    }
  }

  // derived

  // flag to indicate a three-dimensional case (== has toroidal variation)
  lthreed = (ntor > 0);

  // number of Fourier basis functions
  // num_basis = 2**(lthreed + lasym)
  num_basis = 1;
  if (lthreed) {
    num_basis *= 2;
  }
  if (lasym) {
    num_basis *= 2;
  }

  // real-space array sizes

  // [0, 2pi[ --> EXCLUDING endpoint!
  nThetaEven = 2 * (ntheta / 2);

  // [0, pi] --> INCLUDING endpoint!
  nThetaReduced = nThetaEven / 2 + 1;

  if (lasym) {
    nThetaEff = nThetaEven;
  } else {
    // use stellarator- or up/down-symmetry
    // --> only eval on reduced [0, pi] poloidal interval
    nThetaEff = nThetaReduced;
  }

  // surface is always full in toroidal direction
  // but can be reduced in poloidal direction --> nTheta_Eff_
  nZnT = nZeta * nThetaEff;

  // normalization factor for poloidal integrals
  // default case: use stellarator symmetry
  // --> # of gaps between grid points is one less than number of grid points
  // (which INCLUDE endpoint in symmetric case)
  double dnorm3 = 1.0 / (nZeta * (nThetaReduced - 1));
  if (lasym) {
    dnorm3 = 1.0 / (nZeta * nThetaEven);
  }

  wInt.resize(nThetaEff);
  for (int l = 0; l < nThetaEff; ++l) {
    wInt[l] = dnorm3;
    if (!lasym && (l == 0 || l == nThetaReduced - 1)) {
      // weight back to 1 at the endpoints
      wInt[l] /= 2.0;
    }
  }

  mnsize = mpol * (ntor + 1);

  // m = 0: n =  0, 1, ..., ntor --> ntor + 1
  // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
  mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1);

  // --------- Nyquist sizes

  // NEED 2 X NYQUIST FOR FAST HESSIAN CALCULATIONS
  // maximum mode numbers supported by grid
  int mnyq0 = nThetaEven / 2;
  int nnyq0 = nZeta / 2;

  // make sure that mnyq, nnyq are at least twice mpol-1, ntor
  // or large enough to fully represent the information held in realspace
  // (mnyq0, nnyq0)
  mnyq2 = std::max(0, std::max(2 * mnyq0, 2 * (mpol - 1)));
  nnyq2 = std::max(0, std::max(2 * nnyq0, 2 * ntor));

  mnmax_nyq = nnyq2 / 2 + 1 + mnyq2 / 2 * (nnyq2 + 1);

  // COMPUTE NYQUIST-SIZED ARRAYS FOR OUTPUT.
  // RESTORE m,n Nyquist TO 1 X ... (USED IN WROUT, JXBFORCE)
  mnyq = mnyq2 / 2;
  nnyq = nnyq2 / 2;
}

}  // namespace vmecpp
