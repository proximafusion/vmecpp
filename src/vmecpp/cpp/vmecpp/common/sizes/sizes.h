// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_SIZES_SIZES_H_
#define VMECPP_COMMON_SIZES_SIZES_H_

#include <Eigen/Dense>
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

  Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta,
        int mpol_geometry = -1, int ntor_geometry = -1);

  // inputs from INDATA

  // flag to indicate non-symmetric case
  bool lasym;

  // number of toroidal field periods
  int nfp;

  // number of poloidal Fourier harmoncis
  int mpol;

  // number of toroidal Fourier harmoncis
  int ntor;

  // geometry-only resolution: R and Z modes with m >= mpolGeometry or
  // n > ntorGeometry are frozen, while lambda keeps the full mpol/ntor.
  // Equal to mpol/ntor unless a reduced geometry resolution was requested.
  int mpolGeometry;
  int ntorGeometry;

  // Per-(m, n) flag (size mpol*(ntor+1), indexed m*(ntor+1)+n) marking extra
  // geometry modes that are kept free above the mpolGeometry/ntorGeometry cap.
  // Empty when no extra modes were requested.
  std::vector<char> extraGeometryActive;

  // True if (m, n) is one of the explicitly-kept extra geometry modes.
  bool isExtraGeometryMode(int m, int n) const {
    if (extraGeometryActive.empty()) {
      return false;
    }
    return extraGeometryActive[m * (ntor + 1) + n] != 0;
  }

  // When true, lambda is also restricted to the active geometry modes and the
  // transforms / preconditioner skip the frozen modes (sparse computation).
  // Only meaningful together with a geometry cap and/or extra modes.
  bool sparseLambda = false;

  // True if geometry mode (m, n) is free: inside the mpolGeometry/ntorGeometry
  // cap, or one of the explicitly-kept extra modes.
  bool isActiveGeometryMode(int m, int n) const {
    return (m < mpolGeometry && n <= ntorGeometry) || isExtraGeometryMode(m, n);
  }

  // True if any geometry mode at poloidal m is free; used to skip whole
  // poloidal rows in the sparse transforms.
  bool anyActiveGeometryAtM(int m) const {
    if (m < mpolGeometry) {
      return true;
    }
    if (extraGeometryActive.empty()) {
      return false;
    }
    for (int n = 0; n <= ntor; ++n) {
      if (extraGeometryActive[m * (ntor + 1) + n] != 0) {
        return true;
      }
    }
    return false;
  }

  // Highest free toroidal mode at poloidal m, or -1 if none. The sparse
  // transforms only sum n in [0, maxActiveGeometryNAtM]; the frozen modes
  // below it are zero and contribute nothing, so the result is unchanged.
  int maxActiveGeometryNAtM(int m) const {
    int max_n = (m < mpolGeometry) ? ntorGeometry : -1;
    if (!extraGeometryActive.empty()) {
      for (int n = ntor; n > max_n; --n) {
        if (extraGeometryActive[m * (ntor + 1) + n] != 0) {
          max_n = n;
          break;
        }
      }
    }
    return max_n;
  }

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
  Eigen::VectorXd wInt;

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
