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

  Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta);

  // Construct with an explicit active toroidal mode set (sorted ascending,
  // unique, each in [0, ntor], including 0). An empty list selects the dense
  // default {0, ..., ntor}. Primarily used for sparse-toroidal handling and
  // testing.
  Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta,
        const std::vector<int>& active_toroidal_modes);

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
  Eigen::VectorXd wInt;

  // number of Fourier coefficients in one of the 2D arrays (cc, ss, ...)
  // NOTE: in the sparse-toroidal case this is mpol * n_active, i.e. only the
  // active toroidal columns are stored per poloidal mode m.
  int mnsize;

  // number of Fourier coefficients in linearly-indexed array (xm, xn).
  // Always dense (external combined-basis convention): unused toroidal modes
  // appear as zeros in the wout output.
  int mnmax;

  // --------- sparse toroidal mode handling
  //
  // Ordered list of active toroidal mode indices n (the "n" in the n*nfp
  // geometric toroidal mode number), strictly increasing, each in [0, ntor],
  // always including 0. By default this is {0, 1, ..., ntor} and the code
  // behaves exactly as the dense case. A shorter list lets us carry only the
  // non-zero toroidal modes of the internal product-basis coefficients, so the
  // toroidal transform cost scales with n_active instead of (ntor + 1) and high
  // toroidal modes are representable without a dense (m, n) grid.
  Eigen::VectorXi active_n;

  // number of active toroidal modes == active_n.size()
  int n_active;

  // reverse map: n_to_compact[n] is the compact index c such that
  // active_n[c] == n, or -1 if toroidal mode n is not active. Length ntor + 1.
  Eigen::VectorXi n_to_compact;

  // true when fewer toroidal modes are active than the dense (ntor + 1).
  bool is_sparse_toroidal;

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
  // Compute the derived sizes. If `active_toroidal_modes` is non-empty it is
  // used verbatim as the active toroidal mode set (must be sorted, unique, in
  // [0, ntor], include 0); otherwise the dense set {0, ..., ntor} is used.
  void computeDerivedSizes(const std::vector<int>& active_toroidal_modes = {});
};

// Determine which toroidal mode indices n (in [0, ntor]) carry any non-zero
// boundary or axis coefficient in `id`. The +n and -n columns of the combined
// boundary basis are folded onto |n|. The result is sorted, unique, and always
// includes n=0. Used to drive sparse-toroidal handling automatically.
std::vector<int> ActiveToroidalModesFromBoundary(const VmecINDATA& id);

}  // namespace vmecpp

#endif  // VMECPP_COMMON_SIZES_SIZES_H_
