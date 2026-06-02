// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/sizes/sizes.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <string>  // std::to_string
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"

namespace vmecpp {

std::vector<int> ActiveToroidalModesFromBoundary(const VmecINDATA& id) {
  // Auto-derive the active toroidal mode set from the non-zero boundary/axis
  // coefficients. When the boundary populates every toroidal mode 0..ntor (the
  // common case), this returns the dense set and the solver behaves exactly as
  // before. When the boundary is sparse, the solve is restricted to the
  // toroidal subspace spanned by the populated modes.
  //
  // CAVEAT: this is a deliberate approximation. VMEC's nonlinear force balance
  // can in general transfer energy into toroidal modes that are zero at the
  // boundary; restricting the active set drops those couplings. It is intended
  // for exploring perturbations where the active modes are (near-)closed under
  // the equilibrium's nonlinearity -- e.g. a base geometry on multiples of nfp
  // plus a few extra high-n modes. For a faithful general solve, populate (even
  // with zeros) every toroidal mode that should participate.
  //
  // n=0 is always active (offset / axisymmetric component).
  std::set<int> active = {0};

  // Boundary coefficients are [mpol, 2*ntor+1] in the combined basis
  // cos/sin(m*u - n*v): column j maps to n = j - ntor. Fold +-n onto |n|.
  const auto scan_boundary = [&](const RowMatrixXd& coeffs) {
    for (int m = 0; m < coeffs.rows(); ++m) {
      for (int j = 0; j < coeffs.cols(); ++j) {
        if (coeffs(m, j) != 0.0) {
          active.insert(std::abs(j - id.ntor));
        }
      }
    }
  };
  scan_boundary(id.rbc);
  scan_boundary(id.zbs);
  if (id.rbs.has_value()) {
    scan_boundary(*id.rbs);
  }
  if (id.zbc.has_value()) {
    scan_boundary(*id.zbc);
  }

  // Axis coefficients are [ntor+1], indexed directly by n.
  const auto scan_axis = [&](const Eigen::VectorXd& coeffs) {
    for (int n = 0; n < coeffs.size(); ++n) {
      if (coeffs[n] != 0.0) {
        active.insert(n);
      }
    }
  };
  scan_axis(id.raxis_c);
  scan_axis(id.zaxis_s);
  if (id.raxis_s.has_value()) {
    scan_axis(*id.raxis_s);
  }
  if (id.zaxis_c.has_value()) {
    scan_axis(*id.zaxis_c);
  }

  return std::vector<int>(active.begin(), active.end());
}

Sizes::Sizes(const VmecINDATA& id)
    : lasym(id.lasym),
      nfp(id.nfp),
      mpol(id.mpol),
      ntor(id.ntor),
      ntheta(id.ntheta),
      nZeta(id.nzeta) {
  computeDerivedSizes(ActiveToroidalModesFromBoundary(id));
}

Sizes::Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta)
    : lasym(lasym),
      nfp(nfp),
      mpol(mpol),
      ntor(ntor),
      ntheta(ntheta),
      nZeta(nzeta) {
  computeDerivedSizes();
}

Sizes::Sizes(bool lasym, int nfp, int mpol, int ntor, int ntheta, int nzeta,
             const std::vector<int>& active_toroidal_modes)
    : lasym(lasym),
      nfp(nfp),
      mpol(mpol),
      ntor(ntor),
      ntheta(ntheta),
      nZeta(nzeta) {
  computeDerivedSizes(active_toroidal_modes);
}

// Assuming that the key parameters defining the array sizes etc. have been set,
// compute the derived sizes like actual array sizes etc.
void Sizes::computeDerivedSizes(const std::vector<int>& active_toroidal_modes) {
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

  // --------- active toroidal modes (sparse-toroidal handling)
  //
  // Build the ordered active toroidal mode set. When no explicit set is
  // provided, use the dense set {0, 1, ..., ntor} so that all derived sizes and
  // downstream loops behave exactly as in the original dense implementation.
  if (active_toroidal_modes.empty()) {
    active_n.resize(ntor + 1);
    for (int n = 0; n <= ntor; ++n) {
      active_n[n] = n;
    }
  } else {
    // Validate the provided set: sorted ascending, unique, in [0, ntor],
    // including 0.
    CHECK_EQ(active_toroidal_modes.front(), 0)
        << "active toroidal mode set must include n=0";
    for (size_t c = 0; c < active_toroidal_modes.size(); ++c) {
      const int n = active_toroidal_modes[c];
      CHECK_GE(n, 0) << "active toroidal mode must be >= 0";
      CHECK_LE(n, ntor) << "active toroidal mode must be <= ntor=" << ntor;
      if (c > 0) {
        CHECK_LT(active_toroidal_modes[c - 1], n)
            << "active toroidal mode set must be strictly increasing";
      }
    }
    active_n = Eigen::Map<const Eigen::VectorXi>(active_toroidal_modes.data(),
                                                 active_toroidal_modes.size());
  }
  n_active = static_cast<int>(active_n.size());
  is_sparse_toroidal = (n_active != ntor + 1);

  n_to_compact.setConstant(ntor + 1, -1);
  for (int c = 0; c < n_active; ++c) {
    n_to_compact[active_n[c]] = c;
  }

  // INTERNAL product-basis per-surface block: only active toroidal columns are
  // stored per poloidal mode m.
  mnsize = mpol * n_active;

  // EXTERNAL combined-basis linear count stays dense (unused modes are zeros in
  // the wout output):
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
