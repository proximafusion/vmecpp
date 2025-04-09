// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/boundaries/boundaries.h"

#include <iostream>

#include "absl/algorithm/container.h"
#include "vmecpp/vmec/boundaries/guess_magnetic_axis.h"

namespace vmecpp {

Boundaries::Boundaries(const Sizes* s, const FourierBasisFastPoloidal* t,
                       const int sign_of_jacobian)
    : s_(*s), t_(*t), sign_of_jacobian_(sign_of_jacobian) {
  raxis_c.resize(s_.ntor + 1);
  zaxis_s.resize(s_.ntor + 1);
  if (s_.lasym) {
    raxis_s.resize(s_.ntor + 1);
    zaxis_c.resize(s_.ntor + 1);
  }

  rbcc.resize(s_.mpol * (s_.ntor + 1));
  zbsc.resize(s_.mpol * (s_.ntor + 1));
  if (s_.lthreed) {
    rbss.resize(s_.mpol * (s_.ntor + 1));
    zbcs.resize(s_.mpol * (s_.ntor + 1));
  }
  if (s_.lasym) {
    rbsc.resize(s_.mpol * (s_.ntor + 1));
    zbcc.resize(s_.mpol * (s_.ntor + 1));
    if (s_.lthreed) {
      rbcs.resize(s_.mpol * (s_.ntor + 1));
      zbss.resize(s_.mpol * (s_.ntor + 1));
    }
  }
}

bool Boundaries::setupFromIndata(const VmecINDATA& id, bool verbose) {
  parseToInternalArrays(id, verbose);

  bool haveToFlipTheta = checkSignOfJacobian();
  if (haveToFlipTheta) {
    if (verbose) {
      std::cout << "need to flip theta definition of input boundary shape\n";
    }
    flipTheta();
  }

  // activate m=1-constraint
  ensureM1Constrained(0.5);

  return haveToFlipTheta;
}

void Boundaries::parseToInternalArrays(const VmecINDATA& id, bool verbose) {
  // copy over axis from INDATA to this class
  for (int n = 0; n < s_.ntor + 1; ++n) {
    raxis_c[n] = id.raxis_c[n];
    zaxis_s[n] = id.zaxis_s[n];
    if (s_.lasym) {
      raxis_s[n] = id.raxis_s[n];
      zaxis_c[n] = id.zaxis_c[n];
    }
  }

  // check for necessity to shift poloidal angle
  // to achieve unique poloidal angle
  double delta = 0.0;
  if (s_.lasym) {
    // comment by Matt Landreman in Simsopt Slack:
    // > When lasym=.true.,  vmec claims to shift theta for the provided
    // boundary so RBS(n=0,m=1) = ZBC(n=0,m=1) > However the angle shift delta
    // does not correctly do this, due to superfluous absolute value signs. > As
    // a result, each time a boundary shape passes through readin.f, theta keeps
    // getting shifted.

    // get index of element at m=1, n=0
    int m = 1;
    int n = 0;

    // Fortran layout along n in rbc, zbs, ...
    int nIdx = s_.ntor + n;

    // m slow, n fast
    int idx = m * (2 * s_.ntor + 1) + nIdx;

    delta = atan2(id.rbs[idx] - id.zbc[idx], id.rbc[idx] + id.zbs[idx]);

    if (verbose && delta != 0.0) {
      std::cout << "need to shift theta by delta = " << delta << "\n";
      // In this implementation, the theta-shift will be done during sorting of
      // coefficients below.
    }
  }

  absl::c_fill_n(rbcc, s_.mpol * (s_.ntor + 1), 0);
  absl::c_fill_n(zbsc, s_.mpol * (s_.ntor + 1), 0);
  if (s_.lthreed) {
    absl::c_fill_n(rbss, s_.mpol * (s_.ntor + 1), 0);
    absl::c_fill_n(zbcs, s_.mpol * (s_.ntor + 1), 0);
  }
  if (s_.lasym) {
    absl::c_fill_n(rbsc, s_.mpol * (s_.ntor + 1), 0);
    absl::c_fill_n(zbcc, s_.mpol * (s_.ntor + 1), 0);
    if (s_.lthreed) {
      absl::c_fill_n(rbcs, s_.mpol * (s_.ntor + 1), 0);
      absl::c_fill_n(zbss, s_.mpol * (s_.ntor + 1), 0);
    }
  }

  for (int m = 0; m < s_.mpol; ++m) {
    double cosMDelta = 1.0;
    double sinMDelta = 0.0;
    if (delta != 0.0) {
      cosMDelta = cos(m * delta);
      sinMDelta = sin(m * delta);
    }

    for (int n = -s_.ntor; n <= s_.ntor; ++n) {
      // Fortran layout along n in rbc, zbs, ...
      int source_n = s_.ntor + n;

      int target_n = abs(n);
      double sign_n = signum(n);

      double rbc;
      double zbs;
      if (!s_.lasym || delta == 0.0) {
        rbc = id.rbc[m * (2 * s_.ntor + 1) + source_n];
        zbs = id.zbs[m * (2 * s_.ntor + 1) + source_n];
      } else {
        // lasym && delta != 0.0
        rbc = id.rbc[m * (2 * s_.ntor + 1) + source_n] * cosMDelta +
              id.rbs[m * (2 * s_.ntor + 1) + source_n] * sinMDelta;
        zbs = id.zbs[m * (2 * s_.ntor + 1) + source_n] * cosMDelta -
              id.zbc[m * (2 * s_.ntor + 1) + source_n] * sinMDelta;
      }

      const int idx_mn = m * (s_.ntor + 1) + target_n;
      rbcc[idx_mn] += rbc;
      if (m > 0) {
        zbsc[idx_mn] += zbs;
      }
      if (s_.lthreed) {
        if (m > 0) {
          rbss[idx_mn] += sign_n * rbc;
        }
        zbcs[idx_mn] -= sign_n * zbs;
      }

      if (s_.lasym) {
        double rbs;
        double zbc;
        if (delta == 0.0) {
          rbs = id.rbs[m * (2 * s_.ntor + 1) + source_n];
          zbc = id.zbc[m * (2 * s_.ntor + 1) + source_n];
        } else {
          rbs = id.rbs[m * (2 * s_.ntor + 1) + source_n] * cosMDelta -
                id.rbc[m * (2 * s_.ntor + 1) + source_n] * sinMDelta;
          zbc = id.zbc[m * (2 * s_.ntor + 1) + source_n] * cosMDelta +
                id.zbs[m * (2 * s_.ntor + 1) + source_n] * sinMDelta;
        }

        if (m > 0) {
          rbsc[idx_mn] += rbs;
        }
        zbcc[idx_mn] += zbc;
        if (s_.lthreed) {
          rbcs[idx_mn] -= sign_n * rbs;
          if (m > 0) {
            zbss[idx_mn] += sign_n * zbc;
          }
        }
      }
    }  // m
  }  // n
}

bool Boundaries::checkSignOfJacobian() {
  /**
   * current working hypothesis: rTest, zTest are related to the leading terms
   * of d(R,Z)/dTheta at (theta, zeta)=(pi/2, 0) for R and at (theta, zeta)=(0,
   * 0) for Z. If the leading derivatives have the same sign, the path is
   * probably going counter-clockwise, with different signs it is likely going
   * clockwise.
   */

  double rTest = 0.0;
  double zTest = 0.0;
  for (int n = 0; n < s_.ntor + 1; ++n) {
    int m = 1;
    int idx_mn = m * (s_.ntor + 1) + n;
    rTest += rbcc[idx_mn];
    zTest += zbsc[idx_mn];
  }

  // TODO(jons): potentially more robust version of this
  // - eval boundary in a given poloidal plane at equal theta intervals, enough
  // to satisfy Nyquist requirement
  // - compute signed polygon area
  // --> handedness of polygon is given by sign of polygon area

  // for signOfJacobian == -1, need to flip when rTest*zTest < 0
  // ---> this is true when the total sign is positive
  return (rTest * zTest * sign_of_jacobian_ > 0.0);
}

void Boundaries::flipTheta() {
  for (int m = 1; m < s_.mpol; ++m) {
    for (int n = 0; n <= s_.ntor; ++n) {
      int idx_mn = m * (s_.ntor + 1) + n;

      // +1 if m is even, -1 if m is odd
      int m_parity = ((m % 2 == 0) ? 1 : -1);

      rbcc[idx_mn] *= m_parity;
      zbsc[idx_mn] *= -m_parity;
      if (s_.lthreed) {
        rbss[idx_mn] *= -m_parity;
        zbcs[idx_mn] *= m_parity;
      }
      if (s_.lasym) {
        rbsc[idx_mn] *= -m_parity;
        zbcc[idx_mn] *= m_parity;
        if (s_.lthreed) {
          rbcs[idx_mn] *= m_parity;
          zbss[idx_mn] *= -m_parity;
        }
      }
    }  // n
  }  // m
}

/**
 * Make sure that the (m=1) Fourier coefficients of R and Z are coupled to
 * result in a quasi-polar constraint on the boundary shape.
 * This goes hand-in-hand with the shift by delta in parseToInternalArrays().
 *
 * Essentially, the initial boundary is re-scaled to yield a unique poloidal
 * origin.
 */
void Boundaries::ensureM1Constrained(const double scaling_factor) {
  for (int n = 0; n <= s_.ntor; ++n) {
    int m = 1;
    int idx_mn = m * (s_.ntor + 1) + n;
    if (s_.lthreed) {
      double backup_rss = rbss[idx_mn];
      rbss[idx_mn] = (backup_rss + zbcs[idx_mn]) * scaling_factor;
      zbcs[idx_mn] = (backup_rss - zbcs[idx_mn]) * scaling_factor;
    }
    if (s_.lasym) {
      double backup_rsc = rbsc[idx_mn];
      rbsc[idx_mn] = (backup_rsc + zbcc[idx_mn]) * scaling_factor;
      zbcc[idx_mn] = (backup_rsc - zbcc[idx_mn]) * scaling_factor;
    }
  }  // n
}

void Boundaries::RecomputeMagneticAxisToFixJacobianSign(
    const int number_of_flux_surfaces, const int sign_of_jacobian) {
  // NOTE: externalized into guess_magnetic_axis.h
  // to allow more in-depth testing of intermediate quantities
  // without polluting the namespace of Boundaries class
  RecomputeAxisWorkspace w = vmecpp::RecomputeMagneticAxisToFixJacobianSign(
      number_of_flux_surfaces, sign_of_jacobian, s_, t_, rbcc, rbss, rbsc, rbcs,
      zbsc, zbcs, zbcc, zbss, raxis_c, raxis_s, zaxis_s, zaxis_c);

  // Now copy over the Fourier coefficients of the new axis:
  for (int n = 0; n <= s_.ntor; ++n) {
    raxis_c[n] = w.new_raxis_c[n];
    zaxis_s[n] = w.new_zaxis_s[n];
    if (s_.lasym) {
      raxis_s[n] = w.new_raxis_s[n];
      zaxis_c[n] = w.new_zaxis_c[n];
    }
  }  // n
}  // RecomputeMagneticAxisToFixJacobianSign

}  // namespace vmecpp
