// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"

#include <algorithm>
#include <vector>

#include "vmecpp/common/util/util.h"

namespace vmecpp {

FourierGeometry::FourierGeometry(const Sizes* s, const RadialPartitioning* r,
                                 int ns)
    : FourierCoeffs(s, r, r->nsMinF1, r->nsMaxF1, ns),
      rmncc(rcc),
      rmnss(rss),
      rmnsc(rsc),
      rmncs(rcs),

      zmnsc(zsc),
      zmncs(zcs),
      zmncc(zcc),
      zmnss(zss),

      lmnsc(lsc),
      lmncs(lcs),
      lmncc(lcc),
      lmnss(lss) {}

void FourierGeometry::interpFromBoundaryAndAxis(
    const FourierBasisFastPoloidal& t, const Boundaries& b,
    const RadialProfiles& p) {
  for (int jF = nsMin_; jF < nsMax_; ++jF) {
    for (int m = 0; m < s_.mpol; ++m) {
      for (int n = 0; n < s_.ntor + 1; ++n) {
        int idx_bdy = m * (s_.ntor + 1) + n;
        int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

        double basis_norm = 1.0 / (t.mscale[m] * t.nscale[n]);

        if (m == 0) {
          // m=0-terms: only cos(m*theta) contribute

          // interpolate geometry between magnetic axis and LCFS into plasma
          // volume
          double interpolationWeight =
              p.sqrtSF[jF - r_.nsMinF1] * p.sqrtSF[jF - r_.nsMinF1];

          rmncc[idx_fc] =
              basis_norm * (interpolationWeight * b.rbcc[idx_bdy] +
                            (1.0 - interpolationWeight) * b.raxis_c[n]);
          // zmnsc has no m=0-contributions from the axis
          zmnsc[idx_fc] = basis_norm * interpolationWeight * b.zbsc[idx_bdy];
          if (s_.lthreed) {
            // rmnss has no m=0-contributions from the axis
            rmnss[idx_fc] = basis_norm * interpolationWeight * b.rbss[idx_bdy];
            zmncs[idx_fc] =
                basis_norm * (interpolationWeight * b.zbcs[idx_bdy] -
                              (1.0 - interpolationWeight) * b.zaxis_s[n]);
          }
          if (s_.lasym) {
            // rmnsc has no m=0-contributions from the axis
            rmnsc[idx_fc] = basis_norm * interpolationWeight * b.rbsc[idx_bdy];
            zmncc[idx_fc] =
                basis_norm * (interpolationWeight * b.zbcc[idx_bdy] +
                              (1.0 - interpolationWeight) * b.zaxis_c[n]);
            if (s_.lthreed) {
              rmncs[idx_fc] =
                  basis_norm * (interpolationWeight * b.rbcs[idx_bdy] -
                                (1.0 - interpolationWeight) * b.raxis_s[n]);
              // zmnss has no m=0-contributions from the axis
              zmnss[idx_fc] =
                  basis_norm * interpolationWeight * b.zbss[idx_bdy];
            }
          }
        } else {
          // m > 0: no axis terms, since axis has m=0 per definition

          // weighting factor for radial interpolation between 0 at axis and 1
          // at boundary
          double interpolationWeight = pow(p.sqrtSF[jF - r_.nsMinF1], m);

          rmncc[idx_fc] = basis_norm * interpolationWeight * b.rbcc[idx_bdy];
          zmnsc[idx_fc] = basis_norm * interpolationWeight * b.zbsc[idx_bdy];
          if (s_.lthreed) {
            rmnss[idx_fc] = basis_norm * interpolationWeight * b.rbss[idx_bdy];
            zmncs[idx_fc] = basis_norm * interpolationWeight * b.zbcs[idx_bdy];
          }
          if (s_.lasym) {
            rmnsc[idx_fc] = basis_norm * interpolationWeight * b.rbsc[idx_bdy];
            zmncc[idx_fc] = basis_norm * interpolationWeight * b.zbcc[idx_bdy];
            if (s_.lthreed) {
              rmncs[idx_fc] =
                  basis_norm * interpolationWeight * b.rbcs[idx_bdy];
              zmnss[idx_fc] =
                  basis_norm * interpolationWeight * b.zbss[idx_bdy];
            }
          }
        }
      }  // n
    }  // m
  }  // j
}

void FourierGeometry::InitFromState(const FourierBasisFastPoloidal& fb,
                                    const RowMatrixXd& rmnc,
                                    const RowMatrixXd& zmns,
                                    const RowMatrixXd& lmns_full,
                                    const RadialProfiles& p,
                                    const VmecConstants& constants,
                                    const Boundaries* b) {
  // b == nullptr -> free-boundary -> we also initialize the last surface;
  // otherwise skip the last surface here and grab it from b later
  const int max_ns_to_set_rz_on_from_state = (b == nullptr) ? ns : ns - 1;
  const int max_ns_to_set_rz_on_from_state_locally =
      std::min(nsMax_, max_ns_to_set_rz_on_from_state);
  for (int jF = nsMin_; jF < max_ns_to_set_rz_on_from_state_locally; ++jF) {
    const auto& rmnc_row = rmnc.row(jF);
    const auto& rmnc_row_vector =
        std::vector<double>(rmnc_row.data(), rmnc_row.data() + rmnc_row.size());
    std::vector<double> rmncc_at_jF(s_.mpol * (s_.ntor + 1));
    std::vector<double> rmnss_at_jF(s_.mpol * (s_.ntor + 1));
    fb.cos_to_cc_ss(rmnc_row_vector, rmncc_at_jF, rmnss_at_jF, s_.ntor,
                    s_.mpol);

    const auto& zmns_row = zmns.row(jF);
    const auto& zmns_row_vector =
        std::vector<double>(zmns_row.data(), zmns_row.data() + zmns_row.size());
    std::vector<double> zmnsc_at_jF(s_.mpol * (s_.ntor + 1));
    std::vector<double> zmncs_at_jF(s_.mpol * (s_.ntor + 1));
    fb.sin_to_sc_cs(zmns_row_vector, zmnsc_at_jF, zmncs_at_jF, s_.ntor,
                    s_.mpol);

    for (int m = 0; m < s_.mpol; ++m) {
      for (int n = 0; n < s_.ntor + 1; ++n) {
        const int idx_mn = m * (s_.ntor + 1) + n;
        const int idx_jmn = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
        rmncc[idx_jmn] = rmncc_at_jF[idx_mn];
        zmnsc[idx_jmn] = zmnsc_at_jF[idx_mn];
        if (s_.lthreed) {
          rmnss[idx_jmn] = rmnss_at_jF[idx_mn];
          zmncs[idx_jmn] = zmncs_at_jF[idx_mn];
        }
      }
    }
  }

  // This loop always need to go over the full radial range
  // from nsMin_ to nsMax_ in order to copy over
  // the lambda Fourier coefficients on every flux surface,
  // __including__ the plasma boundary.
  for (int jF = nsMin_; jF < nsMax_; ++jF) {
    const auto& lmns_row = lmns_full.row(jF);
    const auto& lmns_row_vector =
        std::vector<double>(lmns_row.data(), lmns_row.data() + lmns_row.size());
    std::vector<double> lmnsc_at_jF(s_.mpol * (s_.ntor + 1));
    std::vector<double> lmncs_at_jF(s_.mpol * (s_.ntor + 1));
    fb.sin_to_sc_cs(lmns_row_vector, lmnsc_at_jF, lmncs_at_jF, s_.ntor,
                    s_.mpol);

    for (int m = 0; m < s_.mpol; ++m) {
      for (int n = 0; n < s_.ntor + 1; ++n) {
        const int idx_mn = m * (s_.ntor + 1) + n;
        const int idx_jmn = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

        // undo lambda un-scaling that was done when writing the wout file
        // contents
        const double lambda_unscaling =
            constants.lamscale / p.phipF[jF - r_.nsMinF1];

        lmnsc[idx_jmn] = lmnsc_at_jF[idx_mn] / lambda_unscaling;
        if (s_.lthreed) {
          lmncs[idx_jmn] = lmncs_at_jF[idx_mn] / lambda_unscaling;
        }
      }
    }
  }

  // boundary from b, if present
  if (b != nullptr && r_.has_boundary()) {
    const int jF = ns - 1;
    const int mnsize = s_.mpol * (s_.ntor + 1);

    auto rmncc_begin = rmncc.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
    std::copy(b->rbcc.begin(), b->rbcc.begin() + mnsize, rmncc_begin);
    auto zmnsc_begin = zmnsc.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
    std::copy(b->zbsc.begin(), b->zbsc.begin() + mnsize, zmnsc_begin);

    if (s_.lthreed) {
      auto rmnss_begin =
          rmnss.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
      std::copy(b->rbss.begin(), b->rbss.begin() + mnsize, rmnss_begin);

      auto zmncs_begin =
          zmncs.begin() + (jF - nsMin_) * s_.mpol * (s_.ntor + 1);
      std::copy(b->zbcs.begin(), b->zbcs.begin() + mnsize, zmncs_begin);
    }

    for (int m = 0; m < s_.mpol; ++m) {
      for (int n = 0; n < s_.ntor + 1; ++n) {
        int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

        double basis_norm = 1.0 / (fb.mscale[m] * fb.nscale[n]);

        rmncc[idx_fc] *= basis_norm;
        zmnsc[idx_fc] *= basis_norm;

        if (s_.lthreed) {
          rmnss[idx_fc] *= basis_norm;
          zmncs[idx_fc] *= basis_norm;
        }
      }
    }
  }

  // Activate the m=1 constraint always for all interior (ns - 1) surfaces.
  // If performing a fixed-boundary hot-restart,
  // the boundary geometry is taken from the Boundaries object,
  // which activates the m=1 constraint already internally.
  // If performing a free-boundary hot-restart,
  // also the boundary geometry is initialized from the given initial state,
  // and hence the m=1 constraint also needs to be activated on the boundary.
  this->m1Constraint(0.5, max_ns_to_set_rz_on_from_state);

  if (nsMin_ == 0) {
    // remove towards-axis-extrapolated m=0 coefficients of lambda (was done
    // when writing wout)
    const int jF = 0;
    const int m = 0;
    for (int n = 0; n < s_.ntor + 1; ++n) {
      int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;
      lmnsc[idx_fc] = 0.0;
      if (s_.lthreed) {
        lmncs[idx_fc] = 0.0;
      }
    }
  }
}  // InitFromState

/** constant extrapolation from first surface towards axis of m=0 of lambda and
 * m=1 of R, Z, lambda */
void FourierGeometry::extrapolateTowardsAxis() {
  if (nsMin_ > 0) {
    // relevant data is not in this thread
    return;
  }

  int axis = 0;
  int firstSurface = 1;
  for (int n = 0; n < s_.ntor + 1; ++n) {
    int m0 = 0;
    int m1 = 1;

    int axis0 = (axis * s_.mpol + m0) * (s_.ntor + 1) + n;
    int axis1 = (axis * s_.mpol + m1) * (s_.ntor + 1) + n;
    int firstSurface0 = (firstSurface * s_.mpol + m0) * (s_.ntor + 1) + n;
    int firstSurface1 = (firstSurface * s_.mpol + m1) * (s_.ntor + 1) + n;

    rmncc[axis1] = rmncc[firstSurface1];
    zmnsc[axis1] = zmnsc[firstSurface1];
    lmnsc[axis1] = lmnsc[firstSurface1];
    if (s_.lthreed) {
      rmnss[axis1] = rmnss[firstSurface1];
      zmncs[axis1] = zmncs[firstSurface1];
      lmncs[axis1] = lmncs[firstSurface1];

      // m=0 component of lambda leftover from chi-force ?
      lmncs[axis0] = lmncs[firstSurface0];
    }
    if (s_.lasym) {
      rmnsc[axis1] = rmnsc[firstSurface1];
      zmncc[axis1] = zmncc[firstSurface1];
      lmncc[axis1] = lmncc[firstSurface1];

      // m=0 component of lambda leftover from chi-force ?
      lmncc[axis0] = lmncc[firstSurface0];
      if (s_.lthreed) {
        rmncs[axis1] = rmncs[firstSurface1];
        zmnss[axis1] = zmnss[firstSurface1];
        lmnss[axis1] = lmnss[firstSurface1];
      }
    }
  }  // n
}

void FourierGeometry::copyFrom(const FourierGeometry& src) {
  for (int jF = nsMin_; jF < nsMax_; ++jF) {
    for (int m = 0; m < s_.mpol; ++m) {
      for (int n = 0; n < s_.ntor + 1; ++n) {
        int idx_fc = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

        rmncc[idx_fc] = src.rmncc[idx_fc];
        zmnsc[idx_fc] = src.zmnsc[idx_fc];
        lmnsc[idx_fc] = src.lmnsc[idx_fc];
        if (s_.lthreed) {
          rmnss[idx_fc] = src.rmnss[idx_fc];
          zmncs[idx_fc] = src.zmncs[idx_fc];
          lmncs[idx_fc] = src.lmncs[idx_fc];
        }
        if (s_.lasym) {
          rmnsc[idx_fc] = src.rmnsc[idx_fc];
          zmncc[idx_fc] = src.zmncc[idx_fc];
          lmncc[idx_fc] = src.lmncc[idx_fc];
          if (s_.lthreed) {
            rmncs[idx_fc] = src.rmncs[idx_fc];
            zmnss[idx_fc] = src.zmnss[idx_fc];
            lmnss[idx_fc] = src.lmnss[idx_fc];
          }
        }
      }  // n
    }  // m
  }  // j
}

void FourierGeometry::ComputeSpectralWidth(
    const FourierBasisFastPoloidal& fourier_basis,
    RadialProfiles& m_radial_profiles, const int p, const int q) const {
  int minimum_j = nsMin_;
  if (nsMin_ == 0) {
    minimum_j = 1;

    // Set axis spectral width to 1 to allow volume average to be computed.
    m_radial_profiles.spectral_width[nsMin_ - r_.nsMinF1] = 1.0;
  }

  // compute only on unique full-grid points
  for (int jF = minimum_j; jF < nsMax_; ++jF) {
    double spectral_width_numerator = 0.0;
    double spectral_width_denominator = 0.0;

    // note that we exclude m = 0
    for (int m = 1; m < s_.mpol; ++m) {
      for (int n = 0; n < s_.ntor + 1; ++n) {
        int fourier_index = ((jF - nsMin_) * s_.mpol + m) * (s_.ntor + 1) + n;

        const double basis_norm =
            fourier_basis.mscale[m] * fourier_basis.nscale[n];

        std::vector<double> r_coefficients(4, 0.0);
        std::vector<double> z_coefficients(4, 0.0);
        int basis_dimension = 0;

        r_coefficients[basis_dimension] = rmncc[fourier_index];
        z_coefficients[basis_dimension] = zmnsc[fourier_index];
        basis_dimension++;

        // CONVERT FROM INTERNAL XC REPRESENTATION FOR m=1 MODES,
        // R+(at rsc) = .5(rsc + zcc),
        // R-(at zcc) = .5(rsc - zcc),
        // TO REQUIRED rsc, zcc FORMS
        if (s_.lthreed) {
          if (m == 1) {
            const double r_plus = rmnss[fourier_index];
            const double r_minus = zmncs[fourier_index];
            // rmnss
            r_coefficients[basis_dimension] = r_plus + r_minus;
            // zmncs
            z_coefficients[basis_dimension] = r_plus - r_minus;
          } else {
            r_coefficients[basis_dimension] = rmnss[fourier_index];
            z_coefficients[basis_dimension] = zmncs[fourier_index];
          }
          basis_dimension++;
        }
        if (s_.lasym) {
          if (m == 1) {
            const double r_plus = rmnsc[fourier_index];
            const double r_minus = zmncc[fourier_index];
            // rmnsc
            r_coefficients[basis_dimension] = r_plus + r_minus;
            // zmncc
            z_coefficients[basis_dimension] = r_plus - r_minus;
          } else {
            r_coefficients[basis_dimension] = rmnsc[fourier_index];
            z_coefficients[basis_dimension] = zmncc[fourier_index];
          }
          basis_dimension++;
        }

        if (s_.lasym && s_.lthreed) {
          r_coefficients[basis_dimension] = rmncs[fourier_index];
          z_coefficients[basis_dimension] = zmnss[fourier_index];
          basis_dimension++;
        }

        double coefficient_norm = 0.0;
        for (int basis_index = 0; basis_index < basis_dimension;
             ++basis_index) {
          coefficient_norm +=
              r_coefficients[basis_index] * r_coefficients[basis_index];
          coefficient_norm +=
              z_coefficients[basis_index] * z_coefficients[basis_index];
        }  // basis_index
        coefficient_norm *= basis_norm * basis_norm;

        spectral_width_numerator += coefficient_norm * std::pow(m, p + q);
        spectral_width_denominator += coefficient_norm * std::pow(m, p);
      }  // m
    }  // n

    m_radial_profiles.spectral_width[jF - r_.nsMinF1] =
        spectral_width_numerator / spectral_width_denominator;
  }  // jF
}  // ComputeSpectralWidth

}  // namespace vmecpp
