// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <span>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/fourier_asymmetric/fourier_asymmetric.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kOddParity;

namespace {

// Set m_h.rCC_LCFS etc. to the corresponding values in the FourierGeometry
// of the last surface, also transposing m and n dimensions to make the
// data layout what Nestor expects.
void HandOverBoundaryGeometry(vmecpp::HandoverStorage& m_h,
                              const vmecpp::FourierGeometry& physical_x,
                              const vmecpp::Sizes& sizes, int offset) {
  const int ntorp1 = sizes.ntor + 1;
  for (int m = 0; m < sizes.mpol; ++m) {
    for (int n = 0; n < ntorp1; ++n) {
      const int idx_mn = m * ntorp1 + n;
      const int idx_nm = n * sizes.mpol + m;
      m_h.rCC_LCFS[idx_nm] = physical_x.rmncc[offset + idx_mn];
      m_h.rSS_LCFS[idx_nm] = physical_x.rmnss[offset + idx_mn];
      m_h.zSC_LCFS[idx_nm] = physical_x.zmnsc[offset + idx_mn];
      m_h.zCS_LCFS[idx_nm] = physical_x.zmncs[offset + idx_mn];

      if (sizes.lasym) {
        m_h.rSC_LCFS[idx_nm] = physical_x.rmnsc[offset + idx_mn];
        m_h.rCS_LCFS[idx_nm] = physical_x.rmncs[offset + idx_mn];
        m_h.zCC_LCFS[idx_nm] = physical_x.zmncc[offset + idx_mn];
        m_h.zSS_LCFS[idx_nm] = physical_x.zmnss[offset + idx_mn];
      }
    }
  }
}  // HandOverBoundaryGeometry

void HandOverMagneticAxis(vmecpp::HandoverStorage& m_h,
                          const std::vector<double>& r1_e,
                          const std::vector<double>& z1_e,
                          const vmecpp::Sizes& s) {
  for (int k = 0; k < s.nZeta; ++k) {
    // we are interested in l == 0
    int idx_kl = k * s.nThetaEff;
    m_h.rAxis[k] = r1_e[idx_kl];
    m_h.zAxis[k] = z1_e[idx_kl];
  }
}  // HandOverMagneticAxis

}  // namespace

void vmecpp::ForcesToFourier3DSymmFastPoloidal(
    const RealSpaceForces& d, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces) {
  // in here, we can safely assume lthreed == true

  // fill target force arrays with zeros
  m_physical_forces.setZero();

  int jMaxRZ = std::min(rp.nsMaxF, fc.ns - 1);

  if (fc.lfreeb &&
      (vacuum_pressure_state == VacuumPressureState::kInitialized ||
       vacuum_pressure_state == VacuumPressureState::kActive)) {
    // free-boundary: up to jMaxRZ=ns
    jMaxRZ = std::min(rp.nsMaxF, fc.ns);
  }

  // axis lambda stays zero (no contribution from any m)
  const int jMinL = 1;

  for (int jF = rp.nsMinF; jF < jMaxRZ; ++jF) {
    const int mmax = jF == 0 ? 1 : s.mpol;
    for (int m = 0; m < mmax; ++m) {
      const bool m_even = m % 2 == 0;

      const auto& armn = m_even ? d.armn_e : d.armn_o;
      const auto& azmn = m_even ? d.azmn_e : d.azmn_o;
      const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
      const auto& brmn = m_even ? d.brmn_e : d.brmn_o;
      const auto& bzmn = m_even ? d.bzmn_e : d.bzmn_o;
      const auto& clmn = m_even ? d.clmn_e : d.clmn_o;
      const auto& crmn = m_even ? d.crmn_e : d.crmn_o;
      const auto& czmn = m_even ? d.czmn_e : d.czmn_o;
      const auto& frcon = m_even ? d.frcon_e : d.frcon_o;
      const auto& fzcon = m_even ? d.fzcon_e : d.fzcon_o;

      for (int k = 0; k < s.nZeta; ++k) {
        double rmkcc = 0.0;
        double rmkcc_n = 0.0;
        double rmkss = 0.0;
        double rmkss_n = 0.0;
        double zmksc = 0.0;
        double zmksc_n = 0.0;
        double zmkcs = 0.0;
        double zmkcs_n = 0.0;
        double lmksc = 0.0;
        double lmksc_n = 0.0;
        double lmkcs = 0.0;
        double lmkcs_n = 0.0;

        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
        const int idx_ml_base = m * s.nThetaReduced;

        // NOTE: nThetaReduced is usually pretty small, 9 for cma.json
        // and 16 for w7x_ref_167_12_12.json, so in our benchmark forcing
        // the compiler to auto-vectorize this loop was a pessimization.
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_kl = idx_kl_base + l;
          const int idx_ml = idx_ml_base + l;

          const double cosmui = fb.cosmui[idx_ml];
          const double sinmui = fb.sinmui[idx_ml];
          const double cosmumi = fb.cosmumi[idx_ml];
          const double sinmumi = fb.sinmumi[idx_ml];

          lmksc += blmn[idx_kl] * cosmumi;   // --> flsc (no A)
          lmkcs += blmn[idx_kl] * sinmumi;   // --> flcs
          lmkcs_n -= clmn[idx_kl] * cosmui;  // --> flcs
          lmksc_n -= clmn[idx_kl] * sinmui;  // --> flsc

          rmkcc_n -= crmn[idx_kl] * cosmui;  // --> frcc
          zmkcs_n -= czmn[idx_kl] * cosmui;  // --> fzcs

          rmkss_n -= crmn[idx_kl] * sinmui;  // --> frss
          zmksc_n -= czmn[idx_kl] * sinmui;  // --> fzsc

          // assemble effective R and Z forces from MHD and spectral
          // condensation contributions
          const double tempR = armn[idx_kl] + xmpq[m] * frcon[idx_kl];
          const double tempZ = azmn[idx_kl] + xmpq[m] * fzcon[idx_kl];

          rmkcc += tempR * cosmui + brmn[idx_kl] * sinmumi;  // --> frcc
          rmkss += tempR * sinmui + brmn[idx_kl] * cosmumi;  // --> frss
          zmksc += tempZ * sinmui + bzmn[idx_kl] * cosmumi;  // --> fzsc
          zmkcs += tempZ * cosmui + bzmn[idx_kl] * sinmumi;  // --> fzcs
        }  // l

        for (int n = 0; n < s.ntor + 1; ++n) {
          const int idx_mn = ((jF - rp.nsMinF) * s.mpol + m) * (s.ntor + 1) + n;
          const int idx_kn = k * (s.nnyq2 + 1) + n;

          const double cosnv = fb.cosnv[idx_kn];
          const double sinnv = fb.sinnv[idx_kn];
          const double cosnvn = fb.cosnvn[idx_kn];
          const double sinnvn = fb.sinnvn[idx_kn];

          m_physical_forces.frcc[idx_mn] += rmkcc * cosnv + rmkcc_n * sinnvn;
          m_physical_forces.frss[idx_mn] += rmkss * sinnv + rmkss_n * cosnvn;
          m_physical_forces.fzsc[idx_mn] += zmksc * cosnv + zmksc_n * sinnvn;
          m_physical_forces.fzcs[idx_mn] += zmkcs * sinnv + zmkcs_n * cosnvn;

          if (jMinL <= jF) {
            m_physical_forces.flsc[idx_mn] += lmksc * cosnv + lmksc_n * sinnvn;
            m_physical_forces.flcs[idx_mn] += lmkcs * sinnv + lmkcs_n * cosnvn;
          }
        }  // n
      }  // k
    }  // m
  }  // jF

  // repeat the above just for jMaxRZ to nsMaxFIncludingLcfs, just for flsc,
  // flcs
  for (int jF = jMaxRZ; jF < rp.nsMaxFIncludingLcfs; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = m % 2 == 0;

      const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
      const auto& clmn = m_even ? d.clmn_e : d.clmn_o;

      for (int k = 0; k < s.nZeta; ++k) {
        double lmksc = 0.0;
        double lmksc_n = 0.0;
        double lmkcs = 0.0;
        double lmkcs_n = 0.0;

        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
        const int idx_ml_base = m * s.nThetaReduced;

        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_kl = idx_kl_base + l;
          const int idx_ml = idx_ml_base + l;

          const double cosmui = fb.cosmui[idx_ml];
          const double sinmui = fb.sinmui[idx_ml];
          const double cosmumi = fb.cosmumi[idx_ml];
          const double sinmumi = fb.sinmumi[idx_ml];

          lmksc += blmn[idx_kl] * cosmumi;   // --> flsc (no A)
          lmkcs += blmn[idx_kl] * sinmumi;   // --> flcs
          lmkcs_n -= clmn[idx_kl] * cosmui;  // --> flcs
          lmksc_n -= clmn[idx_kl] * sinmui;  // --> flsc
        }  // l

        for (int n = 0; n < s.ntor + 1; ++n) {
          const int idx_mn = ((jF - rp.nsMinF) * s.mpol + m) * (s.ntor + 1) + n;
          const int idx_kn = k * (s.nnyq2 + 1) + n;

          const double cosnv = fb.cosnv[idx_kn];
          const double sinnv = fb.sinnv[idx_kn];
          const double cosnvn = fb.cosnvn[idx_kn];
          const double sinnvn = fb.sinnvn[idx_kn];

          m_physical_forces.flsc[idx_mn] += lmksc * cosnv + lmksc_n * sinnvn;
          m_physical_forces.flcs[idx_mn] += lmkcs * sinnv + lmkcs_n * cosnvn;
        }  // n
      }  // k
    }  // m
  }  // jF
}

void vmecpp::FourierToReal3DSymmFastPoloidal(
    const FourierGeometry& physical_x, const std::vector<double>& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, RealSpaceGeometry& m_geometry) {
  // can safely assume lthreed == true in here

  absl::c_fill(m_geometry.r1_e, 0);
  absl::c_fill(m_geometry.r1_o, 0);
  absl::c_fill(m_geometry.ru_e, 0);
  absl::c_fill(m_geometry.ru_o, 0);
  absl::c_fill(m_geometry.rv_e, 0);
  absl::c_fill(m_geometry.rv_o, 0);
  absl::c_fill(m_geometry.z1_e, 0);
  absl::c_fill(m_geometry.z1_o, 0);
  absl::c_fill(m_geometry.zu_e, 0);
  absl::c_fill(m_geometry.zu_o, 0);
  absl::c_fill(m_geometry.zv_e, 0);
  absl::c_fill(m_geometry.zv_o, 0);
  absl::c_fill(m_geometry.lu_e, 0);
  absl::c_fill(m_geometry.lu_o, 0);
  absl::c_fill(m_geometry.lv_e, 0);
  absl::c_fill(m_geometry.lv_o, 0);

  absl::c_fill(m_geometry.rCon, 0);
  absl::c_fill(m_geometry.zCon, 0);

  // NOTE: fix on old VMEC++: need to transform geometry for nsMinF1 ... nsMaxF1
  const int nsMinF1 = r.nsMinF1;
  const int nsMinF = r.nsMinF;
  for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = m % 2 == 0;
      const int idx_ml_base = m * s.nThetaReduced;

      // with sqrtS for odd-m
      const double con_factor =
          m_even ? xmpq[m] : xmpq[m] * rp.sqrtSF[jF - nsMinF1];

      auto& r1 = m_even ? m_geometry.r1_e : m_geometry.r1_o;
      auto& ru = m_even ? m_geometry.ru_e : m_geometry.ru_o;
      auto& rv = m_even ? m_geometry.rv_e : m_geometry.rv_o;
      auto& z1 = m_even ? m_geometry.z1_e : m_geometry.z1_o;
      auto& zu = m_even ? m_geometry.zu_e : m_geometry.zu_o;
      auto& zv = m_even ? m_geometry.zv_e : m_geometry.zv_o;
      auto& lu = m_even ? m_geometry.lu_e : m_geometry.lu_o;
      auto& lv = m_even ? m_geometry.lv_e : m_geometry.lv_o;

      // axis only gets contributions up to m=1
      // --> all larger m contributions enter only from j=1 onwards
      // TODO(jons): why does the axis need m=1?
      int jMin = 1;
      if (m == 0 || m == 1) {
        jMin = 0;
      }

      if (jF < jMin) {
        continue;
      }

      for (int k = 0; k < s.nZeta; ++k) {
        double rmkcc = 0.0;
        double rmkcc_n = 0.0;
        double rmkss = 0.0;
        double rmkss_n = 0.0;
        double zmksc = 0.0;
        double zmksc_n = 0.0;
        double zmkcs = 0.0;
        double zmkcs_n = 0.0;
        double lmksc = 0.0;
        double lmksc_n = 0.0;
        double lmkcs = 0.0;
        double lmkcs_n = 0.0;

        for (int n = 0; n < s.ntor + 1; ++n) {
          // INVERSE TRANSFORM IN N-ZETA, FOR FIXED M

          const int idx_kn = k * (s.nnyq2 + 1) + n;

          double cosnv = fb.cosnv[idx_kn];
          double sinnv = fb.sinnv[idx_kn];
          double sinnvn = fb.sinnvn[idx_kn];
          double cosnvn = fb.cosnvn[idx_kn];

          int idx_mn = ((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1) + n;

          rmkcc += physical_x.rmncc[idx_mn] * cosnv;
          rmkcc_n += physical_x.rmncc[idx_mn] * sinnvn;
          rmkss += physical_x.rmnss[idx_mn] * sinnv;
          rmkss_n += physical_x.rmnss[idx_mn] * cosnvn;
          zmksc += physical_x.zmnsc[idx_mn] * cosnv;
          zmksc_n += physical_x.zmnsc[idx_mn] * sinnvn;
          zmkcs += physical_x.zmncs[idx_mn] * sinnv;
          zmkcs_n += physical_x.zmncs[idx_mn] * cosnvn;
          lmksc += physical_x.lmnsc[idx_mn] * cosnv;
          lmksc_n += physical_x.lmnsc[idx_mn] * sinnvn;
          lmkcs += physical_x.lmncs[idx_mn] * sinnv;
          lmkcs_n += physical_x.lmncs[idx_mn] * cosnvn;
        }  // n

        // INVERSE TRANSFORM IN M-THETA, FOR ALL RADIAL, ZETA VALUES
        const int idx_kl_base = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff;

        // the loop over l is split to help compiler auto-vectorization
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_ml = idx_ml_base + l;

          const double sinmum = fb.sinmum[idx_ml];
          const double cosmum = fb.cosmum[idx_ml];

          const int idx_kl = idx_kl_base + l;
          ru[idx_kl] += rmkcc * sinmum + rmkss * cosmum;
          zu[idx_kl] += zmksc * cosmum + zmkcs * sinmum;
          lu[idx_kl] += lmksc * cosmum + lmkcs * sinmum;
        }  // l

        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_kl = idx_kl_base + l;
          const int idx_ml = idx_ml_base + l;

          const double cosmu = fb.cosmu[idx_ml];
          const double sinmu = fb.sinmu[idx_ml];
          rv[idx_kl] += rmkcc_n * cosmu + rmkss_n * sinmu;
          zv[idx_kl] += zmksc_n * sinmu + zmkcs_n * cosmu;
          // it is here that lv gets a negative sign!
          lv[idx_kl] -= lmksc_n * sinmu + lmkcs_n * cosmu;
        }  // l

        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_ml = idx_ml_base + l;

          const double cosmu = fb.cosmu[idx_ml];
          const double sinmu = fb.sinmu[idx_ml];

          const int idx_kl = idx_kl_base + l;

          r1[idx_kl] += rmkcc * cosmu + rmkss * sinmu;
          z1[idx_kl] += zmksc * sinmu + zmkcs * cosmu;
        }  // l

        if (nsMinF <= jF && jF < r.nsMaxFIncludingLcfs) {
          for (int l = 0; l < s.nThetaReduced; ++l) {
            const int idx_ml = idx_ml_base + l;
            const double cosmu = fb.cosmu[idx_ml];
            const double sinmu = fb.sinmu[idx_ml];

            // spectral condensation is local per flux surface
            // --> no need for numFull1
            const int idx_con = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff + l;
            m_geometry.rCon[idx_con] +=
                (rmkcc * cosmu + rmkss * sinmu) * con_factor;
            m_geometry.zCon[idx_con] +=
                (zmksc * sinmu + zmkcs * cosmu) * con_factor;
          }
        }  // l
      }  // k
    }  // m
  }  // j
}

// Implemented as a free function for easier testing and benchmarking.
void vmecpp::deAliasConstraintForce(
    const vmecpp::RadialPartitioning& rp,
    const vmecpp::FourierBasisFastPoloidal& fb, const vmecpp::Sizes& s_,
    const std::vector<double>& faccon, const std::vector<double>& tcon,
    const std::vector<double>& gConEff, std::vector<double>& m_gsc,
    std::vector<double>& m_gcs, std::vector<double>& m_gcc,
    std::vector<double>& m_gss, std::vector<double>& m_gCon) {
  absl::c_fill_n(m_gCon, (rp.nsMaxF - rp.nsMinF) * s_.nZnT, 0);

  // Temporary array for asymmetric contributions (like gcona in jVMEC)
  std::vector<double> gConAsym;
  if (s_.lasym) {
    gConAsym.resize((rp.nsMaxF - rp.nsMinF) * s_.nZnT, 0.0);
  }

  // no constraint on axis --> has no poloidal angle
  int jMin = 0;
  if (rp.nsMinF == 0) {
    jMin = 1;
  }

  for (int jF = std::max(jMin, rp.nsMinF); jF < rp.nsMaxF; ++jF) {
    for (int m = 1; m < s_.mpol - 1; ++m) {
      absl::c_fill_n(m_gsc, s_.ntor + 1, 0);
      absl::c_fill_n(m_gcs, s_.ntor + 1, 0);
      if (s_.lasym) {
        absl::c_fill_n(m_gcc, s_.ntor + 1, 0);
        absl::c_fill_n(m_gss, s_.ntor + 1, 0);
      }

      for (int k = 0; k < s_.nZeta; ++k) {
        double w0 = 0.0;
        double w1 = 0.0;
        double w2 = 0.0;
        double w3 = 0.0;

        // fwd transform in poloidal direction
        // integrate poloidally to get m-th poloidal Fourier coefficient
        for (int l = 0; l < s_.nThetaReduced; ++l) {
          const int idx_ml = m * s_.nThetaReduced + l;

          int idx_kl = ((jF - rp.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
          w0 += gConEff[idx_kl] * fb.sinmui[idx_ml];
          w1 += gConEff[idx_kl] * fb.cosmui[idx_ml];

          if (s_.lasym) {
            // Handle reflection indices for asymmetric case
            const int kReversed = (s_.nZeta - k) % s_.nZeta;
            const int lReversed = (s_.nThetaReduced - l) % s_.nThetaReduced;
            int idx_kl_rev =
                ((jF - rp.nsMinF) * s_.nZeta + kReversed) * s_.nThetaEff +
                lReversed;
            w2 += gConEff[idx_kl_rev] * fb.cosmui[idx_ml];
            w3 += gConEff[idx_kl_rev] * fb.sinmui[idx_ml];
          }
        }  // l

        // forward Fourier transform in toroidal direction for full set of mode
        // numbers (n = 0, 1, ..., ntor)
        for (int n = 0; n < s_.ntor + 1; ++n) {
          int idx_kn = k * (s_.nnyq2 + 1) + n;

          // NOTE: `tcon` comes into play here
          if (!s_.lasym) {
            m_gsc[n] += fb.cosnv[idx_kn] * w0 * tcon[jF - rp.nsMinF];
            m_gcs[n] += fb.sinnv[idx_kn] * w1 * tcon[jF - rp.nsMinF];
          } else {
            // Asymmetric case with on-the-fly symmetrization
            m_gcc[n] +=
                0.5 * tcon[jF - rp.nsMinF] * fb.cosnv[idx_kn] * (w1 + w2);
            m_gss[n] +=
                0.5 * tcon[jF - rp.nsMinF] * fb.sinnv[idx_kn] * (w0 + w3);
            m_gsc[n] +=
                0.5 * tcon[jF - rp.nsMinF] * fb.cosnv[idx_kn] * (w0 - w3);
            m_gcs[n] +=
                0.5 * tcon[jF - rp.nsMinF] * fb.sinnv[idx_kn] * (w1 - w2);
          }
        }
      }  // k

      // ------------------------------------------
      // need to "wait" (= finish k loop) here
      // to get Fourier coefficients fully defined!
      // ------------------------------------------

      // inverse Fourier-transform from reduced set of mode numbers
      for (int k = 0; k < s_.nZeta; ++k) {
        double w0 = 0.0;
        double w1 = 0.0;
        double w2 = 0.0;
        double w3 = 0.0;

        // collect contribution to current grid point from n-th toroidal mode
        for (int n = 0; n < s_.ntor + 1; ++n) {
          int idx_kn = k * (s_.nnyq2 + 1) + n;
          w2 += m_gcs[n] * fb.sinnv[idx_kn];
          w3 += m_gsc[n] * fb.cosnv[idx_kn];
          if (s_.lasym) {
            w0 += m_gcc[n] * fb.cosnv[idx_kn];
            w1 += m_gss[n] * fb.sinnv[idx_kn];
          }
        }  // n

        // inv transform in poloidal direction
        for (int l = 0; l < s_.nThetaReduced; ++l) {
          int idx_kl = ((jF - rp.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
          const int idx_ml = m * s_.nThetaReduced + l;

          // NOTE: `faccon` comes into play here
          m_gCon[idx_kl] +=
              faccon[m] * (w2 * fb.cosmu[idx_ml] + w3 * fb.sinmu[idx_ml]);

          if (s_.lasym) {
            // Store asymmetric contribution separately
            gConAsym[idx_kl] +=
                faccon[m] * (w0 * fb.cosmu[idx_ml] + w1 * fb.sinmu[idx_ml]);
          }
        }  // l
      }  // k
    }  // m
  }

  // For asymmetric case, extend gCon into theta = [pi, 2*pi] domain
  if (s_.lasym) {
    // Based on jVMEC lines 418-438
    // First, add the asymmetric contribution to theta = [0, pi]
    for (int jF = std::max(jMin, rp.nsMinF); jF < rp.nsMaxF; ++jF) {
      for (int k = 0; k < s_.nZeta; ++k) {
        for (int l = 0; l < s_.nThetaReduced; ++l) {
          int idx_kl = ((jF - rp.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
          m_gCon[idx_kl] += gConAsym[idx_kl];
        }
      }
    }

    // Note: Extension to theta=[pi,2pi] is handled elsewhere in the code
    // through the symrzl functions that extend all quantities consistently
  }
}

namespace vmecpp {

IdealMhdModel::IdealMhdModel(
    FlowControl* m_fc, const Sizes* s, const FourierBasisFastPoloidal* t,
    RadialProfiles* m_p, const VmecConstants* constants,
    ThreadLocalStorage* m_ls, HandoverStorage* m_h, const RadialPartitioning* r,
    FreeBoundaryBase* m_fb, int signOfJacobian, int nvacskip,
    VacuumPressureState* m_vacuum_pressure_state)
    : m_fc_(*m_fc),
      s_(*s),
      t_(*t),
      m_p_(*m_p),
      constants_(*constants),
      m_ls_(*m_ls),
      m_h_(*m_h),
      r_(*r),
      m_fb_(m_fb),
      m_vacuum_pressure_state_(*m_vacuum_pressure_state),
      signOfJacobian(signOfJacobian),
      nvacskip(nvacskip),
      ivacskip(0) {
  CHECK_GE(nvacskip, 0)
      << "Should never happen: should be checked by VmecINDATA";
  if (m_fc_.lfreeb) {
    CHECK(m_fb_ != nullptr)
        << "Free-boundary configuration requires a Free-boundary solver";
  }

  // init members
  ncurr = 0;
  adiabaticIndex = 0.0;
  tcon0 = 0.0;

  // allocate arrays
  xmpq.resize(s_.mpol);
  faccon.resize(s_.mpol);
  for (int m = 0; m < s_.mpol; ++m) {
    xmpq[m] = m * (m - 1);
    if (m > 1) {
      faccon[m - 1] = -0.25 * signOfJacobian / (xmpq[m] * xmpq[m]);
    }
  }

  int nrzt1 = s_.nZnT * (r_.nsMaxF1 - r_.nsMinF1);
  int nrzt = s_.nZnT * (r_.nsMaxF - r_.nsMinF);

  r1_e.resize(nrzt1);
  r1_o.resize(nrzt1);
  ru_e.resize(nrzt1);
  ru_o.resize(nrzt1);
  z1_e.resize(nrzt1);
  z1_o.resize(nrzt1);
  zu_e.resize(nrzt1);
  zu_o.resize(nrzt1);
  lu_e.resize(nrzt1);
  lu_o.resize(nrzt1);

  if (s_.lthreed) {
    rv_e.resize(nrzt1);
    rv_o.resize(nrzt1);
    zv_e.resize(nrzt1);
    zv_o.resize(nrzt1);
    lv_e.resize(nrzt1);
    lv_o.resize(nrzt1);
  }

  int nrztIncludingBoundary = s_.nZnT * (r_.nsMaxFIncludingLcfs - r_.nsMinF);

  ruFull.resize(nrztIncludingBoundary);
  zuFull.resize(nrztIncludingBoundary);

  rCon.resize(nrztIncludingBoundary);
  zCon.resize(nrztIncludingBoundary);

  rCon0.resize(nrztIncludingBoundary);
  zCon0.resize(nrztIncludingBoundary);

  r12.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  ru12.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  zu12.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  rs.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  zs.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  tau.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

  gsqrt.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

  guu.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  if (s_.lthreed) {
    guv.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  }
  gvv.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

  bsupu.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  bsupv.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

  bsubu.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  bsubv.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);

  totalPressure.resize((r_.nsMaxH - r_.nsMinH) * s_.nZnT);
  rBSq.resize(s_.nZnT);

  insideTotalPressure.resize(s_.nZnT);
  delBSq.resize(s_.nZnT);

  // CRITICAL: resize() does not initialize to zero!
  // Java arrays are zero-initialized, but C++ vectors are not.
  // This matches jVMEC behavior where force arrays rely on zero initialization.
  armn_e.resize(nrzt, 0.0);
  armn_o.resize(nrzt, 0.0);
  brmn_e.resize(nrzt, 0.0);
  brmn_o.resize(nrzt, 0.0);
  azmn_e.resize(nrzt, 0.0);
  azmn_o.resize(nrzt, 0.0);
  bzmn_e.resize(nrzt, 0.0);
  bzmn_o.resize(nrzt, 0.0);
  blmn_e.resize(nrztIncludingBoundary, 0.0);
  blmn_o.resize(nrztIncludingBoundary, 0.0);

  if (s_.lthreed) {
    crmn_e.resize(nrzt, 0.0);
    crmn_o.resize(nrzt, 0.0);
    czmn_e.resize(nrzt, 0.0);
    czmn_o.resize(nrzt, 0.0);
    clmn_e.resize(nrztIncludingBoundary, 0.0);
    clmn_o.resize(nrztIncludingBoundary, 0.0);
  }

  // TODO(jons): +1 only if at LCFS
  bLambda.resize(r_.nsMaxF1 - r_.nsMinF1 + 1);
  dLambda.resize(r_.nsMaxF1 - r_.nsMinF1 + 1);
  cLambda.resize(r_.nsMaxF1 - r_.nsMinF1 + 1);
  lambdaPreconditioner.resize((r_.nsMaxFIncludingLcfs - r_.nsMinF) * s_.mpol *
                              (s_.ntor + 1));

  ax.resize((r_.nsMaxH - r_.nsMinH) * 4);
  bx.resize((r_.nsMaxH - r_.nsMinH) * 3);
  cx.resize(r_.nsMaxH - r_.nsMinH);

  arm.resize((r_.nsMaxH - r_.nsMinH) * 2);
  azm.resize((r_.nsMaxH - r_.nsMinH) * 2);
  brm.resize((r_.nsMaxH - r_.nsMinH) * 2);
  bzm.resize((r_.nsMaxH - r_.nsMinH) * 2);

  ard.resize((r_.nsMaxF - r_.nsMinF) * 2);
  brd.resize((r_.nsMaxF - r_.nsMinF) * 2);
  azd.resize((r_.nsMaxF - r_.nsMinF) * 2);
  bzd.resize((r_.nsMaxF - r_.nsMinF) * 2);
  cxd.resize(r_.nsMaxF - r_.nsMinF);

  // leave one entry at beginning as target to put in the data sent from the MPI
  // rank next inside
  ar.resize((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
  az.resize((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
  dr.resize((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
  dz.resize((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
  br.resize((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);
  bz.resize((r_.nsMaxF - r_.nsMinF) * (s_.ntor + 1) * s_.mpol);

  tcon.resize(r_.nsMaxFIncludingLcfs - r_.nsMinF);

  gConEff.resize(nrztIncludingBoundary);
  gsc.resize(s_.ntor + 1);
  gcs.resize(s_.ntor + 1);
  gcc.resize(s_.ntor + 1);
  gss.resize(s_.ntor + 1);
  gCon.resize(nrztIncludingBoundary);

  frcon_e.resize(nrzt);
  frcon_o.resize(nrzt);
  fzcon_e.resize(nrzt);
  fzcon_o.resize(nrzt);

  jMin.resize(s_.mpol * (s_.ntor + 1));
}

void IdealMhdModel::setFromINDATA(int ncurr, double adiabaticIndex,
                                  double tcon0) {
  this->ncurr = ncurr;
  this->adiabaticIndex = adiabaticIndex;
  this->tcon0 = tcon0;
}

void IdealMhdModel::evalFResInvar(const std::vector<double>& localFResInvar) {
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    m_fc_.fResInvar[0] = 0.0;
    m_fc_.fResInvar[1] = 0.0;
    m_fc_.fResInvar[2] = 0.0;
  }

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    m_fc_.fResInvar[0] += localFResInvar[0];
    m_fc_.fResInvar[1] += localFResInvar[1];
    m_fc_.fResInvar[2] += localFResInvar[2];
  }

// this is protecting reads of fResInvar as well as
// writes to m_fc.fsqz which is read before this call
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // set new values
    // TODO(jons): what is `r1scale`?
    constexpr double r1scale = 0.25;

    m_fc_.fsqr = m_fc_.fResInvar[0] * m_h_.fNormRZ * r1scale;
    m_fc_.fsqz = m_fc_.fResInvar[1] * m_h_.fNormRZ * r1scale;
    m_fc_.fsql = m_fc_.fResInvar[2] * m_h_.fNormL;
  }
}

void IdealMhdModel::evalFResPrecd(const std::vector<double>& localFResPrecd) {
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    m_fc_.fResPrecd[0] = 0.0;
    m_fc_.fResPrecd[1] = 0.0;
    m_fc_.fResPrecd[2] = 0.0;
  }

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    m_fc_.fResPrecd[0] += localFResPrecd[0];
    m_fc_.fResPrecd[1] += localFResPrecd[1];
    m_fc_.fResPrecd[2] += localFResPrecd[2];
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    m_fc_.fsqr1 = m_fc_.fResPrecd[0] * m_h_.fNorm1;
    m_fc_.fsqz1 = m_fc_.fResPrecd[1] * m_h_.fNorm1;
    m_fc_.fsql1 = m_fc_.fResPrecd[2] * m_fc_.deltaS;
  }
}

absl::StatusOr<bool> IdealMhdModel::update(
    FourierGeometry& m_decomposed_x, FourierGeometry& m_physical_x,
    FourierForces& m_decomposed_f, FourierForces& m_physical_f,
    bool& m_need_restart, int& m_last_preconditioner_update,
    int& m_last_full_update_nestor, FlowControl& m_fc, const int iter1,
    const int iter2, const VmecCheckpoint& checkpoint,
    const int iterations_before_checkpointing, bool verbose) {
  // preprocess Fourier coefficients of geometry
  m_decomposed_x.decomposeInto(m_physical_x, m_p_.scalxc);
  if (checkpoint == VmecCheckpoint::FOURIER_GEOMETRY_TO_START_WITH &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  // undo m=1 constraint
  m_physical_x.m1Constraint(1.0);

  m_physical_x.extrapolateTowardsAxis();

  // inv-DFT to get realspace geometric quantities
  geometryFromFourier(m_physical_x);
  if (checkpoint == VmecCheckpoint::INV_DFT_GEOMETRY &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  if (iter2 == iter1 &&
      (m_vacuum_pressure_state_ == VacuumPressureState::kOff ||
       m_vacuum_pressure_state_ == VacuumPressureState::kInitializing)) {
    rzConIntoVolume();
  }

  // sets m_fc_.restart_reason
  computeJacobian();
  if (checkpoint == VmecCheckpoint::JACOBIAN &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  if (m_fc_.restart_reason == RestartReason::BAD_JACOBIAN) {
    // bad jacobian and not final iteration yet
    //   (would be indicated by iequi.eq.1)
    // --> need to restart except when computing output file
    // --> in that case, ignore bad jacobian

    return false;
  }

  // start of bcovar (ends in updateForces)

  computeMetricElements();
  if (checkpoint == VmecCheckpoint::METRIC &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  updateDifferentialVolume();

  if (iter2 == 1) {
    computeInitialVolume();
  }
  if (checkpoint == VmecCheckpoint::VOLUME &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  computeBContra();
  if (checkpoint == VmecCheckpoint::B_CONTRA &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  computeBCo();
  if (checkpoint == VmecCheckpoint::B_CO &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  pressureAndEnergies();
  if (checkpoint == VmecCheckpoint::ENERGY &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  radialForceBalance();
  if (checkpoint == VmecCheckpoint::RADIAL_FORCE_BALANCE &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  // This computes the poloidal current close to the axis (rBtor0).
  if (r_.nsMinH == 0) {
    // only in thread that has axis

    // output only
    m_h_.rBtor0 = 1.5 * m_p_.bvcoH[r_.nsMinH - r_.nsMinH] -
                  0.5 * m_p_.bvcoH[r_.nsMinH + 1 - r_.nsMinH];
  }

  if (r_.nsMaxH == m_fc_.ns - 1) {
    // only in thread that has LCFS

    // poloidal current at the boundary; used only as a check for NESTOR; output
    m_h_.rBtor = 1.5 * m_p_.bvcoH[r_.nsMaxH - 1 - r_.nsMinH] -
                 0.5 * m_p_.bvcoH[r_.nsMaxH - 2 - r_.nsMinH];

    // This computes the net toroidal current enclosed by the LCFS (cTor).
    // net toroidal current input to NESTOR
    // TODO(jons): if add_fluxed always works, could use curtor instead and not
    // have to wait for MHD routines to finish for calling NESTOR - more
    // parallelization possible!
    m_h_.cTor = (1.5 * m_p_.bucoH[r_.nsMaxH - 1 - r_.nsMinH] -
                 0.5 * m_p_.bucoH[r_.nsMaxH - 2 - r_.nsMinH]) *
                signOfJacobian * 2.0 * M_PI;
  }

  hybridLambdaForce();
  if (checkpoint == VmecCheckpoint::HYBRID_LAMBDA_FORCE &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  // NOTE: No need to return here in case of iequi != 0,
  // since we don't overwrite stuff in-place in VMEC++.

  if (shouldUpdateRadialPreconditioner(iter1, iter2)) {
#ifdef _OPENMP
#pragma omp single nowait
#endif  // _OPENMP
    {
      m_last_preconditioner_update = iter2;
    }

    updateRadialPreconditioner();
    if (checkpoint == VmecCheckpoint::UPDATE_RADIAL_PRECONDITIONER &&
        iter2 >= iterations_before_checkpointing) {
      return true;
    }

    updateVolume();

    // need preconditioner matrix elements for constraint force multiplier

    computeForceNorms(m_decomposed_x);
    if (checkpoint == VmecCheckpoint::UPDATE_FORCE_NORMS &&
        iter2 >= iterations_before_checkpointing) {
      return true;
    }

    absl::Status s = constraintForceMultiplier();
    if (!s.ok()) {
      return s;
    }
    if (checkpoint == VmecCheckpoint::UPDATE_TCON &&
        iter2 >= iterations_before_checkpointing) {
      return true;
    }
  }  // update radial preconditioner?

  // virtual checkpoint, if maximum_iterations not integer multiple of
  // kPreconditionerUpdateInterval
  if ((checkpoint == VmecCheckpoint::UPDATE_RADIAL_PRECONDITIONER ||
       checkpoint == VmecCheckpoint::UPDATE_FORCE_NORMS ||
       checkpoint == VmecCheckpoint::UPDATE_TCON) &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  // end of bcovar

  // back in funct3d, free-boundary force contribution active?
  // This can even happen in the first iteration when hot-restarted.
  if (m_fc_.lfreeb &&
      (iter2 > 1 || m_vacuum_pressure_state_ != VacuumPressureState::kOff)) {
    ivacskip = (iter2 - iter1) % nvacskip;
    // when R+Z force residuals are <1e-3, enable vacuum contribution
    if (m_vacuum_pressure_state_ != VacuumPressureState::kActive &&
        m_fc_.fsqr + m_fc_.fsqz < 1.0e-3) {
      // vacuum pressure not fully turned on yet
      // Do full vacuum calc on every iteration
      ivacskip = 0;
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
      // Increment ivac, never exceeding VacuumPressureState::kActive
      m_vacuum_pressure_state_ = static_cast<VacuumPressureState>(
          static_cast<int>(m_vacuum_pressure_state_) + 1);
    }

    // EXTEND NVACSKIP AS EQUILIBRIUM CONVERGES
    if (ivacskip == 0) {
      const int new_nvacskip = static_cast<int>(
          1.0 / std::max(0.1, 1.0e11 * (m_fc_.fsqr + m_fc_.fsqz)));
      nvacskip = std::max(nvacskip, new_nvacskip);

#ifdef _OPENMP
#pragma omp single nowait
#endif  // _OPENMP
      {
        m_last_full_update_nestor = iter2;
      }
    }
// protects read of `m_vacuum_pressure_state_` below from the write above
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
    if (m_vacuum_pressure_state_ != VacuumPressureState::kOff) {
      // IF INITIALLY ON, MUST TURN OFF rcon0, zcon0 SLOWLY
      for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
        for (int kl = 0; kl < s_.nZnT; ++kl) {
          int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;

          // gradually turn off rcon0, zcon0
          rCon0[idx_kl] *= 0.9;
          zCon0[idx_kl] *= 0.9;
        }  // kl
      }  // j

      if (r_.nsMaxF1 == m_fc_.ns) {
        // can only get this from thread that has the LCFS !!!

        // TODO(jons): respect lthreed in case of a free-boundary axisymmetric
        // run
        HandOverBoundaryGeometry(
            m_h_, m_physical_x, s_,
            /*offset=*/(r_.nsMaxF1 - 1 - r_.nsMinF1) * s_.mnsize);
      }

      if (r_.nsMinF == 0) {
        // this thread has the magnetic axis
        // Note: axis geometry is zero-th flux surface, l = 0, k fastest index
        HandOverMagneticAxis(m_h_, r1_e, z1_e, s_);
      }

// protect reads of magnetic axis, boundary geometry below from writes above
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
      const double netToroidalCurrent = m_h_.cTor / MU_0;
      bool reached_checkpoint = m_fb_->update(
          m_h_.rCC_LCFS, m_h_.rSS_LCFS, m_h_.rSC_LCFS, m_h_.rCS_LCFS,
          m_h_.zSC_LCFS, m_h_.zCS_LCFS, m_h_.zCC_LCFS, m_h_.zSS_LCFS,
          signOfJacobian, m_h_.rAxis, m_h_.zAxis, &(m_h_.bSubUVac),
          &(m_h_.bSubVVac), netToroidalCurrent, ivacskip, checkpoint,
          iter2 >= iterations_before_checkpointing);
      if (reached_checkpoint) {
        return true;
      }

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
      {
        // In educational_VMEC, this is part of Nestor.
        if (m_vacuum_pressure_state_ == VacuumPressureState::kInitializing) {
          m_vacuum_pressure_state_ = VacuumPressureState::kInitialized;

          if (verbose) {
            // bSubUVac and cTor contain 2*pi already; see Nestor.cc for
            // bSubUVac and above for cTor
            const double fac = 1.0e-6 / MU_0;  // in MA
            std::cout << "\n";
            std::cout << absl::StrFormat(
                "2*pi * a * -BPOL(vac) = %10.2e MA       R * BTOR(vacuum) = "
                "%10.2e\n",
                m_h_.bSubUVac * fac, m_h_.bSubVVac);
            std::cout << absl::StrFormat(
                "     TOROIDAL CURRENT = %10.2e MA       R * BTOR(plasma) = "
                "%10.2e\n",
                m_h_.cTor * fac, m_h_.rBtor);
          }
        }  // fullUpdate printout
      }

      if (m_h_.rBtor * m_h_.bSubVVac < 0.0) {
        return absl::InternalError(
            "IdealMHDModel::update: rbtor and bsubvvac must have the same "
            "sign - maybe flip the sign of phiedge or the sign of the coil "
            "currents");
      } else if (fabs((m_h_.cTor - m_h_.bSubUVac) / m_h_.rBtor) > 0.01) {
        return absl::InternalError(
            "IdealMHDModel::update: VAC-VMEC I_TOR MISMATCH : BOUNDARY MAY "
            "ENCLOSE EXT. COIL");
      }

      // RESET FIRST TIME FOR SOFT START
      if (m_vacuum_pressure_state_ == VacuumPressureState::kInitialized) {
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
        m_fc_.restart_reason = RestartReason::BAD_JACOBIAN;
        m_need_restart = true;
      } else {
        m_need_restart = false;
      }

      if (r_.nsMaxF1 == m_fc_.ns) {
        // MUST NOT BREAK TRI-DIAGONAL RADIAL COUPLING: OFFENDS PRECONDITIONER!
        // double edgePressure = 1.5 * p.presH[r.nsMaxH-1 - r.nsMinH] - 0.5 *
        // p.presH[r.nsMinH - r.nsMinH];
        double edgePressure =
            m_p_.evalMassProfile((m_fc_.ns - 1.5) / (m_fc_.ns - 1.0));
        if (edgePressure != 0.0) {
          edgePressure = m_p_.evalMassProfile(1.0) / edgePressure *
                         m_p_.presH[r_.nsMaxH - 1 - r_.nsMinH];
        }

        for (int kl = 0; kl < s_.nZnT; ++kl) {
          // extrapolate total pressure (from inside) to LCFS
          // TODO(jons): mark that this is bsqsav(lk,3)
          insideTotalPressure[kl] =
              1.5 * totalPressure[(r_.nsMaxH - 1 - r_.nsMinH) * s_.nZnT + kl] -
              0.5 * totalPressure[(r_.nsMaxH - 2 - r_.nsMinH) * s_.nZnT + kl];

          // net pressure from outside on LCFS
          // FIXME(eguiraud) slow loop over Nestor output
          // NOTE: here is the interface between the fast-toroidal setup in
          // Nestor and fast-poloidal setup in VMEC
          const int k = kl / s_.nThetaEff;
          const int l = kl % s_.nThetaEff;
          const int idx_lk = l * s_.nZeta + k;
          double outsideEdgePressure =
              m_h_.vacuum_magnetic_pressure[idx_lk] + edgePressure;

          // term to enter MHD forces
          int idx_kl = (r_.nsMaxF1 - 1 - r_.nsMinF1) * s_.nZnT + kl;
          rBSq[kl] = outsideEdgePressure * (r1_e[idx_kl] + r1_o[idx_kl]) /
                     m_fc_.deltaS;

          // for printout: global mismatch between inside and outside pressure
          delBSq[kl] = fabs(outsideEdgePressure - insideTotalPressure[kl]);
        }

        if (m_vacuum_pressure_state_ == VacuumPressureState::kInitialized) {
          // TODO(jons): implement this !!!

          // initial magnetic field at boundary
          // bsqsav(:nznt,1) = bzmn_o(ns:nrzt:ns)

          // initial NESTOR |B|^2 at boundary
          // bsqsav(:nznt,2) = bsqvac(:nznt)
        }
      }

      if (checkpoint == VmecCheckpoint::RBSQ &&
          iter2 >= iterations_before_checkpointing) {
        return true;
      }
    }
  }  // lfreeb

  // NOTE: if (iequi != 1) { ... continue with code below ...
  // -> iequi==1 computations for outputs are done in OutputQuantities

  effectiveConstraintForce();

  deAliasConstraintForce();
  if (checkpoint == VmecCheckpoint::ALIAS &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  computeMHDForces();

  assembleTotalForces();
  if (checkpoint == VmecCheckpoint::REALSPACE_FORCES &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  forcesToFourier(m_physical_f);
  if (checkpoint == VmecCheckpoint::FWD_DFT_FORCES &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  m_physical_f.decomposeInto(m_decomposed_f, m_p_.scalxc);

  // ----- start of residue

  // re-establish m=1 constraint
  // TODO(jons): why 1/sqrt(2) and not 1/2 ?
  m_decomposed_f.m1Constraint(1.0 / std::numbers::sqrt2);

  // v8.50: ADD iter2<2 so reset=<WOUT_FILE> works
  if (m_fc.fsqz < 1.0e-6 || iter2 < 2) {
    // ensure that the m=1 constraint is satisfied exactly
    // --> the corresponding m=1 coeffs of R,Z are constrained to be zero
    //     and thus must not be "forced" (by the time evol using gc) away from
    //     zero
    m_decomposed_f.zeroZForceForM1();
  }

  if (checkpoint == VmecCheckpoint::PHYSICAL_FORCES &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  // COMPUTE INVARIANT RESIDUALS

  // include edge contribution if the equilibrium has converged very quickly,
  // to prevent a strong force-imbalance at the LCFS-vacuum transition, since
  // the termination criterion based on sum(force residuals) < ftol only
  // considers the inner flux-surfaces, but not the balance with the magnetic
  // pressure in vacuum at the LCFS. This special case includes that force
  // contribution in the first few iterations, preventing termination, to
  // ensure the free-boundary forces have "enough time" to propagate through
  // to the inner surfaces.
  // TODO(jurasic) the hard-coded 50 and 1e-6 are only here for backwards
  // compatibility, ideally vacuum-pressure should always part of the
  // force-balance
  bool almost_converged = (m_fc.fsqr + m_fc.fsqz) < 1.0e-6;
  // In iter==1, the forces are initialized to 1.0 so includeEdgeRZForces
  // wouldn't trigger without special handling for the hot-restart case.
  bool hot_restart = (iter2 == 1 && m_vacuum_pressure_state_ ==
                                        VacuumPressureState::kInitialized);
  bool includeEdgeRZForces =
      ((iter2 - iter1) < 50 && (almost_converged || hot_restart));
  std::vector<double> localFResInvar(3, 0.0);
  m_decomposed_f.residuals(localFResInvar, includeEdgeRZForces);

  evalFResInvar(localFResInvar);

  if (checkpoint == VmecCheckpoint::INVARIANT_RESIDUALS &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  // PERFORM PRECONDITIONING AND COMPUTE RESIDUES

  applyM1Preconditioner(m_decomposed_f);
  if (checkpoint == VmecCheckpoint::APPLY_M1_PRECONDITIONER &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  assembleRZPreconditioner();
  if (checkpoint == VmecCheckpoint::ASSEMBLE_RZ_PRECONDITIONER &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  absl::Status status = applyRZPreconditioner(m_decomposed_f);
  if (!status.ok()) {
    return status;
  }

  applyLambdaPreconditioner(m_decomposed_f);
  if (checkpoint == VmecCheckpoint::APPLY_RADIAL_PRECONDITIONER &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  std::vector<double> localFResPrecd(3, 0.0);
  m_decomposed_f.residuals(localFResPrecd, true);

  evalFResPrecd(localFResPrecd);

  if (checkpoint == VmecCheckpoint::PRECONDITIONED_RESIDUALS &&
      iter2 >= iterations_before_checkpointing) {
    return true;
  }

  // --- end of residue()

  // back in funct3d

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    if (iter2 == 1 && (m_fc.fsqr + m_fc.fsqz + m_fc.fsql) > 1.0e2) {
      // first iteration and gigantic force residuals
      // --> what is going on here?
      m_fc.restart_reason = RestartReason::HUGE_INITIAL_FORCES;
    }
  }

  // end of funct3d

  return false;
}  // NOLINT(readability/fn_size)

/** inverse Fourier transform to get geometry from Fourier coefficients */
void IdealMhdModel::geometryFromFourier(const FourierGeometry& physical_x) {
  // std::cout << "DEBUG geometryFromFourier: lasym=" << s_.lasym
  //           << ", lthreed=" << s_.lthreed << std::endl;

  // symmetric contribution is always needed
  if (s_.lthreed) {
    dft_FourierToReal_3d_symm(physical_x);
  } else {
    dft_FourierToReal_2d_symm(physical_x);
  }

  if (s_.lasym) {
    std::cout << "DEBUG: Processing asymmetric equilibrium with lasym=true" << std::endl;
    
    // asymmetric contribution needed for non-symmetric equilibria
    std::cout << "DEBUG: Processing asymmetric contribution" << std::endl;

    // Check physical_x arrays
    std::cout << "DEBUG: physical_x.rmnsc size=" << physical_x.rmnsc.size()
              << std::endl;
    if (physical_x.rmnsc.size() > 0) {
      std::cout << "  First few rmnsc values: ";
      for (int i = 0;
           i < std::min(5, static_cast<int>(physical_x.rmnsc.size())); ++i) {
        std::cout << physical_x.rmnsc[i] << " ";
      }
      std::cout << std::endl;
    }

    // IMPORTANT: For asymmetric case, following educational_VMEC pattern:
    // 1. Symmetric transform already filled main arrays (r1_e, etc.)
    // 2. Asymmetric transform fills m_ls_ arrays with ONLY asymmetric
    // contributions
    // 3. ADD asymmetric contributions to symmetric arrays (r1s = r1s + r1a
    // pattern)

    // Step 1: Resize asymmetric arrays to accommodate all radial surfaces
    int required_size = s_.nZnT * (r_.nsMaxF1 - r_.nsMinF1);
    std::cout << "DEBUG: Resizing asymmetric arrays from " << m_ls_.r1e_i.size()
              << " to " << required_size << std::endl;

    m_ls_.r1e_i.resize(required_size, 0.0);
    m_ls_.z1e_i.resize(required_size, 0.0);
    m_ls_.lue_i.resize(required_size, 0.0);
    m_ls_.rue_i.resize(required_size, 0.0);
    m_ls_.zue_i.resize(required_size, 0.0);

    // Step 2: Apply asymmetric transform to fill m_ls_ arrays
    if (s_.lthreed) {
      dft_FourierToReal_3d_asymm(physical_x);
    } else {
      std::cout << "DEBUG: Calling 2D asymmetric transform" << std::endl;
      dft_FourierToReal_2d_asymm(physical_x);
    }

    // Step 3: Apply symmetrization to asymmetric arrays
    std::cout << "DEBUG: Calling symrzl_geometry" << std::endl;
    symrzl_geometry(physical_x);

    // Step 4: ADD asymmetric contributions to symmetric arrays
    // (educational_VMEC pattern) Following educational_VMEC: r1s = r1s + r1a,
    // z1s = z1s + z1a, etc. NOTE: Only add position arrays, not derivatives
    // (asymmetric transform doesn't compute derivatives)
    std::cout << "DEBUG: Adding asymmetric contributions to symmetric arrays"
              << std::endl;

    // First, let's check the sizes to debug the bounds error
    int total_symmetric_size = r1_e.size();
    int total_asymmetric_size = m_ls_.r1e_i.size();
    std::cout << "DEBUG: Array sizes - symmetric: " << total_symmetric_size
              << ", asymmetric: " << total_asymmetric_size << std::endl;
    std::cout << "DEBUG: nsMinF1=" << r_.nsMinF1 << ", nsMaxF1=" << r_.nsMaxF1
              << ", nZnT=" << s_.nZnT << std::endl;

    for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {
      int offset = (jF - r_.nsMinF1) * s_.nZnT;
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        int idx = offset + kl;
        if (idx >= total_symmetric_size || idx >= total_asymmetric_size) {
          std::cout << "ERROR: Array index " << idx
                    << " out of bounds! jF=" << jF << ", kl=" << kl
                    << ", offset=" << offset << std::endl;
          continue;
        }

        // Only add position arrays (r1, z1) - these are what asymmetric
        // transform fills
        r1_e[idx] += m_ls_.r1e_i[idx];
        z1_e[idx] += m_ls_.z1e_i[idx];
        // Note: r1_o and z1_o are populated separately from asymmetric
        // coefficients below

        // lu_e array should be added as well since asymmetric transform fills
        // lue_i  
        lu_e[idx] += m_ls_.lue_i[idx];
      }
    }

    // Extract odd parity arrays from asymmetric transform results
    if (s_.lasym) {
      // The asymmetric transform has filled m_ls_ arrays with combined
      // symmetric+asymmetric results for the full theta range [0, 2π]
      // We need to extract the odd parity part using symmetrization
      // Following educational_VMEC's symrzl function
      
      for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {
        int surface_offset = (jF - r_.nsMinF1) * s_.nZnT;
        
        // Loop over theta points in [0, π] range
        int ntheta_half = s_.ntheta / 2;
        for (int kl = 0; kl < ntheta_half; ++kl) {
          int idx_0_to_pi = surface_offset + kl;  // theta in [0, π]
          int idx_pi_to_2pi = surface_offset + kl + ntheta_half;  // theta in [π, 2π]
          
          // Extract symmetric and asymmetric parts using symmetrization:
          // For theta in [0, π]:   combined = symmetric + asymmetric
          // For theta in [π, 2π]:  combined = symmetric - asymmetric
          // Therefore: asymmetric = (combined_0_pi - combined_pi_2pi) / 2
          
          double r_combined_0 = m_ls_.r1e_i[idx_0_to_pi];
          double r_combined_pi = m_ls_.r1e_i[idx_pi_to_2pi];
          double ru_combined_0 = m_ls_.rue_i[idx_0_to_pi];
          double ru_combined_pi = m_ls_.rue_i[idx_pi_to_2pi];
          
          double z_combined_0 = m_ls_.z1e_i[idx_0_to_pi];
          double z_combined_pi = m_ls_.z1e_i[idx_pi_to_2pi];
          double zu_combined_0 = m_ls_.zue_i[idx_0_to_pi];
          double zu_combined_pi = m_ls_.zue_i[idx_pi_to_2pi];
          
          double lu_combined_0 = m_ls_.lue_i[idx_0_to_pi];
          double lu_combined_pi = m_ls_.lue_i[idx_pi_to_2pi];
          
          // Extract symmetric (even parity) part and asymmetric (odd parity) part
          // symmetric = (combined_0_pi + combined_pi_2pi) / 2
          // asymmetric = (combined_0_pi - combined_pi_2pi) / 2
          
          double r_symmetric = (r_combined_0 + r_combined_pi) / 2.0;
          double ru_symmetric = (ru_combined_0 + ru_combined_pi) / 2.0;
          double z_symmetric = (z_combined_0 + z_combined_pi) / 2.0;
          double zu_symmetric = (zu_combined_0 + zu_combined_pi) / 2.0;
          double lu_symmetric = (lu_combined_0 + lu_combined_pi) / 2.0;
          
          double r_asymmetric = (r_combined_0 - r_combined_pi) / 2.0;
          double ru_asymmetric = (ru_combined_0 - ru_combined_pi) / 2.0;
          double z_asymmetric = (z_combined_0 - z_combined_pi) / 2.0;
          double zu_asymmetric = (zu_combined_0 - zu_combined_pi) / 2.0;
          double lu_asymmetric = (lu_combined_0 - lu_combined_pi) / 2.0;
          
          // Add the symmetric part to the even arrays (which already contain
          // contributions from the symmetric transform)
          r1_e[idx_0_to_pi] += r_symmetric;
          ru_e[idx_0_to_pi] += ru_symmetric; 
          z1_e[idx_0_to_pi] += z_symmetric;
          zu_e[idx_0_to_pi] += zu_symmetric;
          lu_e[idx_0_to_pi] += lu_symmetric;
          
          // Store asymmetric part in odd arrays for [0, π]
          // (odd arrays were reset at the beginning)
          r1_o[idx_0_to_pi] += r_asymmetric;
          ru_o[idx_0_to_pi] += ru_asymmetric;
          z1_o[idx_0_to_pi] += z_asymmetric;
          zu_o[idx_0_to_pi] += zu_asymmetric;
          lu_o[idx_0_to_pi] += lu_asymmetric;
          
          // For [π, 2π] range: even gets symmetric, odd gets -asymmetric
          r1_e[idx_pi_to_2pi] = r1_e[idx_0_to_pi];  // Copy symmetric part
          ru_e[idx_pi_to_2pi] = ru_e[idx_0_to_pi];
          z1_e[idx_pi_to_2pi] = z1_e[idx_0_to_pi]; 
          zu_e[idx_pi_to_2pi] = zu_e[idx_0_to_pi];
          lu_e[idx_pi_to_2pi] = lu_e[idx_0_to_pi];
          
          r1_o[idx_pi_to_2pi] = -r1_o[idx_0_to_pi];  // Anti-symmetric
          ru_o[idx_pi_to_2pi] = -ru_o[idx_0_to_pi];
          z1_o[idx_pi_to_2pi] = -z1_o[idx_0_to_pi];
          zu_o[idx_pi_to_2pi] = -zu_o[idx_0_to_pi];
          lu_o[idx_pi_to_2pi] = -lu_o[idx_0_to_pi];
        }
      }
    }

  }  // lasym

  // DEBUG: Check surface population in asymmetric mode
  if (s_.lasym) {
    std::cout << "DEBUG SURFACE POPULATION:\n";
    for (int jF = 0; jF < 3; ++jF) {
      int idx = jF * s_.nZnT + 6;  // kl=6 for each surface
      if (idx < static_cast<int>(r1_e.size())) {
        std::cout << "  Surface jF=" << jF << " r1_e[" << idx
                  << "]=" << r1_e[idx] << " r1_o[" << idx << "]=" << r1_o[idx]
                  << "\n";
      }
    }
    std::cout << "  Array sizes: r1_e.size()=" << r1_e.size()
              << " r1_o.size()=" << r1_o.size() << "\n";
    std::cout << "  nsMinF1=" << r_.nsMinF1 << " nsMaxF1=" << r_.nsMaxF1
              << " nZnT=" << s_.nZnT << "\n";
  }

  // DEBUG: Check geometry arrays for NaN before MHD computation
  if (s_.lasym) {
    bool found_nan_geom = false;
    for (int i = 0; i < std::min(10, static_cast<int>(r1_e.size())); ++i) {
      if (!std::isfinite(r1_e[i]) || !std::isfinite(r1_o[i])) {
        std::cout << "ERROR: Non-finite geometry array at i=" << i
                  << ", r1_e=" << r1_e[i] << ", r1_o=" << r1_o[i] << std::endl;
        found_nan_geom = true;
      }
    }
    if (!found_nan_geom) {
      std::cout << "DEBUG: All geometry arrays are finite (first 10 checked)"
                << std::endl;
    }
  }

  // related post-processing:
  // combine even-m and odd-m to ru, zu into ruFull, zuFull
  for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int idx_kl1 = (jF - r_.nsMinF1) * s_.nZnT + kl;
      int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
      ruFull[idx_kl] =
          ru_e[idx_kl1] + m_p_.sqrtSF[jF - r_.nsMinF1] * ru_o[idx_kl1];
      zuFull[idx_kl] =
          zu_e[idx_kl1] + m_p_.sqrtSF[jF - r_.nsMinF1] * zu_o[idx_kl1];
    }  // kl
  }  // jF

  if (r_.nsMaxF1 == m_fc_.ns) {
    // This thread has the boundary.

    // at theta = 0
    const int outer_index = (m_fc_.ns - 1 - r_.nsMinF1) * s_.nZnT + 0;

    // at theta = pi
    const int inner_index =
        (m_fc_.ns - 1 - r_.nsMinF1) * s_.nZnT + (s_.nThetaReduced - 1);
    RadialExtent radial_extent = {
        .r_outer = r1_e[outer_index] + r1_o[outer_index],
        .r_inner = r1_e[inner_index] + r1_o[inner_index]};
    m_h_.SetRadialExtent(radial_extent);
  }
  if (r_.nsMinF1 == 0) {
    // This thread has the magnetic axis.
    GeometricOffset geometric_offset = {.r_00 = r1_e[0], .z_00 = z1_e[0]};
    m_h_.SetGeometricOffset(geometric_offset);
  }
}

// compute inv-DFTs on unique radial grid points
void IdealMhdModel::dft_FourierToReal_3d_symm(
    const FourierGeometry& physical_x) {
  auto geometry = RealSpaceGeometry{.r1_e = r1_e,
                                    .r1_o = r1_o,
                                    .ru_e = ru_e,
                                    .ru_o = ru_o,
                                    .rv_e = rv_e,
                                    .rv_o = rv_o,
                                    .z1_e = z1_e,
                                    .z1_o = z1_o,
                                    .zu_e = zu_e,
                                    .zu_o = zu_o,
                                    .zv_e = zv_e,
                                    .zv_o = zv_o,
                                    .lu_e = lu_e,
                                    .lu_o = lu_o,
                                    .lv_e = lv_e,
                                    .lv_o = lv_o,
                                    .rCon = rCon,
                                    .zCon = zCon};

  // DEBUG: Check radial partitioning values in asymmetric mode
  if (s_.lasym) {
    std::cout << "DEBUG RADIAL PARTITIONING:\n";
    std::cout << "  nsMinF1 = " << r_.nsMinF1 << "\n";
    std::cout << "  nsMaxF1 = " << r_.nsMaxF1 << "\n";
    std::cout << "  nsMinF = " << r_.nsMinF << "\n";
    std::cout << "  Range: jF = " << r_.nsMinF1 << " to " << (r_.nsMaxF1 - 1)
              << "\n";
  }

  FourierToReal3DSymmFastPoloidal(physical_x, xmpq, r_, s_, m_p_, t_, geometry);
}

// compute inv-DFTs on unique radial grid points
void IdealMhdModel::dft_FourierToReal_2d_symm(
    const FourierGeometry& physical_x) {
  // can safely assume lthreed == false in here

  const int num_realsp = (r_.nsMaxF1 - r_.nsMinF1) * s_.nThetaEff;

  for (auto* v :
       {&r1_e, &r1_o, &ru_e, &ru_o, &z1_e, &z1_o, &zu_e, &zu_o, &lu_e, &lu_o}) {
    absl::c_fill_n(*v, num_realsp, 0);
  }

  int num_con = (r_.nsMaxFIncludingLcfs - r_.nsMinF) * s_.nThetaEff;
  absl::c_fill_n(rCon, num_con, 0);
  absl::c_fill_n(zCon, num_con, 0);

// need to wait for other threads to have filled _i and _o arrays above
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  for (int jF = r_.nsMinF1; jF < r_.nsMaxF1; ++jF) {
    double* src_rcc = &(physical_x.rmncc[(jF - r_.nsMinF1) * s_.mnsize]);
    double* src_zsc = &(physical_x.zmnsc[(jF - r_.nsMinF1) * s_.mnsize]);
    double* src_lsc = &(physical_x.lmnsc[(jF - r_.nsMinF1) * s_.mnsize]);

    for (int l = 0; l < s_.nThetaReduced; ++l) {
      std::array<double, 2> rnkcc = {0.0, 0.0};
      std::array<double, 2> rnkcc_m = {0.0, 0.0};
      std::array<double, 2> znksc = {0.0, 0.0};
      std::array<double, 2> znksc_m = {0.0, 0.0};
      std::array<double, 2> lnksc_m = {0.0, 0.0};

      // NOTE: The axis only gets contributions up to m=1.
      // This is counterintuitive on its own, since the axis is a
      // one-dimensional object, and thus has to poloidal variation of its
      // geometry. As far as we know, this has to do with the innermost
      // half-grid point for computing a better near-axis approximation of the
      // Jacobian.
      //
      // Regular case: all poloidal contributions up to m = mpol - 1.
      int num_m = s_.mpol;
      if (jF == 0) {
        // axis: num_m = 2 -> m = 0, 1
        num_m = 2;
      }

      // TODO(jons): One could go further about optimizing this,
      // but since the axisymmetric case is really not the main deal in VMEC++,
      // I left it as-is for now.

      for (int m = 0; m < num_m; ++m) {
        const int m_parity = m % 2;
        const int idx_ml = m * s_.nThetaReduced + l;
        const double cosmu = t_.cosmu[idx_ml];
        rnkcc[m_parity] += src_rcc[m] * cosmu;
      }

      for (int m = 0; m < num_m; ++m) {
        const int m_parity = m % 2;
        const int idx_ml = m * s_.nThetaReduced + l;
        const double sinmum = t_.sinmum[idx_ml];
        rnkcc_m[m_parity] += src_rcc[m] * sinmum;
      }

      for (int m = 0; m < num_m; ++m) {
        const int m_parity = m % 2;
        const int idx_ml = m * s_.nThetaReduced + l;
        const double sinmu = t_.sinmu[idx_ml];
        znksc[m_parity] += src_zsc[m] * sinmu;
      }

      for (int m = 0; m < num_m; ++m) {
        const int m_parity = m % 2;
        const int idx_ml = m * s_.nThetaReduced + l;
        const double cosmum = t_.cosmum[idx_ml];
        znksc_m[m_parity] += src_zsc[m] * cosmum;
      }

      for (int m = 0; m < num_m; ++m) {
        const int m_parity = m % 2;
        const int idx_ml = m * s_.nThetaReduced + l;
        const double cosmum = t_.cosmum[idx_ml];
        lnksc_m[m_parity] += src_lsc[m] * cosmum;
      }

      const int idx_jl = (jF - r_.nsMinF1) * s_.nThetaEff + l;
      r1_e[idx_jl] += rnkcc[kEvenParity];
      ru_e[idx_jl] += rnkcc_m[kEvenParity];
      z1_e[idx_jl] += znksc[kEvenParity];
      zu_e[idx_jl] += znksc_m[kEvenParity];
      lu_e[idx_jl] += lnksc_m[kEvenParity];
      r1_o[idx_jl] += rnkcc[kOddParity];
      ru_o[idx_jl] += rnkcc_m[kOddParity];
      z1_o[idx_jl] += znksc[kOddParity];
      zu_o[idx_jl] += znksc_m[kOddParity];
      lu_o[idx_jl] += lnksc_m[kOddParity];
    }  // l
  }  // j

  // The DFTs for rCon and zCon are done separately here,
  // since this allows to remove the condition on the radial range from the
  // innermost loops.

  for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
    double* src_rcc = &(physical_x.rmncc[(jF - r_.nsMinF1) * s_.mnsize]);
    double* src_zsc = &(physical_x.zmnsc[(jF - r_.nsMinF1) * s_.mnsize]);

    // NOTE: The axis only gets contributions up to m=1.
    // This is counterintuitive on its own, since the axis is a
    // one-dimensional object, and thus has to poloidal variation of its
    // geometry. As far as we know, this has to do with the innermost
    // half-grid point for computing a better near-axis approximation of the
    // Jacobian.
    //
    // Regular case: all poloidal contributions up to m = mpol - 1.
    int num_m = s_.mpol;
    if (jF == 0) {
      // axis: num_m = 2 -> m = 0, 1
      num_m = 2;
    }

    // TODO(jons): One could go further about optimizing this,
    // but since the axisymmetric case is really not the main deal in VMEC++,
    // I left it as-is for now.

    // In the following, we need to apply a scaling factor only for the
    // odd-parity m contributions:
    //   m_parity == kOddParity(==1) --> * m_p_.sqrtSF[jF - r_.nsMinF1]
    //   m_parity == kEvenParity(==0) --> * 1
    //
    // This expression is 0 if m_parity is 0 (=kEvenParity) and m_p_.sqrtSF[jF -
    // r_.nsMinF1] if m_parity is 1 (==kOddParity):
    //   m_parity * m_p_.sqrtSF[jF - r_.nsMinF1]
    //
    // This expression is 1 if m_parity is 0 and 0 if m_parity is 1:
    //   (1 - m_parity)
    //
    // Hence, we can replace the following conditional statement:
    //   double scale = xmpq[m];
    //   if (m_parity == kOddParity) {
    //       scale *= m_p_.sqrtSF[jF - r_.nsMinF1];
    //   }
    // with the following code:
    //   const double scale = xmpq[m] * (1 - m_parity + m_parity *
    //   m_p_.sqrtSF[jF - r_.nsMinF1]);

    for (int m = 0; m < num_m; ++m) {
      const int m_parity = m % 2;
      const double scale =
          xmpq[m] * (1 - m_parity + m_parity * m_p_.sqrtSF[jF - r_.nsMinF1]);

      for (int l = 0; l < s_.nThetaReduced; ++l) {
        const int idx_ml = m * s_.nThetaReduced + l;
        const double cosmu = t_.cosmu[idx_ml];
        const int idx_con = (jF - r_.nsMinF) * s_.nThetaEff + l;
        rCon[idx_con] += src_rcc[m] * cosmu * scale;
      }  // l

      for (int l = 0; l < s_.nThetaReduced; ++l) {
        const int idx_ml = m * s_.nThetaReduced + l;
        const double sinmu = t_.sinmu[idx_ml];
        const int idx_con = (jF - r_.nsMinF) * s_.nThetaEff + l;
        zCon[idx_con] += src_zsc[m] * sinmu * scale;
      }  // l
    }  // m
  }  // jF
}  // dft_FourierToReal_2d_symm

/** extrapolate (r,z)Con from boundary into volume.
 * Only called on initialization/soft reset to set (r,z)Con0 to a large value.
 * Since (r,z)Con0 are subtracted from (r,z)Con, this effectively disables the
 * constraint. Over the iterations, (r,z)Con0 are gradually reduced to zero,
 * enabling the constraint again.
 */
void IdealMhdModel::rzConIntoVolume() {
  // The CPU which has the LCFS needs to compute (r,z)Con at the LCFS
  // for computing (r,z)Con0 by extrapolation from the LCFS into the volume.

  // step 1: source thread puts rCon, zCon at LCFS into global array
  if (r_.nsMaxF1 == m_fc_.ns) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int lcfs_kl = (m_fc_.ns - 1 - r_.nsMinF) * s_.nZnT + kl;
      m_h_.rCon_LCFS[kl] = rCon[lcfs_kl];
      m_h_.zCon_LCFS[kl] = zCon[lcfs_kl];
    }  // kl
  }

