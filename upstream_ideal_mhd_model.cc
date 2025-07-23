// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

#include <algorithm>
#include <array>
#include <cstdio>
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
    std::vector<double>& m_gcs, std::vector<double>& m_gCon) {
  absl::c_fill_n(m_gCon, (rp.nsMaxF - rp.nsMinF) * s_.nZnT, 0);

  // no constraint on axis --> has no poloidal angle
  int jMin = 0;
  if (rp.nsMinF == 0) {
    jMin = 1;
  }

  for (int jF = std::max(jMin, rp.nsMinF); jF < rp.nsMaxF; ++jF) {
    for (int m = 1; m < s_.mpol - 1; ++m) {
      absl::c_fill_n(m_gsc, s_.ntor + 1, 0);
      absl::c_fill_n(m_gcs, s_.ntor + 1, 0);

      for (int k = 0; k < s_.nZeta; ++k) {
        double w0 = 0.0;
        double w1 = 0.0;

        // fwd transform in poloidal direction
        // integrate poloidally to get m-th poloidal Fourier coefficient
        for (int l = 0; l < s_.nThetaReduced; ++l) {
          const int idx_ml = m * s_.nThetaReduced + l;

          int idx_kl = ((jF - rp.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
          w0 += gConEff[idx_kl] * fb.sinmui[idx_ml];
          w1 += gConEff[idx_kl] * fb.cosmui[idx_ml];
        }  // l

        // forward Fourier transform in toroidal direction for full set of mode
        // numbers (n = 0, 1, ..., ntor)
        for (int n = 0; n < s_.ntor + 1; ++n) {
          int idx_kn = k * (s_.nnyq2 + 1) + n;

          // NOTE: `tcon` comes into play here
          m_gsc[n] += fb.cosnv[idx_kn] * w0 * tcon[jF - rp.nsMinF];
          m_gcs[n] += fb.sinnv[idx_kn] * w1 * tcon[jF - rp.nsMinF];
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

        // collect contribution to current grid point from n-th toroidal mode
        for (int n = 0; n < s_.ntor + 1; ++n) {
          int idx_kn = k * (s_.nnyq2 + 1) + n;
          w0 += m_gsc[n] * fb.cosnv[idx_kn];
          w1 += m_gcs[n] * fb.sinnv[idx_kn];
        }  // n

        // inv transform in poloidal direction
        for (int l = 0; l < s_.nThetaReduced; ++l) {
          int idx_kl = ((jF - rp.nsMinF) * s_.nZeta + k) * s_.nThetaEff + l;
          const int idx_ml = m * s_.nThetaReduced + l;

          // NOTE: `faccon` comes into play here
          m_gCon[idx_kl] +=
              faccon[m] * (w0 * fb.sinmu[idx_ml] + w1 * fb.cosmu[idx_ml]);
        }  // l
      }  // k
    }  // m
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

  armn_e.resize(nrzt);
  armn_o.resize(nrzt);
  brmn_e.resize(nrzt);
  brmn_o.resize(nrzt);
  azmn_e.resize(nrzt);
  azmn_o.resize(nrzt);
  bzmn_e.resize(nrzt);
  bzmn_o.resize(nrzt);
  blmn_e.resize(nrztIncludingBoundary);
  blmn_o.resize(nrztIncludingBoundary);

  if (s_.lthreed) {
    crmn_e.resize(nrzt);
    crmn_o.resize(nrzt);
    czmn_e.resize(nrzt);
    czmn_o.resize(nrzt);
    clmn_e.resize(nrztIncludingBoundary);
    clmn_o.resize(nrztIncludingBoundary);
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
  // symmetric contribution is always needed
  if (s_.lthreed) {
    dft_FourierToReal_3d_symm(physical_x);
  } else {
    dft_FourierToReal_2d_symm(physical_x);
  }

  if (s_.lasym) {
    // FIXME(jons): implement non-symmetric DFT variants
    std::cerr << "asymmetric inv-DFT not implemented yet\n";

    // FIXME(jons): implement symrzl
    std::cerr << "symrzl not implemented yet\n";

#ifdef _OPENMP
    abort();
#else
    exit(-1);
#endif  // _OPENMP
  }  // lasym

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

// wait for thread that has LCFS to have put rzCon at LCFS into array above
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // step 2: all threads interpolate into volume
  for (int jF = std::max(1, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
    double sFull = m_p_.sqrtSF[jF - r_.nsMinF1] * m_p_.sqrtSF[jF - r_.nsMinF1];
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
      rCon0[idx_kl] = m_h_.rCon_LCFS[kl] * sFull;
      zCon0[idx_kl] = m_h_.zCon_LCFS[kl] * sFull;
    }  // kl
  }  // j
}

void IdealMhdModel::computeJacobian() {
  // r12, ru12, zu12, rs, zs, tau

  double minTau = 0.0;
  double maxTau = 0.0;

  // contributions from full-grid surface _i_nside j-th half-grid surface
  int j0 = r_.nsMinF1;
  for (int kl = 0; kl < s_.nZnT; ++kl) {
    m_ls_.r1e_i[kl] = r1_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.r1o_i[kl] = r1_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.z1e_i[kl] = z1_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.z1o_i[kl] = z1_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.rue_i[kl] = ru_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.ruo_i[kl] = ru_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.zue_i[kl] = zu_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.zuo_i[kl] = zu_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
  }

  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    // sqrt(s) on j-th half-grid pos
    double sqrtSH = m_p_.sqrtSH[jH - r_.nsMinH];

    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // contributions from full-grid surface _o_utside j-th half-grid surface
      double r1e_o = r1_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double r1o_o = r1_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double z1e_o = z1_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double z1o_o = z1_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double rue_o = ru_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double ruo_o = ru_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double zue_o = zu_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double zuo_o = zu_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];

      int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;

      // R on half-grid
      r12[iHalf] = 0.5 * ((m_ls_.r1e_i[kl] + r1e_o) +
                          sqrtSH * (m_ls_.r1o_i[kl] + r1o_o));

      // dRdTheta on half-grid
      ru12[iHalf] = 0.5 * ((m_ls_.rue_i[kl] + rue_o) +
                           sqrtSH * (m_ls_.ruo_i[kl] + ruo_o));

      // dZdTheta on half-grid
      zu12[iHalf] = 0.5 * ((m_ls_.zue_i[kl] + zue_o) +
                           sqrtSH * (m_ls_.zuo_i[kl] + zuo_o));

      // \tilde{dRds} on half-grid
      rs[iHalf] =
          ((r1e_o - m_ls_.r1e_i[kl]) + sqrtSH * (r1o_o - m_ls_.r1o_i[kl])) /
          m_fc_.deltaS;

      // \tilde{dZds} on half-grid
      zs[iHalf] =
          ((z1e_o - m_ls_.z1e_i[kl]) + sqrtSH * (z1o_o - m_ls_.z1o_i[kl])) /
          m_fc_.deltaS;

      // sqrt(g)/R on half-grid: assemble as governed by product rule
      double tau1 = ru12[iHalf] * zs[iHalf] - rs[iHalf] * zu12[iHalf];
      double tau2 = ruo_o * z1o_o + m_ls_.ruo_i[kl] * m_ls_.z1o_i[kl] -
                    zuo_o * r1o_o - m_ls_.zuo_i[kl] * m_ls_.r1o_i[kl] +
                    (rue_o * z1o_o + m_ls_.rue_i[kl] * m_ls_.z1o_i[kl] -
                     zue_o * r1o_o - m_ls_.zue_i[kl] * m_ls_.r1o_i[kl]) /
                        sqrtSH;
      double tau_val = tau1 + dSHalfDsInterp * tau2;

      if (tau_val < minTau || minTau == 0.0) {
        minTau = tau_val;
      }
      if (tau_val > maxTau || maxTau == 0.0) {
        maxTau = tau_val;
      }

      tau[iHalf] = tau_val;

      // hand over to next iteration of radial loop
      // --> what was outside in this loop iteration will be inside for next
      // half-grid location
      m_ls_.r1e_i[kl] = r1e_o;
      m_ls_.r1o_i[kl] = r1o_o;
      m_ls_.z1e_i[kl] = z1e_o;
      m_ls_.z1o_i[kl] = z1o_o;
      m_ls_.rue_i[kl] = rue_o;
      m_ls_.ruo_i[kl] = ruo_o;
      m_ls_.zue_i[kl] = zue_o;
      m_ls_.zuo_i[kl] = zuo_o;
    }  // kl
  }  // j

  bool localBadJacobian = (minTau * maxTau < 0.0);

  if (localBadJacobian) {
#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
    {
      m_fc_.restart_reason = RestartReason::BAD_JACOBIAN;
    }
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

void IdealMhdModel::computeMetricElements() {
  // gsqrt
  // guu, guv, gvv

  // contributions from full-grid surface _i_nside j-th half-grid surface
  int j0 = r_.nsMinF1;
  for (int kl = 0; kl < s_.nZnT; ++kl) {
    m_ls_.r1e_i[kl] = r1_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.r1o_i[kl] = r1_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.z1e_i[kl] = z1_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.z1o_i[kl] = z1_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.rue_i[kl] = ru_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.ruo_i[kl] = ru_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.zue_i[kl] = zu_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.zuo_i[kl] = zu_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    if (s_.lthreed) {
      m_ls_.rve_i[kl] = rv_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
      m_ls_.rvo_i[kl] = rv_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
      m_ls_.zve_i[kl] = zv_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
      m_ls_.zvo_i[kl] = zv_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    }
  }

  // s on inner full-grid pos
  double sF_i =
      m_p_.sqrtSF[r_.nsMinH - r_.nsMinF1] * m_p_.sqrtSF[r_.nsMinH - r_.nsMinF1];

  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    // s on outside full-grid pos
    double sF_o =
        m_p_.sqrtSF[jH + 1 - r_.nsMinF1] * m_p_.sqrtSF[jH + 1 - r_.nsMinF1];

    // sqrt(s) on j-th half-grid pos
    double sqrtSH = m_p_.sqrtSH[jH - r_.nsMinH];

    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;

      // Re-use this loop to compute Jacobian gsqrt=tau*R
      // only tau needed to be checked for a sign change,
      // so skip the last part where gsqrt is computed
      // if a sign changed happened by computing it only here
      // (which will only be reached when tau did not change sign).
      gsqrt[iHalf] = tau[iHalf] * r12[iHalf];

      // contributions from full-grid surface _o_utside j-th half-grid surface
      double r1e_o = r1_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double r1o_o = r1_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double rue_o = ru_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double ruo_o = ru_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double zue_o = zu_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double zuo_o = zu_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];

      // g_{\theta,\theta} is needed for both 2D and 3D cases
      guu[iHalf] = 0.5 * ((m_ls_.rue_i[kl] * m_ls_.rue_i[kl] +
                           m_ls_.zue_i[kl] * m_ls_.zue_i[kl]) +
                          (rue_o * rue_o + zue_o * zue_o) +
                          sF_i * (m_ls_.ruo_i[kl] * m_ls_.ruo_i[kl] +
                                  m_ls_.zuo_i[kl] * m_ls_.zuo_i[kl]) +
                          sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
                   sqrtSH * ((m_ls_.rue_i[kl] * m_ls_.ruo_i[kl] +
                              m_ls_.zue_i[kl] * m_ls_.zuo_i[kl]) +
                             (rue_o * ruo_o + zue_o * zuo_o));

      // g_{\zeta,\zeta} reduces to R^2 in the 2D case, so compute this always
      gvv[iHalf] = 0.5 * (m_ls_.r1e_i[kl] * m_ls_.r1e_i[kl] + r1e_o * r1e_o +
                          sF_i * m_ls_.r1o_i[kl] * m_ls_.r1o_i[kl] +
                          sF_o * r1o_o * r1o_o) +
                   sqrtSH * (m_ls_.r1e_i[kl] * m_ls_.r1o_i[kl] + r1e_o * r1o_o);

      if (s_.lthreed) {
        double rve_o = rv_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
        double rvo_o = rv_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
        double zve_o = zv_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
        double zvo_o = zv_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];

        // g_{\theta,\zeta} is only needed for the 3D case
        guv[iHalf] = 0.5 * ((m_ls_.rue_i[kl] * m_ls_.rve_i[kl] +
                             m_ls_.zue_i[kl] * m_ls_.zve_i[kl]) +
                            (rue_o * rve_o + zue_o * zve_o) +
                            sF_i * (m_ls_.ruo_i[kl] * m_ls_.rvo_i[kl] +
                                    m_ls_.zuo_i[kl] * m_ls_.zvo_i[kl]) +
                            sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
                            sqrtSH * ((m_ls_.rue_i[kl] * m_ls_.rvo_i[kl] +
                                       m_ls_.zue_i[kl] * m_ls_.zvo_i[kl]) +
                                      (rue_o * rvo_o + zue_o * zvo_o) +
                                      (m_ls_.rve_i[kl] * m_ls_.ruo_i[kl] +
                                       m_ls_.zve_i[kl] * m_ls_.zuo_i[kl]) +
                                      (rve_o * ruo_o + zve_o * zuo_o)));

        // compute remaining contribution for 3D to g_{\zeta,\zeta}
        gvv[iHalf] += 0.5 * ((m_ls_.rve_i[kl] * m_ls_.rve_i[kl] +
                              m_ls_.zve_i[kl] * m_ls_.zve_i[kl]) +
                             (rve_o * rve_o + zve_o * zve_o) +
                             sF_i * (m_ls_.rvo_i[kl] * m_ls_.rvo_i[kl] +
                                     m_ls_.zvo_i[kl] * m_ls_.zvo_i[kl]) +
                             sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
                      sqrtSH * ((m_ls_.rve_i[kl] * m_ls_.rvo_i[kl] +
                                 m_ls_.zve_i[kl] * m_ls_.zvo_i[kl]) +
                                (rve_o * rvo_o + zve_o * zvo_o));

        // hand over to next iteration of radial loop
        // --> what was outside in this loop iteration will be inside for next
        // half-grid location
        m_ls_.rve_i[kl] = rve_o;
        m_ls_.rvo_i[kl] = rvo_o;
        m_ls_.zve_i[kl] = zve_o;
        m_ls_.zvo_i[kl] = zvo_o;
      }

      // hand over to next iteration of radial loop
      // --> what was outside in this loop iteration will be inside for next
      // half-grid location
      m_ls_.r1e_i[kl] = r1e_o;
      m_ls_.r1o_i[kl] = r1o_o;
      m_ls_.rue_i[kl] = rue_o;
      m_ls_.ruo_i[kl] = ruo_o;
      m_ls_.zue_i[kl] = zue_o;
      m_ls_.zuo_i[kl] = zuo_o;
    }  // kl

    // hand over to next iteration of radial loop
    // --> what was outside in this loop iteration will be inside for next
    // half-grid location
    sF_i = sF_o;
  }  // jH
}

/**
 * Compute radial profile of differential volume
 * via a surface integral:
 * dV/ds = int_u int_v |sqrt(g)| du dv
 */
void IdealMhdModel::updateDifferentialVolume() {
  // dVdsH

  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    m_p_.dVdsH[jH - r_.nsMinH] = 0.0;
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int l = kl % s_.nThetaEff;
      // multiply by surface differential
      m_p_.dVdsH[jH - r_.nsMinH] +=
          gsqrt[(jH - r_.nsMinH) * s_.nZnT + kl] * s_.wInt[l];
    }  // kl

    // cancel signgs contained in gsqrt so that dVds is always positive
    m_p_.dVdsH[jH - r_.nsMinH] *= signOfJacobian;
  }  // jH
}  // updateDifferentialVolume

// first iteration of a multi-grid step
void IdealMhdModel::computeInitialVolume() {
  double localPlasmaVolume = 0.0;
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    // radial integral to get plasma volume
    // This must be done over UNIQUE half-grid points !!!
    // --> The standard partitioning has half-grid points between
    //     neighboring ranks that are handled by both ranks.
    if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
      localPlasmaVolume += m_p_.dVdsH[jH - r_.nsMinH];
    }
  }
  localPlasmaVolume *= m_fc_.deltaS;

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  m_h_.voli = 0.0;
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  m_h_.voli += localPlasmaVolume * (2.0 * M_PI) * (2.0 * M_PI);
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // computeInitialVolume

void IdealMhdModel::updateVolume() {
  double localPlasmaVolume = 0.0;
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    // radial integral to get plasma volume
    // This must be done over UNIQUE half-grid points !!!
    // --> The standard partitioning has half-grid points between
    //     neighboring ranks that are handled by both ranks.
    if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
      localPlasmaVolume += m_p_.dVdsH[jH - r_.nsMinH];
    }
  }
  localPlasmaVolume *= m_fc_.deltaS;

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  m_h_.plasmaVolume = 0.0;
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  m_h_.plasmaVolume += localPlasmaVolume;
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // updateVolume

/**
 * Compute contravariant magnetic field components
 * and apply toroidal current constraint, if enabled.
 */
void IdealMhdModel::computeBContra() {
  // bsupu, bsupv
  // chipH (, iotaH)
  // chipF, iotaF

  int j0 = r_.nsMinH;
  for (int kl = 0; kl < s_.nZnT; ++kl) {
    // undo lambda normalization for first radial location
    lu_e[(j0 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
    lu_o[(j0 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
    if (s_.lthreed) {
      lv_e[(j0 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
      lv_o[(j0 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
    }

    // add phi' to d(lambda)/d(theta) for preparing B^v
    lu_e[(j0 - r_.nsMinF1) * s_.nZnT + kl] += m_p_.phipF[j0 - r_.nsMinF1];

    // contributions from full-grid surface _i_nside j-th half-grid surface
    // starting values: jRel=0
    m_ls_.lue_i[kl] = lu_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    m_ls_.luo_i[kl] = lu_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    if (s_.lthreed) {
      m_ls_.lve_i[kl] = lv_e[(j0 - r_.nsMinF1) * s_.nZnT + kl];
      m_ls_.lvo_i[kl] = lv_o[(j0 - r_.nsMinF1) * s_.nZnT + kl];
    }
  }  // kl

  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    // sqrt(s) on j-th half-grid pos
    double sqrtSH = m_p_.sqrtSH[jH - r_.nsMinH];

    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;

      // undo lambda normalization for next full-grid radial location
      lu_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
      lu_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
      if (s_.lthreed) {
        lv_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
        lv_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] *= constants_.lamscale;
      }

      // add phi' to d(lambda)/d(theta) for preparing B^v
      lu_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl] +=
          m_p_.phipF[jH + 1 - r_.nsMinH];

      // contributions from full-grid surface _o_utside j-th half-grid surface
      double lue_o = lu_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double luo_o = lu_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
      double lve_o = 0.0;
      double lvo_o = 0.0;
      if (s_.lthreed) {
        lve_o = lv_e[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];
        lvo_o = lv_o[(jH + 1 - r_.nsMinF1) * s_.nZnT + kl];

        // first part for B^\theta
        bsupu[iHalf] =
            0.5 *
            ((m_ls_.lve_i[kl] + lve_o) + sqrtSH * (m_ls_.lvo_i[kl] + lvo_o)) /
            gsqrt[iHalf];
      } else {
        // will get a contribution from chip'/sqrt(g) below
        bsupu[iHalf] = 0.0;
      }

      // first part for B^\zeta
      bsupv[iHalf] =
          0.5 *
          ((m_ls_.lue_i[kl] + lue_o) + sqrtSH * (m_ls_.luo_i[kl] + luo_o)) /
          gsqrt[iHalf];

      // hand over to next iteration of radial loop
      // --> what was outside in this loop iteration will be inside for next
      // half-grid location
      m_ls_.lue_i[kl] = lue_o;
      m_ls_.luo_i[kl] = luo_o;
      if (s_.lthreed) {
        m_ls_.lve_i[kl] = lve_o;
        m_ls_.lvo_i[kl] = lvo_o;
      }
    }  // kl
  }  // jH

  if (ncurr == 1) {
    // constrained toroidal current profile
    // --> compute chi' consistent with prescribed toroidal current

    for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
      double jvPlasma = 0.0;
      double avg_guu_gsqrt = 0.0;
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
        int l = kl % s_.nThetaEff;
        if (s_.lthreed) {
          jvPlasma += (guu[iHalf] * bsupu[iHalf] + guv[iHalf] * bsupv[iHalf]) *
                      s_.wInt[l];
        } else {
          jvPlasma += guu[iHalf] * bsupu[iHalf] * s_.wInt[l];
        }
        avg_guu_gsqrt += guu[iHalf] / gsqrt[iHalf] * s_.wInt[l];
      }  // kl

      // add in prescribed toroidal current profile and re-compute chi' and iota
      if (avg_guu_gsqrt != 0.0) {
        m_p_.chipH[jH - r_.nsMinH] =
            (m_p_.currH[jH - r_.nsMinH] - jvPlasma) / avg_guu_gsqrt;
      }

      if (m_p_.phipH[jH - r_.nsMinH] != 0.0) {
        m_p_.iotaH[jH - r_.nsMinH] =
            m_p_.chipH[jH - r_.nsMinH] / m_p_.phipH[jH - r_.nsMinH];
      }
    }  // jH
  } else {
    // constrained iota profile

    // evaluate chi' profile from phi' and prescribed iota profile
    for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
      m_p_.chipH[jH - r_.nsMinH] =
          m_p_.iotaH[jH - r_.nsMinH] * m_p_.phipH[jH - r_.nsMinH];
    }  // jH
  }

  // update full-grid chi'
  for (int jFi = r_.nsMinFi; jFi < r_.nsMaxFi; ++jFi) {
    m_p_.chipF[jFi - r_.nsMinF1] =
        0.5 * (m_p_.chipH[jFi - r_.nsMinH] + m_p_.chipH[jFi - 1 - r_.nsMinH]);
  }
  if (r_.nsMaxF1 == m_fc_.ns) {
    // TODO(jons): inconsistent extrapolation ??? (see below)
    m_p_.chipF[r_.nsMaxF1 - 1 - r_.nsMinF1] =
        2.0 * m_p_.chipH[r_.nsMaxH - 1 - r_.nsMinH] -
        m_p_.chipH[r_.nsMaxH - 2 - r_.nsMinH];
  }

  // update full-grid iota
  if (r_.nsMinF1 == 0) {
    m_p_.iotaF[0] = 1.5 * m_p_.iotaH[0] - 0.5 * m_p_.iotaH[1];
  }
  for (int jFi = r_.nsMinFi; jFi < r_.nsMaxFi; ++jFi) {
    m_p_.iotaF[jFi - r_.nsMinF1] =
        0.5 * (m_p_.iotaH[jFi - r_.nsMinH] + m_p_.iotaH[jFi - 1 - r_.nsMinH]);
  }
  if (r_.nsMaxF1 == m_fc_.ns) {
    // TODO(jons): inconsistent extrapolation ??? (see above)
    m_p_.iotaF[r_.nsMaxF1 - 1 - r_.nsMinF1] =
        1.5 * m_p_.iotaH[r_.nsMaxH - 1 - r_.nsMinH] -
        0.5 * m_p_.iotaH[r_.nsMaxH - 2 - r_.nsMinH];
  }

  // bsupu contains -dLambda/dZeta and now needs to get chip/sqrt(g) added,
  // as outlined in bcovar above the call to this routine.
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
      bsupu[iHalf] += m_p_.chipH[jH - r_.nsMinH] / gsqrt[iHalf];
    }  // kl
  }  // jH
}

// Compute covariant magnetic field components.
void IdealMhdModel::computeBCo() {
  // bsubu, bsubv

  if (s_.lthreed) {
    // 3D case: need all of guu, guv, gvv
    for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
        bsubu[iHalf] = guu[iHalf] * bsupu[iHalf] + guv[iHalf] * bsupv[iHalf];
        bsubv[iHalf] = guv[iHalf] * bsupu[iHalf] + gvv[iHalf] * bsupv[iHalf];
      }  // kl
    }  // jH
  } else {
    // 2D case: can ignore guv (not even allocated)
    for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
        bsubu[iHalf] = guu[iHalf] * bsupu[iHalf];
        bsubv[iHalf] = gvv[iHalf] * bsupv[iHalf];
      }  // kl
    }  // jH
  }  // lthreed
}

void IdealMhdModel::pressureAndEnergies() {
  // presH, totalPressure
  // thermalEnergy, magneticEnergy, mhdEnergy

  double localThermalEnergy = 0.0;
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    // compute pressure from mass, dV/ds and adiabatic index (gamma)
    m_p_.presH[jH - r_.nsMinH] =
        m_p_.massH[jH - r_.nsMinH] /
        pow(m_p_.dVdsH[jH - r_.nsMinH], adiabaticIndex);

    // perform volume integral over kinetic pressure for thermal energy
    // This must be done over UNIQUE half-grid points !!!
    // --> The standard partitioning has half-grid points between
    //     neighboring ranks that are handled by both ranks.
    if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
      localThermalEnergy +=
          m_p_.presH[jH - r_.nsMinH] * m_p_.dVdsH[jH - r_.nsMinH];
    }
  }  // jH

  // 1/(ns-1) is the radial integration differential
  // --> multiply it in here for thermal energy
  localThermalEnergy *= m_fc_.deltaS;

  double localMagneticEnergy = 0.0;
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;

      // magnetic pressure is |B|^2/2 = 0.5*(B^u*B_u + B^v*B_v)
      double magneticPressure =
          0.5 * (bsupu[iHalf] * bsubu[iHalf] + bsupv[iHalf] * bsubv[iHalf]);

      // perform volume integral over magnetic pressure for magnetic energy
      // This must be done over UNIQUE half-grid points !!!
      // --> The standard partitioning has half-grid points between
      //     neighboring ranks that are handled by both ranks.
      if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
        int l = kl % s_.nThetaEff;
        localMagneticEnergy += gsqrt[iHalf] * magneticPressure * s_.wInt[l];
      }

      // now can ADD KINETIC PRESSURE TO MAGNETIC PRESSURE
      // to compute the total pressure
      totalPressure[iHalf] = magneticPressure + m_p_.presH[jH - r_.nsMinH];
    }  // kl
  }  // jH

  // magneticEnergy could be negative due to negative sign of Jacobian (gsqrt)
  // --> could introduce signOfJacobian, but abs() does the job here as well
  localMagneticEnergy = fabs(localMagneticEnergy) * m_fc_.deltaS;

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    m_h_.thermalEnergy = 0.0;
    m_h_.magneticEnergy = 0.0;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    m_h_.thermalEnergy += localThermalEnergy;
    m_h_.magneticEnergy += localMagneticEnergy;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  // compute MHD energy from individual volume integrals
  m_h_.mhdEnergy =
      m_h_.magneticEnergy + m_h_.thermalEnergy / (adiabaticIndex - 1.0);
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

// COMPUTE AVERAGE FORCE BALANCE AND TOROIDAL/POLOIDAL CURRENTS
void IdealMhdModel::radialForceBalance() {
  // Compute profiles of enclosed toroidal current and enclosed poloidal current
  // on half-grid.
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    m_p_.bucoH[jH - r_.nsMinH] = 0.0;
    m_p_.bvcoH[jH - r_.nsMinH] = 0.0;
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
      int l = kl % s_.nThetaEff;
      m_p_.bucoH[jH - r_.nsMinH] += bsubu[iHalf] * s_.wInt[l];
      m_p_.bvcoH[jH - r_.nsMinH] += bsubv[iHalf] * s_.wInt[l];
    }  // kl
  }  // jH

  double signByDeltaS = signOfJacobian / m_fc_.deltaS;

  // Compute derivatives on interior full-grid knots
  // and from them, evaluate radial force balance residual.
  for (int jFi = r_.nsMinFi; jFi < r_.nsMaxFi; ++jFi) {
    // radial derivatives from half-grid to full-grid
    m_p_.jcurvF[jFi - r_.nsMinFi] =
        signByDeltaS *
        (m_p_.bucoH[jFi - r_.nsMinH] - m_p_.bucoH[jFi - 1 - r_.nsMinH]);
    m_p_.jcuruF[jFi - r_.nsMinFi] =
        -signByDeltaS *
        (m_p_.bvcoH[jFi - r_.nsMinH] - m_p_.bvcoH[jFi - 1 - r_.nsMinH]);

    // prescribed pressure gradient from user input
    m_p_.presgradF[jFi - r_.nsMinFi] =
        (m_p_.presH[jFi - r_.nsMinH] - m_p_.presH[jFi - 1 - r_.nsMinH]) /
        m_fc_.deltaS;

    // interpolate dVds onto full grid
    m_p_.dVdsF[jFi - r_.nsMinFi] =
        0.5 * (m_p_.dVdsH[jFi - r_.nsMinH] + m_p_.dVdsH[jFi - 1 - r_.nsMinH]);

    // total resulting radial force-imbalance:
    // <F> = <-j x B + grad(p)>/V'
    m_p_.equiF[jFi - r_.nsMinFi] =
        (m_p_.chipF[jFi - r_.nsMinF1] * m_p_.jcurvF[jFi - r_.nsMinFi] -
         m_p_.phipF[jFi - r_.nsMinF1] * m_p_.jcuruF[jFi - r_.nsMinFi]) /
            m_p_.dVdsF[jFi - r_.nsMinFi] +
        m_p_.presgradF[jFi - r_.nsMinFi];
  }
}

void IdealMhdModel::hybridLambdaForce() {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // obtain first inside point
  int j0 = r_.nsMinF;
  double sqrtSHi = 0.0;
  if (j0 > 0) {
    sqrtSHi = m_p_.sqrtSH[j0 - 1 - r_.nsMinH];
  }
  for (int kl = 0; kl < s_.nZnT; ++kl) {
    if (j0 == 0) {
      // defaults to 0: no contribution from half-grid point inside the axis
      m_ls_.bsubu_i[kl] = 0.0;
      m_ls_.bsubv_i[kl] = 0.0;
      m_ls_.gvv_gsqrt_i[kl] = 0.0;  // gvv / gsqrt
      m_ls_.guv_bsupu_i[kl] = 0.0;  // guv * bsupu
    } else {
      // for the j-th forces full-grid point, the (j-1)-th half-grid point is
      // inside
      int iHalf = (j0 - 1 - r_.nsMinH) * s_.nZnT + kl;
      m_ls_.bsubu_i[kl] = bsubu[iHalf];
      m_ls_.bsubv_i[kl] = bsubv[iHalf];
      m_ls_.gvv_gsqrt_i[kl] = gvv[iHalf] / gsqrt[iHalf];
      if (s_.lthreed) {
        m_ls_.guv_bsupu_i[kl] = guv[iHalf] * bsupu[iHalf];
      }
    }
  }  // kl

  for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
    double sqrtSHo = 0.0;
    if (jF < r_.nsMaxH) {
      sqrtSHo = m_p_.sqrtSH[jF - r_.nsMinH];
    }

    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // obtain next outside point
      // defaults to 0: no contribution from half-grid point outside LCFS
      double bsubv_o = 0.0;
      // gvv / gsqrt
      double gvv_gsqrt_o = 0.0;
      // guv * bsupu
      double guv_bsupu_o = 0.0;
      if (jF < r_.nsMaxH) {
        // for the j-th forces full-grid point, the j-th half-grid point is
        // outside
        int iHalf = (jF - r_.nsMinH) * s_.nZnT + kl;
        bsubv_o = bsubv[iHalf];
        gvv_gsqrt_o = gvv[iHalf] / gsqrt[iHalf];
        if (s_.lthreed) {
          guv_bsupu_o = guv[iHalf] * bsupu[iHalf];
        }
      }

      // alternative way to interpolate bsubv onto the full-grid
      double gvv_gsqrt_lu_e = 0.5 * (m_ls_.gvv_gsqrt_i[kl] + gvv_gsqrt_o) *
                              lu_e[(jF - r_.nsMinF1) * s_.nZnT + kl];
      double gvv_gsqrt_lu_o =
          0.5 * (m_ls_.gvv_gsqrt_i[kl] * sqrtSHi + gvv_gsqrt_o * sqrtSHo) *
          lu_o[(jF - r_.nsMinF1) * s_.nZnT + kl];

      double gvv_gsqrt_lu = gvv_gsqrt_lu_e + gvv_gsqrt_lu_o;
      double bsubv_alternative = gvv_gsqrt_lu;
      if (s_.lthreed) {
        double guv_bsupu = 0.5 * (m_ls_.guv_bsupu_i[kl] + guv_bsupu_o);
        bsubv_alternative += guv_bsupu;
      }

      const double bsubv_average = 0.5 * (bsubv_o + m_ls_.bsubv_i[kl]);

      // blend together two ways of interpolating bsubv
      double _blmn =
          bsubv_average * (1.0 - m_p_.radialBlending[jF - r_.nsMinF1]) +
          bsubv_alternative * m_p_.radialBlending[jF - r_.nsMinF1];

      if (jF > 0) {
        // TODO(jons): no lamscale and (-1) factor for axis lambda force?
        // MINUS SIGN => HESSIAN DIAGONALS ARE POSITIVE
        _blmn *= -constants_.lamscale;
      }

      blmn_e[(jF - r_.nsMinF) * s_.nZnT + kl] = _blmn;
      blmn_o[(jF - r_.nsMinF) * s_.nZnT + kl] =
          _blmn * m_p_.sqrtSF[jF - r_.nsMinF1];

      if (s_.lthreed) {
        // obtain next outside point
        // defaults to 0 for half-grid point outside LCFS
        double bsubu_o = 0.0;
        if (jF < r_.nsMaxH) {
          bsubu_o = bsubu[(jF - r_.nsMinH) * s_.nZnT + kl];
        }

        double _clmn = 0.5 * (bsubu_o + m_ls_.bsubu_i[kl]);

        if (jF > 0) {
          // TODO(jons): no lamscale and (-1) factor for axis lambda force?
          // MINUS SIGN => HESSIAN DIAGONALS ARE POSITIVE
          _clmn *= -constants_.lamscale;
        }

        clmn_e[(jF - r_.nsMinF) * s_.nZnT + kl] = _clmn;
        clmn_o[(jF - r_.nsMinF) * s_.nZnT + kl] =
            _clmn * m_p_.sqrtSF[jF - r_.nsMinF1];

        // shift to next point
        m_ls_.bsubu_i[kl] = bsubu_o;
      }  // lthreed

      // shift to next point
      m_ls_.bsubv_i[kl] = bsubv_o;
      m_ls_.gvv_gsqrt_i[kl] = gvv_gsqrt_o;
      if (s_.lthreed) {
        m_ls_.guv_bsupu_i[kl] = guv_bsupu_o;
      }
    }  // kl
    sqrtSHi = sqrtSHo;
  }  // jF

// }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

// Compute normalization factors for force residuals.
void IdealMhdModel::computeForceNorms(const FourierGeometry& decomposed_x) {
  // r2 in Fortran VMEC
  double energyDensity =
      std::max(m_h_.magneticEnergy, m_h_.thermalEnergy) / m_h_.plasmaVolume;

  double localForceNormSumRZ = 0.0;
  double localForceNormSumL = 0.0;
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;

      // perform volume integral over magnetic pressure for magnetic energy
      // This must be done over UNIQUE half-grid points !!!
      // --> The standard partitioning has half-grid points between
      //     neighboring ranks that are handled by both ranks.
      if (jH < r_.nsMaxH - 1 || jH == m_fc_.ns - 2) {
        int l = kl % s_.nThetaEff;
        localForceNormSumRZ +=
            guu[iHalf] * r12[iHalf] * r12[iHalf] * s_.wInt[l];
        localForceNormSumL +=
            (bsubu[iHalf] * bsubu[iHalf] + bsubv[iHalf] * bsubv[iHalf]) *
            s_.wInt[l];
      }
    }  // kl
  }  // j

  // TODO(jons): exclude axis --> mimic PARVMEC
  // only unique radial points here;
  // decomposed_x is over nsMinF1 ... nsMaxF1 --> would count overlapping
  // elements twice !!!
  const int nsMinHere = r_.nsMinF;
  double localForceNorm1 =
      decomposed_x.rzNorm(false, nsMinHere, r_.nsMaxFIncludingLcfs);

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // re-use target array elements for global accumulation
    m_h_.fNormRZ = 0.0;
    m_h_.fNormL = 0.0;
    m_h_.fNorm1 = 0.0;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    m_h_.fNormRZ += localForceNormSumRZ;
    m_h_.fNormL += localForceNormSumL;
    m_h_.fNorm1 += localForceNorm1;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    m_h_.fNormRZ = 1.0 / (m_h_.fNormRZ * energyDensity * energyDensity);
    m_h_.fNormL =
        1.0 / (m_h_.fNormL * constants_.lamscale * constants_.lamscale);
    m_h_.fNorm1 = 1.0 / m_h_.fNorm1;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

void IdealMhdModel::computeMHDForces() {
  int jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns - 1);
  if (m_fc_.lfreeb) {
    jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns);
  }

  // obtain first inside point
  // stuff gets divided by sqrtSHi, so cannot be 0
  double sqrtSHi = 1.0;
  if (r_.nsMinF > 0) {
    // for the rel-0-th forces full-grid point, the rel-0-th half-grid point is
    // inside
    int j0 = r_.nsMinH;
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int iHalf = (j0 - r_.nsMinH) * s_.nZnT + kl;
      m_ls_.P_i[kl] = r12[iHalf] * totalPressure[iHalf];
      m_ls_.rup_i[kl] = ru12[iHalf] * m_ls_.P_i[kl];
      m_ls_.zup_i[kl] = zu12[iHalf] * m_ls_.P_i[kl];
      m_ls_.rsp_i[kl] = rs[iHalf] * m_ls_.P_i[kl];
      m_ls_.zsp_i[kl] = zs[iHalf] * m_ls_.P_i[kl];
      m_ls_.taup_i[kl] = tau[iHalf] * totalPressure[iHalf];
      m_ls_.gbubu_i[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupu[iHalf];
      m_ls_.gbubv_i[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupv[iHalf];
      m_ls_.gbvbv_i[kl] = gsqrt[iHalf] * bsupv[iHalf] * bsupv[iHalf];
    }  // kl
    sqrtSHi = m_p_.sqrtSH[j0 - r_.nsMinH];
  } else {
    // defaults to 0: no contribution from half-grid point inside the axis
    absl::c_fill_n(m_ls_.P_i, s_.nZnT, 0);
    absl::c_fill_n(m_ls_.rup_i, s_.nZnT, 0);
    absl::c_fill_n(m_ls_.zup_i, s_.nZnT, 0);
    absl::c_fill_n(m_ls_.rsp_i, s_.nZnT, 0);
    absl::c_fill_n(m_ls_.zsp_i, s_.nZnT, 0);
    absl::c_fill_n(m_ls_.taup_i, s_.nZnT, 0);
    absl::c_fill_n(m_ls_.gbubu_i, s_.nZnT, 0);
    absl::c_fill_n(m_ls_.gbubv_i, s_.nZnT, 0);
    absl::c_fill_n(m_ls_.gbvbv_i, s_.nZnT, 0);
  }

  std::vector<double> P_o(s_.nZnT);      //  r12 * totalPressure = P
  std::vector<double> rup_o(s_.nZnT);    // ru12 * P
  std::vector<double> zup_o(s_.nZnT);    // zu12 * P
  std::vector<double> rsp_o(s_.nZnT);    //   rs * P
  std::vector<double> zsp_o(s_.nZnT);    //   zs * P
  std::vector<double> taup_o(s_.nZnT);   //  tau * P
  std::vector<double> gbubu_o(s_.nZnT);  // gsqrt * bsupu * bsupu
  std::vector<double> gbubv_o(s_.nZnT);  // gsqrt * bsupu * bsupv
  std::vector<double> gbvbv_o(s_.nZnT);  // gsqrt * bsupv * bsupv

  for (int jF = r_.nsMinF; jF < jMaxRZ; ++jF) {
    const double sFull =
        m_p_.sqrtSF[jF - r_.nsMinF1] * m_p_.sqrtSF[jF - r_.nsMinF1];
    // stuff gets divided by sqrtSHo, so cannot be 0
    double sqrtSHo = 1.0;
    if (jF < r_.nsMaxH) {
      sqrtSHo = m_p_.sqrtSH[jF - r_.nsMinH];
    }

    if (jF < r_.nsMaxH) {
      const int iHalf_base = (jF - r_.nsMinH) * s_.nZnT;
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // obtain next outside point
        // defaults to 0: no contribution from half-grid point outside LCFS
        int iHalf = iHalf_base + kl;
        P_o[kl] = r12[iHalf] * totalPressure[iHalf];
        rup_o[kl] = ru12[iHalf] * P_o[kl];
        zup_o[kl] = zu12[iHalf] * P_o[kl];
        rsp_o[kl] = rs[iHalf] * P_o[kl];
        zsp_o[kl] = zs[iHalf] * P_o[kl];
        taup_o[kl] = tau[iHalf] * totalPressure[iHalf];
      }  // kl

      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // obtain next outside point
        // defaults to 0: no contribution from half-grid point outside LCFS
        int iHalf = iHalf_base + kl;
        gbubu_o[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupu[iHalf];
        gbubv_o[kl] = gsqrt[iHalf] * bsupu[iHalf] * bsupv[iHalf];
        gbvbv_o[kl] = gsqrt[iHalf] * bsupv[iHalf] * bsupv[iHalf];
      }  // kl
    } else {
      absl::c_fill(P_o, 0.);
      absl::c_fill(rup_o, 0.);
      absl::c_fill(zup_o, 0.);
      absl::c_fill(rsp_o, 0.);
      absl::c_fill(zsp_o, 0.);
      absl::c_fill(taup_o, 0.);
      absl::c_fill(gbubu_o, 0.);
      absl::c_fill(gbubv_o, 0.);
      absl::c_fill(gbvbv_o, 0.);
    }

    // NOTE: the loop over kl is split in many separate loops to help compiler
    // auto-vectorization

    // A_R force
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // index in geometry arrays
      int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

      // index in force arrays
      int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;
      // A_R force
      armn_e[idx_f] =
          (zup_o[kl] - m_ls_.zup_i[kl]) / m_fc_.deltaS +
          0.5 * (taup_o[kl] + m_ls_.taup_i[kl]) -
          0.5 * (gbvbv_o[kl] + m_ls_.gbvbv_i[kl]) * r1_e[idx_g] -
          0.5 * (gbvbv_o[kl] * sqrtSHo + m_ls_.gbvbv_i[kl] * sqrtSHi) *
              r1_o[idx_g];
    }
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // index in geometry arrays
      int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

      // index in force arrays
      int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

      armn_o[idx_f] =
          (zup_o[kl] * sqrtSHo - m_ls_.zup_i[kl] * sqrtSHi) / m_fc_.deltaS -
          0.25 * (P_o[kl] / sqrtSHo + m_ls_.P_i[kl] / sqrtSHi) * zu_e[idx_g] -
          0.25 * (P_o[kl] + m_ls_.P_i[kl]) * zu_o[idx_g] +
          0.5 * (taup_o[kl] * sqrtSHo + m_ls_.taup_i[kl] * sqrtSHi) -
          0.5 * (gbvbv_o[kl] * sqrtSHo + m_ls_.gbvbv_i[kl] * sqrtSHi) *
              r1_e[idx_g] -
          0.5 * (gbvbv_o[kl] + m_ls_.gbvbv_i[kl]) * r1_o[idx_g] * sFull;
    }

    // A_Z force
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // index in force arrays
      int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

      azmn_e[idx_f] = -(rup_o[kl] - m_ls_.rup_i[kl]) / m_fc_.deltaS;
    }
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // index in geometry arrays
      int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

      // index in force arrays
      int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

      azmn_o[idx_f] =
          -(rup_o[kl] * sqrtSHo - m_ls_.rup_i[kl] * sqrtSHi) / m_fc_.deltaS +
          0.25 * (P_o[kl] / sqrtSHo + m_ls_.P_i[kl] / sqrtSHi) * ru_e[idx_g] +
          0.25 * (P_o[kl] + m_ls_.P_i[kl]) * ru_o[idx_g];
    }

    // B_R force
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // index in geometry arrays
      int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

      // index in force arrays
      int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

      brmn_e[idx_f] =
          0.5 * (zsp_o[kl] + m_ls_.zsp_i[kl]) +
          0.25 * (P_o[kl] / sqrtSHo + m_ls_.P_i[kl] / sqrtSHi) * z1_o[idx_g] -
          0.5 * (gbubu_o[kl] + m_ls_.gbubu_i[kl]) * ru_e[idx_g] -
          0.5 * (gbubu_o[kl] * sqrtSHo + m_ls_.gbubu_i[kl] * sqrtSHi) *
              ru_o[idx_g];
    }
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // index in geometry arrays
      int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

      // index in force arrays
      int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

      brmn_o[idx_f] =
          0.5 * (zsp_o[kl] * sqrtSHo + m_ls_.zsp_i[kl] * sqrtSHi) +
          0.25 * (P_o[kl] + m_ls_.P_i[kl]) * z1_o[idx_g] -
          0.5 * (gbubu_o[kl] * sqrtSHo + m_ls_.gbubu_i[kl] * sqrtSHi) *
              ru_e[idx_g] -
          0.5 * (gbubu_o[kl] + m_ls_.gbubu_i[kl]) * ru_o[idx_g] * sFull;
    }

    // B_Z force
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // index in geometry arrays
      int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

      // index in force arrays
      int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

      bzmn_e[idx_f] =
          -0.5 * (rsp_o[kl] + m_ls_.rsp_i[kl]) -
          0.25 * (P_o[kl] / sqrtSHo + m_ls_.P_i[kl] / sqrtSHi) * r1_o[idx_g] -
          0.5 * (gbubu_o[kl] + m_ls_.gbubu_i[kl]) * zu_e[idx_g] -
          0.5 * (gbubu_o[kl] * sqrtSHo + m_ls_.gbubu_i[kl] * sqrtSHi) *
              zu_o[idx_g];
    }
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      // index in geometry arrays
      int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

      // index in force arrays
      int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

      bzmn_o[idx_f] =
          -0.5 * (rsp_o[kl] * sqrtSHo + m_ls_.rsp_i[kl] * sqrtSHi) -
          0.25 * (P_o[kl] + m_ls_.P_i[kl]) * r1_o[idx_g] -
          0.5 * (gbubu_o[kl] * sqrtSHo + m_ls_.gbubu_i[kl] * sqrtSHi) *
              zu_e[idx_g] -
          0.5 * (gbubu_o[kl] + m_ls_.gbubu_i[kl]) * zu_o[idx_g] * sFull;
    }

    if (s_.lthreed) {
      // B_R force
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // index in geometry arrays
        int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

        // index in force arrays
        int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

        brmn_e[idx_f] +=
            -0.5 * (gbubv_o[kl] + m_ls_.gbubv_i[kl]) * rv_e[idx_g] -
            0.5 * (gbubv_o[kl] * sqrtSHo + m_ls_.gbubv_i[kl] * sqrtSHi) *
                rv_o[idx_g];
      }
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // index in geometry arrays
        int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

        // index in force arrays
        int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

        brmn_o[idx_f] +=
            -0.5 * (gbubv_o[kl] * sqrtSHo + m_ls_.gbubv_i[kl] * sqrtSHi) *
                rv_e[idx_g] -
            0.5 * (gbubv_o[kl] + m_ls_.gbubv_i[kl]) * rv_o[idx_g] * sFull;
      }

      // B_Z force
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // index in geometry arrays
        int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

        // index in force arrays
        int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

        bzmn_e[idx_f] +=
            -0.5 * (gbubv_o[kl] + m_ls_.gbubv_i[kl]) * zv_e[idx_g] -
            0.5 * (gbubv_o[kl] * sqrtSHo + m_ls_.gbubv_i[kl] * sqrtSHi) *
                zv_o[idx_g];
      }
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // index in geometry arrays
        int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

        // index in force arrays
        int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

        bzmn_o[idx_f] +=
            -0.5 * (gbubv_o[kl] * sqrtSHo + m_ls_.gbubv_i[kl] * sqrtSHi) *
                zv_e[idx_g] -
            0.5 * (gbubv_o[kl] + m_ls_.gbubv_i[kl]) * zv_o[idx_g] * sFull;
      }

      // C_R force
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // index in geometry arrays
        int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

        // index in force arrays
        int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

        crmn_e[idx_f] =
            0.5 * (gbubv_o[kl] + m_ls_.gbubv_i[kl]) * ru_e[idx_g] +
            0.5 * (gbubv_o[kl] * sqrtSHo + m_ls_.gbubv_i[kl] * sqrtSHi) *
                ru_o[idx_g] +
            0.5 * (gbvbv_o[kl] + m_ls_.gbvbv_i[kl]) * rv_e[idx_g] +
            0.5 * (gbvbv_o[kl] * sqrtSHo + m_ls_.gbvbv_i[kl] * sqrtSHi) *
                rv_o[idx_g];
      }
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // index in geometry arrays
        int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

        // index in force arrays
        int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

        crmn_o[idx_f] =
            0.5 * (gbubv_o[kl] * sqrtSHo + m_ls_.gbubv_i[kl] * sqrtSHi) *
                ru_e[idx_g] +
            0.5 * (gbubv_o[kl] + m_ls_.gbubv_i[kl]) * ru_o[idx_g] * sFull +
            0.5 * (gbvbv_o[kl] * sqrtSHo + m_ls_.gbvbv_i[kl] * sqrtSHi) *
                rv_e[idx_g] +
            0.5 * (gbvbv_o[kl] + m_ls_.gbvbv_i[kl]) * rv_o[idx_g] * sFull;
      }

      // C_Z force
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // index in geometry arrays
        int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

        // index in force arrays
        int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

        czmn_e[idx_f] =
            0.5 * (gbubv_o[kl] + m_ls_.gbubv_i[kl]) * zu_e[idx_g] +
            0.5 * (gbubv_o[kl] * sqrtSHo + m_ls_.gbubv_i[kl] * sqrtSHi) *
                zu_o[idx_g] +
            0.5 * (gbvbv_o[kl] + m_ls_.gbvbv_i[kl]) * zv_e[idx_g] +
            0.5 * (gbvbv_o[kl] * sqrtSHo + m_ls_.gbvbv_i[kl] * sqrtSHi) *
                zv_o[idx_g];
      }
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        // index in geometry arrays
        int idx_g = (jF - r_.nsMinF1) * s_.nZnT + kl;

        // index in force arrays
        int idx_f = (jF - r_.nsMinF) * s_.nZnT + kl;

        czmn_o[idx_f] =
            0.5 * (gbubv_o[kl] * sqrtSHo + m_ls_.gbubv_i[kl] * sqrtSHi) *
                zu_e[idx_g] +
            0.5 * (gbubv_o[kl] + m_ls_.gbubv_i[kl]) * zu_o[idx_g] * sFull +
            0.5 * (gbvbv_o[kl] * sqrtSHo + m_ls_.gbvbv_i[kl] * sqrtSHi) *
                zv_e[idx_g] +
            0.5 * (gbvbv_o[kl] + m_ls_.gbvbv_i[kl]) * zv_o[idx_g] * sFull;
      }
    }  // lthreed

    // shift to next point
    m_ls_.P_i = P_o;
    m_ls_.rup_i = rup_o;
    m_ls_.zup_i = zup_o;
    m_ls_.rsp_i = rsp_o;
    m_ls_.zsp_i = zsp_o;
    m_ls_.taup_i = taup_o;
    m_ls_.gbubu_i = gbubu_o;
    m_ls_.gbubv_i = gbubv_o;
    m_ls_.gbvbv_i = gbvbv_o;

    sqrtSHi = sqrtSHo;
  }  // jF
}

bool IdealMhdModel::shouldUpdateRadialPreconditioner(int iter1,
                                                     int iter2) const {
  return ((iter2 - iter1) % m_fc_.kPreconditionerUpdateInterval == 0);
}

void IdealMhdModel::updateRadialPreconditioner() {
  updateLambdaPreconditioner();

  // compute preconditioning matrix for R
  // TODO(jons): also cos01, rzu_fac for lforbal
  computePreconditioningMatrix(zs, zu12, zu_e, zu_o, z1_o, arm, ard, brm, brd,
                               cxd);

  // compute preconditioning matrix for Z
  // TODO(jons): also sin01, rru_fac for lforbal
  computePreconditioningMatrix(rs, ru12, ru_e, ru_o, r1_o, azm, azd, bzm, bzd,
                               cxd);

  // (compute stuff for lforbal: scaleEqFactor --> later)
}

void IdealMhdModel::updateLambdaPreconditioner() {
  // bLambda, dLambda, cLambda
  // lambdaPreconditioner

  // TODO(jons): what is this ?
  const double pFactor =
      dampingFactor / (4.0 * constants_.lamscale * constants_.lamscale);

  // evaluate preconditioning matrix elements on half-grid
  // on every accessible half-grid point
  // indices are shifted up by 1 to make room at 0 for first target point
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    bLambda[jH + 1 - r_.nsMinH] = 0.0;
    dLambda[jH + 1 - r_.nsMinH] = 0.0;
    cLambda[jH + 1 - r_.nsMinH] = 0.0;
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int idx_kl = (jH - r_.nsMinH) * s_.nZnT + kl;
      int l = kl % s_.nThetaEff;
      bLambda[jH + 1 - r_.nsMinH] += guu[idx_kl] / gsqrt[idx_kl] * s_.wInt[l];
      cLambda[jH + 1 - r_.nsMinH] += gvv[idx_kl] / gsqrt[idx_kl] * s_.wInt[l];
    }  // kl

    if (s_.lthreed) {
      for (int kl = 0; kl < s_.nZnT; ++kl) {
        int idx_kl = (jH - r_.nsMinH) * s_.nZnT + kl;
        int l = kl % s_.nThetaEff;
        dLambda[jH + 1 - r_.nsMinH] += guv[idx_kl] / gsqrt[idx_kl] * s_.wInt[l];
      }  // kl
    }
  }  // jH

  // constant extrapolation towards axis
  // from j=0.5 to j=0
  if (r_.nsMinF == 0) {
    bLambda[0] = bLambda[1];
    dLambda[0] = dLambda[1];
    cLambda[0] = cLambda[1];
  }

  // average onto full grid points
  int jMin = 0;
  if (r_.nsMinF == 0) {
    // skip axis, since constant extrapolation done already above
    jMin = 1;
  }

  for (int jF = std::max(jMin, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
    bLambda[jF - r_.nsMinF] =
        0.5 * (bLambda[jF + 1 - r_.nsMinH] + bLambda[jF - r_.nsMinH]);
    dLambda[jF - r_.nsMinF] =
        0.5 * (dLambda[jF + 1 - r_.nsMinH] + dLambda[jF - r_.nsMinH]);
    cLambda[jF - r_.nsMinF] =
        0.5 * (cLambda[jF + 1 - r_.nsMinH] + cLambda[jF - r_.nsMinH]);
  }

  // assemble lambda preconditioning matrix
  // TODO(jons): maybe not needed, since direct assignments below?
  absl::c_fill_n(lambdaPreconditioner,
                 (r_.nsMaxFIncludingLcfs - r_.nsMinF) * (s_.ntor + 1) * s_.mpol,
                 0);

  for (int jF = std::max(jMin, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
    for (int n = 0; n < s_.ntor + 1; ++n) {
      double tnn = n * s_.nfp * n * s_.nfp;

      for (int m = 0; m < s_.mpol; ++m) {
        if (m == 0 && n == 0) {
          continue;
        }

        int idx_mn = ((jF - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;

        int tmm = m * m;

        // TODO(jons): what is this ? (see below)
        double pwr = std::min(tmm / (16.0 * 16.0), 8.0);
        double tmn = 2.0 * m * n * s_.nfp;

        double faclam =
            tnn * bLambda[jF - r_.nsMinF] +
            tmn * copysign(dLambda[jF - r_.nsMinF], bLambda[jF - r_.nsMinF]) +
            tmm * cLambda[jF - r_.nsMinF];

        // avoid zero eigenvalue (TODO(jons): what is this ?)
        if (faclam == 0.0) {
          faclam = -1.0e-10;
        }

        // Damps m > 16 modes (TODO(jons): why ?)
        // NOTE: This also computes the inverse of each entry in
        // lambdaPreconditioner !
        lambdaPreconditioner[idx_mn] =
            pFactor / faclam * pow(m_p_.sqrtSF[jF - r_.nsMinF1], pwr);
      }  // m
    }  // n
  }  // jF
}

/**
 * Compute the radial preconditioner matrix elements.
 * This is a universal methods used to compute stuff both for R and for Z.
 * Inputs are xs, xu12, xu, x1;
 * Outputs are axm, axd, bxm, bxd and cxd.
 * The off-diagonal terms (..m) are on the half-grid.
 * The diagonal terms (..d) are on the forces full-grid.
 */
void IdealMhdModel::computePreconditioningMatrix(
    const std::vector<double>& xs, const std::vector<double>& xu12,
    const std::vector<double>& xu_e, const std::vector<double>& xu_o,
    const std::vector<double>& x1_o, std::vector<double>& m_axm,
    std::vector<double>& m_axd, std::vector<double>& m_bxm,
    std::vector<double>& m_bxd, std::vector<double>& m_cxd) {
  // zs, zu12, zu, z1 --> arm, ard, brm, brd, cxd
  // rs, ru12, ru, r1 --> azm, azd, bzm, bzd, cxd

  // restored in v8.51
  // TODO(jons): what is this?
  double pFactor = -4.0;

  // zero intermediate work arrays
  absl::c_fill_n(ax, (r_.nsMaxH - r_.nsMinH) * 4, 0);
  absl::c_fill_n(bx, (r_.nsMaxH - r_.nsMinH) * 3, 0);
  absl::c_fill_n(cx, (r_.nsMaxH - r_.nsMinH), 0);

  // all of ax, bx and cxd are on the half-grid here
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int iHalf = (jH - r_.nsMinH) * s_.nZnT + kl;
      int iFull_0 = (jH - r_.nsMinF1) * s_.nZnT + kl;
      int iFull_1 = (jH + 1 - r_.nsMinF1) * s_.nZnT + kl;

      int l = kl % s_.nThetaEff;

      double pTau =
          pFactor * r12[iHalf] * totalPressure[iHalf] / tau[iHalf] * s_.wInt[l];

      // COMPUTE DOMINANT (1/DELTA-S)**2 PRECONDITIONING MATRIX ELEMENTS

      double t1a = xu12[iHalf] / m_fc_.deltaS;
      double t2a =
          0.25 * (xu_e[iFull_1] / m_p_.sqrtSH[jH - r_.nsMinH] + xu_o[iFull_1]) /
          m_p_.sqrtSH[jH - r_.nsMinH];
      double t3a =
          0.25 * (xu_e[iFull_0] / m_p_.sqrtSH[jH - r_.nsMinH] + xu_o[iFull_0]) /
          m_p_.sqrtSH[jH - r_.nsMinH];

      // only even-m
      // both off-diagonal and diagonal
      ax[(jH - r_.nsMinH) * 4 + 0] += pTau * t1a * t1a;

      // only odd-m

      // off-diagonal: mixed
      ax[(jH - r_.nsMinH) * 4 + 1] += pTau * (t1a + t2a) * (-t1a + t3a);

      // diagonal: squared
      ax[(jH - r_.nsMinH) * 4 + 2] += pTau * (t1a + t2a) * (t1a + t2a);

      // diagonal: squared
      ax[(jH - r_.nsMinH) * 4 + 3] += pTau * (-t1a + t3a) * (-t1a + t3a);

      // COMPUTE PRECONDITIONING MATRIX ELEMENTS FOR M**2, N**2 TERMS

      // assemble full radial derivative; incl radial interpolation of odd-m X
      // contrib (?)
      double t1b =
          0.5 * (xs[iHalf] + 0.5 / m_p_.sqrtSH[jH - r_.nsMinH] * x1_o[iFull_1]);
      double t2b =
          0.5 * (xs[iHalf] + 0.5 / m_p_.sqrtSH[jH - r_.nsMinH] * x1_o[iFull_0]);

      // even-m and odd-m
      // off-diagonal: mixed
      bx[(jH - r_.nsMinH) * 3 + 0] += pTau * t1b * t2b;

      // diagonal: squared
      bx[(jH - r_.nsMinH) * 3 + 1] += pTau * t1b * t1b;

      // diagonal: squared
      bx[(jH - r_.nsMinH) * 3 + 2] += pTau * t2b * t2b;

      // even-m and odd-m
      // both off-diagonal and diagonal
      // 0.25 cancels 4 in pFactor; r0scale == 1
      // --> essentially, 0.25 * pFactor simply introduces a (-1) here!
      cx[jH - r_.nsMinH] += 0.25 * pFactor * bsupv[iHalf] * bsupv[iHalf] *
                            gsqrt[iHalf] * s_.wInt[l];
    }  // kl
  }  // jH

  const std::vector<double>& sm = m_p_.sm;
  const std::vector<double>& sp = m_p_.sp;

  // radial assembly of preconditioning matrix element components
  // All this sm, sp logic seems to be related to the odd-m scaling factors...
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    // off-diagonal, d^2/ds^2 (radial), on half-grid
    m_axm[(jH - r_.nsMinH) * 2 + kEvenParity] = -ax[(jH - r_.nsMinH) * 4 + 0];
    m_axm[(jH - r_.nsMinH) * 2 + kOddParity] =
        ax[(jH - r_.nsMinH) * 4 + 1] * sm[jH - r_.nsMinH] * sp[jH - r_.nsMinH];

    // off-diagonal, m^2 (poloidal), on half-grid
    m_bxm[(jH - r_.nsMinH) * 2 + kEvenParity] = bx[(jH - r_.nsMinH) * 3 + 0];
    m_bxm[(jH - r_.nsMinH) * 2 + kOddParity] =
        bx[(jH - r_.nsMinH) * 3 + 0] * sm[jH - r_.nsMinH] * sp[jH - r_.nsMinH];
  }

  for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
    int jH_i = jF - 1 - r_.nsMinH;
    int jH_o = jF - r_.nsMinH;

    // diagonal, d^2/ds^2 (radial), on forces full-grid
    m_axd[(jF - r_.nsMinF) * 2 + kEvenParity] =
        (jF > 0 ? ax[jH_i * 4 + 0] : 0.0) +
        (jF < m_fc_.ns - 1 ? ax[jH_o * 4 + 0] : 0.0);
    m_axd[(jF - r_.nsMinF) * 2 + kOddParity] =
        (jF > 0 ? ax[jH_i * 4 + 2] * sm[jH_i] * sm[jH_i] : 0.0) +
        (jF < m_fc_.ns - 1 ? ax[jH_o * 4 + 3] * sp[jH_o] * sp[jH_o] : 0.0);

    // diagonal, m^2 (poloidal), on forces full-grid
    m_bxd[(jF - r_.nsMinF) * 2 + kEvenParity] =
        (jF > 0 ? bx[jH_i * 3 + 1] : 0.0) +
        (jF < m_fc_.ns - 1 ? bx[jH_o * 3 + 2] : 0.0);
    m_bxd[(jF - r_.nsMinF) * 2 + kOddParity] =
        (jF > 0 ? bx[jH_i * 3 + 1] * sm[jH_i] * sm[jH_i] : 0.0) +
        (jF < m_fc_.ns - 1 ? bx[jH_o * 3 + 2] * sp[jH_o] * sp[jH_o] : 0.0);

    // -------------------------

    // diagonal, n^2 (toroidal), on forces full-grid
    m_cxd[jF - r_.nsMinF] =
        (jF > 0 ? cx[jH_i] : 0.0) + (jF < m_fc_.ns - 1 ? cx[jH_o] : 0.0);
  }
}

/**
 * Compute constraint force multiplier profile.
 * Note that this needs to have the radial preconditioner updated.
 */
absl::Status IdealMhdModel::constraintForceMultiplier() {
  // tcon

  // TODO(jons): some parabola in ns,
  // but why these specific values of the parameters ?
  double tcon_multiplier =
      tcon0 * (1.0 + m_fc_.ns * (1.0 / 60.0 + m_fc_.ns / (200.0 * 120.0)));

  // Scaling of ard, azd (2*r0scale**2);
  // Scaling of cos**2 in alias (4*r0scale**2)
  // TODO(jons): what is this?
  tcon_multiplier /= (4.0 * 4.0);

  // compute constraint force multiplier profile on forces full-grid except axis
  int jMin = 0;
  if (r_.nsMinF == 0) {
    jMin = 1;
  }

  for (int jF = std::max(jMin, r_.nsMinF); jF < r_.nsMaxF; ++jF) {
    double arNorm = 0.0;
    double azNorm = 0.0;
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
      int l = kl % s_.nThetaEff;
      arNorm += ruFull[idx_kl] * ruFull[idx_kl] * s_.wInt[l];
      azNorm += zuFull[idx_kl] * zuFull[idx_kl] * s_.wInt[l];
    }

    if (arNorm == 0.0) {
      return absl::InternalError("arNorm should never be 0.0.");
    }
    if (azNorm == 0.0) {
      return absl::InternalError("azNorm should never be 0.0.");
    }

    double tcon_base =
        std::min(fabs(ard[(jF - r_.nsMinF) * 2 + kEvenParity] / arNorm),
                 fabs(azd[(jF - r_.nsMinF) * 2 + kEvenParity] / azNorm));

    // TODO(jons): why the last term ?
    // --> could be to cancel some terms in ard, azd
    // 32 == 4*4 * 2
    tcon[jF - r_.nsMinF] =
        tcon_base * tcon_multiplier * 32 * m_fc_.deltaS * 32 * m_fc_.deltaS;
  }  // j

  // nsMaxF1 will always include bdy, even in fixed-bdy mode
  if (r_.nsMaxF1 == m_fc_.ns) {
    // TODO(jons): what is this?
    // maybe related to boundary only having MHD force contributions from the
    // inside and not from both sides?
    tcon[r_.nsMaxF1 - 1 - r_.nsMinF] = 0.5 * tcon[r_.nsMaxF1 - 2 - r_.nsMinF];
  }

  return absl::OkStatus();
}

void IdealMhdModel::effectiveConstraintForce() {
  // gConEff

  // no constraint on axis --> has no poloidal angle
  int jMin = 0;
  if (r_.nsMinF == 0) {
    jMin = 1;
  }

  for (int jF = std::max(jMin, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
      gConEff[idx_kl] = (rCon[idx_kl] - rCon0[idx_kl]) * ruFull[idx_kl] +
                        (zCon[idx_kl] - zCon0[idx_kl]) * zuFull[idx_kl];
    }  // kl
  }  // jF
}

// perform Fourier-space bandpass filtering of constraint force
// and apply scaling (tcon[j]) and preconditioning (faccon[m])
void IdealMhdModel::deAliasConstraintForce() {
  vmecpp::deAliasConstraintForce(r_, t_, s_, faccon, tcon, gConEff, gsc, gcs,
                                 gCon);
}

// add constraint force to MHD force
void IdealMhdModel::assembleTotalForces() {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // free-boundary contribution: include force on boundary from NESTOR
  if (m_fc_.lfreeb &&
      (m_vacuum_pressure_state_ == VacuumPressureState::kInitialized ||
       m_vacuum_pressure_state_ == VacuumPressureState::kActive) &&
      r_.nsMaxF1 == m_fc_.ns) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int idx_kl = (r_.nsMaxF - 1 - r_.nsMinF) * s_.nZnT + kl;

      armn_e[idx_kl] += zuFull[idx_kl] * rBSq[kl];
      armn_o[idx_kl] += zuFull[idx_kl] * rBSq[kl];
      azmn_e[idx_kl] -= ruFull[idx_kl] * rBSq[kl];
      azmn_o[idx_kl] -= ruFull[idx_kl] * rBSq[kl];
    }
  }

  for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;

      double brcon = (rCon[idx_kl] - rCon0[idx_kl]) * gCon[idx_kl];
      double bzcon = (zCon[idx_kl] - zCon0[idx_kl]) * gCon[idx_kl];

      brmn_e[idx_kl] += brcon;
      bzmn_e[idx_kl] += bzcon;
      brmn_o[idx_kl] += brcon * m_p_.sqrtSF[jF - r_.nsMinF1];
      bzmn_o[idx_kl] += bzcon * m_p_.sqrtSF[jF - r_.nsMinF1];

      frcon_e[idx_kl] = ruFull[idx_kl] * gCon[idx_kl];
      fzcon_e[idx_kl] = zuFull[idx_kl] * gCon[idx_kl];
      frcon_o[idx_kl] = frcon_e[idx_kl] * m_p_.sqrtSF[jF - r_.nsMinF1];
      fzcon_o[idx_kl] = fzcon_e[idx_kl] * m_p_.sqrtSF[jF - r_.nsMinF1];
    }
  }
}

void IdealMhdModel::forcesToFourier(FourierForces& m_physical_f) {
  // symmetric contribution is always needed
  if (s_.lthreed) {
    dft_ForcesToFourier_3d_symm(m_physical_f);
  } else {
    dft_ForcesToFourier_2d_symm(m_physical_f);
  }

  if (s_.lasym) {
    // FIXME(jons): implement non-symmetric DFT variants
    std::cerr << "asymmetric fwd-DFT not implemented yet\n";

    // FIXME(jons): implement symforce
    std::cerr << "symforce not implemented yet\n";

#ifdef _OPENMP
    abort();
#else
    exit(-1);
#endif  // _OPENMP
  }  // lasym
}

void IdealMhdModel::dft_ForcesToFourier_3d_symm(FourierForces& m_physical_f) {
  const auto input_data = RealSpaceForces{
      .armn_e = armn_e,
      .armn_o = armn_o,
      .azmn_e = azmn_e,
      .azmn_o = azmn_o,
      .blmn_e = blmn_e,
      .blmn_o = blmn_o,
      .brmn_e = brmn_e,
      .brmn_o = brmn_o,
      .bzmn_e = bzmn_e,
      .bzmn_o = bzmn_o,
      .clmn_e = clmn_e,
      .clmn_o = clmn_o,
      .crmn_e = crmn_e,
      .crmn_o = crmn_o,
      .czmn_e = czmn_e,
      .czmn_o = czmn_o,
      .frcon_e = frcon_e,
      .frcon_o = frcon_o,
      .fzcon_e = fzcon_e,
      .fzcon_o = fzcon_o,
  };

  ForcesToFourier3DSymmFastPoloidal(input_data, xmpq, r_, m_fc_, s_, t_,
                                    m_vacuum_pressure_state_, m_physical_f);
}

void IdealMhdModel::dft_ForcesToFourier_2d_symm(FourierForces& m_physical_f) {
  // in here, we can safely assume lthreed == false

  // fill target force arrays with zeros
  m_physical_f.setZero();

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  int jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns - 1);
  if (m_fc_.lfreeb &&
      (m_vacuum_pressure_state_ == VacuumPressureState::kInitialized ||
       m_vacuum_pressure_state_ == VacuumPressureState::kActive)) {
    // free-boundary: up to jMaxRZ=ns
    jMaxRZ = std::min(r_.nsMaxF, m_fc_.ns);
  }

  for (int jF = r_.nsMinF; jF < jMaxRZ; ++jF) {
    // maximum m depends on current surface:
    // --> axis R,Z gets only m=0 contributions
    // --> all other surfaces get the full Fourier spectrum
    int num_m = s_.mpol;
    if (jF == 0) {
      // axis only gets m = 0
      num_m = 1;
    }

    for (int m = 0; m < num_m; ++m) {
      const bool m_even = m % 2 == 0;
      const int idx_jm = (jF - r_.nsMinF) * s_.mpol + m;

      const auto& armn = m_even ? armn_e : armn_o;
      const auto& brmn = m_even ? brmn_e : brmn_o;
      const auto& azmn = m_even ? azmn_e : azmn_o;
      const auto& bzmn = m_even ? bzmn_e : bzmn_o;
      const auto& frcon = m_even ? frcon_e : frcon_o;
      const auto& fzcon = m_even ? fzcon_e : fzcon_o;

      for (int l = 0; l < s_.nThetaReduced; ++l) {
        const int idx_jl = (jF - r_.nsMinF) * s_.nThetaEff + l;

        const double rnkcc = armn[idx_jl];
        const double rnkcc_m = brmn[idx_jl];
        const double znksc = azmn[idx_jl];
        const double znksc_m = bzmn[idx_jl];

        const double rcon_cc = frcon[idx_jl];
        const double zcon_sc = fzcon[idx_jl];

        const int idx_ml = m * s_.nThetaReduced + l;
        const double cosmui = t_.cosmui[idx_ml];
        const double sinmumi = t_.sinmumi[idx_ml];
        const double sinmui = t_.sinmui[idx_ml];
        const double cosmumi = t_.cosmumi[idx_ml];
        // assemble effective R and Z forces from MHD and spectral condensation
        // contributions
        const double _rcc = rnkcc + xmpq[m] * rcon_cc;
        m_physical_f.frcc[idx_jm] += _rcc * cosmui + rnkcc_m * sinmumi;

        const double _zsc = znksc + xmpq[m] * zcon_sc;
        m_physical_f.fzsc[idx_jm] += _zsc * sinmui + znksc_m * cosmumi;
      }  // m
    }  // l
  }  // jF

  // Do the lambda force coefficients separately, as they have different radial
  // ranges.

  // --> axis lambda stays zero (no contribution from any m)
  for (int jF = std::max(1, r_.nsMinF); jF < r_.nsMaxFIncludingLcfs; ++jF) {
    for (int m = 0; m < s_.mpol; ++m) {
      const int m_even = m % 2 == 0;
      const auto& blmn = m_even ? blmn_e : blmn_o;
      const int idx_jm = (jF - r_.nsMinF) * s_.mpol + m;

      for (int l = 0; l < s_.nThetaReduced; ++l) {
        const int idx_jl = (jF - r_.nsMinF) * s_.nThetaEff + l;
        const double lnksc_m = blmn[idx_jl];

        const double cosmumi = t_.cosmumi[m * s_.nThetaReduced + l];
        m_physical_f.flsc[idx_jm] += lnksc_m * cosmumi;
      }  // m
    }  // l
  }  // jF
}  // dft_ForcesToFourier_2d_symm

// ---------------------------

// apply R,Z radial preconditioner for m=1-constraint
void IdealMhdModel::applyM1Preconditioner(FourierForces& m_decomposed_f) {
  if (!s_.lthreed && !s_.lasym) {
    // quick return, if there is nothing to do for us here
    return;
  }

  for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
    for (int n = 0; n < s_.ntor + 1; ++n) {
      int m = 1;
      int mPar = m % 2;

      double denom =
          ard[(jF - r_.nsMinF) * 2 + mPar] + brd[(jF - r_.nsMinF) * 2 + mPar] +
          azd[(jF - r_.nsMinF) * 2 + mPar] + bzd[(jF - r_.nsMinF) * 2 + mPar];
      double forceScaleR = (ard[(jF - r_.nsMinF) * 2 + mPar] +
                            brd[(jF - r_.nsMinF) * 2 + mPar]) /
                           denom;
      double forceScaleZ = (azd[(jF - r_.nsMinF) * 2 + mPar] +
                            bzd[(jF - r_.nsMinF) * 2 + mPar]) /
                           denom;

      int idx_mn = ((jF - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;

      if (s_.lthreed) {
        m_decomposed_f.frss[idx_mn] *= forceScaleR;
        m_decomposed_f.fzcs[idx_mn] *= forceScaleZ;
      }
      if (s_.lasym) {
        m_decomposed_f.frsc[idx_mn] *= forceScaleR;
        m_decomposed_f.fzcc[idx_mn] *= forceScaleZ;
      }
    }  // n
  }  // jF

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

void IdealMhdModel::assembleRZPreconditioner() {
  for (int m = 0; m < s_.mpol; ++m) {
    // magnetic axis only gets m=0 contributions
    // since it has no poloidal dimension/coordinate
    int jMin = 0;
    if (m > 0) {
      jMin = 1;
    }

    for (int n = 0; n < s_.ntor + 1; ++n) {
      int mn = m * (s_.ntor + 1) + n;
      this->jMin[mn] = jMin;
    }
  }

  int jMax = m_fc_.ns - 1;
  if (m_fc_.lfreeb &&
      (m_vacuum_pressure_state_ == VacuumPressureState::kInitialized ||
       m_vacuum_pressure_state_ == VacuumPressureState::kActive)) {
    jMax = m_fc_.ns;
  }

  for (int jF = r_.nsMinF; jF < std::min(r_.nsMaxF, jMax); ++jF) {
    for (int m = 0; m < s_.mpol; ++m) {
      const int m_parity = m % 2;
      for (int n = 0; n < s_.ntor + 1; ++n) {
        int mn = m * (s_.ntor + 1) + n;
        int idx_mn = ((jF - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;
        if (jF >= jMin[mn]) {
          // sup-diagonal: half-grid pos outside jF-th forces full-grid point
          if (jF < r_.nsMaxH) {
            ar[idx_mn] = -(arm[(jF - r_.nsMinH) * 2 + m_parity] +
                           brm[(jF - r_.nsMinH) * 2 + m_parity] * m * m);
            az[idx_mn] = -(azm[(jF - r_.nsMinH) * 2 + m_parity] +
                           bzm[(jF - r_.nsMinH) * 2 + m_parity] * m * m);
          }

          // diagonal: jF-th forces full-grid pos
          dr[idx_mn] = -(ard[(jF - r_.nsMinF) * 2 + m_parity] +
                         brd[(jF - r_.nsMinF) * 2 + m_parity] * m * m +
                         cxd[jF - r_.nsMinF] * n * s_.nfp * n * s_.nfp);
          dz[idx_mn] = -(azd[(jF - r_.nsMinF) * 2 + m_parity] +
                         bzd[(jF - r_.nsMinF) * 2 + m_parity] * m * m +
                         cxd[jF - r_.nsMinF] * n * s_.nfp * n * s_.nfp);

          // sub-diagonal: half-grid pos inside jF-th forces full-grid point
          if (jF > 0) {
            br[idx_mn] = -(arm[(jF - 1 - r_.nsMinH) * 2 + m_parity] +
                           brm[(jF - 1 - r_.nsMinH) * 2 + m_parity] * m * m);
            bz[idx_mn] = -(azm[(jF - 1 - r_.nsMinH) * 2 + m_parity] +
                           bzm[(jF - 1 - r_.nsMinH) * 2 + m_parity] * m * m);
          }

          if (jF == 1 && m == 1) {
            // TODO(jons): maybe this is not actually needed ???
            // related to m=1 constraint ???
            // only at innermost flux surface ???
            dr[idx_mn] += br[idx_mn];
            dz[idx_mn] += bz[idx_mn];
          }
        } else {
          ar[idx_mn] = 0.0;
          az[idx_mn] = 0.0;
          dr[idx_mn] = 0.0;
          dz[idx_mn] = 0.0;
          br[idx_mn] = 0.0;
          bz[idx_mn] = 0.0;
        }
      }  // n
    }  // m
  }  // jF

  // We need to check BOTH for
  // a) if we are in free-boundary mode AND
  // b) if we are in the thread that actually has the boundary data!
  if (r_.nsMaxF == m_fc_.ns) {
    // SMALL EDGE PEDESTAL NEEDED TO IMPROVE CONVERGENCE
    // IN PARTICULAR, NEEDED TO ACCOUNT FOR POTENTIAL ZERO
    // EIGENVALUE DUE TO NEUMANN (GRADIENT) CONDITION AT EDGE
    const double edge_pedestal = 0.05;
    for (int n = 0; n < s_.ntor + 1; ++n) {
      {
        int m = 0;
        int idx_mn =
            ((m_fc_.ns - 1 - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;
        dr[idx_mn] *= 1.0 + edge_pedestal;
        dz[idx_mn] *= 1.0 + edge_pedestal;
      }
      {
        int m = 1;
        int idx_mn =
            ((m_fc_.ns - 1 - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;
        dr[idx_mn] *= 1.0 + edge_pedestal;
        dz[idx_mn] *= 1.0 + edge_pedestal;
      }
      for (int m = 2; m < s_.mpol; ++m) {
        int idx_mn =
            ((m_fc_.ns - 1 - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;
        dr[idx_mn] *= 1.0 + 2.0 * edge_pedestal;
        dz[idx_mn] *= 1.0 + 2.0 * edge_pedestal;
      }
    }

    // STABILIZATION ALGORITHM FOR ZC_00(NS)
    // FOR UNSTABLE CASE, HAVE TO FLIP SIGN OF -FAC -> +FAC FOR CONVERGENCE
    // COEFFICIENT OF < Ru (R Pvac)> ~ -fac*(z-zeq) WHERE fac (EIGENVALUE, OR
    // FIELD INDEX) DEPENDS ON THE EQUILIBRIUM MAGNETIC FIELD AND CURRENT,
    // AND zeq IS THE EQUILIBRIUM EDGE VALUE OF Z00
    double fac = 0.25;
    double multFact = std::min(fac, fac * m_fc_.deltaS * 15.0);

    // METHOD 1: SUBTRACT (INSTABILITY) Pedge ~ fac*z/hs FROM PRECONDITIONER AT
    // EDGE iflag parameter is used in Fortran VMEC to enable this feature only
    // for Z forces !
    int idx_00 = (m_fc_.ns - 1 - r_.nsMinF) * s_.mpol * (s_.ntor + 1);
    dz[idx_00] *= (1.0 - multFact) / (1.0 + edge_pedestal);
  }

  // ACCELERATE (IMPROVE) CONVERGENCE OF FREE BOUNDARY.
  // THIS WAS ADDED TO DEAL WITH CASES WHICH MIGHT OTHERWISE DIVERGE.
  // BY DECREASING THE FSQ TOLERANCE LEVEL WHERE THIS KICKS IN (FTOL_EDGE),
  // THE USER CAN TURN-OFF THIS FEATURE
  //
  // DIAGONALIZE (DX DOMINANT) AND REDUCE FORCE (DX ENHANCED) AT EDGE
  // TO IMPROVE CONVERGENCE FOR N != 0 TERMS

  // ledge = .false.
  // IF ((fsqr+fsqz) .lt. ftol_edge) &
  //   ! only if forces are converged low enough already
  //   ledge = .true.
  //
  // IF ((iter2-iter1).lt.400 .or. vacuum_pressure_state.lt.1) &
  //   ! only starting in late iterations and if NESTOR fully initialized
  //   ledge = .false.
  //
  // IF (ledge) THEN
  //   dx(ns,1:,1:) = 3*dx(ns,1:,1:)
  // END IF

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

// serial variant
absl::Status IdealMhdModel::applyRZPreconditioner(
    FourierForces& m_decomposed_f) {
  std::vector<std::span<double>> cR(s_.num_basis);
  std::vector<std::span<double>> cZ(s_.num_basis);
  {
    int idx_basis = 0;

    cR[idx_basis] = m_decomposed_f.frcc;
    cZ[idx_basis] = m_decomposed_f.fzsc;
    idx_basis++;
    if (s_.lthreed) {
      cR[idx_basis] = m_decomposed_f.frss;
      cZ[idx_basis] = m_decomposed_f.fzcs;
      idx_basis++;
    }
    if (s_.lasym) {
      cR[idx_basis] = m_decomposed_f.frsc;
      cZ[idx_basis] = m_decomposed_f.fzcc;
      idx_basis++;
      if (s_.lthreed) {
        cR[idx_basis] = m_decomposed_f.frcs;
        cZ[idx_basis] = m_decomposed_f.fzss;
        idx_basis++;
      }
    }

    if (idx_basis != s_.num_basis) {
      return absl::InternalError(
          absl::StrFormat("counting error: idx_basis=%d != num_basis=%d",
                          idx_basis, s_.num_basis));
    }
  }

  int jMax = m_fc_.ns - 1;
  if (m_fc_.lfreeb &&
      (m_vacuum_pressure_state_ == VacuumPressureState::kInitialized ||
       m_vacuum_pressure_state_ == VacuumPressureState::kActive)) {
    jMax = m_fc_.ns;
  }

  // gather everything into HandoverStorage
  for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (jF - r_.nsMinF) * s_.mnsize + mn;
      m_h_.all_ar[mn][jF] = ar[idx_mn];
      m_h_.all_az[mn][jF] = az[idx_mn];
      m_h_.all_dr[mn][jF] = dr[idx_mn];
      m_h_.all_dz[mn][jF] = dz[idx_mn];
      m_h_.all_br[mn][jF] = br[idx_mn];
      m_h_.all_bz[mn][jF] = bz[idx_mn];
      for (int idx_basis = 0; idx_basis < s_.num_basis; ++idx_basis) {
        m_h_.all_cr[mn][idx_basis][jF] = cR[idx_basis][idx_mn];
        m_h_.all_cz[mn][idx_basis][jF] = cZ[idx_basis][idx_mn];
      }  // idx_basis
    }  // mn
  }  // jF
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // split range [0, s_.mnsize) among threads
  const int thread_id = r_.get_thread_id();
  const int num_threads = r_.get_num_threads();
  const int mnstep = s_.mnsize / num_threads;
  const int remainder = s_.mnsize % num_threads;
  int mnmin = thread_id * mnstep;
  int mnmax = mnmin + mnstep;

  // add 1 more iteration to the first `remainder` threads
  if (thread_id < remainder) {
    mnmin += thread_id;
    mnmax += thread_id + 1;
  } else {
    mnmin += remainder;
    mnmax += remainder;
  }

  // call serial Thomas solver for every mode number individually
  for (int mn = mnmin; mn < mnmax; ++mn) {
    TridiagonalSolveSerial(m_h_.all_ar[mn], m_h_.all_dr[mn], m_h_.all_br[mn],
                           m_h_.all_cr[mn], jMin[mn], jMax, s_.num_basis);
    TridiagonalSolveSerial(m_h_.all_az[mn], m_h_.all_dz[mn], m_h_.all_bz[mn],
                           m_h_.all_cz[mn], jMin[mn], jMax, s_.num_basis);
  }  // mn
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // re-distribute solution back into threads
  for (int jF = r_.nsMinF; jF < r_.nsMaxF; ++jF) {
    for (int mn = 0; mn < s_.mnsize; ++mn) {
      int idx_mn = (jF - r_.nsMinF) * s_.mnsize + mn;
      for (int idx_basis = 0; idx_basis < s_.num_basis; ++idx_basis) {
        cR[idx_basis][idx_mn] = m_h_.all_cr[mn][idx_basis][jF];
        cZ[idx_basis][idx_mn] = m_h_.all_cz[mn][idx_basis][jF];
      }  // idx_basis
    }  // mn
  }  // jF

  return absl::OkStatus();
}

// commented out for now until parallelized tri-diagonal solver is fixed
// (broken for for fixed-boundary cases)
// void IdealMHDModel::applyRZPreconditioner(FourierForces& decomposed_f,
// std::vector<std::mutex>& mutices) {

//     double *cR[s.num_basis];
//     double *cZ[s.num_basis];
//     {
//         int idx_basis = 0;

//         cR[idx_basis] = decomposed_f.frcc;
//         cZ[idx_basis] = decomposed_f.fzsc;
//         idx_basis++;
//         if (s.lthreed) {
//             cR[idx_basis] = decomposed_f.frss;
//             cZ[idx_basis] = decomposed_f.fzcs;
//             idx_basis++;
//         }
//         if (s.lasym) {
//             cR[idx_basis] = decomposed_f.frsc;
//             cZ[idx_basis] = decomposed_f.fzcc;
//             idx_basis++;
//             if (s.lthreed) {
//                 cR[idx_basis] = decomposed_f.frcs;
//                 cZ[idx_basis] = decomposed_f.fzss;
//                 idx_basis++;
//             }
//         }

//         if (idx_basis != s.num_basis) {
//             LOG(FATAL) << absl::StrFormat("counting error: idx_basis=%d !=
//             num_basis=%d\n", idx_basis, s.num_basis);
//         }
//     }

//     int jMax = r.ns - 1;
//     if (fc.lfreeb && (vacuum_pressure_state ==
//     VacuumPressureState::kInitialized || vacuum_pressure_state ==
//     VacuumPressureState::kActive)) {
//         jMax = r.ns;
//     }

//     int ncpu = r.get_num_threads();
//     int myid = r.get_thread_id();

//     tridiagonalSolve(
//             ar, dr, br, cR,
//             az, dz, bz, cZ,
//             jMin, jMax, s.mnsize, s.num_basis,
//             mutices, ncpu, myid, r.nsMinF, r.nsMaxF,
//             h.handover_aR, h.handover_cR,
//             h.handover_aZ, h.handover_cZ);
// }

/**
 * Apply lambda scaling before computing lambda force residual.
 * Here, the lambda preconditioner is applied. --> see BETAS article for details
 */
void IdealMhdModel::applyLambdaPreconditioner(FourierForces& m_decomposed_f) {
  for (int jF = r_.nsMinF; jF < r_.nsMaxFIncludingLcfs; ++jF) {
    for (int m = 0; m < s_.mpol; ++m) {
      for (int n = 0; n <= s_.ntor; ++n) {
        int idx_mn = ((jF - r_.nsMinF) * s_.mpol + m) * (s_.ntor + 1) + n;

        m_decomposed_f.flsc[idx_mn] *= lambdaPreconditioner[idx_mn];
        if (s_.lthreed) {
          m_decomposed_f.flcs[idx_mn] *= lambdaPreconditioner[idx_mn];
        }
        if (s_.lasym) {
          m_decomposed_f.flcc[idx_mn] *= lambdaPreconditioner[idx_mn];
          if (s_.lthreed) {
            m_decomposed_f.flss[idx_mn] *= lambdaPreconditioner[idx_mn];
          }
        }
      }  // n
    }  // m
  }  // j
}

double IdealMhdModel::get_delbsq() const {
  double delBSqAvg = 0.0;
  if (m_fc_.lfreeb &&
      m_vacuum_pressure_state_ == VacuumPressureState::kActive) {
    double delBSqNorm = 0.0;
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      int l = kl % s_.nThetaEff;
      delBSqAvg += delBSq[kl] * s_.wInt[l];
      delBSqNorm += insideTotalPressure[kl] * s_.wInt[l];
    }
    delBSqAvg /= delBSqNorm;
  }
  return delBSqAvg;
}

int IdealMhdModel::get_ivacskip() const { return ivacskip; }

}  // namespace vmecpp
