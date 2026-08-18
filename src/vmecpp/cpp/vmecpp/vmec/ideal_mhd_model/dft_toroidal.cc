// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/dft_toroidal.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"

namespace vmecpp {

void ForcesToFourier3DSymmFastPoloidal(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
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

void FourierToReal3DSymmFastPoloidal(const FourierGeometry& physical_x,
                                     const Eigen::VectorXd& xmpq,
                                     const RadialPartitioning& r,
                                     const Sizes& s, const RadialProfiles& rp,
                                     const FourierBasisFastPoloidal& fb,
                                     RealSpaceGeometry& m_geometry) {
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

// Non-stellarator-symmetric (lasym) inverse DFT, 3D case. Mirror of
// FourierToReal3DSymmFastPoloidal with the cos<->sin basis swap:
//   R += rmnsc*sin(mu)cos(nv) + rmncs*cos(mu)sin(nv),
//   Z += zmncc*cos(mu)cos(nv) + zmnss*sin(mu)sin(nv),
//   lambda += lmncc*cos(mu)cos(nv) + lmnss*sin(mu)sin(nv),
// plus the matching theta/zeta derivatives. Writes into the *_asym arrays
// carried by m_geometry on the reduced poloidal interval.
void FourierToReal3DAsymFastPoloidal(const FourierGeometry& physical_x,
                                     const Eigen::VectorXd& xmpq,
                                     const RadialPartitioning& r,
                                     const Sizes& s, const RadialProfiles& rp,
                                     const FourierBasisFastPoloidal& fb,
                                     RealSpaceGeometry& m_geometry) {
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

  const int nsMinF1 = r.nsMinF1;
  const int nsMinF = r.nsMinF;
  for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = m % 2 == 0;
      const int idx_ml_base = m * s.nThetaReduced;

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

      int jMin = 1;
      if (m == 0 || m == 1) {
        jMin = 0;
      }
      if (jF < jMin) {
        continue;
      }

      for (int k = 0; k < s.nZeta; ++k) {
        const int idx_kn_base = k * (s.nnyq2 + 1);
        const int idx_mn_base = ((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1);

        auto cosnv_seg = fb.cosnv.segment(idx_kn_base, s.ntor + 1);
        auto sinnv_seg = fb.sinnv.segment(idx_kn_base, s.ntor + 1);
        auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, s.ntor + 1);
        auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, s.ntor + 1);

        auto rmnsc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.rmnsc.data() + idx_mn_base, s.ntor + 1);
        auto rmncs_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.rmncs.data() + idx_mn_base, s.ntor + 1);
        auto zmncc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.zmncc.data() + idx_mn_base, s.ntor + 1);
        auto zmnss_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.zmnss.data() + idx_mn_base, s.ntor + 1);
        auto lmncc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.lmncc.data() + idx_mn_base, s.ntor + 1);
        auto lmnss_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.lmnss.data() + idx_mn_base, s.ntor + 1);

        double rmksc = rmnsc_seg.dot(cosnv_seg);
        double rmksc_n = rmnsc_seg.dot(sinnvn_seg);
        double rmkcs = rmncs_seg.dot(sinnv_seg);
        double rmkcs_n = rmncs_seg.dot(cosnvn_seg);
        double zmkcc = zmncc_seg.dot(cosnv_seg);
        double zmkcc_n = zmncc_seg.dot(sinnvn_seg);
        double zmkss = zmnss_seg.dot(sinnv_seg);
        double zmkss_n = zmnss_seg.dot(cosnvn_seg);
        double lmkcc = lmncc_seg.dot(cosnv_seg);
        double lmkcc_n = lmncc_seg.dot(sinnvn_seg);
        double lmkss = lmnss_seg.dot(sinnv_seg);
        double lmkss_n = lmnss_seg.dot(cosnvn_seg);

        const int idx_kl_base = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff;

        auto sinmum_seg = fb.sinmum.segment(idx_ml_base, s.nThetaReduced);
        auto cosmum_seg = fb.cosmum.segment(idx_ml_base, s.nThetaReduced);

        auto ru_seg = Eigen::Map<Eigen::VectorXd>(ru.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto zu_seg = Eigen::Map<Eigen::VectorXd>(zu.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto lu_seg = Eigen::Map<Eigen::VectorXd>(lu.data() + idx_kl_base,
                                                  s.nThetaReduced);

        ru_seg += rmksc * cosmum_seg + rmkcs * sinmum_seg;
        zu_seg += zmkcc * sinmum_seg + zmkss * cosmum_seg;
        lu_seg += lmkcc * sinmum_seg + lmkss * cosmum_seg;

        auto cosmu_seg = fb.cosmu.segment(idx_ml_base, s.nThetaReduced);
        auto sinmu_seg = fb.sinmu.segment(idx_ml_base, s.nThetaReduced);

        auto rv_seg = Eigen::Map<Eigen::VectorXd>(rv.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto zv_seg = Eigen::Map<Eigen::VectorXd>(zv.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto lv_seg = Eigen::Map<Eigen::VectorXd>(lv.data() + idx_kl_base,
                                                  s.nThetaReduced);

        rv_seg += rmksc_n * sinmu_seg + rmkcs_n * cosmu_seg;
        zv_seg += zmkcc_n * cosmu_seg + zmkss_n * sinmu_seg;
        lv_seg -= lmkcc_n * cosmu_seg + lmkss_n * sinmu_seg;

        auto r1_seg = Eigen::Map<Eigen::VectorXd>(r1.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto z1_seg = Eigen::Map<Eigen::VectorXd>(z1.data() + idx_kl_base,
                                                  s.nThetaReduced);

        r1_seg += rmksc * sinmu_seg + rmkcs * cosmu_seg;
        z1_seg += zmkcc * cosmu_seg + zmkss * sinmu_seg;

        if (nsMinF <= jF && jF < r.nsMaxFIncludingLcfs) {
          const int idx_con_base = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff;

          auto rCon_seg = Eigen::Map<Eigen::VectorXd>(
              m_geometry.rCon.data() + idx_con_base, s.nThetaReduced);
          auto zCon_seg = Eigen::Map<Eigen::VectorXd>(
              m_geometry.zCon.data() + idx_con_base, s.nThetaReduced);

          rCon_seg += (rmksc * sinmu_seg + rmkcs * cosmu_seg) * con_factor;
          zCon_seg += (zmkcc * cosmu_seg + zmkss * sinmu_seg) * con_factor;
        }
      }  // k
    }  // m
  }  // j
}

// Non-stellarator-symmetric (lasym) forward DFT, 3D case. Mirror of
// ForcesToFourier3DSymmFastPoloidal with the cos<->sin basis swap; projects the
// antisymmetric-parity force halves onto the frsc / frcs / fzcc / fzss / flcc /
// flss coefficients (educational_VMEC tomnspa). The force arrays in d are the
// reversed-parity halves produced by symforce.
void ForcesToFourier3DAsymFastPoloidal(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces) {
  // can safely assume lthreed == true in here

  int jMaxRZ = std::min(rp.nsMaxF, fc.ns - 1);
  if (fc.lfreeb &&
      (vacuum_pressure_state == VacuumPressureState::kInitialized ||
       vacuum_pressure_state == VacuumPressureState::kActive)) {
    jMaxRZ = std::min(rp.nsMaxF, fc.ns);
  }

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
        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
        const int idx_ml_base = m * s.nThetaReduced;

        auto cosmui_seg = fb.cosmui.segment(idx_ml_base, s.nThetaReduced);
        auto sinmui_seg = fb.sinmui.segment(idx_ml_base, s.nThetaReduced);
        auto cosmumi_seg = fb.cosmumi.segment(idx_ml_base, s.nThetaReduced);
        auto sinmumi_seg = fb.sinmumi.segment(idx_ml_base, s.nThetaReduced);

        auto blmn_seg = Eigen::Map<const Eigen::VectorXd>(
            blmn.data() + idx_kl_base, s.nThetaReduced);
        auto clmn_seg = Eigen::Map<const Eigen::VectorXd>(
            clmn.data() + idx_kl_base, s.nThetaReduced);
        auto crmn_seg = Eigen::Map<const Eigen::VectorXd>(
            crmn.data() + idx_kl_base, s.nThetaReduced);
        auto czmn_seg = Eigen::Map<const Eigen::VectorXd>(
            czmn.data() + idx_kl_base, s.nThetaReduced);
        auto armn_seg = Eigen::Map<const Eigen::VectorXd>(
            armn.data() + idx_kl_base, s.nThetaReduced);
        auto azmn_seg = Eigen::Map<const Eigen::VectorXd>(
            azmn.data() + idx_kl_base, s.nThetaReduced);
        auto brmn_seg = Eigen::Map<const Eigen::VectorXd>(
            brmn.data() + idx_kl_base, s.nThetaReduced);
        auto bzmn_seg = Eigen::Map<const Eigen::VectorXd>(
            bzmn.data() + idx_kl_base, s.nThetaReduced);
        auto frcon_seg = Eigen::Map<const Eigen::VectorXd>(
            frcon.data() + idx_kl_base, s.nThetaReduced);
        auto fzcon_seg = Eigen::Map<const Eigen::VectorXd>(
            fzcon.data() + idx_kl_base, s.nThetaReduced);

        double lmkcc = blmn_seg.dot(sinmumi_seg);
        double lmkss = blmn_seg.dot(cosmumi_seg);
        double lmkcc_n = -clmn_seg.dot(cosmui_seg);
        double lmkss_n = -clmn_seg.dot(sinmui_seg);

        double rmksc_n = -crmn_seg.dot(sinmui_seg);
        double zmkcc_n = -czmn_seg.dot(cosmui_seg);
        double rmkcs_n = -crmn_seg.dot(cosmui_seg);
        double zmkss_n = -czmn_seg.dot(sinmui_seg);

        const Eigen::VectorXd tempR_seg =
            (armn_seg + xmpq[m] * frcon_seg).eval();
        const Eigen::VectorXd tempZ_seg =
            (azmn_seg + xmpq[m] * fzcon_seg).eval();

        double rmksc = tempR_seg.dot(sinmui_seg) + brmn_seg.dot(cosmumi_seg);
        double rmkcs = tempR_seg.dot(cosmui_seg) + brmn_seg.dot(sinmumi_seg);
        double zmkcc = tempZ_seg.dot(cosmui_seg) + bzmn_seg.dot(sinmumi_seg);
        double zmkss = tempZ_seg.dot(sinmui_seg) + bzmn_seg.dot(cosmumi_seg);

        const int ntorp1 = s.ntor + 1;
        const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;
        const int idx_kn_base = k * (s.nnyq2 + 1);

        auto cosnv_seg = fb.cosnv.segment(idx_kn_base, ntorp1);
        auto sinnv_seg = fb.sinnv.segment(idx_kn_base, ntorp1);
        auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, ntorp1);
        auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, ntorp1);

        Eigen::Map<Eigen::VectorXd> frsc_seg(
            m_physical_forces.frsc.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> frcs_seg(
            m_physical_forces.frcs.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> fzcc_seg(
            m_physical_forces.fzcc.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> fzss_seg(
            m_physical_forces.fzss.data() + idx_mn_base, ntorp1);

        frsc_seg += rmksc * cosnv_seg + rmksc_n * sinnvn_seg;
        frcs_seg += rmkcs * sinnv_seg + rmkcs_n * cosnvn_seg;
        fzcc_seg += zmkcc * cosnv_seg + zmkcc_n * sinnvn_seg;
        fzss_seg += zmkss * sinnv_seg + zmkss_n * cosnvn_seg;

        if (jMinL <= jF) {
          Eigen::Map<Eigen::VectorXd> flcc_seg(
              m_physical_forces.flcc.data() + idx_mn_base, ntorp1);
          Eigen::Map<Eigen::VectorXd> flss_seg(
              m_physical_forces.flss.data() + idx_mn_base, ntorp1);
          flcc_seg += lmkcc * cosnv_seg + lmkcc_n * sinnvn_seg;
          flss_seg += lmkss * sinnv_seg + lmkss_n * cosnvn_seg;
        }
      }  // k
    }  // m
  }  // jF

  // lambda-only section for jMaxRZ .. nsMaxFIncludingLcfs
  for (int jF = jMaxRZ; jF < rp.nsMaxFIncludingLcfs; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = m % 2 == 0;

      const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
      const auto& clmn = m_even ? d.clmn_e : d.clmn_o;

      for (int k = 0; k < s.nZeta; ++k) {
        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
        const int idx_ml_base = m * s.nThetaReduced;

        auto cosmui_seg = fb.cosmui.segment(idx_ml_base, s.nThetaReduced);
        auto sinmui_seg = fb.sinmui.segment(idx_ml_base, s.nThetaReduced);
        auto cosmumi_seg = fb.cosmumi.segment(idx_ml_base, s.nThetaReduced);
        auto sinmumi_seg = fb.sinmumi.segment(idx_ml_base, s.nThetaReduced);

        auto blmn_seg = Eigen::Map<const Eigen::VectorXd>(
            blmn.data() + idx_kl_base, s.nThetaReduced);
        auto clmn_seg = Eigen::Map<const Eigen::VectorXd>(
            clmn.data() + idx_kl_base, s.nThetaReduced);

        double lmkcc = blmn_seg.dot(sinmumi_seg);
        double lmkss = blmn_seg.dot(cosmumi_seg);
        double lmkcc_n = -clmn_seg.dot(cosmui_seg);
        double lmkss_n = -clmn_seg.dot(sinmui_seg);

        const int ntorp1 = s.ntor + 1;
        const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;
        const int idx_kn_base = k * (s.nnyq2 + 1);

        auto cosnv_seg = fb.cosnv.segment(idx_kn_base, ntorp1);
        auto sinnv_seg = fb.sinnv.segment(idx_kn_base, ntorp1);
        auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, ntorp1);
        auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, ntorp1);

        Eigen::Map<Eigen::VectorXd> flcc_seg(
            m_physical_forces.flcc.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> flss_seg(
            m_physical_forces.flss.data() + idx_mn_base, ntorp1);

        flcc_seg += lmkcc * cosnv_seg + lmkcc_n * sinnvn_seg;
        flss_seg += lmkss * sinnv_seg + lmkss_n * cosnvn_seg;
      }  // k
    }  // m
  }  // jF
}

}  // namespace vmecpp
