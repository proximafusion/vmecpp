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
        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
        const int idx_ml_base = m * s.nThetaReduced;

        // Vectorized poloidal loop using Eigen operations
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

        double lmksc = blmn_seg.dot(cosmumi_seg);
        double lmkcs = blmn_seg.dot(sinmumi_seg);
        double lmkcs_n = -clmn_seg.dot(cosmui_seg);
        double lmksc_n = -clmn_seg.dot(sinmui_seg);

        double rmkcc_n = -crmn_seg.dot(cosmui_seg);
        double zmkcs_n = -czmn_seg.dot(cosmui_seg);

        double rmkss_n = -crmn_seg.dot(sinmui_seg);
        double zmksc_n = -czmn_seg.dot(sinmui_seg);

        // Assemble effective R and Z forces from MHD and spectral condensation
        // contributions. Materialize to avoid re-evaluation in each dot
        // product.
        // Per-thread scratch reused across iterations instead of a heap
        // temporary in this innermost loop; still materialized once and then
        // used in the two dot products below.
        thread_local Eigen::VectorXd tempR_seg;
        thread_local Eigen::VectorXd tempZ_seg;
        tempR_seg = armn_seg + xmpq[m] * frcon_seg;
        tempZ_seg = azmn_seg + xmpq[m] * fzcon_seg;

        double rmkcc = tempR_seg.dot(cosmui_seg) + brmn_seg.dot(sinmumi_seg);
        double rmkss = tempR_seg.dot(sinmui_seg) + brmn_seg.dot(cosmumi_seg);
        double zmksc = tempZ_seg.dot(sinmui_seg) + bzmn_seg.dot(cosmumi_seg);
        double zmkcs = tempZ_seg.dot(cosmui_seg) + bzmn_seg.dot(sinmumi_seg);

        // Vectorized toroidal scatter: segment ops replace scalar n-loop
        const int ntorp1 = s.ntor + 1;
        const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;
        const int idx_kn_base = k * (s.nnyq2 + 1);

        auto cosnv_seg = fb.cosnv.segment(idx_kn_base, ntorp1);
        auto sinnv_seg = fb.sinnv.segment(idx_kn_base, ntorp1);
        auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, ntorp1);
        auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, ntorp1);

        Eigen::Map<Eigen::VectorXd> frcc_seg(
            m_physical_forces.frcc.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> frss_seg(
            m_physical_forces.frss.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> fzsc_seg(
            m_physical_forces.fzsc.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> fzcs_seg(
            m_physical_forces.fzcs.data() + idx_mn_base, ntorp1);

        frcc_seg += rmkcc * cosnv_seg + rmkcc_n * sinnvn_seg;
        frss_seg += rmkss * sinnv_seg + rmkss_n * cosnvn_seg;
        fzsc_seg += zmksc * cosnv_seg + zmksc_n * sinnvn_seg;
        fzcs_seg += zmkcs * sinnv_seg + zmkcs_n * cosnvn_seg;

        if (jMinL <= jF) {
          Eigen::Map<Eigen::VectorXd> flsc_seg(
              m_physical_forces.flsc.data() + idx_mn_base, ntorp1);
          Eigen::Map<Eigen::VectorXd> flcs_seg(
              m_physical_forces.flcs.data() + idx_mn_base, ntorp1);
          flsc_seg += lmksc * cosnv_seg + lmksc_n * sinnvn_seg;
          flcs_seg += lmkcs * sinnv_seg + lmkcs_n * cosnvn_seg;
        }
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
        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
        const int idx_ml_base = m * s.nThetaReduced;

        // Vectorized poloidal loop using Eigen operations
        auto cosmui_seg = fb.cosmui.segment(idx_ml_base, s.nThetaReduced);
        auto sinmui_seg = fb.sinmui.segment(idx_ml_base, s.nThetaReduced);
        auto cosmumi_seg = fb.cosmumi.segment(idx_ml_base, s.nThetaReduced);
        auto sinmumi_seg = fb.sinmumi.segment(idx_ml_base, s.nThetaReduced);

        auto blmn_seg = Eigen::Map<const Eigen::VectorXd>(
            blmn.data() + idx_kl_base, s.nThetaReduced);
        auto clmn_seg = Eigen::Map<const Eigen::VectorXd>(
            clmn.data() + idx_kl_base, s.nThetaReduced);

        double lmksc = blmn_seg.dot(cosmumi_seg);
        double lmkcs = blmn_seg.dot(sinmumi_seg);
        double lmkcs_n = -clmn_seg.dot(cosmui_seg);
        double lmksc_n = -clmn_seg.dot(sinmui_seg);

        // Vectorized toroidal scatter for lambda-only section
        const int ntorp1 = s.ntor + 1;
        const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;
        const int idx_kn_base = k * (s.nnyq2 + 1);

        auto cosnv_seg = fb.cosnv.segment(idx_kn_base, ntorp1);
        auto sinnv_seg = fb.sinnv.segment(idx_kn_base, ntorp1);
        auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, ntorp1);
        auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, ntorp1);

        Eigen::Map<Eigen::VectorXd> flsc_seg(
            m_physical_forces.flsc.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> flcs_seg(
            m_physical_forces.flcs.data() + idx_mn_base, ntorp1);

        flsc_seg += lmksc * cosnv_seg + lmksc_n * sinnvn_seg;
        flcs_seg += lmkcs * sinnv_seg + lmkcs_n * cosnvn_seg;
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
        // INVERSE TRANSFORM IN N-ZETA, FOR FIXED M
        // Vectorized toroidal accumulation loop
        const int idx_kn_base = k * (s.nnyq2 + 1);
        const int idx_mn_base = ((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1);

        auto cosnv_seg = fb.cosnv.segment(idx_kn_base, s.ntor + 1);
        auto sinnv_seg = fb.sinnv.segment(idx_kn_base, s.ntor + 1);
        auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, s.ntor + 1);
        auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, s.ntor + 1);

        auto rmncc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.rmncc.data() + idx_mn_base, s.ntor + 1);
        auto rmnss_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.rmnss.data() + idx_mn_base, s.ntor + 1);
        auto zmnsc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.zmnsc.data() + idx_mn_base, s.ntor + 1);
        auto zmncs_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.zmncs.data() + idx_mn_base, s.ntor + 1);
        auto lmnsc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.lmnsc.data() + idx_mn_base, s.ntor + 1);
        auto lmncs_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.lmncs.data() + idx_mn_base, s.ntor + 1);

        double rmkcc = rmncc_seg.dot(cosnv_seg);
        double rmkcc_n = rmncc_seg.dot(sinnvn_seg);
        double rmkss = rmnss_seg.dot(sinnv_seg);
        double rmkss_n = rmnss_seg.dot(cosnvn_seg);
        double zmksc = zmnsc_seg.dot(cosnv_seg);
        double zmksc_n = zmnsc_seg.dot(sinnvn_seg);
        double zmkcs = zmncs_seg.dot(sinnv_seg);
        double zmkcs_n = zmncs_seg.dot(cosnvn_seg);
        double lmksc = lmnsc_seg.dot(cosnv_seg);
        double lmksc_n = lmnsc_seg.dot(sinnvn_seg);
        double lmkcs = lmncs_seg.dot(sinnv_seg);
        double lmkcs_n = lmncs_seg.dot(cosnvn_seg);

        // INVERSE TRANSFORM IN M-THETA, FOR ALL RADIAL, ZETA VALUES
        const int idx_kl_base = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff;

        // Vectorized poloidal loops using Eigen operations
        auto sinmum_seg = fb.sinmum.segment(idx_ml_base, s.nThetaReduced);
        auto cosmum_seg = fb.cosmum.segment(idx_ml_base, s.nThetaReduced);

        auto ru_seg = Eigen::Map<Eigen::VectorXd>(ru.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto zu_seg = Eigen::Map<Eigen::VectorXd>(zu.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto lu_seg = Eigen::Map<Eigen::VectorXd>(lu.data() + idx_kl_base,
                                                  s.nThetaReduced);

        // NOTE: element-wise multiplication
        ru_seg += rmkcc * sinmum_seg + rmkss * cosmum_seg;
        zu_seg += zmksc * cosmum_seg + zmkcs * sinmum_seg;
        lu_seg += lmksc * cosmum_seg + lmkcs * sinmum_seg;

        auto cosmu_seg = fb.cosmu.segment(idx_ml_base, s.nThetaReduced);
        auto sinmu_seg = fb.sinmu.segment(idx_ml_base, s.nThetaReduced);

        auto rv_seg = Eigen::Map<Eigen::VectorXd>(rv.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto zv_seg = Eigen::Map<Eigen::VectorXd>(zv.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto lv_seg = Eigen::Map<Eigen::VectorXd>(lv.data() + idx_kl_base,
                                                  s.nThetaReduced);

        // NOTE: element-wise multiplication
        rv_seg += rmkcc_n * cosmu_seg + rmkss_n * sinmu_seg;
        zv_seg += zmksc_n * sinmu_seg + zmkcs_n * cosmu_seg;
        // it is here that lv gets a negative sign!
        lv_seg -= lmksc_n * sinmu_seg + lmkcs_n * cosmu_seg;

        auto r1_seg = Eigen::Map<Eigen::VectorXd>(r1.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto z1_seg = Eigen::Map<Eigen::VectorXd>(z1.data() + idx_kl_base,
                                                  s.nThetaReduced);

        r1_seg += rmkcc * cosmu_seg + rmkss * sinmu_seg;
        z1_seg += zmksc * sinmu_seg + zmkcs * cosmu_seg;

        if (nsMinF <= jF && jF < r.nsMaxFIncludingLcfs) {
          // spectral condensation is local per flux surface
          const int idx_con_base = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff;

          auto rCon_seg = Eigen::Map<Eigen::VectorXd>(
              m_geometry.rCon.data() + idx_con_base, s.nThetaReduced);
          auto zCon_seg = Eigen::Map<Eigen::VectorXd>(
              m_geometry.zCon.data() + idx_con_base, s.nThetaReduced);

          rCon_seg += (rmkcc * cosmu_seg + rmkss * sinmu_seg) * con_factor;
          zCon_seg += (zmksc * sinmu_seg + zmkcs * cosmu_seg) * con_factor;
        }
      }  // k
    }  // m
  }  // j
}

}  // namespace vmecpp
