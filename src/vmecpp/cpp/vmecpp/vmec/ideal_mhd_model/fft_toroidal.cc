// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"

#include <mkl_dfti.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include "absl/algorithm/container.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

// ---------------------------------------------------------------------------
// ToroidalFftPlans
// ---------------------------------------------------------------------------

ToroidalFftPlans::ToroidalFftPlans(int n_in, int nfp_in)
    : n(n_in), nhalf(n_in / 2 + 1), nfp(nfp_in) {
  // Real-domain, double-precision, 1-D descriptor of length n.
  // CCE (complex conjugate-even) packing is used for the half-spectrum side.
  // The CCE buffer layout is identical to interleaved double[2] complex:
  //   [re0, 0, re1, im1, ..., re(N/2), 0]  (2*nhalf doubles)
  // so the Fill* helpers and accumulation code work directly on double[2].
  //
  // Backward transform (synthesis): CCE input -> real output.
  DftiCreateDescriptor(&desc_c2r, DFTI_DOUBLE, DFTI_REAL, 1,
                       static_cast<MKL_LONG>(n));
  DftiSetValue(desc_c2r, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_CCE_FORMAT);
  DftiSetValue(desc_c2r, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiCommitDescriptor(desc_c2r);

  // Forward transform (analysis): real input -> CCE output.
  DftiCreateDescriptor(&desc_r2c, DFTI_DOUBLE, DFTI_REAL, 1,
                       static_cast<MKL_LONG>(n));
  DftiSetValue(desc_r2c, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_CCE_FORMAT);
  DftiSetValue(desc_r2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiCommitDescriptor(desc_r2c);
}

ToroidalFftPlans::~ToroidalFftPlans() {
  DftiFreeDescriptor(&desc_c2r);
  DftiFreeDescriptor(&desc_r2c);
}

// ---------------------------------------------------------------------------
// Internal helpers for filling the half-spectrum (c2r input / r2c output)
// ---------------------------------------------------------------------------
//
// MKL CCE half-spectrum layout: [re0, 0, re1, im1, ..., re(N/2), 0]
// = 2*nhalf doubles, indexed as double[2] per bin.
//
// MKL backward (c2r) produces:
//   f[k] = X[0] + 2*Sum_{n=1}^{N/2-1}[Re(X[n])*cos(2*pi*n*k/N)
//                                       - Im(X[n])*sin(2*pi*n*k/N)]
//          + Re(X[N/2])*(-1)^k
//
// Mapping to VMEC toroidal basis (zeta_k = 2*pi*k/N):
//   DCT: f[k] = Sigma_n spec[n]*nscale[n]*cos(n*phi_k)
//     -> X[0][0] = spec[0]*nscale[0], X[n][0] = spec[n]*nscale[n]/2 (n>=1)
//
//   DST: f[k] = Sigma_n spec[n]*nscale[n]*sin(n*phi_k)
//     -> X[n][1] = -spec[n]*nscale[n]/2  (Im negative)
//
//   DCT-deriv (d/dzeta of cosine series, = sinnvn contraction):
//     -> X[n][1] = +spec[n]*n*nfp*nscale[n]/2
//
//   DST-deriv (d/dzeta of sine series, = cosnvn contraction):
//     -> X[n][0] = spec[n]*n*nfp*nscale[n]/2

namespace {

using CplxBuf = double[2];

inline void FillDct(const double* spec, const Eigen::VectorXd& nscale, int ntor,
                    int nhalf, CplxBuf* X) {
  X[0][0] = spec[0] * nscale[0];
  X[0][1] = 0.0;
  for (int n = 1; n <= ntor; ++n) {
    X[n][0] = spec[n] * nscale[n] * 0.5;
    X[n][1] = 0.0;
  }
  for (int n = ntor + 1; n < nhalf; ++n) {
    X[n][0] = 0.0;
    X[n][1] = 0.0;
  }
}

inline void FillDst(const double* spec, const Eigen::VectorXd& nscale, int ntor,
                    int nhalf, CplxBuf* X) {
  X[0][0] = 0.0;
  X[0][1] = 0.0;
  for (int n = 1; n <= ntor; ++n) {
    X[n][0] = 0.0;
    X[n][1] = -spec[n] * nscale[n] * 0.5;
  }
  for (int n = ntor + 1; n < nhalf; ++n) {
    X[n][0] = 0.0;
    X[n][1] = 0.0;
  }
}

inline void FillDctDeriv(const double* spec, const Eigen::VectorXd& nscale,
                         int ntor, int nhalf, int nfp, CplxBuf* X) {
  X[0][0] = 0.0;
  X[0][1] = 0.0;
  for (int n = 1; n <= ntor; ++n) {
    X[n][0] = 0.0;
    X[n][1] = spec[n] * (n * nfp) * nscale[n] * 0.5;
  }
  for (int n = ntor + 1; n < nhalf; ++n) {
    X[n][0] = 0.0;
    X[n][1] = 0.0;
  }
}

inline void FillDstDeriv(const double* spec, const Eigen::VectorXd& nscale,
                         int ntor, int nhalf, int nfp, CplxBuf* X) {
  X[0][0] = 0.0;
  X[0][1] = 0.0;
  for (int n = 1; n <= ntor; ++n) {
    X[n][0] = spec[n] * (n * nfp) * nscale[n] * 0.5;
    X[n][1] = 0.0;
  }
  for (int n = ntor + 1; n < nhalf; ++n) {
    X[n][0] = 0.0;
    X[n][1] = 0.0;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// FourierToReal3DSymmFastPoloidalFft
// ---------------------------------------------------------------------------

void FourierToReal3DSymmFastPoloidalFft(
    const FourierGeometry& physical_x, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, const ToroidalFftPlans& plans,
    RealSpaceGeometry& m_geometry) {
  // This function matches the logic of FourierToReal3DSymmFastPoloidal but
  // replaces the O(nZeta * ntor) inner-n dot-product loop with
  // O(nZeta * log(nZeta)) MKL DFTI backward (c2r) transforms.

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
  const int ntor = s.ntor;
  const int nhalf = plans.nhalf;
  const int nfp = plans.nfp;

  // Half-spectrum scratch: 2*nhalf doubles = nhalf CplxBuf entries.
  std::vector<double> X_buf(2 * nhalf);
  CplxBuf* X = reinterpret_cast<CplxBuf*>(X_buf.data());

  Eigen::VectorXd rmkcc(s.nZeta), rmkss(s.nZeta);
  Eigen::VectorXd rmkcc_n(s.nZeta), rmkss_n(s.nZeta);
  Eigen::VectorXd zmksc(s.nZeta), zmkcs(s.nZeta);
  Eigen::VectorXd zmksc_n(s.nZeta), zmkcs_n(s.nZeta);
  Eigen::VectorXd lmksc(s.nZeta), lmkcs(s.nZeta);
  Eigen::VectorXd lmksc_n(s.nZeta), lmkcs_n(s.nZeta);

  for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = (m % 2 == 0);

      const int jMin = (m == 0 || m == 1) ? 0 : 1;
      if (jF < jMin) {
        continue;
      }

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

      const int idx_mn_base = ((jF - nsMinF1) * s.mpol + m) * (ntor + 1);

      const double* rmncc_ptr = physical_x.rmncc.data() + idx_mn_base;
      const double* rmnss_ptr = physical_x.rmnss.data() + idx_mn_base;
      const double* zmnsc_ptr = physical_x.zmnsc.data() + idx_mn_base;
      const double* zmncs_ptr = physical_x.zmncs.data() + idx_mn_base;
      const double* lmnsc_ptr = physical_x.lmnsc.data() + idx_mn_base;
      const double* lmncs_ptr = physical_x.lmncs.data() + idx_mn_base;

      FillDct(rmncc_ptr, fb.nscale, ntor, nhalf, X);
      DftiComputeBackward(plans.desc_c2r, X, rmkcc.data());

      FillDst(rmnss_ptr, fb.nscale, ntor, nhalf, X);
      DftiComputeBackward(plans.desc_c2r, X, rmkss.data());

      FillDctDeriv(rmncc_ptr, fb.nscale, ntor, nhalf, nfp, X);
      DftiComputeBackward(plans.desc_c2r, X, rmkcc_n.data());

      FillDstDeriv(rmnss_ptr, fb.nscale, ntor, nhalf, nfp, X);
      DftiComputeBackward(plans.desc_c2r, X, rmkss_n.data());

      FillDct(zmnsc_ptr, fb.nscale, ntor, nhalf, X);
      DftiComputeBackward(plans.desc_c2r, X, zmksc.data());

      FillDst(zmncs_ptr, fb.nscale, ntor, nhalf, X);
      DftiComputeBackward(plans.desc_c2r, X, zmkcs.data());

      FillDctDeriv(zmnsc_ptr, fb.nscale, ntor, nhalf, nfp, X);
      DftiComputeBackward(plans.desc_c2r, X, zmksc_n.data());

      FillDstDeriv(zmncs_ptr, fb.nscale, ntor, nhalf, nfp, X);
      DftiComputeBackward(plans.desc_c2r, X, zmkcs_n.data());

      FillDct(lmnsc_ptr, fb.nscale, ntor, nhalf, X);
      DftiComputeBackward(plans.desc_c2r, X, lmksc.data());

      FillDst(lmncs_ptr, fb.nscale, ntor, nhalf, X);
      DftiComputeBackward(plans.desc_c2r, X, lmkcs.data());

      FillDctDeriv(lmnsc_ptr, fb.nscale, ntor, nhalf, nfp, X);
      DftiComputeBackward(plans.desc_c2r, X, lmksc_n.data());

      FillDstDeriv(lmncs_ptr, fb.nscale, ntor, nhalf, nfp, X);
      DftiComputeBackward(plans.desc_c2r, X, lmkcs_n.data());

      const int idx_ml_base = m * s.nThetaReduced;

      auto sinmum_seg = fb.sinmum.segment(idx_ml_base, s.nThetaReduced);
      auto cosmum_seg = fb.cosmum.segment(idx_ml_base, s.nThetaReduced);
      auto cosmu_seg = fb.cosmu.segment(idx_ml_base, s.nThetaReduced);
      auto sinmu_seg = fb.sinmu.segment(idx_ml_base, s.nThetaReduced);

      for (int k = 0; k < s.nZeta; ++k) {
        const int idx_kl_base = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff;

        auto ru_seg = Eigen::Map<Eigen::VectorXd>(ru.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto zu_seg = Eigen::Map<Eigen::VectorXd>(zu.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto lu_seg = Eigen::Map<Eigen::VectorXd>(lu.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto rv_seg = Eigen::Map<Eigen::VectorXd>(rv.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto zv_seg = Eigen::Map<Eigen::VectorXd>(zv.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto lv_seg = Eigen::Map<Eigen::VectorXd>(lv.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto r1_seg = Eigen::Map<Eigen::VectorXd>(r1.data() + idx_kl_base,
                                                  s.nThetaReduced);
        auto z1_seg = Eigen::Map<Eigen::VectorXd>(z1.data() + idx_kl_base,
                                                  s.nThetaReduced);

        ru_seg += rmkcc[k] * sinmum_seg + rmkss[k] * cosmum_seg;
        zu_seg += zmksc[k] * cosmum_seg + zmkcs[k] * sinmum_seg;
        lu_seg += lmksc[k] * cosmum_seg + lmkcs[k] * sinmum_seg;

        rv_seg += rmkcc_n[k] * cosmu_seg + rmkss_n[k] * sinmu_seg;
        zv_seg += zmksc_n[k] * sinmu_seg + zmkcs_n[k] * cosmu_seg;
        lv_seg -= lmksc_n[k] * sinmu_seg + lmkcs_n[k] * cosmu_seg;

        r1_seg += rmkcc[k] * cosmu_seg + rmkss[k] * sinmu_seg;
        z1_seg += zmksc[k] * sinmu_seg + zmkcs[k] * cosmu_seg;

        if (nsMinF <= jF && jF < r.nsMaxFIncludingLcfs) {
          const int idx_con_base = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff;

          auto rCon_seg = Eigen::Map<Eigen::VectorXd>(
              m_geometry.rCon.data() + idx_con_base, s.nThetaReduced);
          auto zCon_seg = Eigen::Map<Eigen::VectorXd>(
              m_geometry.zCon.data() + idx_con_base, s.nThetaReduced);

          rCon_seg +=
              (rmkcc[k] * cosmu_seg + rmkss[k] * sinmu_seg) * con_factor;
          zCon_seg +=
              (zmksc[k] * sinmu_seg + zmkcs[k] * cosmu_seg) * con_factor;
        }
      }  // k
    }  // m
  }  // jF
}

// ---------------------------------------------------------------------------
// ForcesToFourier3DSymmFastPoloidalFft
// ---------------------------------------------------------------------------

void ForcesToFourier3DSymmFastPoloidalFft(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb, const ToroidalFftPlans& plans,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces) {
  // This function matches the logic of ForcesToFourier3DSymmFastPoloidal but
  // replaces the O(nZeta * ntor) toroidal scatter loop with
  // O(nZeta * log(nZeta)) MKL DFTI forward (r2c) transforms.

  m_physical_forces.setZero();

  int jMaxRZ = std::min(rp.nsMaxF, fc.ns - 1);
  if (fc.lfreeb &&
      (vacuum_pressure_state == VacuumPressureState::kInitialized ||
       vacuum_pressure_state == VacuumPressureState::kActive)) {
    jMaxRZ = std::min(rp.nsMaxF, fc.ns);
  }

  const int jMinL = 1;
  const int ntor = s.ntor;
  const int nhalf = plans.nhalf;
  const int nfp = plans.nfp;

  Eigen::VectorXd rmkcc_buf(s.nZeta), rmkss_buf(s.nZeta);
  Eigen::VectorXd rmkcc_n_buf(s.nZeta), rmkss_n_buf(s.nZeta);
  Eigen::VectorXd zmksc_buf(s.nZeta), zmkcs_buf(s.nZeta);
  Eigen::VectorXd zmksc_n_buf(s.nZeta), zmkcs_n_buf(s.nZeta);
  Eigen::VectorXd lmksc_buf(s.nZeta), lmkcs_buf(s.nZeta);
  Eigen::VectorXd lmksc_n_buf(s.nZeta), lmkcs_n_buf(s.nZeta);

  // Half-spectrum output buffers: 2*nhalf doubles = nhalf CplxBuf entries.
  std::vector<double> F_rmkcc_raw(2 * nhalf), F_rmkss_raw(2 * nhalf);
  std::vector<double> F_rmkcc_n_raw(2 * nhalf), F_rmkss_n_raw(2 * nhalf);
  std::vector<double> F_zmksc_raw(2 * nhalf), F_zmkcs_raw(2 * nhalf);
  std::vector<double> F_zmksc_n_raw(2 * nhalf), F_zmkcs_n_raw(2 * nhalf);
  std::vector<double> F_lmksc_raw(2 * nhalf), F_lmkcs_raw(2 * nhalf);
  std::vector<double> F_lmksc_n_raw(2 * nhalf), F_lmkcs_n_raw(2 * nhalf);

  auto* F_rmkcc = reinterpret_cast<CplxBuf*>(F_rmkcc_raw.data());
  auto* F_rmkss = reinterpret_cast<CplxBuf*>(F_rmkss_raw.data());
  auto* F_rmkcc_n = reinterpret_cast<CplxBuf*>(F_rmkcc_n_raw.data());
  auto* F_rmkss_n = reinterpret_cast<CplxBuf*>(F_rmkss_n_raw.data());
  auto* F_zmksc = reinterpret_cast<CplxBuf*>(F_zmksc_raw.data());
  auto* F_zmkcs = reinterpret_cast<CplxBuf*>(F_zmkcs_raw.data());
  auto* F_zmksc_n = reinterpret_cast<CplxBuf*>(F_zmksc_n_raw.data());
  auto* F_zmkcs_n = reinterpret_cast<CplxBuf*>(F_zmkcs_n_raw.data());
  auto* F_lmksc = reinterpret_cast<CplxBuf*>(F_lmksc_raw.data());
  auto* F_lmkcs = reinterpret_cast<CplxBuf*>(F_lmkcs_raw.data());
  auto* F_lmksc_n = reinterpret_cast<CplxBuf*>(F_lmksc_n_raw.data());
  auto* F_lmkcs_n = reinterpret_cast<CplxBuf*>(F_lmkcs_n_raw.data());

  for (int jF = rp.nsMinF; jF < jMaxRZ; ++jF) {
    const int mmax = (jF == 0) ? 1 : s.mpol;
    for (int m = 0; m < mmax; ++m) {
      const bool m_even = (m % 2 == 0);

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

      const int idx_ml_base = m * s.nThetaReduced;
      auto cosmui_seg = fb.cosmui.segment(idx_ml_base, s.nThetaReduced);
      auto sinmui_seg = fb.sinmui.segment(idx_ml_base, s.nThetaReduced);
      auto cosmumi_seg = fb.cosmumi.segment(idx_ml_base, s.nThetaReduced);
      auto sinmumi_seg = fb.sinmumi.segment(idx_ml_base, s.nThetaReduced);

      for (int k = 0; k < s.nZeta; ++k) {
        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;

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

        lmksc_buf[k] = blmn_seg.dot(cosmumi_seg);
        lmkcs_buf[k] = blmn_seg.dot(sinmumi_seg);
        lmkcs_n_buf[k] = -clmn_seg.dot(cosmui_seg);
        lmksc_n_buf[k] = -clmn_seg.dot(sinmui_seg);

        rmkcc_n_buf[k] = -crmn_seg.dot(cosmui_seg);
        zmkcs_n_buf[k] = -czmn_seg.dot(cosmui_seg);
        rmkss_n_buf[k] = -crmn_seg.dot(sinmui_seg);
        zmksc_n_buf[k] = -czmn_seg.dot(sinmui_seg);

        const Eigen::VectorXd tempR = (armn_seg + xmpq[m] * frcon_seg).eval();
        const Eigen::VectorXd tempZ = (azmn_seg + xmpq[m] * fzcon_seg).eval();

        rmkcc_buf[k] = tempR.dot(cosmui_seg) + brmn_seg.dot(sinmumi_seg);
        rmkss_buf[k] = tempR.dot(sinmui_seg) + brmn_seg.dot(cosmumi_seg);
        zmksc_buf[k] = tempZ.dot(sinmui_seg) + bzmn_seg.dot(cosmumi_seg);
        zmkcs_buf[k] = tempZ.dot(cosmui_seg) + bzmn_seg.dot(sinmumi_seg);
      }  // k

      // MKL forward (r2c): F[n] = Sum_k f[k]*exp(-i*2*pi*n*k/N)
      //   Re(F[n]) = Sum_k f[k]*cos(n*zeta_k)  (DCT projection)
      //   Im(F[n]) = -Sum_k f[k]*sin(n*zeta_k) (negative DST projection)
      //
      // frcc[n] += nscale[n]*[Re(F_rmkcc[n]) + n*nfp*Im(F_rmkcc_n[n])]
      // frss[n] += nscale[n]*[-Im(F_rmkss[n]) + n*nfp*Re(F_rmkss_n[n])]
      DftiComputeForward(plans.desc_r2c, rmkcc_buf.data(), F_rmkcc);
      DftiComputeForward(plans.desc_r2c, rmkss_buf.data(), F_rmkss);
      DftiComputeForward(plans.desc_r2c, rmkcc_n_buf.data(), F_rmkcc_n);
      DftiComputeForward(plans.desc_r2c, rmkss_n_buf.data(), F_rmkss_n);
      DftiComputeForward(plans.desc_r2c, zmksc_buf.data(), F_zmksc);
      DftiComputeForward(plans.desc_r2c, zmkcs_buf.data(), F_zmkcs);
      DftiComputeForward(plans.desc_r2c, zmksc_n_buf.data(), F_zmksc_n);
      DftiComputeForward(plans.desc_r2c, zmkcs_n_buf.data(), F_zmkcs_n);

      if (jMinL <= jF) {
        DftiComputeForward(plans.desc_r2c, lmksc_buf.data(), F_lmksc);
        DftiComputeForward(plans.desc_r2c, lmkcs_buf.data(), F_lmkcs);
        DftiComputeForward(plans.desc_r2c, lmksc_n_buf.data(), F_lmksc_n);
        DftiComputeForward(plans.desc_r2c, lmkcs_n_buf.data(), F_lmkcs_n);
      }

      const int ntorp1 = ntor + 1;
      const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;

      Eigen::Map<Eigen::VectorXd> frcc_seg(
          m_physical_forces.frcc.data() + idx_mn_base, ntorp1);
      Eigen::Map<Eigen::VectorXd> frss_seg(
          m_physical_forces.frss.data() + idx_mn_base, ntorp1);
      Eigen::Map<Eigen::VectorXd> fzsc_seg(
          m_physical_forces.fzsc.data() + idx_mn_base, ntorp1);
      Eigen::Map<Eigen::VectorXd> fzcs_seg(
          m_physical_forces.fzcs.data() + idx_mn_base, ntorp1);

      for (int n = 0; n <= ntor; ++n) {
        const double ns = fb.nscale[n];
        const double nfp_n = static_cast<double>(n) * nfp;

        frcc_seg[n] += ns * (F_rmkcc[n][0] + nfp_n * F_rmkcc_n[n][1]);
        frss_seg[n] += ns * (-F_rmkss[n][1] + nfp_n * F_rmkss_n[n][0]);
        fzsc_seg[n] += ns * (F_zmksc[n][0] + nfp_n * F_zmksc_n[n][1]);
        fzcs_seg[n] += ns * (-F_zmkcs[n][1] + nfp_n * F_zmkcs_n[n][0]);
      }  // n

      if (jMinL <= jF) {
        Eigen::Map<Eigen::VectorXd> flsc_seg(
            m_physical_forces.flsc.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> flcs_seg(
            m_physical_forces.flcs.data() + idx_mn_base, ntorp1);

        for (int n = 0; n <= ntor; ++n) {
          const double ns = fb.nscale[n];
          const double nfp_n = static_cast<double>(n) * nfp;

          flsc_seg[n] += ns * (F_lmksc[n][0] + nfp_n * F_lmksc_n[n][1]);
          flcs_seg[n] += ns * (-F_lmkcs[n][1] + nfp_n * F_lmkcs_n[n][0]);
        }  // n
      }  // jMinL
    }  // m
  }  // jF (main loop)

  // Lambda-only tail: jMaxRZ to nsMaxFIncludingLcfs.
  for (int jF = jMaxRZ; jF < rp.nsMaxFIncludingLcfs; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = (m % 2 == 0);

      const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
      const auto& clmn = m_even ? d.clmn_e : d.clmn_o;

      const int idx_ml_base = m * s.nThetaReduced;
      auto cosmui_seg = fb.cosmui.segment(idx_ml_base, s.nThetaReduced);
      auto sinmui_seg = fb.sinmui.segment(idx_ml_base, s.nThetaReduced);
      auto cosmumi_seg = fb.cosmumi.segment(idx_ml_base, s.nThetaReduced);
      auto sinmumi_seg = fb.sinmumi.segment(idx_ml_base, s.nThetaReduced);

      for (int k = 0; k < s.nZeta; ++k) {
        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;
        auto blmn_seg = Eigen::Map<const Eigen::VectorXd>(
            blmn.data() + idx_kl_base, s.nThetaReduced);
        auto clmn_seg = Eigen::Map<const Eigen::VectorXd>(
            clmn.data() + idx_kl_base, s.nThetaReduced);

        lmksc_buf[k] = blmn_seg.dot(cosmumi_seg);
        lmkcs_buf[k] = blmn_seg.dot(sinmumi_seg);
        lmkcs_n_buf[k] = -clmn_seg.dot(cosmui_seg);
        lmksc_n_buf[k] = -clmn_seg.dot(sinmui_seg);
      }  // k

      DftiComputeForward(plans.desc_r2c, lmksc_buf.data(), F_lmksc);
      DftiComputeForward(plans.desc_r2c, lmkcs_buf.data(), F_lmkcs);
      DftiComputeForward(plans.desc_r2c, lmksc_n_buf.data(), F_lmksc_n);
      DftiComputeForward(plans.desc_r2c, lmkcs_n_buf.data(), F_lmkcs_n);

      const int ntorp1 = ntor + 1;
      const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;

      Eigen::Map<Eigen::VectorXd> flsc_seg(
          m_physical_forces.flsc.data() + idx_mn_base, ntorp1);
      Eigen::Map<Eigen::VectorXd> flcs_seg(
          m_physical_forces.flcs.data() + idx_mn_base, ntorp1);

      for (int n = 0; n <= ntor; ++n) {
        const double ns = fb.nscale[n];
        const double nfp_n = static_cast<double>(n) * nfp;

        flsc_seg[n] += ns * (F_lmksc[n][0] + nfp_n * F_lmksc_n[n][1]);
        flcs_seg[n] += ns * (-F_lmkcs[n][1] + nfp_n * F_lmkcs_n[n][0]);
      }  // n
    }  // m
  }  // jF (lambda-only tail)
}

}  // namespace vmecpp
