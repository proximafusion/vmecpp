// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"

#include <mkl_dfti.h>

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

// ---------------------------------------------------------------------------
// ToroidalFftPlans
// ---------------------------------------------------------------------------
//
// MKL DFTI 1-D real transforms with CCE (complex conjugate-even) packing.
// CCE layout (2*nhalf doubles per spectrum) is byte-identical to the FFTW
// half-complex layout: [re0, 0, re1, im1, ..., re(N/2), 0]. This lets the
// fill helpers and the accumulation code address the half-spectrum as
// CplxBuf (double[2]) per bin.
//
// Batched transforms are configured via:
//   DFTI_NUMBER_OF_TRANSFORMS = howmany
//   DFTI_INPUT_DISTANCE / DFTI_OUTPUT_DISTANCE = stride between batches
// Real-domain spectra (CCE) are 2*nhalf doubles long, real-domain signals
// are n doubles long.

ToroidalFftPlans::ToroidalFftPlans(int n_in, int nfp_in, int mpol_in)
    : n(n_in), nhalf(n_in / 2 + 1), nfp(nfp_in), mpol(mpol_in) {
  // Single-transform descriptor (c2r: backward, CCE -> real).
  DftiCreateDescriptor(&desc_c2r, DFTI_DOUBLE, DFTI_REAL, 1,
                       static_cast<MKL_LONG>(n));
  DftiSetValue(desc_c2r, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_CCE_FORMAT);
  DftiSetValue(desc_c2r, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiCommitDescriptor(desc_c2r);

  // Single-transform descriptor (r2c: forward, real -> CCE).
  DftiCreateDescriptor(&desc_r2c, DFTI_DOUBLE, DFTI_REAL, 1,
                       static_cast<MKL_LONG>(n));
  DftiSetValue(desc_r2c, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_CCE_FORMAT);
  DftiSetValue(desc_r2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiCommitDescriptor(desc_r2c);

  // Per-surface batched descriptors: 12 * mpol transforms in one call.
  // For c2r the input is CCE (2*nhalf doubles per spectrum) and the output
  // is real (n doubles per signal). For r2c the directions swap.
  const MKL_LONG full_count = static_cast<MKL_LONG>(kBatch * mpol);
  const MKL_LONG cce_dist = static_cast<MKL_LONG>(2 * nhalf);
  const MKL_LONG real_dist = static_cast<MKL_LONG>(n);

  DftiCreateDescriptor(&desc_full_c2r, DFTI_DOUBLE, DFTI_REAL, 1,
                       static_cast<MKL_LONG>(n));
  DftiSetValue(desc_full_c2r, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_CCE_FORMAT);
  DftiSetValue(desc_full_c2r, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiSetValue(desc_full_c2r, DFTI_NUMBER_OF_TRANSFORMS, full_count);
  DftiSetValue(desc_full_c2r, DFTI_INPUT_DISTANCE, cce_dist);
  DftiSetValue(desc_full_c2r, DFTI_OUTPUT_DISTANCE, real_dist);
  DftiCommitDescriptor(desc_full_c2r);

  DftiCreateDescriptor(&desc_full_r2c, DFTI_DOUBLE, DFTI_REAL, 1,
                       static_cast<MKL_LONG>(n));
  DftiSetValue(desc_full_r2c, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_CCE_FORMAT);
  DftiSetValue(desc_full_r2c, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiSetValue(desc_full_r2c, DFTI_NUMBER_OF_TRANSFORMS, full_count);
  DftiSetValue(desc_full_r2c, DFTI_INPUT_DISTANCE, real_dist);
  DftiSetValue(desc_full_r2c, DFTI_OUTPUT_DISTANCE, cce_dist);
  DftiCommitDescriptor(desc_full_r2c);
}

ToroidalFftPlans::~ToroidalFftPlans() {
  DftiFreeDescriptor(&desc_c2r);
  DftiFreeDescriptor(&desc_r2c);
  DftiFreeDescriptor(&desc_full_c2r);
  DftiFreeDescriptor(&desc_full_r2c);
}

// ---------------------------------------------------------------------------
// Internal helpers for filling the half-spectrum (c2r input)
// ---------------------------------------------------------------------------
//
// MKL backward (c2r) produces:
//   f[k] = X[0] + 2*Sum_{n=1}^{N/2-1}[Re(X[n])*cos(2*pi*n*k/N)
//                                       - Im(X[n])*sin(2*pi*n*k/N)]
//          + Re(X[N/2])*(-1)^k
//
// Mapping to VMEC toroidal basis (zeta_k = 2*pi*k/N):
//   DCT: f[k] = Sigma_n spec[n]*nscale[n]*cos(n*phi_k)
//     -> X[0][0] = spec[0]*nscale[0], X[n][0] = spec[n]*nscale[n]/2 (n>=1)
//   DST: f[k] = Sigma_n spec[n]*nscale[n]*sin(n*phi_k)
//     -> X[n][1] = -spec[n]*nscale[n]/2  (Im negative)
//   DCT-deriv (= sinnvn contraction):
//     -> X[n][1] = +spec[n]*n*nfp*nscale[n]/2
//   DST-deriv (= cosnvn contraction):
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
  // Replaces the O(nZeta * ntor) inner-n dot-product loop with one
  // 12*mpol-wide MKL DFTI batched backward (c2r) transform per surface.

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
  const int nZeta = s.nZeta;
  constexpr int kBatch = ToroidalFftPlans::kBatch;

  const int mpol = s.mpol;
  const int full_count = kBatch * mpol;

  // Thread-local scratch reused across calls.
  // X_batch holds 12*mpol contiguous half-spectra (each 2*nhalf doubles).
  // Y_batch holds 12*mpol contiguous real signals (each nZeta doubles).
  thread_local std::vector<double> X_batch;
  thread_local std::vector<double> Y_batch;
  if (static_cast<int>(X_batch.size()) < 2 * full_count * nhalf) {
    X_batch.resize(2 * full_count * nhalf);
  }
  if (static_cast<int>(Y_batch.size()) < full_count * nZeta) {
    Y_batch.resize(full_count * nZeta);
  }
  CplxBuf* X = reinterpret_cast<CplxBuf*>(X_batch.data());

  // Slot indices for the 12 quantities transformed per m.
  enum Slot {
    kRmkcc = 0,
    kRmkss = 1,
    kRmkccN = 2,
    kRmkssN = 3,
    kZmksc = 4,
    kZmkcs = 5,
    kZmkscN = 6,
    kZmkcsN = 7,
    kLmksc = 8,
    kLmkcs = 9,
    kLmkscN = 10,
    kLmkcsN = 11,
  };
  // Slot index = m * kBatch + quantity (all 12 quantities of one m
  // contiguous, then next m, ...). The batched FFT handles all m's.
  auto X_slot = [&](int m_idx, int q_idx) {
    return X + (m_idx * kBatch + q_idx) * nhalf;
  };
  auto Y_slot = [&](int m_idx, int q_idx) {
    return Y_batch.data() + (m_idx * kBatch + q_idx) * nZeta;
  };

  for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {
    // === Pack all 12*mpol half-spectra for this surface ===
    for (int m = 0; m < mpol; ++m) {
      const int jMin = (m == 0 || m == 1) ? 0 : 1;
      if (jF < jMin) {
        // Zero out spectra so the batched FFT produces zeros for this m.
        // (We still skip the accumulation step below for these m's.)
        for (int q = 0; q < kBatch; ++q) {
          CplxBuf* slot = X_slot(m, q);
          for (int n = 0; n < nhalf; ++n) {
            slot[n][0] = 0.0;
            slot[n][1] = 0.0;
          }
        }
        continue;
      }

      const int idx_mn_base = ((jF - nsMinF1) * s.mpol + m) * (ntor + 1);

      const double* rmncc_ptr = physical_x.rmncc.data() + idx_mn_base;
      const double* rmnss_ptr = physical_x.rmnss.data() + idx_mn_base;
      const double* zmnsc_ptr = physical_x.zmnsc.data() + idx_mn_base;
      const double* zmncs_ptr = physical_x.zmncs.data() + idx_mn_base;
      const double* lmnsc_ptr = physical_x.lmnsc.data() + idx_mn_base;
      const double* lmncs_ptr = physical_x.lmncs.data() + idx_mn_base;

      FillDct(rmncc_ptr, fb.nscale, ntor, nhalf, X_slot(m, kRmkcc));
      FillDst(rmnss_ptr, fb.nscale, ntor, nhalf, X_slot(m, kRmkss));
      FillDctDeriv(rmncc_ptr, fb.nscale, ntor, nhalf, nfp, X_slot(m, kRmkccN));
      FillDstDeriv(rmnss_ptr, fb.nscale, ntor, nhalf, nfp, X_slot(m, kRmkssN));
      FillDct(zmnsc_ptr, fb.nscale, ntor, nhalf, X_slot(m, kZmksc));
      FillDst(zmncs_ptr, fb.nscale, ntor, nhalf, X_slot(m, kZmkcs));
      FillDctDeriv(zmnsc_ptr, fb.nscale, ntor, nhalf, nfp, X_slot(m, kZmkscN));
      FillDstDeriv(zmncs_ptr, fb.nscale, ntor, nhalf, nfp, X_slot(m, kZmkcsN));
      FillDct(lmnsc_ptr, fb.nscale, ntor, nhalf, X_slot(m, kLmksc));
      FillDst(lmncs_ptr, fb.nscale, ntor, nhalf, X_slot(m, kLmkcs));
      FillDctDeriv(lmnsc_ptr, fb.nscale, ntor, nhalf, nfp, X_slot(m, kLmkscN));
      FillDstDeriv(lmncs_ptr, fb.nscale, ntor, nhalf, nfp, X_slot(m, kLmkcsN));
    }

    // Single 12*mpol batched c2r for the entire surface.
    DftiComputeBackward(plans.desc_full_c2r, X_batch.data(), Y_batch.data());

    // === Poloidal accumulation per m ===
    for (int m = 0; m < mpol; ++m) {
      const int jMin = (m == 0 || m == 1) ? 0 : 1;
      if (jF < jMin) {
        continue;
      }

      const bool m_even = (m % 2 == 0);
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

      const double* rmkcc = Y_slot(m, kRmkcc);
      const double* rmkss = Y_slot(m, kRmkss);
      const double* rmkcc_n = Y_slot(m, kRmkccN);
      const double* rmkss_n = Y_slot(m, kRmkssN);
      const double* zmksc = Y_slot(m, kZmksc);
      const double* zmkcs = Y_slot(m, kZmkcs);
      const double* zmksc_n = Y_slot(m, kZmkscN);
      const double* zmkcs_n = Y_slot(m, kZmkcsN);
      const double* lmksc = Y_slot(m, kLmksc);
      const double* lmkcs = Y_slot(m, kLmkcs);
      const double* lmksc_n = Y_slot(m, kLmkscN);
      const double* lmkcs_n = Y_slot(m, kLmkcsN);

      const int idx_ml_base = m * s.nThetaReduced;

      auto sinmum_seg = fb.sinmum.segment(idx_ml_base, s.nThetaReduced);
      auto cosmum_seg = fb.cosmum.segment(idx_ml_base, s.nThetaReduced);
      auto cosmu_seg = fb.cosmu.segment(idx_ml_base, s.nThetaReduced);
      auto sinmu_seg = fb.sinmu.segment(idx_ml_base, s.nThetaReduced);

      for (int k = 0; k < nZeta; ++k) {
        const int idx_kl_base = ((jF - nsMinF1) * nZeta + k) * s.nThetaEff;

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
          const int idx_con_base = ((jF - nsMinF) * nZeta + k) * s.nThetaEff;

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
  // Replaces the O(nZeta * ntor) toroidal scatter loop with one
  // 12*mpol-wide MKL DFTI batched forward (r2c) transform per surface.

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
  constexpr int kBatch = ToroidalFftPlans::kBatch;

  enum Slot {
    kRmkcc = 0,
    kRmkss = 1,
    kRmkccN = 2,
    kRmkssN = 3,
    kZmksc = 4,
    kZmkcs = 5,
    kZmkscN = 6,
    kZmkcsN = 7,
    kLmksc = 8,
    kLmkcs = 9,
    kLmkscN = 10,
    kLmkcsN = 11,
  };

  const int mpol = s.mpol;
  const int full_count = kBatch * mpol;

  // Thread-local scratch reused across calls.
  thread_local std::vector<double> in_batch;
  thread_local std::vector<double> out_batch_real;
  if (static_cast<int>(in_batch.size()) < full_count * s.nZeta) {
    in_batch.resize(full_count * s.nZeta);
  }
  if (static_cast<int>(out_batch_real.size()) < 2 * full_count * nhalf) {
    out_batch_real.resize(2 * full_count * nhalf);
  }
  CplxBuf* out_batch = reinterpret_cast<CplxBuf*>(out_batch_real.data());

  auto in_slot = [&](int m_idx, int q_idx) {
    return in_batch.data() + (m_idx * kBatch + q_idx) * s.nZeta;
  };
  auto out_slot = [&](int m_idx, int q_idx) {
    return out_batch + (m_idx * kBatch + q_idx) * nhalf;
  };

  for (int jF = rp.nsMinF; jF < jMaxRZ; ++jF) {
    const int mmax = (jF == 0) ? 1 : s.mpol;

    // === Fill all input slots for this surface ===
    // For m >= mmax (axis case), zero the inputs so the batched FFT
    // produces zeros; the corresponding outputs are skipped on accumulation.
    if (mmax < mpol) {
      std::fill(in_batch.data() + mmax * kBatch * s.nZeta,
                in_batch.data() + mpol * kBatch * s.nZeta, 0.0);
    }
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

      double* rmkcc_buf = in_slot(m, kRmkcc);
      double* rmkss_buf = in_slot(m, kRmkss);
      double* rmkcc_n_buf = in_slot(m, kRmkccN);
      double* rmkss_n_buf = in_slot(m, kRmkssN);
      double* zmksc_buf = in_slot(m, kZmksc);
      double* zmkcs_buf = in_slot(m, kZmkcs);
      double* zmksc_n_buf = in_slot(m, kZmkscN);
      double* zmkcs_n_buf = in_slot(m, kZmkcsN);
      double* lmksc_buf = in_slot(m, kLmksc);
      double* lmkcs_buf = in_slot(m, kLmkcs);
      double* lmksc_n_buf = in_slot(m, kLmkscN);
      double* lmkcs_n_buf = in_slot(m, kLmkcsN);

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
    }  // m (fill)

    // Single 12*mpol batched r2c for the entire surface.
    DftiComputeForward(plans.desc_full_r2c, in_batch.data(),
                       out_batch_real.data());

    // === Accumulate r2c outputs into Fourier force arrays ===
    for (int m = 0; m < mmax; ++m) {
      const CplxBuf* F_rmkcc = out_slot(m, kRmkcc);
      const CplxBuf* F_rmkss = out_slot(m, kRmkss);
      const CplxBuf* F_rmkcc_n = out_slot(m, kRmkccN);
      const CplxBuf* F_rmkss_n = out_slot(m, kRmkssN);
      const CplxBuf* F_zmksc = out_slot(m, kZmksc);
      const CplxBuf* F_zmkcs = out_slot(m, kZmkcs);
      const CplxBuf* F_zmksc_n = out_slot(m, kZmkscN);
      const CplxBuf* F_zmkcs_n = out_slot(m, kZmkcsN);
      const CplxBuf* F_lmksc = out_slot(m, kLmksc);
      const CplxBuf* F_lmkcs = out_slot(m, kLmkcs);
      const CplxBuf* F_lmksc_n = out_slot(m, kLmkscN);
      const CplxBuf* F_lmkcs_n = out_slot(m, kLmkcsN);

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
    }  // m (accumulate)
  }  // jF (main loop)

  // Lambda-only tail: jMaxRZ to nsMaxFIncludingLcfs.
  for (int jF = jMaxRZ; jF < rp.nsMaxFIncludingLcfs; ++jF) {
    // Fill all m's; we only read the lambda-output slots, so the other 8
    // input slots per m can stay as-is (the FFT cost is paid regardless).
    for (int m = 0; m < mpol; ++m) {
      const bool m_even = (m % 2 == 0);

      const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
      const auto& clmn = m_even ? d.clmn_e : d.clmn_o;

      const int idx_ml_base = m * s.nThetaReduced;
      auto cosmui_seg = fb.cosmui.segment(idx_ml_base, s.nThetaReduced);
      auto sinmui_seg = fb.sinmui.segment(idx_ml_base, s.nThetaReduced);
      auto cosmumi_seg = fb.cosmumi.segment(idx_ml_base, s.nThetaReduced);
      auto sinmumi_seg = fb.sinmumi.segment(idx_ml_base, s.nThetaReduced);

      double* lmksc_buf = in_slot(m, kLmksc);
      double* lmkcs_buf = in_slot(m, kLmkcs);
      double* lmksc_n_buf = in_slot(m, kLmkscN);
      double* lmkcs_n_buf = in_slot(m, kLmkcsN);

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
    }  // m (fill)

    // Single 12*mpol batched r2c (we only use the 4 lambda slots per m).
    DftiComputeForward(plans.desc_full_r2c, in_batch.data(),
                       out_batch_real.data());

    for (int m = 0; m < mpol; ++m) {
      const CplxBuf* F_lmksc = out_slot(m, kLmksc);
      const CplxBuf* F_lmkcs = out_slot(m, kLmkcs);
      const CplxBuf* F_lmksc_n = out_slot(m, kLmkscN);
      const CplxBuf* F_lmkcs_n = out_slot(m, kLmkcsN);

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
    }  // m (accumulate)
  }  // jF (lambda-only tail)
}

}  // namespace vmecpp
