// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"

// FFT path is gated on VMECPP_USE_FFTX; without it this entire TU is empty
// and IdealMhdModel uses the partial-DFT free functions instead.
#ifdef VMECPP_USE_FFTX

#include <algorithm>

#include "absl/algorithm/container.h"
#include "fftx_iprdftbat_cpu_public.h"
#include "fftx_minimal.hpp"
#include "fftx_prdftbat_cpu_public.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

// Interleaved real/imag pair
using FftComplex = double[2];

// ---------------------------------------------------------------------------
// ToroidalFftPlans
// ---------------------------------------------------------------------------

ToroidalFftPlans::ToroidalFftPlans(int n_in, int nfp_in, int mpol_in)
    : n(n_in), nhalf(n_in / 2 + 1), nfp(nfp_in), mpol(mpol_in) {
  const int full_count = kBatch * mpol;
  fftx::point_t<4> req;
  req[0] = n;
  req[1] = full_count;
  req[2] = 0;  // APar read stride
  req[3] = 0;  // APar write stride
  if (transformTuple_t* tup_c2r = fftx_iprdftbat_cpu_Tuple(req)) {
    (*tup_c2r->initfp)();
    fftx_full_c2r_run = tup_c2r->runfp;
    fftx_full_c2r_destroy = tup_c2r->destroyfp;
    std::free(tup_c2r);
  }
  if (transformTuple_t* tup_r2c = fftx_prdftbat_cpu_Tuple(req)) {
    (*tup_r2c->initfp)();
    fftx_full_r2c_run = tup_r2c->runfp;
    fftx_full_r2c_destroy = tup_r2c->destroyfp;
    std::free(tup_r2c);
  }
}

ToroidalFftPlans::~ToroidalFftPlans() {
  if (fftx_full_c2r_destroy) (*fftx_full_c2r_destroy)();
  if (fftx_full_r2c_destroy) (*fftx_full_r2c_destroy)();
}

// ---------------------------------------------------------------------------
// Internal helpers for filling the complex half-spectrum (c2r input)
// ---------------------------------------------------------------------------
//
// FFTX iprdftbat produces:
//   f[k] = X[0] + 2*Sum_{n=1}^{N/2-1}[Re(X[n])*cos(2*pi*n*k/N)
//                                       - Im(X[n])*sin(2*pi*n*k/N)]
//          + Re(X[N/2])*(-1)^k
//
// Mapping to VMEC toroidal basis (zeta_k = 2*pi*k/N, phi_n = n*zeta_k):
//   DCT (cosine synthesis): Sigma_n spec[n]*nscale[n]*cos(n*phi_k)
//     -> X[0] = spec[0]*nscale[0], X[n] = spec[n]*nscale[n]/2 (Re only)
//
//   DST (sine synthesis): Sigma_n spec[n]*nscale[n]*sin(n*phi_k)
//     -> X[0] = 0, Im(X[n]) = -spec[n]*nscale[n]/2 (Im negative)
//
// The derivative arrays in VMEC are:
//   cosnvn[k,n] = n*nfp * cos(n*zeta_k) * nscale[n]    -> DCT with n*nfp
//   factor sinnvn[k,n] = -n*nfp * sin(n*zeta_k) * nscale[n]   -> DST with
//   -n*nfp factor
//
// So the derivative of a DCT quantity  (e.g. rmkcc_n = Sigma rmncc * sinnvn):
//   rmkcc_n = Sigma_n rmncc[n]*(-n*nfp)*nscale[n]*sin(n*zeta_k)
//     -> X[0]=0, Im(X[n]) = +rmncc[n]*n*nfp*nscale[n]/2 (note: positive Im)
//
// And derivative of a DST quantity (e.g. rmkss_n = Sigma rmnss * cosnvn):
//   rmkss_n = Sigma_n rmnss[n]*(n*nfp)*nscale[n]*cos(n*zeta_k)
//     -> X[0]=0, X[n] = rmnss[n]*n*nfp*nscale[n]/2 (Re only)

namespace {

// Fill the c2r input for a DCT synthesis:
//   f[k] = Sigma_{n=0}^{ntor} spec[n] * nscale[n] * cos(n*phi_k)
// X[0] = spec[0]*nscale[0], X[n] = spec[n]*nscale[n]/2 for n=1..ntor.
inline void FillDct(const double* spec, const Eigen::VectorXd& nscale, int ntor,
                    int nhalf, FftComplex* X) {
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

// Fill the c2r input for a DST synthesis:
//   f[k] = Sigma_{n=1}^{ntor} spec[n] * nscale[n] * sin(n*phi_k)
// X[n] has Im = -spec[n]*nscale[n]/2, Re = 0.
inline void FillDst(const double* spec, const Eigen::VectorXd& nscale, int ntor,
                    int nhalf, FftComplex* X) {
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

// Fill the c2r input for the zeta-derivative of a DCT quantity:
//   f[k] = Sigma_{n=1}^{ntor} spec[n] * (-n*nfp) * nscale[n] * sin(n*phi_k)
//        = rmkcc_n (using sinnvn[k,n] = -n*nfp*sin*nscale)
// X[n] has Im = +spec[n]*n*nfp*nscale[n]/2, Re = 0.
inline void FillDctDeriv(const double* spec, const Eigen::VectorXd& nscale,
                         int ntor, int nhalf, int nfp, FftComplex* X) {
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

// Fill the c2r input for the zeta-derivative of a DST quantity:
//   f[k] = Sigma_{n=1}^{ntor} spec[n] * (n*nfp) * nscale[n] * cos(n*phi_k)
//        = rmkss_n (using cosnvn[k,n] = n*nfp*cos*nscale)
// X[n] has Re = spec[n]*n*nfp*nscale[n]/2, Im = 0.
inline void FillDstDeriv(const double* spec, const Eigen::VectorXd& nscale,
                         int ntor, int nhalf, int nfp, FftComplex* X) {
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
  // O(nZeta * log(nZeta)) FFTX c2r IFFTs.

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

  // Thread-local scratch reused across calls. Buffers are sized on first use
  // and re-sized only if dimensions grow. They are local to each thread that
  // executes this function (OpenMP threads keep their own copies).
  const int mpol = s.mpol;
  const int full_count = kBatch * mpol;
  thread_local std::vector<double> X_batch;
  thread_local std::vector<double> Y_batch;
  if (static_cast<int>(X_batch.size()) < 2 * full_count * nhalf) {
    X_batch.resize(2 * full_count * nhalf);
  }
  if (static_cast<int>(Y_batch.size()) < full_count * nZeta) {
    Y_batch.resize(full_count * nZeta);
  }
  FftComplex* X = reinterpret_cast<FftComplex*>(X_batch.data());
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
  // contiguous, then next m, ...). This batches all mpol*12 transforms
  // for one radial surface into a single FFTX call.
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
        for (int q = 0; q < kBatch; ++q) {
          FftComplex* slot = X_slot(m, q);
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

    // Single 12*mpol batched c2r for the entire surface via FFTX.
    // FFTX input does NOT get destroyed.
    plans.fftx_full_c2r_run(Y_batch.data(), X_batch.data());

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
  // This function matches the logic of ForcesToFourier3DSymmFastPoloidal but
  // replaces the O(nZeta * ntor) toroidal scatter loop with
  // O(nZeta * log(nZeta)) FFTX r2c DFTs.

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
  const int nZeta = s.nZeta;
  constexpr int kBatch = ToroidalFftPlans::kBatch;

  // Slot indices for the 12 batched r2c transforms (matching FourierToReal).
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
  if (static_cast<int>(in_batch.size()) < full_count * nZeta) {
    in_batch.resize(full_count * nZeta);
  }
  if (static_cast<int>(out_batch_real.size()) < 2 * full_count * nhalf) {
    out_batch_real.resize(2 * full_count * nhalf);
  }
  FftComplex* out_batch = reinterpret_cast<FftComplex*>(out_batch_real.data());
  // Slot index = m * kBatch + quantity (all 12 quantities of one m
  // contiguous, then next m, ...). One full batched FFT covers all m's.
  auto in_slot = [&](int m_idx, int q_idx) {
    return in_batch.data() + (m_idx * kBatch + q_idx) * nZeta;
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
      std::fill(in_batch.data() + mmax * kBatch * nZeta,
                in_batch.data() + mpol * kBatch * nZeta, 0.0);
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

      const double xmpq_m = xmpq[m];
      for (int k = 0; k < s.nZeta; ++k) {
        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;

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

        // Fused poloidal reduction, matching ForcesToFourier3DSymmFastPoloidal:
        // one pass over the short theta axis, no per-iteration heap temporary.
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_kl = idx_kl_base + l;
          const int idx_ml = idx_ml_base + l;

          const double cosmui = fb.cosmui[idx_ml];
          const double sinmui = fb.sinmui[idx_ml];
          const double cosmumi = fb.cosmumi[idx_ml];
          const double sinmumi = fb.sinmumi[idx_ml];

          lmksc += blmn[idx_kl] * cosmumi;
          lmkcs += blmn[idx_kl] * sinmumi;
          lmkcs_n -= clmn[idx_kl] * cosmui;
          lmksc_n -= clmn[idx_kl] * sinmui;

          rmkcc_n -= crmn[idx_kl] * cosmui;
          zmkcs_n -= czmn[idx_kl] * cosmui;
          rmkss_n -= crmn[idx_kl] * sinmui;
          zmksc_n -= czmn[idx_kl] * sinmui;

          const double tempR = armn[idx_kl] + xmpq_m * frcon[idx_kl];
          const double tempZ = azmn[idx_kl] + xmpq_m * fzcon[idx_kl];

          rmkcc += tempR * cosmui + brmn[idx_kl] * sinmumi;
          rmkss += tempR * sinmui + brmn[idx_kl] * cosmumi;
          zmksc += tempZ * sinmui + bzmn[idx_kl] * cosmumi;
          zmkcs += tempZ * cosmui + bzmn[idx_kl] * sinmumi;
        }  // l

        rmkcc_buf[k] = rmkcc;
        rmkss_buf[k] = rmkss;
        rmkcc_n_buf[k] = rmkcc_n;
        rmkss_n_buf[k] = rmkss_n;
        zmksc_buf[k] = zmksc;
        zmkcs_buf[k] = zmkcs;
        zmksc_n_buf[k] = zmksc_n;
        zmkcs_n_buf[k] = zmkcs_n;
        lmksc_buf[k] = lmksc;
        lmkcs_buf[k] = lmkcs;
        lmksc_n_buf[k] = lmksc_n;
        lmkcs_n_buf[k] = lmkcs_n;
      }  // k
    }  // m (fill)

    // Single 12*mpol batched r2c for the entire surface via FFTX.
    plans.fftx_full_r2c_run(out_batch_real.data(), in_batch.data());

    // === Accumulate r2c outputs into Fourier force arrays ===
    for (int m = 0; m < mmax; ++m) {
      const FftComplex* F_rmkcc = out_slot(m, kRmkcc);
      const FftComplex* F_rmkss = out_slot(m, kRmkss);
      const FftComplex* F_rmkcc_n = out_slot(m, kRmkccN);
      const FftComplex* F_rmkss_n = out_slot(m, kRmkssN);
      const FftComplex* F_zmksc = out_slot(m, kZmksc);
      const FftComplex* F_zmkcs = out_slot(m, kZmkcs);
      const FftComplex* F_zmksc_n = out_slot(m, kZmkscN);
      const FftComplex* F_zmkcs_n = out_slot(m, kZmkcsN);
      const FftComplex* F_lmksc = out_slot(m, kLmksc);
      const FftComplex* F_lmkcs = out_slot(m, kLmkcs);
      const FftComplex* F_lmksc_n = out_slot(m, kLmkscN);
      const FftComplex* F_lmkcs_n = out_slot(m, kLmkcsN);

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

      // FFTX prdftbat folds nscale[n] into its kernel output
      // (RCDiag(FList(...)) post-multiply baked into SPIRAL spec).  No
      // additional ns multiply here.
      for (int n = 0; n <= ntor; ++n) {
        const double nfp_n = static_cast<double>(n) * nfp;

        frcc_seg[n] += F_rmkcc[n][0] + nfp_n * F_rmkcc_n[n][1];
        frss_seg[n] += -F_rmkss[n][1] + nfp_n * F_rmkss_n[n][0];
        fzsc_seg[n] += F_zmksc[n][0] + nfp_n * F_zmksc_n[n][1];
        fzcs_seg[n] += -F_zmkcs[n][1] + nfp_n * F_zmkcs_n[n][0];
      }  // n

      if (jMinL <= jF) {
        Eigen::Map<Eigen::VectorXd> flsc_seg(
            m_physical_forces.flsc.data() + idx_mn_base, ntorp1);
        Eigen::Map<Eigen::VectorXd> flcs_seg(
            m_physical_forces.flcs.data() + idx_mn_base, ntorp1);

        for (int n = 0; n <= ntor; ++n) {
          const double nfp_n = static_cast<double>(n) * nfp;

          flsc_seg[n] += F_lmksc[n][0] + nfp_n * F_lmksc_n[n][1];
          flcs_seg[n] += -F_lmkcs[n][1] + nfp_n * F_lmkcs_n[n][0];
        }  // n
      }  // jMinL
    }  // m (accumulate)
  }  // jF (main loop)

  // Repeat for jMaxRZ to nsMaxFIncludingLcfs: lambda forces only.
  for (int jF = jMaxRZ; jF < rp.nsMaxFIncludingLcfs; ++jF) {
    // Fill all m's; we only read the lambda-output slots, so the other 8
    // input slots per m can stay as-is (the FFT cost is paid regardless).
    for (int m = 0; m < mpol; ++m) {
      const bool m_even = (m % 2 == 0);

      const auto& blmn = m_even ? d.blmn_e : d.blmn_o;
      const auto& clmn = m_even ? d.clmn_e : d.clmn_o;

      const int idx_ml_base = m * s.nThetaReduced;

      double* lmksc_buf = in_slot(m, kLmksc);
      double* lmkcs_buf = in_slot(m, kLmkcs);
      double* lmksc_n_buf = in_slot(m, kLmkscN);
      double* lmkcs_n_buf = in_slot(m, kLmkcsN);

      for (int k = 0; k < s.nZeta; ++k) {
        const int idx_kl_base = ((jF - rp.nsMinF) * s.nZeta + k) * s.nThetaEff;

        double lmksc = 0.0;
        double lmksc_n = 0.0;
        double lmkcs = 0.0;
        double lmkcs_n = 0.0;

        // Fused poloidal reduction (see main-loop note).
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_kl = idx_kl_base + l;
          const int idx_ml = idx_ml_base + l;

          const double cosmui = fb.cosmui[idx_ml];
          const double sinmui = fb.sinmui[idx_ml];
          const double cosmumi = fb.cosmumi[idx_ml];
          const double sinmumi = fb.sinmumi[idx_ml];

          lmksc += blmn[idx_kl] * cosmumi;
          lmkcs += blmn[idx_kl] * sinmumi;
          lmkcs_n -= clmn[idx_kl] * cosmui;
          lmksc_n -= clmn[idx_kl] * sinmui;
        }  // l

        lmksc_buf[k] = lmksc;
        lmkcs_buf[k] = lmkcs;
        lmksc_n_buf[k] = lmksc_n;
        lmkcs_n_buf[k] = lmkcs_n;
      }  // k
    }  // m (fill)

    // Single 12*mpol batched r2c (we only use the 4 lambda slots per m).
    plans.fftx_full_r2c_run(out_batch_real.data(), in_batch.data());

    for (int m = 0; m < mpol; ++m) {
      const FftComplex* F_lmksc = out_slot(m, kLmksc);
      const FftComplex* F_lmkcs = out_slot(m, kLmkcs);
      const FftComplex* F_lmksc_n = out_slot(m, kLmkscN);
      const FftComplex* F_lmkcs_n = out_slot(m, kLmkcsN);

      const int ntorp1 = ntor + 1;
      const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * ntorp1;

      Eigen::Map<Eigen::VectorXd> flsc_seg(
          m_physical_forces.flsc.data() + idx_mn_base, ntorp1);
      Eigen::Map<Eigen::VectorXd> flcs_seg(
          m_physical_forces.flcs.data() + idx_mn_base, ntorp1);

      // FFTX prdftbat folds nscale[n] into its kernel; no extra ns multiply.
      for (int n = 0; n <= ntor; ++n) {
        const double nfp_n = static_cast<double>(n) * nfp;

        flsc_seg[n] += F_lmksc[n][0] + nfp_n * F_lmksc_n[n][1];
        flcs_seg[n] += -F_lmkcs[n][1] + nfp_n * F_lmkcs_n[n][0];
      }  // n
    }  // m (accumulate)
  }  // jF (lambda-only tail)
}

}  // namespace vmecpp

#endif  // VMECPP_USE_FFTX
