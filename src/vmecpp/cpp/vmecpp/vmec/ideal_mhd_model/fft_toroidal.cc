// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <span>

#include "absl/algorithm/container.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

// ---------------------------------------------------------------------------
// ToroidalFftPlans
// ---------------------------------------------------------------------------

ToroidalFftPlans::ToroidalFftPlans(int n_in, int nfp_in)
    : n(n_in), nhalf(n_in / 2 + 1), nfp(nfp_in) {
  // Allocate temporary aligned buffers for plan creation.
  // Plans are safe to execute concurrently from multiple threads with separate
  // input/output buffers (via fftw_execute_dft_c2r / fftw_execute_dft_r2c).
  fftw_complex* in_c2r =
      static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nhalf));
  double* out_c2r = static_cast<double*>(fftw_malloc(sizeof(double) * n));

  plan_c2r = fftw_plan_dft_c2r_1d(n, in_c2r, out_c2r, FFTW_ESTIMATE);

  double* in_r2c = static_cast<double*>(fftw_malloc(sizeof(double) * n));
  fftw_complex* out_r2c =
      static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * nhalf));

  plan_r2c = fftw_plan_dft_r2c_1d(n, in_r2c, out_r2c, FFTW_ESTIMATE);

  fftw_free(in_c2r);
  fftw_free(out_c2r);
  fftw_free(in_r2c);
  fftw_free(out_r2c);
}

ToroidalFftPlans::~ToroidalFftPlans() {
  fftw_destroy_plan(plan_c2r);
  fftw_destroy_plan(plan_r2c);
}

// ---------------------------------------------------------------------------
// Internal helpers for filling the complex half-spectrum (c2r input)
// ---------------------------------------------------------------------------
//
// FFTW c2r produces:
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
//   cosnvn[k,n] = n*nfp * cos(n*zeta_k) * nscale[n]    -> DCT with n*nfp factor
//   sinnvn[k,n] = -n*nfp * sin(n*zeta_k) * nscale[n]   -> DST with -n*nfp
//   factor
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
                    int nhalf, fftw_complex* X) {
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
                    int nhalf, fftw_complex* X) {
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
                         int ntor, int nhalf, int nfp, fftw_complex* X) {
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
                         int ntor, int nhalf, int nfp, fftw_complex* X) {
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

// fftw_complex is double[2], a C array type that is not copy-constructible and
// therefore incompatible with std::vector on libc++ (macOS). Use fftw_malloc
// instead to get properly aligned storage and wrap it with a custom deleter.
struct FftwComplexBuffer {
  explicit FftwComplexBuffer(int n)
      : ptr(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n))) {
  }
  ~FftwComplexBuffer() { fftw_free(ptr); }
  FftwComplexBuffer(const FftwComplexBuffer&) = delete;
  FftwComplexBuffer& operator=(const FftwComplexBuffer&) = delete;
  fftw_complex* data() { return ptr; }
  fftw_complex& operator[](int i) { return ptr[i]; }

 private:
  fftw_complex* ptr;
};

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
  // O(nZeta * log(nZeta)) FFTW c2r IFFTs.

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

  // Per-thread scratch buffers for FFTW input/output.
  // Allocated here (once per call) to avoid repeated heap allocation in the
  // inner loop and to keep memory usage predictable.
  FftwComplexBuffer X(nhalf);

  // Per-(jF, m) toroidal profiles (size nZeta each).
  Eigen::VectorXd rmkcc(s.nZeta), rmkss(s.nZeta);
  Eigen::VectorXd rmkcc_n(s.nZeta), rmkss_n(s.nZeta);
  Eigen::VectorXd zmksc(s.nZeta), zmkcs(s.nZeta);
  Eigen::VectorXd zmksc_n(s.nZeta), zmkcs_n(s.nZeta);
  Eigen::VectorXd lmksc(s.nZeta), lmkcs(s.nZeta);
  Eigen::VectorXd lmksc_n(s.nZeta), lmkcs_n(s.nZeta);

  for (int jF = nsMinF1; jF < r.nsMaxF1; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = (m % 2 == 0);

      // Axis only gets contributions up to m=1.
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

      // === Compute toroidal profiles via c2r IFFTs ===

      // rmkcc[k] = Sum_n rmncc[n]*nscale[n]*cos(n*zeta_k)
      FillDct(rmncc_ptr, fb.nscale, ntor, nhalf, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), rmkcc.data());

      // rmkss[k] = Sum_n rmnss[n]*nscale[n]*sin(n*zeta_k)
      FillDst(rmnss_ptr, fb.nscale, ntor, nhalf, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), rmkss.data());

      // rmkcc_n[k] = Sum_n rmncc[n]*(-n*nfp)*nscale[n]*sin(n*zeta_k)
      //            (= rmncc_seg.dot(sinnvn_seg) in the DFT version)
      FillDctDeriv(rmncc_ptr, fb.nscale, ntor, nhalf, nfp, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), rmkcc_n.data());

      // rmkss_n[k] = Sum_n rmnss[n]*(n*nfp)*nscale[n]*cos(n*zeta_k)
      //            (= rmnss_seg.dot(cosnvn_seg) in the DFT version)
      FillDstDeriv(rmnss_ptr, fb.nscale, ntor, nhalf, nfp, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), rmkss_n.data());

      // zmksc[k] = Sum_n zmnsc[n]*nscale[n]*cos(n*zeta_k)
      FillDct(zmnsc_ptr, fb.nscale, ntor, nhalf, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), zmksc.data());

      // zmkcs[k] = Sum_n zmncs[n]*nscale[n]*sin(n*zeta_k)
      FillDst(zmncs_ptr, fb.nscale, ntor, nhalf, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), zmkcs.data());

      // zmksc_n[k] = Sum_n zmnsc[n]*(-n*nfp)*nscale[n]*sin(n*zeta_k)
      FillDctDeriv(zmnsc_ptr, fb.nscale, ntor, nhalf, nfp, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), zmksc_n.data());

      // zmkcs_n[k] = Sum_n zmncs[n]*(n*nfp)*nscale[n]*cos(n*zeta_k)
      FillDstDeriv(zmncs_ptr, fb.nscale, ntor, nhalf, nfp, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), zmkcs_n.data());

      // lmksc[k] = Sum_n lmnsc[n]*nscale[n]*cos(n*zeta_k)
      FillDct(lmnsc_ptr, fb.nscale, ntor, nhalf, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), lmksc.data());

      // lmkcs[k] = Sum_n lmncs[n]*nscale[n]*sin(n*zeta_k)
      FillDst(lmncs_ptr, fb.nscale, ntor, nhalf, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), lmkcs.data());

      // lmksc_n[k] = Sum_n lmnsc[n]*(-n*nfp)*nscale[n]*sin(n*zeta_k)
      FillDctDeriv(lmnsc_ptr, fb.nscale, ntor, nhalf, nfp, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), lmksc_n.data());

      // lmkcs_n[k] = Sum_n lmncs[n]*(n*nfp)*nscale[n]*cos(n*zeta_k)
      FillDstDeriv(lmncs_ptr, fb.nscale, ntor, nhalf, nfp, X.data());
      fftw_execute_dft_c2r(plans.plan_c2r, X.data(), lmkcs_n.data());

      // === Poloidal accumulation (identical to
      // FourierToReal3DSymmFastPoloidal) ===
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
  // O(nZeta * log(nZeta)) FFTW r2c DFTs.

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

  // Per-thread scratch buffers.
  // For each (jF, m) we accumulate the k-profiles of the poloidal integrals
  // into these, then do the r2c FFT to project onto toroidal modes.
  Eigen::VectorXd rmkcc_buf(s.nZeta), rmkss_buf(s.nZeta);
  Eigen::VectorXd rmkcc_n_buf(s.nZeta), rmkss_n_buf(s.nZeta);
  Eigen::VectorXd zmksc_buf(s.nZeta), zmkcs_buf(s.nZeta);
  Eigen::VectorXd zmksc_n_buf(s.nZeta), zmkcs_n_buf(s.nZeta);
  Eigen::VectorXd lmksc_buf(s.nZeta), lmkcs_buf(s.nZeta);
  Eigen::VectorXd lmksc_n_buf(s.nZeta), lmkcs_n_buf(s.nZeta);

  // FFT output: half-complex spectrum for each k-profile.
  FftwComplexBuffer F_rmkcc(nhalf), F_rmkss(nhalf);
  FftwComplexBuffer F_rmkcc_n(nhalf), F_rmkss_n(nhalf);
  FftwComplexBuffer F_zmksc(nhalf), F_zmkcs(nhalf);
  FftwComplexBuffer F_zmksc_n(nhalf), F_zmkcs_n(nhalf);
  FftwComplexBuffer F_lmksc(nhalf), F_lmkcs(nhalf);
  FftwComplexBuffer F_lmksc_n(nhalf), F_lmkcs_n(nhalf);

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

      // Poloidal-integration step: for each k, compute the scalar projections
      // (identical to the DFT version) and store in the k-profile buffers.
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

      // === Toroidal projection via r2c FFTs ===
      //
      // The DFT version used:
      //   frcc[n] += rmkcc(k)*cosnv[k,n] + rmkcc_n(k)*sinnvn[k,n]
      // where cosnv[k,n] = cos(n*zeta_k)*nscale[n]
      //   and sinnvn[k,n] = -n*nfp*sin(n*zeta_k)*nscale[n].
      //
      // r2c FFT: F[n] = Sum_k f[k]*exp(-i*2*pi*n*k/N)
      //   Re(F[n]) = Sum_k f[k]*cos(n*zeta_k)  = DCT(f)[n]
      //   Im(F[n]) = -Sum_k f[k]*sin(n*zeta_k) = -DST(f)[n]
      //   -> DST(f)[n] = -Im(F[n])
      //
      // Therefore:
      //   Sum_k rmkcc(k)*cos(n*zeta_k) = Re(r2c(rmkcc))[n]
      //   Sum_k rmkcc_n(k)*(-n*nfp)*sin(n*zeta_k)
      //     = (-n*nfp)*(-Im(r2c(rmkcc_n))[n]) = n*nfp*Im(r2c(rmkcc_n))[n]
      //
      // So: frcc[n] += nscale[n] * [Re(r2c(rmkcc))[n]
      //                             + n*nfp * Im(r2c(rmkcc_n))[n]]

      fftw_execute_dft_r2c(plans.plan_r2c, rmkcc_buf.data(), F_rmkcc.data());
      fftw_execute_dft_r2c(plans.plan_r2c, rmkss_buf.data(), F_rmkss.data());
      fftw_execute_dft_r2c(plans.plan_r2c, rmkcc_n_buf.data(),
                           F_rmkcc_n.data());
      fftw_execute_dft_r2c(plans.plan_r2c, rmkss_n_buf.data(),
                           F_rmkss_n.data());
      fftw_execute_dft_r2c(plans.plan_r2c, zmksc_buf.data(), F_zmksc.data());
      fftw_execute_dft_r2c(plans.plan_r2c, zmkcs_buf.data(), F_zmkcs.data());
      fftw_execute_dft_r2c(plans.plan_r2c, zmksc_n_buf.data(),
                           F_zmksc_n.data());
      fftw_execute_dft_r2c(plans.plan_r2c, zmkcs_n_buf.data(),
                           F_zmkcs_n.data());

      if (jMinL <= jF) {
        fftw_execute_dft_r2c(plans.plan_r2c, lmksc_buf.data(), F_lmksc.data());
        fftw_execute_dft_r2c(plans.plan_r2c, lmkcs_buf.data(), F_lmkcs.data());
        fftw_execute_dft_r2c(plans.plan_r2c, lmksc_n_buf.data(),
                             F_lmksc_n.data());
        fftw_execute_dft_r2c(plans.plan_r2c, lmkcs_n_buf.data(),
                             F_lmkcs_n.data());
      }

      // Accumulate r2c outputs into the Fourier force coefficient arrays.
      // FFTW r2c gives the unnormalized DFT: F[n] = Sum_k
      // f[k]*exp(-i*2*pi*n*k/N). This matches exactly what the DFT k-loop
      // computes:
      //   Sum_k rmkcc(k)*cosnv(k,n) = nscale[n] * Sum_k rmkcc(k)*cos(n*zeta_k)
      //                             = nscale[n] * Re(r2c(rmkcc)[n])
      // No division by N is needed.
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

      // frcc[n] += nscale[n] * [DCT(rmkcc)[n] + n*nfp * DST(rmkcc_n)[n]]
      //          + nscale[n] * [n*nfp * DCT(rmkss_n)[n] - DST(rmkss)[n]
      //                        * n*nfp / n*nfp ... no, let me re-derive.
      //
      // From the DFT version:
      //   frcc_seg += rmkcc * cosnv_seg + rmkcc_n * sinnvn_seg
      // where cosnv[k,n] = cos(n*zeta_k)*nscale[n],
      //       sinnvn[k,n] = -n*nfp*sin(n*zeta_k)*nscale[n]
      //
      //   frcc[n] += nscale[n]*Sum_k rmkcc(k)*cos(n*zeta_k)
      //              + nscale[n]*(-n*nfp)*Sum_k rmkcc_n(k)*sin(n*zeta_k)
      //            = nscale[n]*(DCT(rmkcc)[n] + n*nfp*Im(r2c(rmkcc_n))[n])
      //   (using -Im(r2c(f)[n]) = DST(f)[n], so Sum_k f*sin = -Im(r2c(f)[n]),
      //    and frcc gets n*nfp * (-(-Im)) = n*nfp * Im)
      //
      //   frss[n] += nscale[n]*Sum_k rmkss(k)*sin(n*zeta_k)
      //              + nscale[n]*(n*nfp)*Sum_k rmkss_n(k)*cos(n*zeta_k)
      //            = nscale[n]*(-Im(r2c(rmkss))[n] + n*nfp*Re(r2c(rmkss_n))[n])
      for (int n = 0; n <= ntor; ++n) {
        const double ns = fb.nscale[n];
        const double nfp_n = static_cast<double>(n) * nfp;

        // frcc[n] += nscale[n] * [Re(r2c(rmkcc))[n] + n*nfp *
        // Im(r2c(rmkcc_n))[n]]
        frcc_seg[n] += ns * (F_rmkcc[n][0] + nfp_n * F_rmkcc_n[n][1]);

        // frss[n] += nscale[n] * [-Im(r2c(rmkss))[n] + n*nfp *
        // Re(r2c(rmkss_n))[n]]
        frss_seg[n] += ns * (-F_rmkss[n][1] + nfp_n * F_rmkss_n[n][0]);

        // fzsc[n] += nscale[n] * [Re(r2c(zmksc))[n] + n*nfp *
        // Im(r2c(zmksc_n))[n]]
        fzsc_seg[n] += ns * (F_zmksc[n][0] + nfp_n * F_zmksc_n[n][1]);

        // fzcs[n] += nscale[n] * [-Im(r2c(zmkcs))[n] + n*nfp *
        // Re(r2c(zmkcs_n))[n]]
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

          // flsc[n] += nscale[n] * [Re(r2c(lmksc))[n] + n*nfp *
          // Im(r2c(lmksc_n))[n]]
          flsc_seg[n] += ns * (F_lmksc[n][0] + nfp_n * F_lmksc_n[n][1]);

          // flcs[n] += nscale[n] * [-Im(r2c(lmkcs))[n] + n*nfp *
          // Re(r2c(lmkcs_n))[n]]
          flcs_seg[n] += ns * (-F_lmkcs[n][1] + nfp_n * F_lmkcs_n[n][0]);
        }  // n
      }  // jMinL
    }  // m
  }  // jF (main loop)

  // Repeat for jMaxRZ to nsMaxFIncludingLcfs: lambda forces only.
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

      fftw_execute_dft_r2c(plans.plan_r2c, lmksc_buf.data(), F_lmksc.data());
      fftw_execute_dft_r2c(plans.plan_r2c, lmkcs_buf.data(), F_lmkcs.data());
      fftw_execute_dft_r2c(plans.plan_r2c, lmksc_n_buf.data(),
                           F_lmksc_n.data());
      fftw_execute_dft_r2c(plans.plan_r2c, lmkcs_n_buf.data(),
                           F_lmkcs_n.data());

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
