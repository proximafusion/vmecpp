#include "vmecpp/vmec/ideal_mhd_model/vectorized_dft_functions.h"

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"

void vmecpp::ForcesToFourier3DSymmFastPoloidal_avx(
    const RealSpaceForces& d, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb, int ivac,
    FourierForces& physical_forces) {
  // in here, we can safely assume lthreed == true

  // fill target force arrays with zeros
  physical_forces.setZero();

  int jMaxRZ = std::min(rp.nsMaxF, fc.ns - 1);

  if (fc.lfreeb && ivac >= 1) {
    // free-boundary: up to jMaxRZ=ns
    jMaxRZ = std::min(rp.nsMaxF, fc.ns);
  }

  const double* p_cosmui = fb.cosmui.data();
  const double* p_sinmui = fb.sinmui.data();
  const double* p_cosmumi = fb.cosmumi.data();
  const double* p_sinmumi = fb.sinmumi.data();
  const double* p_cosnv = fb.cosnv.data();
  const double* p_sinnv = fb.sinnv.data();
  const double* p_cosnvn = fb.cosnvn.data();
  const double* p_sinnvn = fb.sinnvn.data();
  double* p_frcc = physical_forces.frcc.data();
  double* p_frss = physical_forces.frss.data();
  double* p_fzsc = physical_forces.fzsc.data();
  double* p_fzcs = physical_forces.fzcs.data();
  double* p_flsc = physical_forces.flsc.data();
  double* p_flcs = physical_forces.flcs.data();

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

      const __m256d _xmpq = _mm256_set1_pd(xmpq[m]);

      const double* p_armn = armn.data();
      const double* p_azmn = azmn.data();
      const double* p_blmn = blmn.data();
      const double* p_brmn = brmn.data();
      const double* p_bzmn = bzmn.data();
      const double* p_clmn = clmn.data();
      const double* p_crmn = crmn.data();
      const double* p_czmn = czmn.data();
      const double* p_frcon = frcon.data();
      const double* p_fzcon = fzcon.data();

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
        for (int l = VECTORSIZE * (s.nThetaReduced / VECTORSIZE);
             l < s.nThetaReduced; l++) {
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
        }

        __m256d acc_lmksc = _mm256_setzero_pd();
        __m256d acc_lmkcs = _mm256_setzero_pd();
        __m256d acc_lmkcs_n = _mm256_setzero_pd();
        __m256d acc_lmksc_n = _mm256_setzero_pd();
        __m256d acc_rmkcc_n = _mm256_setzero_pd();
        __m256d acc_zmkcs_n = _mm256_setzero_pd();
        __m256d acc_rmkss_n = _mm256_setzero_pd();
        __m256d acc_zmksc_n = _mm256_setzero_pd();
        __m256d acc_rmkcc = _mm256_setzero_pd();
        __m256d acc_rmkss = _mm256_setzero_pd();
        __m256d acc_zmksc = _mm256_setzero_pd();
        __m256d acc_zmkcs = _mm256_setzero_pd();

        for (int l = 0; l < VECTORSIZE * (s.nThetaReduced / VECTORSIZE);
             l += VECTORSIZE) {
          const int idx_kl = idx_kl_base + l;
          const int idx_ml = idx_ml_base + l;

          const __m256d _cosmui = _mm256_loadu_pd(p_cosmui + idx_ml);
          const __m256d _sinmui = _mm256_loadu_pd(p_sinmui + idx_ml);
          const __m256d _cosmumi = _mm256_loadu_pd(p_cosmumi + idx_ml);
          const __m256d _sinmumi = _mm256_loadu_pd(p_sinmumi + idx_ml);

          const __m256d _armn = _mm256_loadu_pd(p_armn + idx_kl);
          const __m256d _azmn = _mm256_loadu_pd(p_azmn + idx_kl);
          const __m256d _blmn = _mm256_loadu_pd(p_blmn + idx_kl);
          const __m256d _brmn = _mm256_loadu_pd(p_brmn + idx_kl);
          const __m256d _bzmn = _mm256_loadu_pd(p_bzmn + idx_kl);
          const __m256d _clmn = _mm256_loadu_pd(p_clmn + idx_kl);
          const __m256d _crmn = _mm256_loadu_pd(p_crmn + idx_kl);
          const __m256d _czmn = _mm256_loadu_pd(p_czmn + idx_kl);

          // _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)  // a * b + c
          acc_lmksc = _mm256_fmadd_pd(_blmn, _cosmumi, acc_lmksc);
          acc_lmkcs = _mm256_fmadd_pd(_blmn, _sinmumi, acc_lmkcs);
          acc_lmkcs_n = _mm256_fmadd_pd(_clmn, _cosmui, acc_lmkcs_n);
          acc_lmksc_n = _mm256_fmadd_pd(_clmn, _sinmui, acc_lmksc_n);
          acc_rmkcc_n = _mm256_fmadd_pd(_crmn, _cosmui, acc_rmkcc_n);
          acc_zmkcs_n = _mm256_fmadd_pd(_czmn, _cosmui, acc_zmkcs_n);
          acc_rmkss_n = _mm256_fmadd_pd(_crmn, _sinmui, acc_rmkss_n);
          acc_zmksc_n = _mm256_fmadd_pd(_czmn, _sinmui, acc_zmksc_n);

          // assemble effective R and Z forces from MHD and spectral
          // condensation contributions
          const __m256d _frcon = _mm256_loadu_pd(p_frcon + idx_kl);
          const __m256d _fzcon = _mm256_loadu_pd(p_fzcon + idx_kl);
          const __m256d _tempR = _mm256_fmadd_pd(_xmpq, _frcon, _armn);
          const __m256d _tempZ = _mm256_fmadd_pd(_xmpq, _fzcon, _azmn);
          const __m256d tmpR_mul_cosmui = _mm256_mul_pd(_tempR, _cosmui);
          const __m256d tmpR_mul_sinmui = _mm256_mul_pd(_tempR, _sinmui);
          const __m256d tmpZ_mul_cosmui = _mm256_mul_pd(_tempZ, _cosmui);
          const __m256d tmpZ_mul_sinmui = _mm256_mul_pd(_tempZ, _sinmui);
          const __m256d brmn_mul_sinmumi = _mm256_mul_pd(_brmn, _sinmumi);
          const __m256d brmn_mul_cosmumi = _mm256_mul_pd(_brmn, _cosmumi);
          const __m256d bzmn_mul_sinmumi = _mm256_mul_pd(_bzmn, _sinmumi);
          const __m256d bzmn_mul_cosmumi = _mm256_mul_pd(_bzmn, _cosmumi);

          acc_rmkcc = _mm256_add_pd(
              acc_rmkcc, _mm256_add_pd(tmpR_mul_cosmui, brmn_mul_sinmumi));
          acc_rmkss = _mm256_add_pd(
              acc_rmkss, _mm256_add_pd(tmpR_mul_sinmui, brmn_mul_cosmumi));
          acc_zmksc = _mm256_add_pd(
              acc_zmksc, _mm256_add_pd(tmpZ_mul_sinmui, bzmn_mul_cosmumi));
          acc_zmkcs = _mm256_add_pd(
              acc_zmkcs, _mm256_add_pd(tmpZ_mul_cosmui, bzmn_mul_sinmumi));
        }

        lmksc += avx_hadd(acc_lmksc);
        lmkcs += avx_hadd(acc_lmkcs);
        lmkcs_n -= avx_hadd(acc_lmkcs_n);
        lmksc_n -= avx_hadd(acc_lmksc_n);
        rmkcc_n -= avx_hadd(acc_rmkcc_n);
        zmkcs_n -= avx_hadd(acc_zmkcs_n);
        rmkss_n -= avx_hadd(acc_rmkss_n);
        zmksc_n -= avx_hadd(acc_zmksc_n);
        rmkcc += avx_hadd(acc_rmkcc);
        rmkss += avx_hadd(acc_rmkss);
        zmksc += avx_hadd(acc_zmksc);
        zmkcs += avx_hadd(acc_zmkcs);

        const __m256d _rmkcc = _mm256_set1_pd(rmkcc);
        const __m256d _rmkcc_n = _mm256_set1_pd(rmkcc_n);
        const __m256d _rmkss = _mm256_set1_pd(rmkss);
        const __m256d _rmkss_n = _mm256_set1_pd(rmkss_n);
        const __m256d _zmksc = _mm256_set1_pd(zmksc);
        const __m256d _zmksc_n = _mm256_set1_pd(zmksc_n);
        const __m256d _zmkcs = _mm256_set1_pd(zmkcs);
        const __m256d _zmkcs_n = _mm256_set1_pd(zmkcs_n);
        const __m256d _lmksc = _mm256_set1_pd(lmksc);
        const __m256d _lmksc_n = _mm256_set1_pd(lmksc_n);
        const __m256d _lmkcs = _mm256_set1_pd(lmkcs);
        const __m256d _lmkcs_n = _mm256_set1_pd(lmkcs_n);

        int idx_mn = ((jF - rp.nsMinF) * s.mpol + m) * (s.ntor + 1);
        int idx_kn = k * (s.nnyq2 + 1);

        for (int n = 0; n < VECTORSIZE * ((s.ntor + 1) / VECTORSIZE);
             n += VECTORSIZE) {
          const __m256d _cosnv = _mm256_loadu_pd(p_cosnv + idx_kn);
          const __m256d _sinnv = _mm256_loadu_pd(p_sinnv + idx_kn);
          const __m256d _cosnvn = _mm256_loadu_pd(p_cosnvn + idx_kn);
          const __m256d _sinnvn = _mm256_loadu_pd(p_sinnvn + idx_kn);
          const __m256d _frcc = _mm256_loadu_pd(p_frcc + idx_mn);
          const __m256d _frss = _mm256_loadu_pd(p_frss + idx_mn);
          const __m256d _fzsc = _mm256_loadu_pd(p_fzsc + idx_mn);
          const __m256d _fzcs = _mm256_loadu_pd(p_fzcs + idx_mn);

          const __m256d rmkcc_mul_cosnv = _mm256_mul_pd(_rmkcc, _cosnv);
          const __m256d rmkcc_n_mul_sinnvn = _mm256_mul_pd(_rmkcc_n, _sinnvn);
          const __m256d rmkss_mul_sinnv = _mm256_mul_pd(_rmkss, _sinnv);
          const __m256d rmkss_n_mul_cosnvn = _mm256_mul_pd(_rmkss_n, _cosnvn);
          const __m256d zmksc_mul_cosnv = _mm256_mul_pd(_zmksc, _cosnv);
          const __m256d zmksc_n_mul_sinnvn = _mm256_mul_pd(_zmksc_n, _sinnvn);
          const __m256d zmkcs_mul_sinnv = _mm256_mul_pd(_zmkcs, _sinnv);
          const __m256d zmkcs_n_mul_cosnvn = _mm256_mul_pd(_zmkcs_n, _cosnvn);

          _mm256_storeu_pd(
              p_frcc + idx_mn,
              _mm256_add_pd(_mm256_add_pd(rmkcc_mul_cosnv, rmkcc_n_mul_sinnvn),
                            _frcc));
          _mm256_storeu_pd(
              p_frss + idx_mn,
              _mm256_add_pd(_mm256_add_pd(rmkss_mul_sinnv, rmkss_n_mul_cosnvn),
                            _frss));
          _mm256_storeu_pd(
              p_fzsc + idx_mn,
              _mm256_add_pd(_mm256_add_pd(zmksc_mul_cosnv, zmksc_n_mul_sinnvn),
                            _fzsc));
          _mm256_storeu_pd(
              p_fzcs + idx_mn,
              _mm256_add_pd(_mm256_add_pd(zmkcs_mul_sinnv, zmkcs_n_mul_cosnvn),
                            _fzcs));

          if (jMinL <= jF) {
            const __m256d _flsc = _mm256_loadu_pd(p_flsc + idx_mn);
            const __m256d _flcs = _mm256_loadu_pd(p_flcs + idx_mn);
            const __m256d lmksc_mul_cosnv = _mm256_mul_pd(_lmksc, _cosnv);
            const __m256d lmksc_n_mul_sinnvn = _mm256_mul_pd(_lmksc_n, _sinnvn);
            const __m256d lmkcs_mul_sinnv = _mm256_mul_pd(_lmkcs, _sinnv);
            const __m256d lmkcs_n_mul_cosnvn = _mm256_mul_pd(_lmkcs_n, _cosnvn);
            _mm256_storeu_pd(
                p_flsc + idx_mn,
                _mm256_add_pd(
                    _mm256_add_pd(lmksc_mul_cosnv, lmksc_n_mul_sinnvn), _flsc));
            _mm256_storeu_pd(
                p_flcs + idx_mn,
                _mm256_add_pd(
                    _mm256_add_pd(lmkcs_mul_sinnv, lmkcs_n_mul_cosnvn), _flcs));
          }

          idx_mn += VECTORSIZE;
          idx_kn += VECTORSIZE;
        }  // n

        for (int n = VECTORSIZE * ((s.ntor + 1) / VECTORSIZE); n < (s.ntor + 1);
             ++n) {
          const double cosnv = fb.cosnv[idx_kn];
          const double sinnv = fb.sinnv[idx_kn];
          const double cosnvn = fb.cosnvn[idx_kn];
          const double sinnvn = fb.sinnvn[idx_kn];

          physical_forces.frcc[idx_mn] += rmkcc * cosnv + rmkcc_n * sinnvn;
          physical_forces.frss[idx_mn] += rmkss * sinnv + rmkss_n * cosnvn;
          physical_forces.fzsc[idx_mn] += zmksc * cosnv + zmksc_n * sinnvn;
          physical_forces.fzcs[idx_mn] += zmkcs * sinnv + zmkcs_n * cosnvn;

          if (jMinL <= jF) {
            physical_forces.flsc[idx_mn] += lmksc * cosnv + lmksc_n * sinnvn;
            physical_forces.flcs[idx_mn] += lmkcs * sinnv + lmkcs_n * cosnvn;
          }
          idx_mn++;
          idx_kn++;
        }  // n
      }    // k
    }      // m
  }        // jF

  // repeat the above just for jMaxRZ to nsMaxFIncludingLcfs, just for flsc,
  // flcs
  // ** NOT vectorized; does not pay off
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
        }
        int idx_mn = ((jF - rp.nsMinF) * s.mpol + m) * (s.ntor + 1);
        int idx_kn = k * (s.nnyq2 + 1);

        for (int n = 0; n < s.ntor + 1; ++n) {
          const double cosnv = fb.cosnv[idx_kn];
          const double sinnv = fb.sinnv[idx_kn];
          const double cosnvn = fb.cosnvn[idx_kn];
          const double sinnvn = fb.sinnvn[idx_kn];

          physical_forces.flsc[idx_mn] += lmksc * cosnv + lmksc_n * sinnvn;
          physical_forces.flcs[idx_mn] += lmkcs * sinnv + lmkcs_n * cosnvn;
          idx_mn++;
          idx_kn++;
        }  // n
      }    // k
    }      // m
  }        // jF
}

void vmecpp::FourierToReal3DSymmFastPoloidal_avx(
    const FourierGeometry& physical_x, const std::vector<double>& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, RealSpaceGeometry& g) {
  // can safely assume lthreed == true in here

  absl::c_fill(g.r1_e, 0);
  absl::c_fill(g.r1_o, 0);
  absl::c_fill(g.ru_e, 0);
  absl::c_fill(g.ru_o, 0);
  absl::c_fill(g.rv_e, 0);
  absl::c_fill(g.rv_o, 0);
  absl::c_fill(g.z1_e, 0);
  absl::c_fill(g.z1_o, 0);
  absl::c_fill(g.zu_e, 0);
  absl::c_fill(g.zu_o, 0);
  absl::c_fill(g.zv_e, 0);
  absl::c_fill(g.zv_o, 0);
  absl::c_fill(g.lu_e, 0);
  absl::c_fill(g.lu_o, 0);
  absl::c_fill(g.lv_e, 0);
  absl::c_fill(g.lv_o, 0);

  absl::c_fill(g.rCon, 0);
  absl::c_fill(g.zCon, 0);

  const double* p_cosnv = fb.cosnv.data();
  const double* p_sinnv = fb.sinnv.data();
  const double* p_cosnvn = fb.cosnvn.data();
  const double* p_sinnvn = fb.sinnvn.data();
  const double* p_rmncc = physical_x.rmncc.data();
  const double* p_rmnss = physical_x.rmnss.data();
  const double* p_zmnsc = physical_x.zmnsc.data();
  const double* p_zmncs = physical_x.zmncs.data();
  const double* p_lmnsc = physical_x.lmnsc.data();
  const double* p_lmncs = physical_x.lmncs.data();

  const double* p_sinmum = fb.sinmum.data();
  const double* p_cosmum = fb.cosmum.data();
  const double* p_cosmu = fb.cosmu.data();
  const double* p_sinmu = fb.sinmu.data();

  double* p_rCon = g.rCon.data();
  double* p_zCon = g.zCon.data();

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

      const __m256d _con_factor = _mm256_set1_pd(con_factor);

      auto& r1 = m_even ? g.r1_e : g.r1_o;
      auto& ru = m_even ? g.ru_e : g.ru_o;
      auto& rv = m_even ? g.rv_e : g.rv_o;
      auto& z1 = m_even ? g.z1_e : g.z1_o;
      auto& zu = m_even ? g.zu_e : g.zu_o;
      auto& zv = m_even ? g.zv_e : g.zv_o;
      auto& lu = m_even ? g.lu_e : g.lu_o;
      auto& lv = m_even ? g.lv_e : g.lv_o;

      double* p_r1 = r1.data();
      double* p_ru = ru.data();
      double* p_rv = rv.data();
      double* p_z1 = z1.data();
      double* p_zu = zu.data();
      double* p_zv = zv.data();
      double* p_lu = lu.data();
      double* p_lv = lv.data();

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
        __m256d acc_rmkcc = _mm256_setzero_pd();
        __m256d acc_rmkcc_n = _mm256_setzero_pd();
        __m256d acc_rmkss = _mm256_setzero_pd();
        __m256d acc_rmkss_n = _mm256_setzero_pd();
        __m256d acc_zmksc = _mm256_setzero_pd();
        __m256d acc_zmksc_n = _mm256_setzero_pd();
        __m256d acc_zmkcs = _mm256_setzero_pd();
        __m256d acc_zmkcs_n = _mm256_setzero_pd();
        __m256d acc_lmksc = _mm256_setzero_pd();
        __m256d acc_lmksc_n = _mm256_setzero_pd();
        __m256d acc_lmkcs = _mm256_setzero_pd();
        __m256d acc_lmkcs_n = _mm256_setzero_pd();

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

        int idx_kn = k * (s.nnyq2 + 1);
        int idx_mn = ((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1);

        for (int n = 0; n < VECTORSIZE * ((s.ntor + 1) / VECTORSIZE);
             n += VECTORSIZE) {
          // INVERSE TRANSFORM IN N-ZETA, FOR FIXED M
          const __m256d _cosnv = _mm256_loadu_pd(p_cosnv + idx_kn);
          const __m256d _sinnv = _mm256_loadu_pd(p_sinnv + idx_kn);
          const __m256d _cosnvn = _mm256_loadu_pd(p_cosnvn + idx_kn);
          const __m256d _sinnvn = _mm256_loadu_pd(p_sinnvn + idx_kn);

          const __m256d _rmncc = _mm256_loadu_pd(p_rmncc + idx_mn);
          const __m256d _rmnss = _mm256_loadu_pd(p_rmnss + idx_mn);
          const __m256d _zmnsc = _mm256_loadu_pd(p_zmnsc + idx_mn);
          const __m256d _zmncs = _mm256_loadu_pd(p_zmncs + idx_mn);
          const __m256d _lmnsc = _mm256_loadu_pd(p_lmnsc + idx_mn);
          const __m256d _lmncs = _mm256_loadu_pd(p_lmncs + idx_mn);

          // _mm256_fmadd_pd(__m256d a, __m256d b, __m256d c)  // a * b + c
          acc_rmkcc = _mm256_fmadd_pd(_rmncc, _cosnv, acc_rmkcc);
          acc_rmkcc_n = _mm256_fmadd_pd(_rmncc, _sinnvn, acc_rmkcc_n);
          acc_rmkss = _mm256_fmadd_pd(_rmnss, _sinnv, acc_rmkss);
          acc_rmkss_n = _mm256_fmadd_pd(_rmnss, _cosnvn, acc_rmkss_n);
          acc_zmksc = _mm256_fmadd_pd(_zmnsc, _cosnv, acc_zmksc);
          acc_zmksc_n = _mm256_fmadd_pd(_zmnsc, _sinnvn, acc_zmksc_n);
          acc_zmkcs = _mm256_fmadd_pd(_zmncs, _sinnv, acc_zmkcs);
          acc_zmkcs_n = _mm256_fmadd_pd(_zmncs, _cosnvn, acc_zmkcs_n);
          acc_lmksc = _mm256_fmadd_pd(_lmnsc, _cosnv, acc_lmksc);
          acc_lmksc_n = _mm256_fmadd_pd(_lmnsc, _sinnvn, acc_lmksc_n);
          acc_lmkcs = _mm256_fmadd_pd(_lmncs, _sinnv, acc_lmkcs);
          acc_lmkcs_n = _mm256_fmadd_pd(_lmncs, _cosnvn, acc_lmkcs_n);
          idx_kn += VECTORSIZE;
          idx_mn += VECTORSIZE;
        }  // n

        rmkcc += avx_hadd(acc_rmkcc);
        rmkcc_n += avx_hadd(acc_rmkcc_n);
        rmkss += avx_hadd(acc_rmkss);
        rmkss_n += avx_hadd(acc_rmkss_n);
        zmksc += avx_hadd(acc_zmksc);
        zmksc_n += avx_hadd(acc_zmksc_n);
        zmkcs += avx_hadd(acc_zmkcs);
        zmkcs_n += avx_hadd(acc_zmkcs_n);
        lmksc += avx_hadd(acc_lmksc);
        lmksc_n += avx_hadd(acc_lmksc_n);
        lmkcs += avx_hadd(acc_lmkcs);
        lmkcs_n += avx_hadd(acc_lmkcs_n);

        for (int n = VECTORSIZE * ((s.ntor + 1) / VECTORSIZE); n < (s.ntor + 1);
             ++n) {
          // INVERSE TRANSFORM IN N-ZETA, FOR FIXED M
          double cosnv = fb.cosnv[idx_kn];
          double sinnv = fb.sinnv[idx_kn];
          double sinnvn = fb.sinnvn[idx_kn];
          double cosnvn = fb.cosnvn[idx_kn];
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
          idx_kn++;
          idx_mn++;
        }  // n

        const __m256d _rmkcc = _mm256_set1_pd(rmkcc);
        const __m256d _rmkcc_n = _mm256_set1_pd(rmkcc_n);
        const __m256d _rmkss = _mm256_set1_pd(rmkss);
        const __m256d _rmkss_n = _mm256_set1_pd(rmkss_n);
        const __m256d _zmksc = _mm256_set1_pd(zmksc);
        const __m256d _zmksc_n = _mm256_set1_pd(zmksc_n);
        const __m256d _zmkcs = _mm256_set1_pd(zmkcs);
        const __m256d _zmkcs_n = _mm256_set1_pd(zmkcs_n);
        const __m256d _lmksc = _mm256_set1_pd(lmksc);
        const __m256d _lmksc_n = _mm256_set1_pd(lmksc_n);
        const __m256d _lmkcs = _mm256_set1_pd(lmkcs);
        const __m256d _lmkcs_n = _mm256_set1_pd(lmkcs_n);

        // INVERSE TRANSFORM IN M-THETA, FOR ALL RADIAL, ZETA VALUES
        const int idx_kl_base = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff;

        for (int l = 0; l < VECTORSIZE * (s.nThetaReduced / VECTORSIZE);
             l += VECTORSIZE) {
          const int idx_ml = idx_ml_base + l;
          const int idx_kl = idx_kl_base + l;

          const __m256d _sinmum = _mm256_loadu_pd(p_sinmum + idx_ml);
          const __m256d _cosmum = _mm256_loadu_pd(p_cosmum + idx_ml);
          const __m256d _cosmu = _mm256_loadu_pd(p_cosmu + idx_ml);
          const __m256d _sinmu = _mm256_loadu_pd(p_sinmu + idx_ml);

          // This is the bottleneck here
          // Constantly loading and evicting cache just to accumulate values
          // in-memory, would be better to do this somehow within registers
          // load_from_mem, add, store_to_mem, over and over again at different
          // adresses close to memory-bound, backend stalls
          const __m256d _ru = _mm256_loadu_pd(p_ru + idx_kl);
          const __m256d _zu = _mm256_loadu_pd(p_zu + idx_kl);
          const __m256d _lu = _mm256_loadu_pd(p_lu + idx_kl);
          const __m256d _rv = _mm256_loadu_pd(p_rv + idx_kl);
          const __m256d _zv = _mm256_loadu_pd(p_zv + idx_kl);
          const __m256d _lv = _mm256_loadu_pd(p_lv + idx_kl);
          const __m256d _r1 = _mm256_loadu_pd(p_r1 + idx_kl);
          const __m256d _z1 = _mm256_loadu_pd(p_z1 + idx_kl);
          // _mm256_fmadd_pd(aa, bb, cc)  // a*b + c
          _mm256_storeu_pd(
              p_ru + idx_kl,
              _mm256_add_pd(_ru,
                            _mm256_fmadd_pd(_rmkcc, _sinmum,
                                            _mm256_mul_pd(_rmkss, _cosmum))));
          _mm256_storeu_pd(
              p_zu + idx_kl,
              _mm256_add_pd(_zu,
                            _mm256_fmadd_pd(_zmkcs, _sinmum,
                                            _mm256_mul_pd(_zmksc, _cosmum))));
          _mm256_storeu_pd(
              p_lu + idx_kl,
              _mm256_add_pd(_lu,
                            _mm256_fmadd_pd(_lmkcs, _sinmum,
                                            _mm256_mul_pd(_lmksc, _cosmum))));
          _mm256_storeu_pd(
              p_rv + idx_kl,
              _mm256_add_pd(_rv,
                            _mm256_fmadd_pd(_rmkss_n, _sinmu,
                                            _mm256_mul_pd(_rmkcc_n, _cosmu))));
          _mm256_storeu_pd(
              p_zv + idx_kl,
              _mm256_add_pd(_zv,
                            _mm256_fmadd_pd(_zmkcs_n, _cosmu,
                                            _mm256_mul_pd(_zmksc_n, _sinmu))));
          _mm256_storeu_pd(
              p_lv + idx_kl,
              _mm256_sub_pd(_lv,
                            _mm256_fmadd_pd(_lmkcs_n, _cosmu,
                                            _mm256_mul_pd(_lmksc_n, _sinmu))));
          _mm256_storeu_pd(
              p_r1 + idx_kl,
              _mm256_add_pd(_r1,
                            _mm256_fmadd_pd(_rmkss, _sinmu,
                                            _mm256_mul_pd(_rmkcc, _cosmu))));
          _mm256_storeu_pd(
              p_z1 + idx_kl,
              _mm256_add_pd(_z1,
                            _mm256_fmadd_pd(_zmkcs, _cosmu,
                                            _mm256_mul_pd(_zmksc, _sinmu))));
        }  // l

        for (int l = VECTORSIZE * (s.nThetaReduced / VECTORSIZE);
             l < s.nThetaReduced; ++l) {
          const int idx_ml = idx_ml_base + l;
          const int idx_kl = idx_kl_base + l;

          const double sinmum = fb.sinmum[idx_ml];
          const double cosmum = fb.cosmum[idx_ml];
          const double cosmu = fb.cosmu[idx_ml];
          const double sinmu = fb.sinmu[idx_ml];

          ru[idx_kl] += rmkcc * sinmum + rmkss * cosmum;
          zu[idx_kl] += zmksc * cosmum + zmkcs * sinmum;
          lu[idx_kl] += lmksc * cosmum + lmkcs * sinmum;
          rv[idx_kl] += rmkcc_n * cosmu + rmkss_n * sinmu;
          zv[idx_kl] += zmksc_n * sinmu + zmkcs_n * cosmu;
          // it is here that lv gets a negative sign!
          lv[idx_kl] -= lmksc_n * sinmu + lmkcs_n * cosmu;
          r1[idx_kl] += rmkcc * cosmu + rmkss * sinmu;
          z1[idx_kl] += zmksc * sinmu + zmkcs * cosmu;
        }  // l

        if (nsMinF <= jF && jF < r.nsMaxFIncludingLcfs) {
          int idx_con = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff;
          for (int l = 0; l < VECTORSIZE * (s.nThetaReduced / VECTORSIZE);
               l += VECTORSIZE) {
            const int idx_ml = idx_ml_base + l;
            const __m256d _cosmu = _mm256_loadu_pd(p_cosmu + idx_ml);
            const __m256d _sinmu = _mm256_loadu_pd(p_sinmu + idx_ml);
            __m256d _rCon = _mm256_loadu_pd(p_rCon + idx_con);
            __m256d _zCon = _mm256_loadu_pd(p_zCon + idx_con);

            // spectral condensation is local per flux surface
            // --> no need for numFull1
            // _mm256_fmadd_pd(__m256d a, __m256d b, __m256d c)  // a*b+c
            _rCon = _mm256_fmadd_pd(
                _mm256_fmadd_pd(_rmkcc, _cosmu, _mm256_mul_pd(_rmkss, _sinmu)),
                _con_factor, _rCon);
            _zCon = _mm256_fmadd_pd(
                _mm256_fmadd_pd(_zmksc, _sinmu, _mm256_mul_pd(_zmkcs, _cosmu)),
                _con_factor, _zCon);

            _mm256_storeu_pd(p_rCon + idx_con, _rCon);
            _mm256_storeu_pd(p_zCon + idx_con, _zCon);

            idx_con += VECTORSIZE;
          }  // l
          for (int l = VECTORSIZE * (s.nThetaReduced / VECTORSIZE);
               l < s.nThetaReduced; ++l) {
            const int idx_ml = idx_ml_base + l;
            const double cosmu = fb.cosmu[idx_ml];
            const double sinmu = fb.sinmu[idx_ml];

            // spectral condensation is local per flux surface
            // --> no need for numFull1
            g.rCon[idx_con] += (rmkcc * cosmu + rmkss * sinmu) * con_factor;
            g.zCon[idx_con] += (zmksc * sinmu + zmkcs * cosmu) * con_factor;

            idx_con++;
          }
        }  // k
      }    // m
    }      // j
  }
}

void vmecpp::deAliasConstraintForce_avx(
    const vmecpp::RadialPartitioning& rp,
    const vmecpp::FourierBasisFastPoloidal& fb, const vmecpp::Sizes& s_,
    std::vector<double>& faccon, std::vector<double>& tcon,
    std::vector<double>& gConEff, std::vector<double>& gsc,
    std::vector<double>& gcs, std::vector<double>& gCon) {
  absl::c_fill_n(gCon, (rp.nsMaxF - rp.nsMinF) * s_.nZnT, 0);

  double* p_gConEff = gConEff.data();
  const double* p_sinmui = fb.sinmui.data();
  const double* p_cosmui = fb.cosmui.data();
  const double* p_sinmu = fb.sinmu.data();
  const double* p_cosmu = fb.cosmu.data();
  const double* p_sinnv = fb.sinnv.data();
  const double* p_cosnv = fb.cosnv.data();
  double* p_gsc = gsc.data();
  double* p_gcs = gcs.data();
  double* p_gCon = gCon.data();

  // no constraint on axis --> has no poloidal angle
  int jMin = 0;
  if (rp.nsMinF == 0) {
    jMin = 1;
  }

  for (int jF = std::max(jMin, rp.nsMinF); jF < rp.nsMaxF; ++jF) {
    const int idx_j = jF - rp.nsMinF;
    double dtcon = tcon[idx_j];
    for (int m = 1; m < s_.mpol - 1; ++m) {
      const int idx_m = m * s_.nThetaReduced;
      absl::c_fill_n(gsc, s_.ntor + 1, 0);
      absl::c_fill_n(gcs, s_.ntor + 1, 0);
      const double d_faccon = faccon[m];
      __m256d _d_faccon = _mm256_set1_pd(d_faccon);

      for (int k = 0; k < s_.nZeta; ++k) {
        int const idx_jk = (idx_j * s_.nZeta + k) * s_.nThetaEff;
        __m256d _w0 = _mm256_setzero_pd();
        __m256d _w1 = _mm256_setzero_pd();

        // fwd transform in poloidal direction
        // integrate poloidally to get m-th poloidal Fourier coefficient
        for (int l = 0; l < VECTORSIZE * (s_.nThetaReduced / VECTORSIZE);
             l += VECTORSIZE) {
          const int idx_ml = idx_m + l;
          const int idx_jkl = idx_jk + l;
          const __m256d _sinmui = _mm256_loadu_pd(p_sinmui + idx_ml);
          const __m256d _cosmui = _mm256_loadu_pd(p_cosmui + idx_ml);
          const __m256d _gConEff = _mm256_loadu_pd(p_gConEff + idx_jkl);
          // _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)  // a * b + c
          _w0 = _mm256_fmadd_pd(_gConEff, _sinmui, _w0);
          _w1 = _mm256_fmadd_pd(_gConEff, _cosmui, _w1);
        }
        double w0 = avx_hadd(_w0);
        double w1 = avx_hadd(_w1);

        for (int l = VECTORSIZE * (s_.nThetaReduced / VECTORSIZE);
             l < s_.nThetaReduced; ++l) {
          const int idx_ml = idx_m + l;
          int idx_jkl = idx_jk + l;
          w0 += gConEff[idx_jkl] * fb.sinmui[idx_ml];
          w1 += gConEff[idx_jkl] * fb.cosmui[idx_ml];
        }

        _w0 = _mm256_set1_pd(w0);
        _w1 = _mm256_set1_pd(w1);
        const double w0_mul_tcon = w0 * dtcon;
        const double w1_mul_tcon = w1 * dtcon;
        const __m256d _w0_mul_tcon = _mm256_set1_pd(w0_mul_tcon);
        const __m256d _w1_mul_tcon = _mm256_set1_pd(w1_mul_tcon);

        // forward Fourier transform in toroidal direction for full set of mode
        // numbers (n = 0, 1, ..., ntor)
        const int idx_k = k * (s_.nnyq2 + 1);
        for (int n = 0; n < VECTORSIZE * ((s_.ntor + 1) / VECTORSIZE);
             n += VECTORSIZE) {
          const int idx_kn = idx_k + n;
          const __m256d _sinnv = _mm256_loadu_pd(p_sinnv + idx_kn);
          const __m256d _cosnv = _mm256_loadu_pd(p_cosnv + idx_kn);
          __m256d _gsc = _mm256_loadu_pd(p_gsc + n);
          __m256d _gcs = _mm256_loadu_pd(p_gcs + n);
          _gsc = _mm256_fmadd_pd(_cosnv, _w0_mul_tcon, _gsc);
          _gcs = _mm256_fmadd_pd(_sinnv, _w1_mul_tcon, _gcs);
          _mm256_storeu_pd(p_gsc + n, _gsc);
          _mm256_storeu_pd(p_gcs + n, _gcs);
        }
        for (int n = VECTORSIZE * ((s_.ntor + 1) / VECTORSIZE);
             n < (s_.ntor + 1); ++n) {
          const int idx_kn = idx_k + n;
          gsc[n] += fb.cosnv[idx_kn] * w0_mul_tcon;
          gcs[n] += fb.sinnv[idx_kn] * w1_mul_tcon;
        }
      }  // k

      // ------------------------------------------
      // need to "wait" (= finish k loop) here
      // to get Fourier coefficients fully defined!
      // ------------------------------------------

      // inverse Fourier-transform from reduced set of mode numbers
      for (int k = 0; k < s_.nZeta; ++k) {
        const int idx_k = k * (s_.nnyq2 + 1);
        const int idx_jk = (idx_j * s_.nZeta + k) * s_.nThetaEff;
        __m256d _w0 = _mm256_setzero_pd();
        __m256d _w1 = _mm256_setzero_pd();

        // collect contribution to current grid point from n-th toroidal mode
        for (int n = 0; n < VECTORSIZE * ((s_.ntor + 1) / VECTORSIZE);
             n += VECTORSIZE) {
          int idx_kn = idx_k + n;
          const __m256d _sinnv = _mm256_loadu_pd(p_sinnv + idx_kn);
          const __m256d _cosnv = _mm256_loadu_pd(p_cosnv + idx_kn);
          __m256d _gsc = _mm256_loadu_pd(p_gsc + n);
          __m256d _gcs = _mm256_loadu_pd(p_gcs + n);
          _w0 = _mm256_fmadd_pd(_cosnv, _gsc, _w0);
          _w1 = _mm256_fmadd_pd(_sinnv, _gcs, _w1);
        }
        double w0 = avx_hadd(_w0);
        double w1 = avx_hadd(_w1);
        for (int n = VECTORSIZE * ((s_.ntor + 1) / VECTORSIZE);
             n < (s_.ntor + 1); ++n) {
          int idx_kn = idx_k + n;
          w0 += gsc[n] * fb.cosnv[idx_kn];
          w1 += gcs[n] * fb.sinnv[idx_kn];
        }  // n
        _w0 = _mm256_set1_pd(w0);
        _w1 = _mm256_set1_pd(w1);

        // inv transform in poloidal direction
        for (int l = 0; l < VECTORSIZE * (s_.nThetaReduced / VECTORSIZE);
             l += VECTORSIZE) {
          const int idx_kl = idx_jk + l;
          const int idx_ml = idx_m + l;
          const __m256d _sinmu = _mm256_loadu_pd(p_sinmu + idx_ml);
          const __m256d _cosmu = _mm256_loadu_pd(p_cosmu + idx_ml);
          __m256d _gCon = _mm256_loadu_pd(p_gCon + idx_kl);
          _gCon = _mm256_fmadd_pd(
              _d_faccon,
              _mm256_fmadd_pd(_w0, _sinmu, _mm256_mul_pd(_w1, _cosmu)), _gCon);
          _mm256_storeu_pd(p_gCon + idx_kl, _gCon);
        }
        for (int l = VECTORSIZE * (s_.nThetaReduced / VECTORSIZE);
             l < s_.nThetaReduced; ++l) {
          const int idx_kl = idx_jk + l;
          const int idx_ml = idx_m + l;
          gCon[idx_kl] +=
              d_faccon * (w0 * fb.sinmu[idx_ml] + w1 * fb.cosmu[idx_ml]);
        }  // l
      }    // k
    }      // m
  }
}
