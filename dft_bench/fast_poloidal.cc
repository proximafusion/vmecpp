#include "vmecpp_tools/dft_bench/fast_poloidal.h"

#include <algorithm>  // std::min

static constexpr int m_evn = 0;

// This version comes from the old VMEC++ repo
void dft_bench::fast_poloidal::ForcesToFourier3D(
    vmecpp::FourierForces& m_physical_f, const IdealMHDData& d,
    const Inputs& inputs) {
  const auto& [r, s, fb, fc, _1, _2] = inputs;

  // fill target force arrays with zeros
  m_physical_f.setZero();

  int nZetaEven = s.nZeta > 1 ? 2 * (s.nZeta / 2) : 1;

  int jMaxRZ = std::min(r.nsMaxF, fc.ns - 1);

  for (int jF = r.nsMinF; jF < r.nsMaxF; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      int m_parity = m % 2;

      // At which surfaces starts what poloidal contributions?
      // --> axis lambda stays zero (no contribution from any m)
      // --> axis R,Z gets only m=0 contributions
      // --> all other surfaces get the full Fourier spectrum
      int jMinL = 1;
      int jMinRZ = -1;
      if (m == 0) {
        jMinRZ = 0;
      } else {
        jMinRZ = 1;
      }

      for (int k = 0; k < nZetaEven; ++k) {
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

        for (int l = 0; l < s.nThetaReduced; ++l) {
          int idx_kl = ((jF - r.nsMinF) * nZetaEven + k) * s.nThetaReduced + l;

          // NOTE: Changed to new layout in Fourier basis
          // This is not the optimal layout anymore!
          const int idx_lm = l * (s.mnyq2 + 1) + m;

          double cosmui = fb.cosmui[idx_lm];
          double sinmui = fb.sinmui[idx_lm];
          double cosmumi = fb.cosmumi[idx_lm];
          double sinmumi = fb.sinmumi[idx_lm];

          if (m_parity == /*m_evn*/ 0) {
            // assemble effective R and Z forces from MHD and spectral
            // condensation contributions
            double tempR = d.armn_e[idx_kl] + d.xmpq[m] * d.frcon_e[idx_kl];
            double tempZ = d.azmn_e[idx_kl] + d.xmpq[m] * d.fzcon_e[idx_kl];

            rmkcc += tempR * cosmui + d.brmn_e[idx_kl] * sinmumi;  // --> frcc
            rmkcc_n += -d.crmn_e[idx_kl] * cosmui;                 // --> frcc
            rmkss += tempR * sinmui + d.brmn_e[idx_kl] * cosmumi;  // --> frss
            rmkss_n += -d.crmn_e[idx_kl] * sinmui;                 // --> frss
            zmksc += tempZ * sinmui + d.bzmn_e[idx_kl] * cosmumi;  // --> fzsc
            zmksc_n += -d.czmn_e[idx_kl] * sinmui;                 // --> fzsc
            zmkcs += tempZ * cosmui + d.bzmn_e[idx_kl] * sinmumi;  // --> fzcs
            zmkcs_n += -d.czmn_e[idx_kl] * cosmui;                 // --> fzcs
            lmksc += d.blmn_e[idx_kl] * cosmumi;    // --> flsc (no A)
            lmksc_n += -d.clmn_e[idx_kl] * sinmui;  // --> flsc
            lmkcs += d.blmn_e[idx_kl] * sinmumi;    // --> flcs
            lmkcs_n += -d.clmn_e[idx_kl] * cosmui;  // --> flcs
          } else {                                  // m_odd
            // assemble effective R and Z forces from MHD and spectral
            // condensation contributions
            double tempR = d.armn_o[idx_kl] + d.xmpq[m] * d.frcon_o[idx_kl];
            double tempZ = d.azmn_o[idx_kl] + d.xmpq[m] * d.fzcon_o[idx_kl];

            rmkcc += tempR * cosmui + d.brmn_o[idx_kl] * sinmumi;  // --> frcc
            rmkcc_n += -d.crmn_o[idx_kl] * cosmui;                 // --> frcc
            rmkss += tempR * sinmui + d.brmn_o[idx_kl] * cosmumi;  // --> frss
            rmkss_n += -d.crmn_o[idx_kl] * sinmui;                 // --> frss
            zmksc += tempZ * sinmui + d.bzmn_o[idx_kl] * cosmumi;  // --> fzsc
            zmksc_n += -d.czmn_o[idx_kl] * sinmui;                 // --> fzsc
            zmkcs += tempZ * cosmui + d.bzmn_o[idx_kl] * sinmumi;  // --> fzcs
            zmkcs_n += -d.czmn_o[idx_kl] * cosmui;                 // --> fzcs
            lmksc += d.blmn_o[idx_kl] * cosmumi;    // --> flsc (no A)
            lmksc_n += -d.clmn_o[idx_kl] * sinmui;  // --> flsc
            lmkcs += d.blmn_o[idx_kl] * sinmumi;    // --> flcs
            lmkcs_n += -d.clmn_o[idx_kl] * cosmui;  // --> flcs
          }
        }  // l

        for (int n = 0; n < s.ntor + 1; ++n) {
          int idx_mn = ((jF - r.nsMinF) * s.mpol + m) * (s.ntor + 1) + n;

          // NOTE: Changed to new layout in Fourier basis
          // This is not the optimal layout anymore!
          const int idx_nk = n * s.nZeta + k;

          double cosnv = fb.cosnv[idx_nk];
          double sinnv = fb.sinnv[idx_nk];
          double cosnvn = fb.cosnvn[idx_nk];
          double sinnvn = fb.sinnvn[idx_nk];

          if (jF >= jMinRZ && jF < jMaxRZ) {
            m_physical_f.frcc[idx_mn] += rmkcc * cosnv + rmkcc_n * sinnvn;
            m_physical_f.frss[idx_mn] += rmkss * sinnv + rmkss_n * cosnvn;
            m_physical_f.fzsc[idx_mn] += zmksc * cosnv + zmksc_n * sinnvn;
            m_physical_f.fzcs[idx_mn] += zmkcs * sinnv + zmkcs_n * cosnvn;
          }
          if (jF >= jMinL) {
            m_physical_f.flsc[idx_mn] += lmksc * cosnv + lmksc_n * sinnvn;
            m_physical_f.flcs[idx_mn] += lmkcs * sinnv + lmkcs_n * cosnvn;
          }
        }  // n
      }  // k
    }  // m
  }  // jF
}

// This version comes from the old vmecpp repo.
// The only changes that have been applied were to adapt variable to the new
// naming convention, or to substitute raw pointers with vector::data()
// invocations.
void dft_bench::fast_poloidal::FourierToReal3D(IdealMHDData& d,
                                               const Inputs& inputs) {
  const auto& [radial_partitioning, s, fb, _, radial_profiles, physical_x] =
      inputs;

  const int nsMinF1 = radial_partitioning.nsMinF1;
  const int nsMaxF1 = radial_partitioning.nsMaxF1;

  const int nsMinF = radial_partitioning.nsMinF;
  const int nsMaxF = radial_partitioning.nsMaxF;

  // only zero target elements filled in this thread
  d.r1_e.setZero();
  d.r1_o.setZero();
  d.ru_e.setZero();
  d.ru_o.setZero();
  d.rv_e.setZero();
  d.rv_o.setZero();
  d.z1_e.setZero();
  d.z1_o.setZero();
  d.zu_e.setZero();
  d.zu_o.setZero();
  d.zv_e.setZero();
  d.zv_o.setZero();
  d.lu_e.setZero();
  d.lu_o.setZero();
  d.lv_e.setZero();
  d.lv_o.setZero();

  int num_con = (nsMaxF - nsMinF) * s.nZnT;
  d.rCon.head(num_con).setZero();
  d.zCon.head(num_con).setZero();

  // NOTE: fix on old VMEC++: need to transform geometry for nsMinF1 ... nsMaxF1
  for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      int m_parity = m % 2;

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

      const int nZetaEven = s.nZeta > 1 ? 2 * (s.nZeta / 2) : 1;
      for (int k = 0; k < nZetaEven; ++k) {
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

          // NOTE: Changed to new layout in Fourier basis
          // This is not the optimal layout anymore!
          const int idx_nk = n * s.nZeta + k;

          double cosnv = fb.cosnv[idx_nk];
          double sinnv = fb.sinnv[idx_nk];
          double sinnvn = fb.sinnvn[idx_nk];
          double cosnvn = fb.cosnvn[idx_nk];

          int idx_mn = ((jF - nsMinF) * s.mpol + m) * (s.ntor + 1) + n;

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
        for (int l = 0; l < s.nThetaReduced; ++l) {
          // NOTE: Changed to new layout in Fourier basis
          // This is not the optimal layout anymore!
          const int idx_lm = l * (s.mnyq2 + 1) + m;

          double cosmu = fb.cosmu[idx_lm];
          double sinmu = fb.sinmu[idx_lm];
          double sinmum = fb.sinmum[idx_lm];
          double cosmum = fb.cosmum[idx_lm];

          double _r1 = rmkcc * cosmu + rmkss * sinmu;
          double _ru = rmkcc * sinmum + rmkss * cosmum;
          double _rv = rmkcc_n * cosmu + rmkss_n * sinmu;
          double _z1 = zmksc * sinmu + zmkcs * cosmu;
          double _zu = zmksc * cosmum + zmkcs * sinmum;
          double _zv = zmksc_n * sinmu + zmkcs_n * cosmu;
          double _lu = lmksc * cosmum + lmkcs * sinmum;
          double _lv = lmksc_n * sinmu + lmkcs_n * cosmu;

          int idx_kl = ((jF - nsMinF1) * nZetaEven + k) * s.nThetaReduced + l;

          if (m_parity == m_evn) {
            d.r1_e[idx_kl] += _r1;
            d.ru_e[idx_kl] += _ru;
            d.rv_e[idx_kl] += _rv;
            d.z1_e[idx_kl] += _z1;
            d.zu_e[idx_kl] += _zu;
            d.zv_e[idx_kl] += _zv;
            d.lu_e[idx_kl] += _lu;
            // it is here that lv gets a negative sign!
            d.lv_e[idx_kl] -= _lv;
          } else {  // m_odd
            d.r1_o[idx_kl] += _r1;
            d.ru_o[idx_kl] += _ru;
            d.rv_o[idx_kl] += _rv;
            d.z1_o[idx_kl] += _z1;
            d.zu_o[idx_kl] += _zu;
            d.zv_o[idx_kl] += _zv;
            d.lu_o[idx_kl] += _lu;
            // it is here that lv gets a negative sign!
            d.lv_o[idx_kl] -= _lv;
          }

          if (nsMinF <= jF && jF < nsMaxF) {
            // with sqrtS for odd-m !
            double _rCon = _r1 * d.xmpq[m];
            double _zCon = _z1 * d.xmpq[m];

            // spectral condensation is local per flux surface
            // --> no need for numFull1
            int idx_con = ((jF - nsMinF) * nZetaEven + k) * s.nThetaReduced + l;

            if (m_parity == m_evn) {
              d.rCon[idx_con] += _rCon;
              d.zCon[idx_con] += _zCon;
            } else {
              d.rCon[idx_con] += _rCon * radial_profiles.sqrtSF[jF - nsMinF1];
              d.zCon[idx_con] += _zCon * radial_profiles.sqrtSF[jF - nsMinF1];
            }
          }
        }  // l
      }  // k
    }  // m
  }  // j
}
