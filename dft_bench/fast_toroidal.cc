#include "vmecpp_tools/dft_bench/fast_toroidal.h"

#include <algorithm>

static constexpr int m_evn = 0;
static constexpr int m_odd = 1;

void dft_bench::fast_toroidal::ForcesToFourier3D(
    vmecpp::FourierForces& m_physical_f, const IdealMHDData& d,
    const Inputs& inputs) {
  const auto& [r, s, fb, fc, _1, _2] = inputs;

  // fill target force arrays with zeros
  m_physical_f.setZero();

  int jMaxRZ = std::min(r.nsMaxF, fc.ns - 1);
  if (fc.lfreeb && d.ivac >= 1) {
    // free-boundary: up to jMaxRZ=ns
    jMaxRZ = std::min(r.nsMaxF, fc.ns);
  }

  for (int jF = r.nsMinF; jF < r.nsMaxFIncludingLcfs; ++jF) {
    for (int n = 0; n < s.ntor + 1; ++n) {
      for (int l = 0; l < s.nThetaReduced; ++l) {
        std::array<double, 2> rnkcc = {0.0, 0.0};
        std::array<double, 2> rnkcc_m = {0.0, 0.0};
        std::array<double, 2> rnkss = {0.0, 0.0};
        std::array<double, 2> rnkss_m = {0.0, 0.0};
        std::array<double, 2> znksc = {0.0, 0.0};
        std::array<double, 2> znksc_m = {0.0, 0.0};
        std::array<double, 2> znkcs = {0.0, 0.0};
        std::array<double, 2> znkcs_m = {0.0, 0.0};
        std::array<double, 2> lnksc = {0.0, 0.0};
        std::array<double, 2> lnksc_m = {0.0, 0.0};
        std::array<double, 2> lnkcs = {0.0, 0.0};
        std::array<double, 2> lnkcs_m = {0.0, 0.0};

        std::array<double, 2> rcon_cc = {0.0, 0.0};
        std::array<double, 2> rcon_ss = {0.0, 0.0};
        std::array<double, 2> zcon_sc = {0.0, 0.0};
        std::array<double, 2> zcon_cs = {0.0, 0.0};

        for (int k = 0; k < s.nZeta; ++k) {
          int idx_kl = ((jF - r.nsMinF) * s.nThetaEff + l) * s.nZeta + k;

          double cosnv = fb.cosnv[n * s.nZeta + k];
          double sinnv = fb.sinnv[n * s.nZeta + k];
          double cosnvn = fb.cosnvn[n * s.nZeta + k];
          double sinnvn = fb.sinnvn[n * s.nZeta + k];

          if (jF < jMaxRZ) {
            rnkcc[m_evn] += d.armn_e[idx_kl] * cosnv;
            rnkcc_m[m_evn] += d.brmn_e[idx_kl] * cosnv;
            znksc[m_evn] += d.azmn_e[idx_kl] * cosnv;
            znksc_m[m_evn] += d.bzmn_e[idx_kl] * cosnv;
            rcon_cc[m_evn] += d.frcon_e[idx_kl] * cosnv;
            zcon_sc[m_evn] += d.fzcon_e[idx_kl] * cosnv;

            rnkcc[m_odd] += d.armn_o[idx_kl] * cosnv;
            rnkcc_m[m_odd] += d.brmn_o[idx_kl] * cosnv;
            znksc[m_odd] += d.azmn_o[idx_kl] * cosnv;
            znksc_m[m_odd] += d.bzmn_o[idx_kl] * cosnv;
            rcon_cc[m_odd] += d.frcon_o[idx_kl] * cosnv;
            zcon_sc[m_odd] += d.fzcon_o[idx_kl] * cosnv;
          }

          lnksc_m[m_evn] += d.blmn_e[idx_kl] * cosnv;
          lnksc_m[m_odd] += d.blmn_o[idx_kl] * cosnv;

          if (s.lthreed) {
            if (jF < jMaxRZ) {
              rnkcc[m_evn] += -d.crmn_e[idx_kl] * sinnvn;
              rnkss[m_evn] +=
                  d.armn_e[idx_kl] * sinnv - d.crmn_e[idx_kl] * cosnvn;
              rnkss_m[m_evn] += d.brmn_e[idx_kl] * sinnv;
              znksc[m_evn] += -d.czmn_e[idx_kl] * sinnvn;
              znkcs[m_evn] +=
                  d.azmn_e[idx_kl] * sinnv - d.czmn_e[idx_kl] * cosnvn;
              znkcs_m[m_evn] += d.bzmn_e[idx_kl] * sinnv;
              rcon_ss[m_evn] += d.frcon_e[idx_kl] * sinnv;
              zcon_cs[m_evn] += d.fzcon_e[idx_kl] * sinnv;

              rnkcc[m_odd] += -d.crmn_o[idx_kl] * sinnvn;
              rnkss[m_odd] +=
                  d.armn_o[idx_kl] * sinnv - d.crmn_o[idx_kl] * cosnvn;
              rnkss_m[m_odd] += d.brmn_o[idx_kl] * sinnv;
              znksc[m_odd] += -d.czmn_o[idx_kl] * sinnvn;
              znkcs[m_odd] +=
                  d.azmn_o[idx_kl] * sinnv - d.czmn_o[idx_kl] * cosnvn;
              znkcs_m[m_odd] += d.bzmn_o[idx_kl] * sinnv;
              rcon_ss[m_odd] += d.frcon_o[idx_kl] * sinnv;
              zcon_cs[m_odd] += d.fzcon_o[idx_kl] * sinnv;
            }

            lnksc[m_evn] += -d.clmn_e[idx_kl] * sinnvn;
            lnkcs[m_evn] += -d.clmn_e[idx_kl] * cosnvn;
            lnkcs_m[m_evn] += d.blmn_e[idx_kl] * sinnv;

            lnksc[m_odd] += -d.clmn_o[idx_kl] * sinnvn;
            lnkcs[m_odd] += -d.clmn_o[idx_kl] * cosnvn;
            lnkcs_m[m_odd] += d.blmn_o[idx_kl] * sinnv;
          }  // lthreed
        }  // k

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

          int idx_nm = ((jF - r.nsMinF) * (s.ntor + 1) + n) * s.mpol + m;

          double cosmui = fb.cosmui[l * (s.mnyq2 + 1) + m];
          double sinmui = fb.sinmui[l * (s.mnyq2 + 1) + m];
          double cosmumi = fb.cosmumi[l * (s.mnyq2 + 1) + m];
          double sinmumi = fb.sinmumi[l * (s.mnyq2 + 1) + m];

          if (jMinRZ <= jF && jF < jMaxRZ) {
            // assemble effective R and Z forces from MHD and spectral
            // condensation contributions
            double _rcc = rnkcc[m_parity] + d.xmpq[m] * rcon_cc[m_parity];
            double _zsc = znksc[m_parity] + d.xmpq[m] * zcon_sc[m_parity];

            m_physical_f.frcc[idx_nm] +=
                _rcc * cosmui + rnkcc_m[m_parity] * sinmumi;
            m_physical_f.fzsc[idx_nm] +=
                _zsc * sinmui + znksc_m[m_parity] * cosmumi;

            if (s.lthreed) {
              double _rss = rnkss[m_parity] + d.xmpq[m] * rcon_ss[m_parity];
              double _zcs = znkcs[m_parity] + d.xmpq[m] * zcon_cs[m_parity];

              m_physical_f.frss[idx_nm] +=
                  _rss * sinmui + rnkss_m[m_parity] * cosmumi;
              m_physical_f.fzcs[idx_nm] +=
                  _zcs * cosmui + znkcs_m[m_parity] * sinmumi;
            }  // lthreed
          }
          if (jMinL <= jF) {
            m_physical_f.flsc[idx_nm] += lnksc_m[m_parity] * cosmumi;
            if (s.lthreed) {
              m_physical_f.flsc[idx_nm] += lnksc[m_parity] * sinmui;
              m_physical_f.flcs[idx_nm] +=
                  lnkcs[m_parity] * cosmui + lnkcs_m[m_parity] * sinmumi;
            }  // lthreed
          }
        }  // m
      }  // l
    }  // n
  }  // jF
}
