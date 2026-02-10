#include "vmecpp_tools/dft_bench/fast_toroidal_vectorized.h"

#include <algorithm>  // std::min
#include <array>
#include <span>

#include "absl/algorithm/container.h"

static constexpr int m_evn = 0;
static constexpr int m_odd = 1;

// Notable modifications w.r.t. the original code:
// - one `#pragma omp barrier` was removed from the beginning of the function
// - input data is passed as an argument instead of being read from data members
void dft_bench::fast_toroidal_vectorized::ForcesToFourier3D(
    vmecpp::FourierForces& m_physical_f, const IdealMHDData& d,
    const Inputs& inputs) {
  const auto& [r, s, fb, fc, _1, _2] = inputs;

  // in here, we can safely assume lthreed == true

  // fill target force arrays with zeros
  m_physical_f.setZero();

  int jMaxRZ = std::min(r.nsMaxF, fc.ns - 1);
  if (fc.lfreeb && d.ivac >= 1) {
    // free-boundary: up to jMaxRZ=ns
    jMaxRZ = std::min(r.nsMaxF, fc.ns);
  }

  for (int jF = r.nsMinF; jF < jMaxRZ; ++jF) {
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

        std::array<double, 2> rcon_cc = {0.0, 0.0};
        std::array<double, 2> rcon_ss = {0.0, 0.0};
        std::array<double, 2> zcon_sc = {0.0, 0.0};
        std::array<double, 2> zcon_cs = {0.0, 0.0};

        // first even-m ...

        const int idx_kl_base = ((jF - r.nsMinF) * s.nThetaEff + l) * s.nZeta;
        const int sincos_offset = n * s.nZeta;

        // acc_prods(a, b, oa, ob, len) returns the sum of the products of the
        // elements of a and b: sum(a[oa + k] * b[ob + k] for k in 0..len). This
        // is a common operation performed in the hot loop of
        // dft_ForcesToFourier_3d_symm. However, it looks like gcc (up to 13.2.1
        // at least) is having trouble auto-vectorizing this operation because
        // of the accumulation into a single variable. Here we instead use a set
        // of N accumulation buffers that we can sum into with vectorized
        // fuse-multiply-add instructions.
        auto acc_prods = [offset_a = idx_kl_base, offset_b = sincos_offset,
                          len = s.nZeta](const std::span<const double> a,
                                         const std::span<const double> b) {
          // local accumulation buffers
          // AVX registers are typically 256 or 512 bit, so we'll do 4 doubles
          // at a time
          std::array<double, 4> buf{};

          int k = 0;
          for (; k + 3 < len; k += 4) {
            buf[0] += a[offset_a + k + 0] * b[offset_b + k + 0];
            buf[1] += a[offset_a + k + 1] * b[offset_b + k + 1];
            buf[2] += a[offset_a + k + 2] * b[offset_b + k + 2];
            buf[3] += a[offset_a + k + 3] * b[offset_b + k + 3];
          }

          // we need to run over the last nZeta % 4 elements
          if (k != len) [[unlikely]] {  // NOLINT(readability/braces)
            for (; k < len; ++k) {
              buf[0] += a[offset_a + k] * b[offset_b + k];
            }
          }

          return buf[0] + buf[1] + buf[2] + buf[3];
        };

        // first even-m ...
        rnkcc_m[m_evn] = acc_prods(d.brmn_e, fb.cosnv);
        rnkcc[m_evn] =
            acc_prods(d.armn_e, fb.cosnv) - acc_prods(d.crmn_e, fb.sinnvn);
        rnkss[m_evn] =
            acc_prods(d.armn_e, fb.sinnv) - acc_prods(d.crmn_e, fb.cosnvn);
        rnkss_m[m_evn] = acc_prods(d.brmn_e, fb.sinnv);
        znksc[m_evn] =
            acc_prods(d.azmn_e, fb.cosnv) - acc_prods(d.czmn_e, fb.sinnvn);
        znksc_m[m_evn] = acc_prods(d.bzmn_e, fb.cosnv);
        znkcs[m_evn] =
            acc_prods(d.azmn_e, fb.sinnv) - acc_prods(d.czmn_e, fb.cosnvn);
        znkcs_m[m_evn] = acc_prods(d.bzmn_e, fb.sinnv);
        rcon_cc[m_evn] = acc_prods(d.frcon_e, fb.cosnv);
        rcon_ss[m_evn] = acc_prods(d.frcon_e, fb.sinnv);
        zcon_sc[m_evn] = acc_prods(d.fzcon_e, fb.cosnv);
        zcon_cs[m_evn] = acc_prods(d.fzcon_e, fb.sinnv);

        // ... and now odd-m
        rnkcc[m_odd] =
            acc_prods(d.armn_o, fb.cosnv) - acc_prods(d.crmn_o, fb.sinnvn);
        rnkcc_m[m_odd] = acc_prods(d.brmn_o, fb.cosnv);
        rnkss[m_odd] =
            acc_prods(d.armn_o, fb.sinnv) - acc_prods(d.crmn_o, fb.cosnvn);
        rnkss_m[m_odd] = acc_prods(d.brmn_o, fb.sinnv);
        znksc[m_odd] =
            acc_prods(d.azmn_o, fb.cosnv) - acc_prods(d.czmn_o, fb.sinnvn);
        znksc_m[m_odd] = acc_prods(d.bzmn_o, fb.cosnv);
        znkcs[m_odd] =
            acc_prods(d.azmn_o, fb.sinnv) - acc_prods(d.czmn_o, fb.cosnvn);
        znkcs_m[m_odd] = acc_prods(d.bzmn_o, fb.sinnv);
        rcon_cc[m_odd] = acc_prods(d.frcon_o, fb.cosnv);
        rcon_ss[m_odd] = acc_prods(d.frcon_o, fb.sinnv);
        zcon_sc[m_odd] = acc_prods(d.fzcon_o, fb.cosnv);
        zcon_cs[m_odd] = acc_prods(d.fzcon_o, fb.sinnv);

        // maximum m depends on current surface:
        // --> axis R,Z gets only m=0 contributions
        // --> all other surfaces get the full Fourier spectrum
        int num_m = s.mpol;
        if (jF == 0) {
          // axis only gets m = 0
          num_m = 1;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_nm = ((jF - r.nsMinF) * (s.ntor + 1) + n) * s.mpol + m;
          const double cosmui = fb.cosmui[l * (s.mnyq2 + 1) + m];
          const double sinmumi = fb.sinmumi[l * (s.mnyq2 + 1) + m];
          // assemble effective R and Z forces from MHD and spectral
          // condensation contributions
          const double _rcc = rnkcc[m_parity] + d.xmpq[m] * rcon_cc[m_parity];
          m_physical_f.frcc[idx_nm] +=
              _rcc * cosmui + rnkcc_m[m_parity] * sinmumi;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_nm = ((jF - r.nsMinF) * (s.ntor + 1) + n) * s.mpol + m;
          const double sinmui = fb.sinmui[l * (s.mnyq2 + 1) + m];
          const double cosmumi = fb.cosmumi[l * (s.mnyq2 + 1) + m];
          // assemble effective R and Z forces from MHD and spectral
          // condensation contributions
          const double _rss = rnkss[m_parity] + d.xmpq[m] * rcon_ss[m_parity];
          m_physical_f.frss[idx_nm] +=
              _rss * sinmui + rnkss_m[m_parity] * cosmumi;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_nm = ((jF - r.nsMinF) * (s.ntor + 1) + n) * s.mpol + m;
          const double sinmui = fb.sinmui[l * (s.mnyq2 + 1) + m];
          const double cosmumi = fb.cosmumi[l * (s.mnyq2 + 1) + m];
          // assemble effective R and Z forces from MHD and spectral
          // condensation contributions
          const double _zsc = znksc[m_parity] + d.xmpq[m] * zcon_sc[m_parity];
          m_physical_f.fzsc[idx_nm] +=
              _zsc * sinmui + znksc_m[m_parity] * cosmumi;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_nm = ((jF - r.nsMinF) * (s.ntor + 1) + n) * s.mpol + m;
          const double cosmui = fb.cosmui[l * (s.mnyq2 + 1) + m];
          const double sinmumi = fb.sinmumi[l * (s.mnyq2 + 1) + m];
          // assemble effective R and Z forces from MHD and spectral
          // condensation contributions
          const double _zcs = znkcs[m_parity] + d.xmpq[m] * zcon_cs[m_parity];
          m_physical_f.fzcs[idx_nm] +=
              _zcs * cosmui + znkcs_m[m_parity] * sinmumi;
        }
      }  // l
    }  // n
  }  // jF

  // Do the lambda force coefficients separately, as they have different radial
  // ranges.

  // --> axis lambda stays zero (no contribution from any m)
  for (int jF = std::max(1, r.nsMinF); jF < r.nsMaxFIncludingLcfs; ++jF) {
    for (int n = 0; n < s.ntor + 1; ++n) {
      for (int l = 0; l < s.nThetaReduced; ++l) {
        std::array<double, 2> lnksc = {0.0, 0.0};
        std::array<double, 2> lnksc_m = {0.0, 0.0};
        std::array<double, 2> lnkcs = {0.0, 0.0};
        std::array<double, 2> lnkcs_m = {0.0, 0.0};

        const int idx_kl_base = ((jF - r.nsMinF) * s.nThetaEff + l) * s.nZeta;
        const int sincos_offset = n * s.nZeta;

        // see description of acc_prods above.
        // the definition is repeated here because extracting it to a helper
        // functor incurs in a visible performance overhead (gcc 13.2.1).
        auto acc_prods = [offset_a = idx_kl_base, offset_b = sincos_offset,
                          len = s.nZeta](const std::span<const double> a,
                                         const std::span<const double> b) {
          // local accumulation buffers
          // AVX registers are typically 256 or 512 bit, so we'll do 4 doubles
          // at a time
          std::array<double, 4> buf{};

          int k = 0;
          for (; k + 3 < len; k += 4) {
            buf[0] += a[offset_a + k + 0] * b[offset_b + k + 0];
            buf[1] += a[offset_a + k + 1] * b[offset_b + k + 1];
            buf[2] += a[offset_a + k + 2] * b[offset_b + k + 2];
            buf[3] += a[offset_a + k + 3] * b[offset_b + k + 3];
          }

          // we need to run over the last nZeta % 4 elements
          if (k != len) [[unlikely]] {  // NOLINT(readability/braces)
            for (; k < len; ++k) {
              buf[0] += a[offset_a + k] * b[offset_b + k];
            }
          }

          return buf[0] + buf[1] + buf[2] + buf[3];
        };

        lnksc_m[m_evn] = acc_prods(d.blmn_e, fb.cosnv);
        lnksc[m_evn] = -acc_prods(d.clmn_e, fb.sinnvn);

        lnkcs[m_evn] = -acc_prods(d.clmn_e, fb.cosnvn);
        lnkcs_m[m_evn] = acc_prods(d.blmn_e, fb.sinnv);

        lnksc_m[m_odd] = acc_prods(d.blmn_o, fb.cosnv);
        lnksc[m_odd] = -acc_prods(d.clmn_o, fb.sinnvn);

        lnkcs[m_odd] = -acc_prods(d.clmn_o, fb.cosnvn);
        lnkcs_m[m_odd] = acc_prods(d.blmn_o, fb.sinnv);

        // ---------- accumulation from all toroidal grid points k done
        // ---------------
        // ---------- now evaluate for poloidal modes (all m) ---------------

        for (int m = 0; m < s.mpol; ++m) {
          const int m_parity = m % 2;
          const int idx_nm = ((jF - r.nsMinF) * (s.ntor + 1) + n) * s.mpol + m;
          const double cosmumi = fb.cosmumi[l * (s.mnyq2 + 1) + m];
          const double sinmui = fb.sinmui[l * (s.mnyq2 + 1) + m];
          m_physical_f.flsc[idx_nm] +=
              lnksc_m[m_parity] * cosmumi + lnksc[m_parity] * sinmui;
        }

        for (int m = 0; m < s.mpol; ++m) {
          const int m_parity = m % 2;
          const int idx_nm = ((jF - r.nsMinF) * (s.ntor + 1) + n) * s.mpol + m;
          const double cosmui = fb.cosmui[l * (s.mnyq2 + 1) + m];
          const double sinmumi = fb.sinmumi[l * (s.mnyq2 + 1) + m];
          m_physical_f.flcs[idx_nm] +=
              lnkcs[m_parity] * cosmui + lnkcs_m[m_parity] * sinmumi;
        }  // m
      }  // l
    }  // n
  }  // jF
}

// Standalone version of IdealMHDModel::dft_FourierToReal_3d_symm
// Notable modifications:
// - one `#pragma omp barrier` was removed from around the beginning of the
// function
// - input data is passed as an argument instead of being read from
// IdealMHDModel data members
void dft_bench::fast_toroidal_vectorized::FourierToReal3D(
    dft_bench::IdealMHDData& m_mhd_data, const Inputs& inputs) {
  const auto& [radial_partitioning, s, fb, _, radial_profiles, physical_x] =
      inputs;

  // can safely assume lthreed == true in here

  const int num_realsp =
      (radial_partitioning.nsMaxF1 - radial_partitioning.nsMinF1) * s.nZnT;
  for (auto* v :
       {&m_mhd_data.r1_e, &m_mhd_data.r1_o, &m_mhd_data.ru_e, &m_mhd_data.ru_o,
        &m_mhd_data.rv_e, &m_mhd_data.rv_o, &m_mhd_data.z1_e, &m_mhd_data.z1_o,
        &m_mhd_data.zu_e, &m_mhd_data.zu_o, &m_mhd_data.zv_e, &m_mhd_data.zv_o,
        &m_mhd_data.lu_e, &m_mhd_data.lu_o, &m_mhd_data.lv_e,
        &m_mhd_data.lv_o}) {
    absl::c_fill_n(*v, num_realsp, 0);
  }

  int num_con =
      (radial_partitioning.nsMaxFIncludingLcfs - radial_partitioning.nsMinF) *
      s.nZnT;
  absl::c_fill_n(m_mhd_data.rCon, num_con, 0);
  absl::c_fill_n(m_mhd_data.zCon, num_con, 0);

  for (int jF = radial_partitioning.nsMinF1; jF < radial_partitioning.nsMaxF1;
       ++jF) {
    double* src_rcc =
        &(physical_x.rmncc[(jF - radial_partitioning.nsMinF1) * s.mnsize]);
    double* src_rss =
        &(physical_x.rmnss[(jF - radial_partitioning.nsMinF1) * s.mnsize]);
    double* src_zsc =
        &(physical_x.zmnsc[(jF - radial_partitioning.nsMinF1) * s.mnsize]);
    double* src_zcs =
        &(physical_x.zmncs[(jF - radial_partitioning.nsMinF1) * s.mnsize]);
    double* src_lsc =
        &(physical_x.lmnsc[(jF - radial_partitioning.nsMinF1) * s.mnsize]);
    double* src_lcs =
        &(physical_x.lmncs[(jF - radial_partitioning.nsMinF1) * s.mnsize]);

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

        // NOTE: The axis only gets contributions up to m=1.
        // This is counterintuitive on its own, since the axis is a
        // one-dimensional object, and thus has to poloidal variation of its
        // geometry. As far as we know, this has to do with the innermost
        // half-grid point for computing a better near-axis approximation of the
        // Jacobian.
        //
        // Regular case: all poloidal contributions up to m = mpol - 1.
        int num_m = s.mpol;
        if (jF == 0) {
          // axis: num_m = 2 -> m = 0, 1
          num_m = 2;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
          rnkcc[m_parity] += src_rcc[idx_mn] * cosmu;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double sinmum = fb.sinmum[l * (s.mnyq2 + 1) + m];
          rnkcc_m[m_parity] += src_rcc[idx_mn] * sinmum;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];
          rnkss[m_parity] += src_rss[idx_mn] * sinmu;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double cosmum = fb.cosmum[l * (s.mnyq2 + 1) + m];
          rnkss_m[m_parity] += src_rss[idx_mn] * cosmum;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];
          znksc[m_parity] += src_zsc[idx_mn] * sinmu;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double cosmum = fb.cosmum[l * (s.mnyq2 + 1) + m];
          znksc_m[m_parity] += src_zsc[idx_mn] * cosmum;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
          znkcs[m_parity] += src_zcs[idx_mn] * cosmu;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double sinmum = fb.sinmum[l * (s.mnyq2 + 1) + m];
          znkcs_m[m_parity] += src_zcs[idx_mn] * sinmum;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double cosmum = fb.cosmum[l * (s.mnyq2 + 1) + m];
          lnksc_m[m_parity] += src_lsc[idx_mn] * cosmum;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];
          lnksc[m_parity] += src_lsc[idx_mn] * sinmu;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
          lnkcs[m_parity] += src_lcs[idx_mn] * cosmu;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double sinmum = fb.sinmum[l * (s.mnyq2 + 1) + m];
          lnkcs_m[m_parity] += src_lcs[idx_mn] * sinmum;
        }

        // ---------- accumulation from poloidal modes (all m) done
        // ---------------
        // ---------- now evaluate for all toroidal grid points k
        // ---------------

        // first the even-m parts...

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.r1_e[idx_kl] +=
              rnkcc[m_evn] * cosnv + rnkss[m_evn] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.ru_e[idx_kl] +=
              rnkcc_m[m_evn] * cosnv + rnkss_m[m_evn] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double sinnvn = fb.sinnvn[n * s.nZeta + k];
          const double cosnvn = fb.cosnvn[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.rv_e[idx_kl] +=
              rnkcc[m_evn] * sinnvn + rnkss[m_evn] * cosnvn;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.z1_e[idx_kl] +=
              znksc[m_evn] * cosnv + znkcs[m_evn] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.zu_e[idx_kl] +=
              znksc_m[m_evn] * cosnv + znkcs_m[m_evn] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double sinnvn = fb.sinnvn[n * s.nZeta + k];
          const double cosnvn = fb.cosnvn[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.zv_e[idx_kl] +=
              znksc[m_evn] * sinnvn + znkcs[m_evn] * cosnvn;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.lu_e[idx_kl] +=
              lnksc_m[m_evn] * cosnv + lnkcs_m[m_evn] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double sinnvn = fb.sinnvn[n * s.nZeta + k];
          const double cosnvn = fb.cosnvn[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          // it is here that lv gets a negative sign!
          m_mhd_data.lv_e[idx_kl] -=
              lnksc[m_evn] * sinnvn + lnkcs[m_evn] * cosnvn;
        }

        // ... and now come the odd-m parts

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.r1_o[idx_kl] +=
              rnkcc[m_odd] * cosnv + rnkss[m_odd] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.ru_o[idx_kl] +=
              rnkcc_m[m_odd] * cosnv + rnkss_m[m_odd] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double sinnvn = fb.sinnvn[n * s.nZeta + k];
          const double cosnvn = fb.cosnvn[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.rv_o[idx_kl] +=
              rnkcc[m_odd] * sinnvn + rnkss[m_odd] * cosnvn;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.z1_o[idx_kl] +=
              znksc[m_odd] * cosnv + znkcs[m_odd] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.zu_o[idx_kl] +=
              znksc_m[m_odd] * cosnv + znkcs_m[m_odd] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double sinnvn = fb.sinnvn[n * s.nZeta + k];
          const double cosnvn = fb.cosnvn[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.zv_o[idx_kl] +=
              znksc[m_odd] * sinnvn + znkcs[m_odd] * cosnvn;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.lu_o[idx_kl] +=
              lnksc_m[m_odd] * cosnv + lnkcs_m[m_odd] * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double sinnvn = fb.sinnvn[n * s.nZeta + k];
          const double cosnvn = fb.cosnvn[n * s.nZeta + k];
          const int idx_kl =
              ((jF - radial_partitioning.nsMinF1) * s.nThetaEff + l) * s.nZeta +
              k;
          // it is here that lv gets a negative sign!
          m_mhd_data.lv_o[idx_kl] -=
              lnksc[m_odd] * sinnvn + lnkcs[m_odd] * cosnvn;
        }
      }  // l
    }  // n
  }  // j

  // The DFTs for rCon and zCon are done separately here,
  // since this allows to remove the condition on the radial range from the
  // innermost loops.

  for (int jF = radial_partitioning.nsMinF;
       jF < radial_partitioning.nsMaxFIncludingLcfs; ++jF) {
    double* src_rcc =
        &(physical_x.rmncc[(jF - radial_partitioning.nsMinF1) * s.mnsize]);
    double* src_rss =
        &(physical_x.rmnss[(jF - radial_partitioning.nsMinF1) * s.mnsize]);
    double* src_zsc =
        &(physical_x.zmnsc[(jF - radial_partitioning.nsMinF1) * s.mnsize]);
    double* src_zcs =
        &(physical_x.zmncs[(jF - radial_partitioning.nsMinF1) * s.mnsize]);

    for (int n = 0; n < s.ntor + 1; ++n) {
      for (int l = 0; l < s.nThetaReduced; ++l) {
        double rcon_cc = 0.0;
        double rcon_ss = 0.0;
        double zcon_sc = 0.0;
        double zcon_cs = 0.0;

        // NOTE: The axis only gets contributions up to m=1.
        // This is counterintuitive on its own, since the axis is a
        // one-dimensional object, and thus has to poloidal variation of its
        // geometry. As far as we know, this has to do with the innermost
        // half-grid point for computing a better near-axis approximation of the
        // Jacobian.
        //
        // Regular case: all poloidal contributions up to m = mpol - 1.
        int num_m = s.mpol;
        if (jF == 0) {
          // axis: num_m = 2 -> m = 0, 1
          num_m = 2;
        }

        /* The following code is ugly for a reason.

           Manually splitting the loops provides a 2x-3x speed-up w.r.t.
           a single big loop for `vmec_standalone w7x_ref_167_12_12.json`,
           running at lower thread counts, and should provide no performance
           regression otherwise.

           The reason for the speed-up is twofold:
           1) loop splitting avoids a 4K aliasing issue, e.g. between a store
              to zv_o[idx_kl] and an immediately successive read of
           lv_o[idx_kl]. 2) at least for gcc 13.2.1, the many fuse-multiply-add
           operations are auto-vectorized with loop splitting, and they are not
           otherwise.

           See git log for more information.
        */

        // In the following, we need to apply a scaling factor only for the
        // odd-parity m contributions:
        //   m_parity == m_odd(==1) --> * radial_profiles.sqrtSF[jF - r.nsMinF1]
        //   m_parity == m_evn(==0) --> * 1
        //
        // This expression is 0 if m_parity is 0 (=m_evn) and
        // radial_profiles.sqrtSF[jF - r.nsMinF1] if m_parity is 1 (==m_odd):
        //   m_parity * radial_profiles.sqrtSF[jF - r.nsMinF1]
        //
        // This expression is 1 if m_parity is 0 and 0 if m_parity is 1:
        //   (1 - m_parity)
        //
        // Hence, we can replace the following conditional statement:
        //   double scale = xmpq[m];
        //   if (m_parity == m_odd) {
        //       scale *= radial_profiles.sqrtSF[jF - r.nsMinF1];
        //   }
        // with the following code:
        //   const double scale = xmpq[m] * (1 - m_parity + m_parity *
        //   radial_profiles.sqrtSF[jF - r.nsMinF1]);

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double scale =
              m_mhd_data.xmpq[m] *
              (1 - m_parity +
               m_parity *
                   radial_profiles.sqrtSF[jF - radial_partitioning.nsMinF1]);
          const double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
          rcon_cc += src_rcc[idx_mn] * cosmu * scale;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double scale =
              m_mhd_data.xmpq[m] *
              (1 - m_parity +
               m_parity *
                   radial_profiles.sqrtSF[jF - radial_partitioning.nsMinF1]);
          const double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];
          rcon_ss += src_rss[idx_mn] * sinmu * scale;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double scale =
              m_mhd_data.xmpq[m] *
              (1 - m_parity +
               m_parity *
                   radial_profiles.sqrtSF[jF - radial_partitioning.nsMinF1]);
          const double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];
          zcon_sc += src_zsc[idx_mn] * sinmu * scale;
        }

        for (int m = 0; m < num_m; ++m) {
          const int m_parity = m % 2;
          const int idx_mn = n * s.mpol + m;
          const double scale =
              m_mhd_data.xmpq[m] *
              (1 - m_parity +
               m_parity *
                   radial_profiles.sqrtSF[jF - radial_partitioning.nsMinF1]);
          const double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
          zcon_cs += src_zcs[idx_mn] * cosmu * scale;
        }

        // ---------- accumulation from poloidal modes (all m) done
        // ---------------
        // ---------- now evaluate for all toroidal grid points k
        // ---------------

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_con =
              ((jF - radial_partitioning.nsMinF) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.rCon[idx_con] += rcon_cc * cosnv + rcon_ss * sinnv;
        }

        for (int k = 0; k < s.nZeta; ++k) {
          const double cosnv = fb.cosnv[n * s.nZeta + k];
          const double sinnv = fb.sinnv[n * s.nZeta + k];
          const int idx_con =
              ((jF - radial_partitioning.nsMinF) * s.nThetaEff + l) * s.nZeta +
              k;
          m_mhd_data.zCon[idx_con] += zcon_sc * cosnv + zcon_cs * sinnv;
        }
      }  // l
    }  // n
  }  // j
}  // NOLINT(readability/fn_size)
