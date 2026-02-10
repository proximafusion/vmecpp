#include <random>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"  // IsCloseRelAbs
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp_tools/dft_bench/fast_poloidal.h"
#include "vmecpp_tools/dft_bench/fast_toroidal.h"
#include "vmecpp_tools/dft_bench/fast_toroidal_vectorized.h"
#include "vmecpp_tools/dft_bench/ideal_mhd_data.h"
#include "vmecpp_tools/dft_bench/inputs.h"

static constexpr const char *config_path =
    "alphasim_core/test_data/vmecpp/cma.json";

using dft_bench::IdealMHDData;
using dft_bench::Inputs;

enum class DataOrder { FAST_ZETA_M, FAST_THETA_N };

void CheckAllClose(vmecpp::FourierForces &forces_a,
                   vmecpp::FourierForces &forces_b, const Inputs &inputs,
                   DataOrder order, double tolerance) {
  const std::size_t size = forces_a.frcc.size();
  ASSERT_EQ(size, forces_a.frss.size());
  ASSERT_EQ(size, forces_a.fzsc.size());
  ASSERT_EQ(size, forces_a.fzcs.size());
  ASSERT_EQ(size, forces_a.flsc.size());
  ASSERT_EQ(size, forces_a.flcs.size());

  ASSERT_EQ(size, forces_b.frcc.size());
  ASSERT_EQ(size, forces_b.frss.size());
  ASSERT_EQ(size, forces_b.fzsc.size());
  ASSERT_EQ(size, forces_b.fzcs.size());
  ASSERT_EQ(size, forces_b.flsc.size());
  ASSERT_EQ(size, forces_b.flcs.size());

  const int nsMinF = inputs.radial_partitioning.nsMinF;
  const int nsMaxF = inputs.radial_partitioning.nsMaxF;

  const int mpol = inputs.sizes.mpol;
  const int ntor = inputs.sizes.ntor;

  for (int j = nsMinF; j < nsMaxF; ++j) {
    for (int m = 0; m < mpol; ++m) {
      for (int n = 0; n < ntor + 1; ++n) {
        int index_mn = 0;
        if (order == DataOrder::FAST_ZETA_M) {
          index_mn = ((j - nsMinF) * (ntor + 1) + n) * mpol + m;
        } else if (order == DataOrder::FAST_THETA_N) {
          index_mn = ((j - nsMinF) * mpol + m) * (ntor + 1) + n;
        } else {
          FAIL() << "unknown data order";
        }

        EXPECT_TRUE(testing::IsCloseRelAbs(forces_a.frcc[index_mn],
                                           forces_b.frcc[index_mn], tolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(forces_a.frss[index_mn],
                                           forces_b.frss[index_mn], tolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(forces_a.fzsc[index_mn],
                                           forces_b.fzsc[index_mn], tolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(forces_a.fzsc[index_mn],
                                           forces_b.fzsc[index_mn], tolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(forces_a.flsc[index_mn],
                                           forces_b.flsc[index_mn], tolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(forces_a.flcs[index_mn],
                                           forces_b.flcs[index_mn], tolerance));
      }
    }
  }
}

IdealMHDData InitializeRealspaceForces(
    const vmecpp::Sizes &sizes,
    const vmecpp::RadialPartitioning &radial_partitioning,
    const DataOrder &data_order) {
  IdealMHDData mhd_data(sizes, radial_partitioning);

  // Standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  const int nsMinF = radial_partitioning.nsMinF;
  const int nsMaxF = radial_partitioning.nsMaxF;
  const int nsMaxFB = radial_partitioning.nsMaxFIncludingLcfs;

  const int nTheta = sizes.nThetaReduced;
  const int nZeta = sizes.nZeta;

  for (int j = nsMinF; j < nsMaxF; ++j) {
    for (int l = 0; l < nTheta; ++l) {
      for (int k = 0; k < nZeta; ++k) {
        // No need to rush things and optimize for data layout
        // in the unit test that only checks for correctness.
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        // uniform random data between -1 and +1
        mhd_data.armn_e[index_kl] = dist(rng);
        mhd_data.armn_o[index_kl] = dist(rng);
        mhd_data.brmn_e[index_kl] = dist(rng);
        mhd_data.brmn_o[index_kl] = dist(rng);
        mhd_data.crmn_e[index_kl] = dist(rng);
        mhd_data.crmn_o[index_kl] = dist(rng);

        mhd_data.azmn_e[index_kl] = dist(rng);
        mhd_data.azmn_o[index_kl] = dist(rng);
        mhd_data.bzmn_e[index_kl] = dist(rng);
        mhd_data.bzmn_o[index_kl] = dist(rng);
        mhd_data.czmn_e[index_kl] = dist(rng);
        mhd_data.czmn_o[index_kl] = dist(rng);

        mhd_data.frcon_e[index_kl] = dist(rng);
        mhd_data.frcon_o[index_kl] = dist(rng);

        mhd_data.fzcon_e[index_kl] = dist(rng);
        mhd_data.fzcon_o[index_kl] = dist(rng);
      }  // k
    }  // l
  }  // j

  for (int j = nsMinF; j < nsMaxFB; ++j) {
    for (int l = 0; l < nTheta; ++l) {
      for (int k = 0; k < nZeta; ++k) {
        // No need to rush things and optimize for data layout
        // in the unit test that only checks for correctness.
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        // uniform random data between -1 and +1
        mhd_data.blmn_e[index_kl] = dist(rng);
        mhd_data.blmn_o[index_kl] = dist(rng);
        mhd_data.clmn_e[index_kl] = dist(rng);
        mhd_data.clmn_o[index_kl] = dist(rng);
      }  // k
    }  // l
  }  // j

  return mhd_data;
}

void ReferenceForcesToFourier3D(vmecpp::FourierForces &m_physical_f,
                                const Inputs &inputs, const IdealMHDData &d,
                                const DataOrder &data_order) {
  const auto &[r, s, fb, fc, _1, _2] = inputs;

  // target storage always needs to be initialized to zero
  m_physical_f.setZero();

  // for all flux surfaces ...
  for (int j = r.nsMinF; j < r.nsMaxF; ++j) {
    // ... for all force Fourier coefficients on that surface
    for (int m = 0; m < s.mpol; ++m) {
      for (int n = 0; n < s.ntor + 1; ++n) {
        int index_mn = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_mn = ((j - r.nsMinF) * (s.ntor + 1) + n) * s.mpol + m;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_mn = ((j - r.nsMinF) * s.mpol + m) * (s.ntor + 1) + n;
        } else {
          FAIL() << "unknown data order";
        }

        // ... compute surface integrals for Fourier coefficients
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaReduced; ++l) {
            // No need to rush things and optimize for data layout
            // in the unit test that only checks for correctness.
            int index_kl = 0;
            if (data_order == DataOrder::FAST_ZETA_M) {
              index_kl = ((j - r.nsMinF) * s.nThetaReduced + l) * s.nZeta + k;
            } else if (data_order == DataOrder::FAST_THETA_N) {
              index_kl = ((j - r.nsMinF) * s.nZeta + k) * s.nThetaReduced + l;
            } else {
              LOG(FATAL) << "unknown data order";
            }

            double a_r = 0.0;
            double b_r = 0.0;
            double c_r = 0.0;
            double a_z = 0.0;
            double b_z = 0.0;
            double c_z = 0.0;
            double b_l = 0.0;
            double c_l = 0.0;
            if (m % 2 == 0) {
              // even-m
              a_r = d.armn_e[index_kl] + d.xmpq[m] * d.frcon_e[index_kl];
              b_r = d.brmn_e[index_kl];
              c_r = d.crmn_e[index_kl];

              a_z = d.azmn_e[index_kl] + d.xmpq[m] * d.fzcon_e[index_kl];
              b_z = d.bzmn_e[index_kl];
              c_z = d.czmn_e[index_kl];

              b_l = d.blmn_e[index_kl];
              c_l = d.clmn_e[index_kl];
            } else {
              // odd-m
              a_r = d.armn_o[index_kl] + d.xmpq[m] * d.frcon_o[index_kl];
              b_r = d.brmn_o[index_kl];
              c_r = d.crmn_o[index_kl];

              a_z = d.azmn_o[index_kl] + d.xmpq[m] * d.fzcon_o[index_kl];
              b_z = d.bzmn_o[index_kl];
              c_z = d.czmn_o[index_kl];

              b_l = d.blmn_o[index_kl];
              c_l = d.clmn_o[index_kl];
            }

            const int idx_lm = l * (s.mnyq2 + 1) + m;
            const double cosmui = fb.cosmui[idx_lm];
            const double sinmui = fb.sinmui[idx_lm];
            const double cosmumi = fb.cosmumi[idx_lm];
            const double sinmumi = fb.sinmumi[idx_lm];

            const int idx_nk = n * s.nZeta + k;
            const double cosnv = fb.cosnv[idx_nk];
            const double sinnv = fb.sinnv[idx_nk];
            const double cosnvn = fb.cosnvn[idx_nk];
            const double sinnvn = fb.sinnvn[idx_nk];

            // R, Z: axis only gets m=0; all other surfaces get all m
            if ((j == 0 && m == 0) || j > 0) {
              m_physical_f.frcc[index_mn] +=
                  (a_r * cosmui + b_r * sinmumi) * cosnv -
                  c_r * cosmui * sinnvn;
              m_physical_f.frss[index_mn] +=
                  (a_r * sinmui + b_r * cosmumi) * sinnv -
                  c_r * sinmui * cosnvn;

              m_physical_f.fzsc[index_mn] +=
                  (a_z * sinmui + b_z * cosmumi) * cosnv -
                  c_z * sinmui * sinnvn;
              m_physical_f.fzcs[index_mn] +=
                  (a_z * cosmui + b_z * sinmumi) * sinnv -
                  c_z * cosmui * cosnvn;
            }

            // lambda: zero on axis -> skip for j=0
            if (j > 0) {
              m_physical_f.flsc[index_mn] +=
                  b_l * cosmumi * cosnv - c_l * sinmui * sinnvn;
              m_physical_f.flcs[index_mn] +=
                  b_l * sinmumi * sinnv - c_l * cosmui * cosnvn;
            }
          }  // l
        }  // k
      }  // n
    }  // m
  }  // j
}  // RefereceForcesToFourier3D

void InitializeFourierGeometry(const DataOrder &data_order, Inputs &m_inputs) {
  auto &[r, s, fb, fc, p, m_fg] = m_inputs;

  // Standard mersenne_twister_engine seeded with rd()
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  const int nsMinF1 = r.nsMinF1;
  const int nsMaxF1 = r.nsMaxF1;

  const int mpol = s.mpol;
  const int ntor = s.ntor;

  for (int j = nsMinF1; j < nsMaxF1; ++j) {
    for (int m = 0; m < mpol; ++m) {
      for (int n = 0; n < ntor + 1; ++n) {
        // No need to rush things and optimize for data layout
        // in the unit test that only checks for correctness.
        int index_mn = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_mn = ((j - nsMinF1) * (ntor + 1) + n) * mpol + m;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_mn = ((j - nsMinF1) * mpol + m) * (ntor + 1) + n;
        } else {
          FAIL() << "unknown data order";
        }

        // uniform random data between -1 and +1
        m_fg.rmncc[index_mn] = dist(rng);
        m_fg.rmnss[index_mn] = dist(rng);
        m_fg.zmnsc[index_mn] = dist(rng);
        m_fg.zmncs[index_mn] = dist(rng);
        m_fg.lmnsc[index_mn] = dist(rng);
        m_fg.lmncs[index_mn] = dist(rng);
      }  // n
    }  // m
  }  // j
}  // InitializeFourierGeometry

void ReferenceFourierToReal3D(dft_bench::IdealMHDData &m_mhd_data,
                              const Inputs &inputs,
                              const DataOrder &data_order) {
  const auto &[r, s, fb, _, p, x] = inputs;

  const int nsMinF1 = r.nsMinF1;
  const int nsMaxF1 = r.nsMaxF1;
  const int nsMinF = r.nsMinF;
  const int nsMaxF = r.nsMaxF;
  const int nsMaxFB = r.nsMaxFIncludingLcfs;

  const int nTheta = s.nThetaReduced;
  const int nZeta = s.nZeta;

  const int mpol = s.mpol;
  const int ntor = s.ntor;

  // zero target storage
  const int num_realsp = (nsMaxF1 - nsMinF1) * s.nZnT;
  for (auto *v : {&m_mhd_data.r1_e, &m_mhd_data.r1_o, &m_mhd_data.ru_e,
                  &m_mhd_data.ru_o, &m_mhd_data.rv_e, &m_mhd_data.rv_o,
                  &m_mhd_data.z1_e, &m_mhd_data.z1_o, &m_mhd_data.zu_e,
                  &m_mhd_data.zu_o, &m_mhd_data.zv_e, &m_mhd_data.zv_o}) {
    absl::c_fill_n(*v, num_realsp, 0);
  }

  const int num_lambda = (nsMaxFB - nsMinF) * s.nZnT;
  for (auto *v : {&m_mhd_data.lu_e, &m_mhd_data.lu_o, &m_mhd_data.lv_e,
                  &m_mhd_data.lv_o}) {
    absl::c_fill_n(*v, num_lambda, 0);
  }

  const int num_con = (nsMaxF - nsMinF) * s.nZnT;
  for (auto *v : {&m_mhd_data.rCon, &m_mhd_data.zCon}) {
    absl::c_fill_n(*v, num_con, 0);
  }

  // for all flux surfaces ...
  for (int j = nsMinF1; j < nsMaxF1; ++j) {
    // ... and all grid point on that surface...
    for (int k = 0; k < nZeta; ++k) {
      for (int l = 0; l < nTheta; ++l) {
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF1) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF1) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        int index_kl_local = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl_local = ((j - nsMinF) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl_local = ((j - nsMinF) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        // ... add in the contributions from all Fourier coefficients

        // axis only get m = 0 and m = 1
        // all other surfaces get all coefficients
        int num_m = mpol;
        if (j == 0) {
          num_m = 2;
        }

        for (int m = 0; m < num_m; ++m) {
          for (int n = 0; n < ntor + 1; ++n) {
            int index_mn = 0;
            if (data_order == DataOrder::FAST_ZETA_M) {
              index_mn = ((j - nsMinF1) * (ntor + 1) + n) * mpol + m;
            } else if (data_order == DataOrder::FAST_THETA_N) {
              index_mn = ((j - nsMinF1) * mpol + m) * (ntor + 1) + n;
            } else {
              FAIL() << "unknown data order";
            }

            const int idx_lm = l * (s.mnyq2 + 1) + m;
            const double cosmu = fb.cosmu[idx_lm];
            const double sinmu = fb.sinmu[idx_lm];
            const double cosmum = fb.cosmum[idx_lm];
            const double sinmum = fb.sinmum[idx_lm];

            const int idx_nk = n * s.nZeta + k;
            const double cosnv = fb.cosnv[idx_nk];
            const double sinnv = fb.sinnv[idx_nk];
            const double cosnvn = fb.cosnvn[idx_nk];
            const double sinnvn = fb.sinnvn[idx_nk];

            const double r1 = x.rmncc[index_mn] * cosmu * cosnv +
                              x.rmnss[index_mn] * sinmu * sinnv;
            const double z1 = x.zmnsc[index_mn] * sinmu * cosnv +
                              x.zmncs[index_mn] * cosmu * sinnv;

            // R, Z: nsMinF1 ... nsMaxF1 -> whole radial loop
            if (m % 2 == 0) {
              // even-m

              m_mhd_data.r1_e[index_kl] += r1;
              m_mhd_data.ru_e[index_kl] += x.rmncc[index_mn] * sinmum * cosnv +
                                           x.rmnss[index_mn] * cosmum * sinnv;
              m_mhd_data.rv_e[index_kl] += x.rmncc[index_mn] * cosmu * sinnvn +
                                           x.rmnss[index_mn] * sinmu * cosnvn;

              m_mhd_data.z1_e[index_kl] += z1;
              m_mhd_data.zu_e[index_kl] += x.zmnsc[index_mn] * cosmum * cosnv +
                                           x.zmncs[index_mn] * sinmum * sinnv;
              m_mhd_data.zv_e[index_kl] += x.zmnsc[index_mn] * sinmu * sinnvn +
                                           x.zmncs[index_mn] * cosmu * cosnvn;
            } else {
              // odd-m

              m_mhd_data.r1_o[index_kl] += r1;
              m_mhd_data.ru_o[index_kl] += x.rmncc[index_mn] * sinmum * cosnv +
                                           x.rmnss[index_mn] * cosmum * sinnv;
              m_mhd_data.rv_o[index_kl] += x.rmncc[index_mn] * cosmu * sinnvn +
                                           x.rmnss[index_mn] * sinmu * cosnvn;

              m_mhd_data.z1_o[index_kl] += z1;
              m_mhd_data.zu_o[index_kl] += x.zmnsc[index_mn] * cosmum * cosnv +
                                           x.zmncs[index_mn] * sinmum * sinnv;
              m_mhd_data.zv_o[index_kl] += x.zmnsc[index_mn] * sinmu * sinnvn +
                                           x.zmncs[index_mn] * cosmu * cosnvn;
            }

            // lambda: nsMinF ... nsMaxFB
            // lambda forces are local per flux surface
            if (nsMinF <= j && j < nsMaxFB) {
              if (m % 2 == 0) {
                // even-m
                m_mhd_data.lu_e[index_kl] +=
                    x.lmnsc[index_mn] * cosmum * cosnv +
                    x.lmncs[index_mn] * sinmum * sinnv;
                // NOTE: lv has a negative sign built-in here
                m_mhd_data.lv_e[index_kl] -=
                    x.lmnsc[index_mn] * sinmu * sinnvn +
                    x.lmncs[index_mn] * cosmu * cosnvn;
              } else {
                // odd-m
                m_mhd_data.lu_o[index_kl] +=
                    x.lmnsc[index_mn] * cosmum * cosnv +
                    x.lmncs[index_mn] * sinmum * sinnv;
                // NOTE: lv has a negative sign built-in here
                m_mhd_data.lv_o[index_kl] -=
                    x.lmnsc[index_mn] * sinmu * sinnvn +
                    x.lmncs[index_mn] * cosmu * cosnvn;
              }
            }

            // r/zCon: nsMinF ... nsMaxF
            // spectral condensation is local per flux surface
            if (nsMinF <= j && j < nsMaxF) {
              const double rCon = r1 * m_mhd_data.xmpq[m];
              const double zCon = z1 * m_mhd_data.xmpq[m];

              if (m % 2 == 0) {
                // even-m
                m_mhd_data.rCon[index_kl_local] += rCon;
                m_mhd_data.zCon[index_kl_local] += zCon;
              } else {
                // odd-m
                m_mhd_data.rCon[index_kl_local] += rCon * p.sqrtSF[j - nsMinF1];
                m_mhd_data.zCon[index_kl_local] += zCon * p.sqrtSF[j - nsMinF1];
              }
            }
          }  // n
        }  // m
      }  // l
    }  // k
  }  // j
}  // ReferenceFourierToReal3D

TEST(ForcesToFourier, FastToroidal) {
  const Inputs inputs = Inputs::FromConfig(config_path);
  const DataOrder data_order = DataOrder::FAST_ZETA_M;

  const IdealMHDData ideal_mhd_data = InitializeRealspaceForces(
      inputs.sizes, inputs.radial_partitioning, data_order);

  // implementation under test
  vmecpp::FourierForces output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);
  dft_bench::fast_toroidal::ForcesToFourier3D(output_forces, ideal_mhd_data,
                                              inputs);

  // reference implementation
  vmecpp::FourierForces reference_output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);
  ReferenceForcesToFourier3D(reference_output_forces, inputs, ideal_mhd_data,
                             data_order);

  CheckAllClose(output_forces, reference_output_forces, inputs, data_order,
                /*tolerance=*/1e-12);
}

TEST(ForcesToFourier, FastToroidalVectorized) {
  const Inputs inputs = Inputs::FromConfig(config_path);
  const DataOrder data_order = DataOrder::FAST_ZETA_M;

  const IdealMHDData ideal_mhd_data = InitializeRealspaceForces(
      inputs.sizes, inputs.radial_partitioning, data_order);

  // implementation under test
  vmecpp::FourierForces output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);
  dft_bench::fast_toroidal_vectorized::ForcesToFourier3D(
      output_forces, ideal_mhd_data, inputs);

  // reference implementation
  vmecpp::FourierForces reference_output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);
  ReferenceForcesToFourier3D(reference_output_forces, inputs, ideal_mhd_data,
                             data_order);

  CheckAllClose(output_forces, reference_output_forces, inputs, data_order,
                /*tolerance=*/1e-12);
}

TEST(ForcesToFourier, FastPoloidal) {
  const auto inputs = Inputs::FromConfig(config_path);
  const DataOrder data_order = DataOrder::FAST_THETA_N;

  const IdealMHDData ideal_mhd_data = InitializeRealspaceForces(
      inputs.sizes, inputs.radial_partitioning, data_order);

  // implementation under test
  vmecpp::FourierForces output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);

  dft_bench::fast_poloidal::ForcesToFourier3D(output_forces, ideal_mhd_data,
                                              inputs);

  // reference implementation
  vmecpp::FourierForces reference_output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);
  ReferenceForcesToFourier3D(reference_output_forces, inputs, ideal_mhd_data,
                             data_order);

  CheckAllClose(output_forces, reference_output_forces, inputs, data_order,
                /*tolerance*/ 1e-12);
}

TEST(FourierToReal, FastToroidalVectorized) {
  const DataOrder data_order = DataOrder::FAST_ZETA_M;

  auto inputs = dft_bench::Inputs::FromConfig(config_path);
  InitializeFourierGeometry(data_order, inputs);

  dft_bench::IdealMHDData mhd_data(inputs.sizes, inputs.radial_partitioning);
  dft_bench::fast_toroidal_vectorized::FourierToReal3D(mhd_data, inputs);

  dft_bench::IdealMHDData reference_mhd_data(inputs.sizes,
                                             inputs.radial_partitioning);
  ReferenceFourierToReal3D(reference_mhd_data, inputs, data_order);

  const int nsMinF1 = inputs.radial_partitioning.nsMinF1;
  const int nsMaxF1 = inputs.radial_partitioning.nsMaxF1;

  const int nsMinF = inputs.radial_partitioning.nsMinF;
  const int nsMaxF = inputs.radial_partitioning.nsMaxF;
  const int nsMaxFB = inputs.radial_partitioning.nsMaxFIncludingLcfs;

  const int nTheta = inputs.sizes.nThetaReduced;
  const int nZeta = inputs.sizes.nZeta;

  const double kTolerance = 1.0e-12;

  for (int j = nsMinF1; j < nsMaxF1; ++j) {
    for (int k = 0; k < nZeta; ++k) {
      for (int l = 0; l < nTheta; ++l) {
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF1) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF1) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.r1_e[index_kl],
                                           mhd_data.r1_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.r1_o[index_kl],
                                           mhd_data.r1_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.ru_e[index_kl],
                                           mhd_data.ru_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.ru_o[index_kl],
                                           mhd_data.ru_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.rv_e[index_kl],
                                           mhd_data.rv_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.rv_o[index_kl],
                                           mhd_data.rv_o[index_kl],
                                           kTolerance));

        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.z1_e[index_kl],
                                           mhd_data.z1_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.z1_o[index_kl],
                                           mhd_data.z1_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zu_e[index_kl],
                                           mhd_data.zu_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zu_o[index_kl],
                                           mhd_data.zu_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zv_e[index_kl],
                                           mhd_data.zv_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zv_o[index_kl],
                                           mhd_data.zv_o[index_kl],
                                           kTolerance));
      }
    }
  }

  for (int j = nsMinF; j < nsMaxFB; ++j) {
    for (int k = 0; k < nZeta; ++k) {
      for (int l = 0; l < nTheta; ++l) {
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF1) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF1) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.lu_e[index_kl],
                                           mhd_data.lu_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.lu_o[index_kl],
                                           mhd_data.lu_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.lv_e[index_kl],
                                           mhd_data.lv_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.lv_o[index_kl],
                                           mhd_data.lv_o[index_kl],
                                           kTolerance));
      }
    }
  }

  for (int j = nsMinF; j < nsMaxF; ++j) {
    for (int k = 0; k < nZeta; ++k) {
      for (int l = 0; l < nTheta; ++l) {
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF1) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF1) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.rCon[index_kl],
                                           mhd_data.rCon[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zCon[index_kl],
                                           mhd_data.zCon[index_kl],
                                           kTolerance));
      }
    }
  }
}

TEST(FourierToReal, FastPoloidal) {
  const DataOrder data_order = DataOrder::FAST_THETA_N;

  auto inputs = dft_bench::Inputs::FromConfig(config_path);
  InitializeFourierGeometry(data_order, inputs);

  dft_bench::IdealMHDData mhd_data(inputs.sizes, inputs.radial_partitioning);
  dft_bench::fast_poloidal::FourierToReal3D(mhd_data, inputs);

  dft_bench::IdealMHDData reference_mhd_data(inputs.sizes,
                                             inputs.radial_partitioning);
  ReferenceFourierToReal3D(reference_mhd_data, inputs, data_order);

  const int nsMinF1 = inputs.radial_partitioning.nsMinF1;
  const int nsMaxF1 = inputs.radial_partitioning.nsMaxF1;

  const int nsMinF = inputs.radial_partitioning.nsMinF;
  const int nsMaxF = inputs.radial_partitioning.nsMaxF;
  const int nsMaxFB = inputs.radial_partitioning.nsMaxFIncludingLcfs;

  const int nTheta = inputs.sizes.nThetaReduced;
  const int nZeta = inputs.sizes.nZeta;

  const double kTolerance = 1.0e-12;

  for (int j = nsMinF1; j < nsMaxF1; ++j) {
    for (int k = 0; k < nZeta; ++k) {
      for (int l = 0; l < nTheta; ++l) {
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF1) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF1) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.r1_e[index_kl],
                                           mhd_data.r1_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.r1_o[index_kl],
                                           mhd_data.r1_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.ru_e[index_kl],
                                           mhd_data.ru_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.ru_o[index_kl],
                                           mhd_data.ru_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.rv_e[index_kl],
                                           mhd_data.rv_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.rv_o[index_kl],
                                           mhd_data.rv_o[index_kl],
                                           kTolerance));

        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.z1_e[index_kl],
                                           mhd_data.z1_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.z1_o[index_kl],
                                           mhd_data.z1_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zu_e[index_kl],
                                           mhd_data.zu_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zu_o[index_kl],
                                           mhd_data.zu_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zv_e[index_kl],
                                           mhd_data.zv_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zv_o[index_kl],
                                           mhd_data.zv_o[index_kl],
                                           kTolerance));
      }
    }
  }

  for (int j = nsMinF; j < nsMaxFB; ++j) {
    for (int k = 0; k < nZeta; ++k) {
      for (int l = 0; l < nTheta; ++l) {
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF1) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF1) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.lu_e[index_kl],
                                           mhd_data.lu_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.lu_o[index_kl],
                                           mhd_data.lu_o[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.lv_e[index_kl],
                                           mhd_data.lv_e[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.lv_o[index_kl],
                                           mhd_data.lv_o[index_kl],
                                           kTolerance));
      }
    }
  }

  for (int j = nsMinF; j < nsMaxF; ++j) {
    for (int k = 0; k < nZeta; ++k) {
      for (int l = 0; l < nTheta; ++l) {
        int index_kl = 0;
        if (data_order == DataOrder::FAST_ZETA_M) {
          index_kl = ((j - nsMinF1) * nTheta + l) * nZeta + k;
        } else if (data_order == DataOrder::FAST_THETA_N) {
          index_kl = ((j - nsMinF1) * nZeta + k) * nTheta + l;
        } else {
          LOG(FATAL) << "unknown data order";
        }

        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.rCon[index_kl],
                                           mhd_data.rCon[index_kl],
                                           kTolerance));
        EXPECT_TRUE(testing::IsCloseRelAbs(reference_mhd_data.zCon[index_kl],
                                           mhd_data.zCon[index_kl],
                                           kTolerance));
      }
    }
  }
}
